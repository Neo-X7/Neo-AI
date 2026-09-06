import re, math
from datetime import datetime

from lancedb_store import search_similarity, search_entity_vectors
from neo_ollama import get_embedding
from db import get_db
from entity_memory import resolve_entity_query, _cosine_sim


# ---------------------------------------------------------------------------
# Decay scoring — how relevant an old memory still is
# ---------------------------------------------------------------------------

def decay_score(last_accessed_at, access_count=0, half_life_days=30.0):
    """Score drops toward 0 the longer a memory goes untouched — halves
    every half_life_days. Frequently-accessed memories get a boost that
    resists decay (diminishing returns via log1p, so going from 1->2
    accesses matters more than 100->101). Used by storage.py's
    purge_decayed to decide what to forget."""
    if not last_accessed_at:
        last_accessed_at = datetime.now().isoformat()
    age_days = (datetime.now() - datetime.fromisoformat(last_accessed_at)).total_seconds() / 86400
    decay = math.exp(-math.log(2) * age_days / half_life_days)
    boost = 1 + math.log1p(access_count) * 0.3
    return decay * boost


def _semantic_to_row(s):
    """Reshapes a semantic_memory row (topic/summary) into the same dict
    shape as an ai_history row (prompt/response), so the rest of the
    retrieval pipeline can treat both identically."""
    return {
        "id": f"sem_{s['id']}", "prompt": s["topic"] or "", "response": s["summary"],
        "compressed_keywords": s["summary"], "entities": "", "embedding_id": s["embedding_id"],
        "timestamp": s["created_at"], "access_count": 0, "last_accessed_at": s["last_updated"],
    }


# ---------------------------------------------------------------------------
# Main retrieval
# ---------------------------------------------------------------------------

def retrieve_similar(query_keywords, top_k_search=25, top_k_final=4, raw_query=None):
    """Find the most relevant past memories for a query.

    Order of attempts:
    1. Direct fact lookup (e.g. "what's my age?") - fast path, skips
       everything else if it can answer directly.
    2. Entity-vector search - rows whose extracted entities are similar.
    3. General similarity search - rows whose keywords/content are similar.
    4. Semantic memory - condensed topic summaries, if any match.
    All non-direct candidates get merged, deduped, and reranked together.
    """
    embed_input = raw_query or query_keywords

    with get_db() as conn:
        direct = resolve_entity_query(embed_input, conn)
    if direct:
        return [{"prompt": embed_input, "response": direct, "compressed_keywords": "", "entities": direct,
                  "embedding_id": None, "timestamp": datetime.now().isoformat(), "access_count": 0,
                  "last_accessed_at": datetime.now().isoformat(), "id": -1}]

    query_vector = get_embedding(embed_input)

    # --- entity-vector search ---
    entity_ids = search_entity_vectors(query_vector, top_k=10)
    with get_db() as conn:
        entity_hits = _fetch_history_rows(conn, entity_ids) if entity_ids else []

    # --- general similarity search ---
    # search_similarity returns results already ordered best-to-worst by
    # LanceDB. We keep that order (embedding_ids) so we can record each
    # row's TRUE rank before anything gets merged with other sources below
    # — merge order is not the same as similarity rank, and scoring off
    # merge position was a bug (see vector_rank_by_id below).
    embedding_ids = search_similarity(query_vector, top_k=top_k_search)
    uuid_pattern = re.compile(r'^[0-9a-f-]{36}$')
    embedding_ids = [e for e in embedding_ids if e and uuid_pattern.match(e)]

    with get_db() as conn:
        vector_rows = _fetch_history_rows_by_embedding(conn, embedding_ids) if embedding_ids else []
        row_by_embedding_id = {r["embedding_id"]: r for r in vector_rows}
        ordered_vector = [row_by_embedding_id[eid] for eid in embedding_ids if eid in row_by_embedding_id]

        # True similarity rank, captured here before merging (0 = best match).
        # entity_hits and semantic rows didn't come from this search, so
        # they simply have no entry here — heuristic_rerank treats that as
        # "no vector rank signal available" rather than faking one.
        vector_rank_by_id = {r["id"]: i for i, r in enumerate(ordered_vector)}

        ids = [r["id"] for r in vector_rows]
        if ids:
            now = datetime.now().isoformat()
            conn.execute(
                f"UPDATE ai_history SET access_count = access_count + 1, last_accessed_at = ? WHERE id IN ({','.join('?' * len(ids))})",
                [now, *ids])

        sem_rows = _fetch_semantic_rows(conn, embedding_ids) if embedding_ids else []

    # --- merge and dedupe (entity hits first, then vector hits, then semantic) ---
    seen = set()
    combined = []
    for r in entity_hits + ordered_vector + [_semantic_to_row(s) for s in sem_rows]:
        if r["id"] not in seen:
            seen.add(r["id"])
            combined.append(r)

    if not combined:
        return [{"prompt": embed_input, "response": "Context not found.", "compressed_keywords": "",
                 "entities": "", "embedding_id": None, "timestamp": datetime.now().isoformat(),
                 "access_count": 0, "last_accessed_at": datetime.now().isoformat(), "id": -1}]

    row_vectors = {}
    for r in combined:
        if r["embedding_id"]:
            try:
                row_vectors[r["id"]] = get_embedding(r["compressed_keywords"] or r["entities"])
            except Exception:
                pass

    return heuristic_rerank(
        embed_input, combined, top_k_final,
        query_vector=query_vector, row_vectors=row_vectors,
        vector_rank_by_id=vector_rank_by_id, total_ranked=len(ordered_vector),
    )


def _fetch_history_rows(conn, ids):
    placeholders = ",".join("?" * len(ids))
    return conn.execute(
        f"SELECT id, prompt, response, compressed_keywords, entities, embedding_id, timestamp, access_count, last_accessed_at "
        f"FROM ai_history WHERE id IN ({placeholders}) AND retracted=0", ids
    ).fetchall()


def _fetch_history_rows_by_embedding(conn, embedding_ids):
    placeholders = ",".join("?" * len(embedding_ids))
    return conn.execute(
        f"SELECT id, prompt, response, compressed_keywords, entities, embedding_id, timestamp, access_count, last_accessed_at "
        f"FROM ai_history WHERE embedding_id IN ({placeholders}) AND retracted=0", embedding_ids
    ).fetchall()


def _fetch_semantic_rows(conn, embedding_ids):
    placeholders = ",".join("?" * len(embedding_ids))
    return conn.execute(
        f"SELECT id, topic, summary, embedding_id, created_at, last_updated "
        f"FROM semantic_memory WHERE embedding_id IN ({placeholders})", embedding_ids
    ).fetchall()


# ---------------------------------------------------------------------------
# Reranking
# ---------------------------------------------------------------------------

def heuristic_rerank(query_keywords, rows, top_k_final=4, query_vector=None, row_vectors=None,
                      vector_rank_by_id=None, total_ranked=0, wj=0.4, wc=0.2, wv=0.4):
    """Score each candidate row and return the top top_k_final.

    Three signals blended per row:
      jaccard (wj) - word overlap between query and row's keywords/entities
      cos     (wc) - cosine similarity between query and row embeddings
      vrank   (wv) - the row's TRUE similarity rank from the original vector
                      search (0 = best), if it came from that search at all.
                      Rows with no vector-search rank (entity/semantic hits)
                      get 0 here instead of a fake merge-position score —
                      previously this used merge-list position, which wrongly
                      favored whichever source got listed first.
    """
    vector_rank_by_id = vector_rank_by_id or {}
    query_words = set(query_keywords.lower().split())
    scored = []

    for row in rows:
        row_words = set((row["compressed_keywords"] or "").lower().replace(",", "").split())
        entity_words = set((row["entities"] or "").lower().replace(",", "").split())
        row_words |= entity_words
        jaccard = len(query_words & row_words) / len(query_words | row_words) if row_words else 0.0

        vector_rank_score = 0.0
        if row["id"] in vector_rank_by_id and total_ranked > 0:
            rank = vector_rank_by_id[row["id"]]
            vector_rank_score = (total_ranked - rank) / total_ranked

        cos = 0.0
        if query_vector is not None and row_vectors and row["id"] in row_vectors:
            cos = _cosine_sim(query_vector, row_vectors[row["id"]])

        combined = wj * jaccard + wc * cos + wv * vector_rank_score
        scored.append((row, combined))

    scored.sort(key=lambda x: x[1], reverse=True)
    return [row for row, _ in scored[:top_k_final]]