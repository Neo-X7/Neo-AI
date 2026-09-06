import uuid
from datetime import datetime

from db import get_db
from extraction import extract_keywords, process_message
from entity_memory import upsert_entity_profile, get_primary_entity
from retrieval import decay_score
from lancedb_store import insert_vector, delete_vector, insert_entity_vector
from neo_ollama import get_embedding
from logger import ai_log_info

# Tracks the last entity seen in this chat session, so entity extraction
# can resolve pronouns/follow-ups across messages. Reset only on restart.
_last_entity_state = {"value": None}


def save_message(conn, prompt, response, source="chat"):
    """Write one prompt/response pair to ai_history, plus its extracted
    entities and embeddings. source='chat' triggers entity extraction;
    other sources (e.g. search results) skip it."""
    timestamp = datetime.now().isoformat()
    embedding_id = str(uuid.uuid4())
    full_text = prompt + " " + response
    compressed_keywords = extract_keywords(full_text)

    if source != "chat":
        return
    primary_entity = get_primary_entity(conn)
    entities, profiles, new_last_entity = process_message(
        full_text,
        primary_entity=primary_entity,
        session_last_entity=_last_entity_state["value"],
        )
    _last_entity_state["value"] = new_last_entity

    conn.execute(
        "INSERT INTO ai_history(prompt, response, timestamp, compressed_keywords, "
        "entities, embedding_id, access_count, last_accessed_at, source) "
        "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?)",
        (prompt, response, timestamp, compressed_keywords, entities, embedding_id, timestamp, source),
    )

    if profiles:
        # NOTE: writes to entity_profile and write_gate.py
        #each fact is actually allowed to be written
        upsert_entity_profile(conn, profiles, source,primary_entity=primary_entity)

    if compressed_keywords:
        try:
            vector = get_embedding(compressed_keywords)
            insert_vector(embedding_id, vector)
        except Exception as e:
            ai_log_info(f"Embedding failed for {embedding_id}: {e}", level="WARNING", module="MEMORY")

    if entities:
        try:
            ent_vector = get_embedding(entities)
            row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            insert_entity_vector(row_id, ent_vector)
        except Exception as e:
            ai_log_info(f"Entity embedding failed: {e}", level="WARNING", module="MEMORY")


def forget(row_id: int) -> bool:
    """Delete one row from ai_history and its matching vector. Returns
    False if the row doesn't exist."""
    with get_db() as conn:
        row = conn.execute("SELECT embedding_id FROM ai_history WHERE id=?", (row_id,)).fetchone()
        if row is None:
            return False
        conn.execute("DELETE FROM ai_history WHERE id=?", (row_id,))

    if row["embedding_id"]:
        try:
            delete_vector(row["embedding_id"])
        except Exception as e:
            ai_log_info(f"Vector delete failed for {row['embedding_id']}: {e}", level="WARNING", module="MEMORY")

    ai_log_info(f"Forgot row {row_id}", level="INFO", module="MEMORY")
    return True


def purge_decayed(threshold: float = 0.1) -> int:
    """Delete every ai_history row whose decay_score has dropped below
    threshold (old + rarely accessed). Returns how many rows were purged."""
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, embedding_id, last_accessed_at, access_count FROM ai_history"
        ).fetchall()
        to_delete = [
            dict(r) for r in rows
            if decay_score(r["last_accessed_at"], r["access_count"]) < threshold
        ]
        ids = [r["id"] for r in to_delete]
        if ids:
            conn.execute(f"DELETE FROM ai_history WHERE id IN ({','.join('?' * len(ids))})", ids)

    for r in to_delete:
        if r["embedding_id"]:
            try:
                delete_vector(r["embedding_id"])
            except Exception as e:
                ai_log_info(f"Vector delete failed for {r['embedding_id']}: {e}", level="WARNING", module="MEMORY")

    ai_log_info(f"Purged {len(to_delete)} decayed rows", level="INFO", module="MEMORY")
    return len(to_delete)


def backfill_entity_vectors() -> int:
    """Re-embed and re-index the 'entities' column for every row that has
    one. Used after a schema/model change, to rebuild the entity vector
    index from what's already in ai_history."""
    with get_db() as conn:
        rows = conn.execute("SELECT id, entities FROM ai_history WHERE entities != ''").fetchall()

    for r in rows:
        v = get_embedding(r["entities"])
        insert_entity_vector(r["id"], v)

    return len(rows)


def backfill_entity_profiles() -> tuple[int, int]:
    """Wipe and rebuild entity_profile (and entity_attrs) from scratch by
    replaying every row in ai_history through process_message again.
    Returns (rows_processed, error_count)."""
    with get_db() as conn:
        primary_entity = get_primary_entity(conn)
        conn.execute("DELETE FROM entity_mentions")
        conn.execute("DELETE FROM entity_attrs")  # see note in save_message re: this table
        conn.execute("DELETE FROM entity_profile")
        rows = conn.execute("SELECT id, prompt, response FROM ai_history ORDER BY id ASC").fetchall()

    errors = []
    last_entity = None
    with get_db() as conn:
        for r in rows:
            try:
                _, profiles, last_entity = process_message(
                    r["prompt"] + " " + r["response"],
                    primary_entity=primary_entity,
                    session_last_entity=last_entity,
                )
                upsert_entity_profile(conn, profiles, source="chat", primary_entity=primary_entity)
            except Exception as e:
                errors.append((r["id"], str(e)))

    if errors:
        ai_log_info(f"Backfill errors: {errors}", level="WARNING", module="MEMORY")

    return len(rows), len(errors)