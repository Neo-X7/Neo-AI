import json
import uuid
from datetime import datetime

from sklearn.cluster import DBSCAN

from neo_ollama import get_embedding, simple_completion
from lancedb_store import insert_vector


def init_semantic_table(conn):
    """Create the semantic_memory table if it doesn't exist yet."""
    conn.execute("""CREATE TABLE IF NOT EXISTS semantic_memory(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        topic TEXT, summary TEXT, source_ids TEXT,
        embedding_id TEXT, created_at TEXT, last_updated TEXT)""")


def get_unconsolidated_rows(conn):
    """Fetch every ai_history row that hasn't been folded into a semantic_memory cluster yet."""
    query = """
        SELECT id, prompt, response, compressed_keywords, embedding_id
        FROM ai_history
        WHERE id NOT IN (
            SELECT json_each.value
            FROM semantic_memory, json_each(source_ids)
            WHERE source_ids IS NOT NULL
        )
    """
    return conn.execute(query).fetchall()


def embed_rows(rows):
    """Get one embedding vector per row. Uses compressed_keywords if present, else the raw prompt."""
    vectors = []
    for row in rows:
        text = row["compressed_keywords"] or row["prompt"]
        vectors.append(get_embedding(text))
    return vectors


def cluster_rows(rows, vectors, min_cluster_size, eps):
    """Group rows into clusters using DBSCAN. Rows labeled -1 (noise) are dropped."""
    labels = DBSCAN(eps=eps, min_samples=min_cluster_size, metric="cosine").fit_predict(vectors)

    clusters = {}
    for row, label in zip(rows, labels):
        if label == -1:
            continue
        clusters.setdefault(label, []).append(row)

    return clusters


def summarize_cluster(cluster_rows):
    """Ask the model to summarize a cluster of exchanges into one factual summary."""
    exchanges_text = "\n".join(f"{r['prompt']} -> {r['response']}" for r in cluster_rows)
    prompt = (
        "Summarize these exchanges. Reuse key terms/names verbatim. "
        f"Be factual, no embellishment:\n{exchanges_text}"
    )
    # This is a one-off summarization call, not a real chat turn — it does
    # not get logged to ai_history, and deliberately skips Neo's persona/
    # facts/search-clause machinery (see simple_completion in neo_ollama.py).
    return simple_completion(prompt)


def save_cluster_summary(conn, cluster_rows, summary):
    """Embed the summary, store its vector, and write the semantic_memory row."""
    embedding = get_embedding(summary)
    embedding_id = str(uuid.uuid4())
    insert_vector(embedding_id, embedding)

    source_ids = json.dumps([row["id"] for row in cluster_rows])
    now = datetime.now().isoformat()

    conn.execute(
        """INSERT INTO semantic_memory(topic, summary, source_ids, embedding_id, created_at, last_updated)
           VALUES (?,?,?,?,?,?)""",
        (None, summary, source_ids, embedding_id, now, now),
    )


def run_consolidation(conn, min_cluster_size=3, eps=0.3):
    """
    Main entry point. Finds unconsolidated ai_history rows, clusters similar ones together,
    summarizes each cluster, and saves the summaries to semantic_memory.
    Returns the number of clusters created.
    """
    init_semantic_table(conn)

    rows = get_unconsolidated_rows(conn)
    if len(rows) < min_cluster_size:
        return 0

    vectors = embed_rows(rows)
    clusters = cluster_rows(rows, vectors, min_cluster_size, eps)

    for rows_in_cluster in clusters.values():
        summary = summarize_cluster(rows_in_cluster)
        save_cluster_summary(conn, rows_in_cluster, summary)

    return len(clusters)