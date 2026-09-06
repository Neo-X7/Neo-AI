import json
import numpy as np
from db import get_db
from neo_ollama import get_embedding


def _cosine_sim(a, b):
    """Cosine similarity between two vectors."""
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def _avg_pairwise_cosine(vectors):
    """Average cosine similarity across every pair of vectors. Measures how tight a cluster is."""
    if len(vectors) < 2:
        return 1.0
    sims = []
    for i in range(len(vectors)):
        for j in range(i + 1, len(vectors)):
            sims.append(_cosine_sim(vectors[i], vectors[j]))
    return sum(sims) / len(sims)


def get_cluster_rows(conn, source_ids):
    """Fetch the original ai_history rows that belong to one semantic_memory cluster."""
    placeholders = ",".join("?" * len(source_ids))
    query = f"SELECT prompt, response, compressed_keywords FROM ai_history WHERE id IN ({placeholders})"
    return conn.execute(query, source_ids).fetchall()


def score_cluster(cluster_row, source_rows):
    """
    Compute three scores for one cluster:
    - coherence: how similar the source rows are to each other
    - alignment: how well the summary matches the average (centroid) of the source rows
    - faithfulness_max: best-case match, the single source row the summary matches best
    - faithfulness_avg: overall match, the summary's average similarity across all source rows
    """
    vectors = [get_embedding(r["compressed_keywords"] or r["prompt"]) for r in source_rows]

    coherence = _avg_pairwise_cosine(vectors)

    centroid = np.mean(vectors, axis=0)
    summary_vec = get_embedding(cluster_row["summary"])
    alignment = _cosine_sim(summary_vec, centroid)

    per_row_sims = [_cosine_sim(summary_vec, v) for v in vectors]
    faithfulness_max = max(per_row_sims)
    faithfulness_avg = sum(per_row_sims) / len(per_row_sims)

    return {
        "id": cluster_row["id"],
        "coherence": round(coherence, 3),
        "alignment": round(alignment, 3),
        "faithfulness_max": round(faithfulness_max, 3),
        "faithfulness_avg": round(faithfulness_avg, 3),
    }


def eval_consolidation(conn):
    """Run coherence/alignment/faithfulness scoring over every cluster in semantic_memory."""
    clusters = conn.execute("SELECT id, source_ids, summary FROM semantic_memory").fetchall()

    results = []
    for cluster_row in clusters:
        source_ids = json.loads(cluster_row["source_ids"])
        source_rows = get_cluster_rows(conn, source_ids)
        results.append(score_cluster(cluster_row, source_rows))

    return results


if __name__ == "__main__":

    with get_db() as conn:
        results = eval_consolidation(conn)

    for r in results:
        print(r)

    passed = sum(
        1 for r in results
        if r["coherence"] > 0.7 and r["alignment"] > 0.7 and r["faithfulness_avg"] > 0.7
    )
    print(f"\n{passed}/{len(results)} clusters passed thresholds")