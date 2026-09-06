from storage import connect_db
from neo_ollama import get_embedding
from lancedb_store import insert_vector, clear_all_vectors
clear_all_vectors()
conn = connect_db()
rows = conn.execute("SELECT embedding_id, compressed_keywords FROM ai_history WHERE compressed_keywords IS NOT NULL AND compressed_keywords != ''").fetchall()
rows = [dict(r) for r in rows]
conn.close()
print(f"Re-embedding {len(rows)} rows...")
for row in rows:
    vector = get_embedding(row["compressed_keywords"])
    insert_vector(row["embedding_id"], vector)
print("Done.")