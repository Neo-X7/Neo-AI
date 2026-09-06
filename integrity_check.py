from db import get_db
from lancedb_store import get_lance_table


def check():
    with get_db() as conn:
        sqlite_ids = {r["embedding_id"] for r in conn.execute("SELECT embedding_id FROM ai_history").fetchall() if r["embedding_id"]}
    t = get_lance_table()
    lance_ids = set(t.to_pandas()["embedding_id"])
    orphan_sqlite = sqlite_ids - lance_ids
    orphan_lance = lance_ids - sqlite_ids
    print(f"SQLite rows missing vector: {len(orphan_sqlite)}")
    print(f"LanceDB vectors missing row: {len(orphan_lance)}")
    return orphan_sqlite, orphan_lance


if __name__ == "__main__":
    check()