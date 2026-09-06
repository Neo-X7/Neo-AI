# migrate_entity_attrs.py
# Run once: python migrate_entity_attrs.py
import sqlite3

DB_PATH = "/home/yorichii/neo-AI/ai_history.db"

def migrate():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cols = [r[1] for r in conn.execute("PRAGMA table_info(entity_attrs)").fetchall()]

    if "version" not in cols:
        conn.execute("ALTER TABLE entity_attrs ADD COLUMN version INTEGER")
        # backfill: for each (entity_name, slot) group, assign version by insertion order (id ASC = oldest = version 1)
        rows = conn.execute(
            "SELECT id, entity_name, slot FROM entity_attrs ORDER BY entity_name, slot, id ASC"
        ).fetchall()
        counters = {}
        for row in rows:
            key = (row["entity_name"], row["slot"])
            counters[key] = counters.get(key, 0) + 1
            conn.execute(
                "UPDATE entity_attrs SET version = ? WHERE id = ?",
                (counters[key], row["id"])
            )
        print(f"[migrate] entity_attrs: backfilled version for {len(rows)} rows")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_locations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT NOT NULL,
            value TEXT NOT NULL,
            ts REAL NOT NULL,
            version INTEGER NOT NULL
        )
    """)
    print("[migrate] entity_locations: table ready")

    conn.commit()
    conn.close()
    print("[migrate] done")

if __name__ == "__main__":
    migrate()