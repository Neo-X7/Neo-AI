import sqlite3, os
from typing import Generator
from contextlib import contextmanager
from logger import ai_log_info

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
_DB_PATH = os.path.join(_BASE_DIR, "ai_history.db")

_schema_ready = False  # set True after initialise_db() runs once


def _connect() -> sqlite3.Connection:
    """Open the sqlite file. row_factory lets us do row["prompt"]
    instead of row[1]."""
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _create_tables(conn: sqlite3.Connection) -> None:
    """Create all tables if they don't exist yet. Safe to call every time.
    """
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ai_history(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt TEXT NOT NULL,
            response TEXT NOT NULL,
            timestamp NOT NULL,
            compressed_keywords TEXT,
            entities TEXT DEFAULT '',
            embedding_id TEXT,
            access_count INTEGER DEFAULT 0,
            last_accessed_at TEXT,
            source TEXT DEFAULT 'chat',
            retracted INTEGER DEFAULT 0
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_profile(
            entity_name TEXT PRIMARY KEY,
            attributes TEXT,
            last_updated TEXT
        )
    """)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS semantic_memory(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            topic TEXT,
            summary TEXT,
            source_ids TEXT,
            embedding_id TEXT,
            
            created_at TEXT,
            last_updated TEXT
        )
    """)

    # entity_attrs and entity_mentions are NOT created here. They're owned
    # by entity_memory.py's create_entity_tables(), which is what
    # neo.py's initialise() actually calls. Having the same table defined
    # in two places let them silently diverge (db.py's copy had a dead
    # 'superseded' column nothing ever read or wrote) — one source of
    # truth now. See entity_memory.py for the real schema.

    # Previously only created by migrate_entity_attrs.py, outside db.py's
    # control. Moved here so db.py is the single source of schema truth.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS entity_locations(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            entity_name TEXT NOT NULL,
            value TEXT NOT NULL,
            ts REAL NOT NULL,
            version INTEGER NOT NULL
        )
    """)


def _migrate_old_columns(conn: sqlite3.Connection) -> None:
    """One-time patch for databases created before these columns existed.
    If a column is already there, this does nothing. Safe to call every time.
    """
    cols = [row["name"] for row in conn.execute("PRAGMA table_info(ai_history)")]
    if "access_count" not in cols:
        conn.execute("ALTER TABLE ai_history ADD COLUMN access_count INTEGER DEFAULT 0")
    if "last_accessed_at" not in cols:
        conn.execute("ALTER TABLE ai_history ADD COLUMN last_accessed_at TEXT")
        conn.execute("UPDATE ai_history SET last_accessed_at = timestamp WHERE last_accessed_at IS NULL")
    if "entities" not in cols:
        conn.execute("ALTER TABLE ai_history ADD COLUMN entities TEXT DEFAULT ''")
        ai_log_info("Migrated: added 'entities' column to ai_history", level="INFO", module="MEMORY")
    if "source" not in cols:
        conn.execute("ALTER TABLE ai_history ADD COLUMN source TEXT DEFAULT 'chat'")


def initialise_db(conn: sqlite3.Connection) -> None:
    """Make sure the database is ready to use: WAL mode, all tables exist,
    old databases get missing columns patched in."""
    conn.execute("PRAGMA journal_mode=WAL")
    _create_tables(conn)
    _migrate_old_columns(conn)
    conn.commit()


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Use as: with get_db() as conn: ...
    Opens a connection and hands it to you. Schema setup (tables,
    migrations) only runs once per process — on the first call — not on
    every connection open, so this stays fast on repeated use.
    If your code block finishes with no error, changes are saved
    (commit). If it throws, changes are undone (rollback) and the error
    is re-raised. Connection is always closed after.
    """
    global _schema_ready
    conn = _connect()
    try:
        if not _schema_ready:
            initialise_db(conn)
            _schema_ready = True
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        ai_log_info("Rollback occurred", level="INFO", module="MEMORY")
        raise
    finally:
        conn.close()