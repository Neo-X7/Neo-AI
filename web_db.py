import sqlite3
import json
import time
import os
from logger import log_info

_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE_DIR, "neo_web.db")


def init_web():
    """Creates the web_data table if it doesn't exist. Call once at
    startup. Stores cached web search results, separate from ai_history
    by design — keeps search-sourced data isolated from personal memory."""
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS web_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query TEXT NOT NULL,
            results TEXT NOT NULL,
            ts REAL NOT NULL,
            version INTEGER NOT NULL
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_web_query ON web_data(query, version DESC)")
    log_info("Web search module has been created", level="INFO", module="WEB SEARCH STORAGE")
    conn.commit()
    conn.close()


def _next_version(conn, query: str) -> int:
    """Same query searched again gets the next version number, instead
    of overwriting the previous result."""
    row = conn.execute(
        "SELECT MAX(version) FROM web_data WHERE query = ?", (query,)
    ).fetchone()
    return (row[0] or 0) + 1


def store_web_result(query: str, results: list[dict]):
    """Saves one search result set.unfiltered. No verification/fact-checking
    is run before this - that step (search_verify.py) was removed.
    Called from query_classifier.web_search_multi() 
    per search variant and is the sole write path into web_data"""
    conn = sqlite3.connect(DB_PATH)
    version = _next_version(conn, query)
    conn.execute(
        "INSERT INTO web_data (query, results, ts, version) VALUES (?, ?, ?, ?)",
        (query, json.dumps(results), time.time(), version))
    log_info("Web query has been stored", level="INFO", module="WEB SEARCH STORAGE")
    conn.commit()
    conn.close()