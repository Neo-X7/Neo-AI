import sqlite3
from datetime import datetime
import yake
from logger import log_info,ai_log_info
import uuid
from typing import Generator
from contextlib import contextmanager
import os
_BASE_DIR=os.path.dirname(os.path.abspath(__file__))
_DB_PATH=os.path.join(_BASE_DIR,"ai_history.db")
def connect_db()->sqlite3.Connection:
    conn=sqlite3.connect(_DB_PATH)
    conn.row_factory=sqlite3.Row
    return conn
def initialise_db(conn : sqlite3.Connection)->None:
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""create table if not exists ai_history(id integer primary key autoincrement, prompt text not null, response text not null,timestamp not null,compressed_keywords text, embeddings_id text)""")
    conn.commit()
@contextmanager
def get_db()-> Generator[sqlite3.Connection, None, None]:
    conn = connect_db()
    initialise_db(conn)
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        log_info("Rollback occured", level="INFO", module="MEMORY")
        raise e
    finally:
        conn.close()
def extract_keywords(text):
    if not text or not text.strip():
        return ""
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=10)
    keywords = kw_extractor.extract_keywords(text)
    if not keywords:
        kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=10)
        keywords = kw_extractor.extract_keywords(text)
    return ", ".join([kw for kw, score in keywords])
def save_message(conn, prompt, response):
    from lancedb_store import insert_vector
    from neo_ollama import get_embedding
    timestamp = datetime.now().isoformat()
    embedding_id = str(uuid.uuid4())
    compressed_keywords = extract_keywords(prompt + " " + response)
    conn.execute("""INSERT INTO ai_history (prompt, response, timestamp, compressed_keywords,embeddings_id) VALUES (?, ?, ?, ?,?)""", (prompt, response, timestamp, compressed_keywords, embedding_id))
    if compressed_keywords:
        try:
            vector = get_embedding(compressed_keywords)
            insert_vector(embedding_id, vector)
            ai_log_info(f"Embedding successful for {embedding_id}", level="INFO", module="MEMORY")
        except Exception as e:
            ai_log_info(f"Embedding failed for {embedding_id}: {e}", level="WARNING", module="MEMORY")
def retrieve_similar(query_keywords:str)->list:
    from lancedb_store import search_similarity
    from neo_ollama import get_embedding
    query_vector=get_embedding(query_keywords)
    embedding_ids=search_similarity(query_vector)
    if not embedding_ids:
        return []
    import re
    uuid_pattern=re.compile(r'^[0-9a-f-]{36}$')
    embedding_ids=[e for e in embedding_ids if uuid_pattern.match(e)]
    if not embedding_ids:
        return []
    placeholders=",".join("?"*len(embedding_ids))
    conn=connect_db()
    try:
        rows=conn.execute(f"""SELECT prompt, response FROM ai_history WHERE embeddings_id IN ({placeholders}) ORDER BY timestamp DESC""",embedding_ids).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()