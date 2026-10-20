import sqlite3
from datetime import datetime
import yake
from logger import log_info,ai_log_info
import uuid
from lancedb_store import insert_vector,search_with_scores
from neo_ollama import get_embedding,decide_action
from typing import Generator
from contextlib import contextmanager
import os
_BASE_DIR=os.path.dirname(os.path.abspath(__file__))
_DB_PATH=os.path.join(_BASE_DIR,"ai_history.db")
def connect_db()->sqlite3.Connection:
    """Open a SQLite connection to ai_history.db. Rows can be read by column name."""
    conn=sqlite3.connect(_DB_PATH)
    conn.row_factory=sqlite3.Row
    return conn
def initialise_db(conn : sqlite3.Connection)->None:
    """Enable WAL mode and create the ai_history and facts tables if they do not exist."""
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("""create table if not exists ai_history(id integer primary key autoincrement, prompt text not null, response text not null,timestamp not null,compressed_keywords text, embeddings_id text)""")
    conn.execute("""create table if not exists facts(id integer primary key autoincrement, fact_text text not null, created_at text not null, status text not null, embedding_id text, retracted_at text)""")
    conn.commit()
@contextmanager
def get_db()-> Generator[sqlite3.Connection, None, None]:
    """Context manager yielding an initialised connection. Commits on normal exit, rolls back on error, always closes."""
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
        
def add_fact(conn, fact_text : str)->int:
    """Insert an active fact and store its embedding in LanceDB. Returns the new row id.

    Raises ValueError for empty text. An embedding failure is logged and the row is kept.
    """

    timestamp=datetime.now().isoformat()
    embedding_id=str(uuid.uuid4())
    fact_text=fact_text.strip()
    status="active"
    if not fact_text:
        raise ValueError("Fact cannot be Empty")
    cursor=conn.execute("""INSERT into facts (fact_text,created_at,status,embedding_id) VALUES (?,?,?,?)""", (fact_text,timestamp,status,embedding_id))
    try:
        vector=get_embedding(fact_text)
        insert_vector(embedding_id,vector)
        ai_log_info(f"Embedding successful for {embedding_id}", level="INFO", module="MEMORY")
    except Exception as e:
        ai_log_info(f"Embedding failed for {embedding_id}: {e}", level="WARNING", module="MEMORY")
    return cursor.lastrowid

def list_active_facts(conn)->list[dict]:
    """Return all facts with status 'active', oldest first, as dicts."""
    cursor=conn.execute("""SELECT id, fact_text, created_at, embedding_id FROM facts WHERE status='active' ORDER BY id""")
    rows=cursor.fetchall()
    return [dict(row) for row in rows]

def retract_fact(conn, fact_id : int)->bool:
    """Mark an active fact as retracted. Returns True if a row was changed."""
    timestamp=datetime.now().isoformat()
    cursor=conn.execute("""
    UPDATE facts SET status='retracted', retracted_at=? WHERE id=?
    AND status='active'""",(timestamp,fact_id))
    return cursor.rowcount>0

def extract_keywords(text : str)->str:
    """Return up to 10 YAKE keywords as a comma-separated string, or '' for empty input."""
    if not text or not text.strip():
        return ""
    kw_extractor = yake.KeywordExtractor(lan="en", n=2, top=10)
    keywords = kw_extractor.extract_keywords(text)
    if not keywords:
        kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=10)
        keywords = kw_extractor.extract_keywords(text)
    return ", ".join([kw for kw, _ in keywords])

def save_message(conn, prompt : str, response : str)-> str:
    """Store a prompt/response pair in ai_history with its keywords, and embed the keywords. Returns nothing."""
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

def retrieve_similar(query_keywords:str)->list[dict]:
    """Return the active facts closest to the query as [{'fact_text': ...}], best match first.

    The query is the full prompt, not YAKE keywords. Results beyond the distance cutoff in
    search_similarity are dropped, so this can return []. Retracted facts never appear.
    """
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
        rows=conn.execute(f"""SELECT embedding_id, fact_text FROM facts WHERE embedding_id IN ({placeholders}) AND status='active'""",embedding_ids).fetchall()
        lookup={}
        for row in rows:
            lookup[row["embedding_id"]]=row["fact_text"]
        results=[]
        for eid in embedding_ids:
            if eid in lookup:
                results.append({"fact_text":lookup[eid]})
        return results
    finally:
        conn.close()

def find_candidate(conn,new_fact_text : str)->list[dict]:
    """Find active facts similar (cosine >= 0.5) to a new fact.

    Returns dicts with id, fact_text and similarity, or [] when none match. If the lookup
    raises, the error is logged and None is returned.
    """

    vector=get_embedding(new_fact_text)
    hits=search_with_scores(vector)
    store=[h for h in hits if h["similarity"]>=0.5]
    ids=[h["embedding_id"] for h in store]
    try:
        if not ids:
            return []
        placeholders=",".join("?"*len(ids))

        sim={h["embedding_id"] : h["similarity"] for h in store}

        rows=conn.execute(f"""SELECT id, fact_text, embedding_id FROM facts WHERE embedding_id IN ({placeholders}) AND status='active'""",ids).fetchall()

        result=[{"id" : row["id"], "fact_text" : row["fact_text"], "similarity"  :sim[row["embedding_id"]]}for row in rows]
        return result
    
    except Exception as e:
        ai_log_info(f"ERROR : {e}", level="INFO", module="MEMORY")

def apply_decision(conn, new_fact_text, decision : dict, candidates : list[dict])->str:
    """Carry out the gate's decision (ADD, UPDATE, DELETE or NONE) and return the action applied.

    Unknown actions, and UPDATE or DELETE targets that are not among the candidates, fall
    back to ADD. An UPDATE onto an identical fact becomes NONE.
    """
    action=decision.get("action")
    target_id=decision.get("target_id")
    try:
        target_id=int(target_id)
    except (TypeError,ValueError):
        target_id=None
    candidate_id=[c["id"] for c in candidates]
    if action not in ("ADD", "UPDATE", "DELETE", "NONE"):
        action=("ADD")
    if action in ("UPDATE","DELETE") and target_id not in candidate_id:
        action=("ADD")
    if action=="UPDATE":
        target_text = next((c.get("fact_text","")for c in candidates if c["id"] == target_id), "")
        if target_text.strip().lower() == new_fact_text.strip().lower():
            action = "NONE"
    if action==("ADD"):
        add_fact(conn,new_fact_text)
    elif action==("DELETE"):
        retract_fact(conn,target_id)
    elif action==("UPDATE"):
        retract_fact(conn,target_id)
        add_fact(conn, new_fact_text)
    return action

def process_fact(conn,new_fact_text : str)->str:
    """Write gate for one fact: find candidates, ask decide_action (plain ADD when there are none), apply the decision, and return the action."""

    candidates=find_candidate(conn,new_fact_text)
    if not candidates:
        decision={"action" : "ADD", "target_id" : None}
    else:
        decision=decide_action(new_fact_text,candidates)
    applied=apply_decision(conn, new_fact_text, decision, candidates)
    return applied