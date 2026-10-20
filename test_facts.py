import pytest
import storage
from storage import get_db, add_fact, list_active_facts,retract_fact,apply_decision, process_fact
import neo_ollama,lancedb_store
import neo
from neo import delete_memory
from neo_ollama import extract_facts,chat
from lancedb_store import get_lance_table

class FakeTable:
    def count_rows(self):
        return 2

    def search(self, vector):
        return self

    def limit(self, n):
        return self

    def to_list(self):
        return [
            {"embedding_id": "a", "_distance": 300},
            {"embedding_id": "b", "_distance": 600},
        ]

def test_storage(monkeypatch, tmp_path):

    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)

    texts=["I am Joe", "I love watching trains", "I do live in Australia"]

    with get_db() as conn:
        for text in texts:
            add_fact(conn,text)

    with get_db() as conn:
        rows=list_active_facts(conn)
    assert len(rows)==3
    assert[row["fact_text"] for row in rows]==texts
    assert calls==[row["embedding_id"]for row in rows]

def test_correct_one_fact(monkeypatch, tmp_path):

    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)

    texts=["I am Joe", "I love watching trains", "I live in Australia"]

    with get_db() as conn:
        ids=[add_fact(conn,t) for t in texts]
        retract_fact(conn,ids[1])

    with get_db() as conn:
        assert retract_fact(conn, ids[1])==False

    with get_db() as conn:
        rows=list_active_facts(conn)
        gone=conn.execute("SELECT status, retracted_at FROM facts WHERE id =?", (ids[1],)).fetchone()

    assert [row["fact_text"]for row in rows]==[texts[0],texts[2]]
    assert gone["status"]=="retracted"
    assert gone["retracted_at"] is not None

def test_retract_survives_restart(monkeypatch,tmp_path):

    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)

    texts=["I love to swim in the lakes", "I do work on the Indian Railways", "I do live in India"]

    with get_db() as conn:
        ids=[add_fact(conn, text) for text in texts]
        retract_fact(conn, ids[0])

    with get_db() as conn:
        rows=list_active_facts(conn)

    assert[row["fact_text"]for row in rows]==[texts[1], texts[2]]
    assert len(rows)==2

def test_apply_decision(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)

    with get_db() as conn:
        id2 = add_fact(conn, "I live in Bengaluru")
        result = apply_decision(conn, "I moved to Pune", {"action": "UPDATE", "target_id": str(id2)}, [{"id": id2}])
        rows = list_active_facts(conn)
    assert result == "UPDATE"
    assert [r["fact_text"] for r in rows] == ["I moved to Pune"]
    texts = [r["fact_text"] for r in rows]
    assert "I moved to Pune" in texts
    assert "I live in Bengaluru" not in texts

def test_no_candidates_adds(monkeypatch,tmp_path):
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    def find_candidate(conn, new_text_fact): return []
    def decide_action(new_text_fact, candidates): raise AssertionError("decide_action should not be called")
    monkeypatch.setattr(storage, "find_candidate", find_candidate)
    monkeypatch.setattr(storage, "decide_action", decide_action)
    with get_db() as conn:
        action=process_fact(conn ,"I like Tea")
        assert action=="ADD"
        assert len(calls)==1

def test_candidate_llm_says_add(monkeypatch,tmp_path):
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    def find_candidate(conn, new_text_fact): return rows
    def decide_action(new_text_fact, candidates): return {"action" : "ADD", "target_id" : None}
    monkeypatch.setattr(storage, "find_candidate", find_candidate)
    monkeypatch.setattr(storage, "decide_action", decide_action)
    with get_db() as conn:
        fact=add_fact(conn,"I like Tea")
        rows=list_active_facts(conn)
        action=process_fact(conn ,"I like coffee")
        after=list_active_facts(conn)
        assert action=="ADD"
        assert len(after)==2

def test_candidate_update_gate(monkeypatch,tmp_path):
    
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    def find_candidate(conn, new_text_fact): return rows
    def decide_action(new_text_fact, candidates): return {"action" : "UPDATE", "target_id" : seed_id}
    monkeypatch.setattr(storage, "find_candidate", find_candidate)
    monkeypatch.setattr(storage, "decide_action", decide_action)
    with get_db() as conn:
        seed_id= add_fact(conn,"I love to watch trains")
        rows=list_active_facts(conn)
        result=process_fact(conn,"I love to watch flights")
        after=list_active_facts(conn)
        assert result=="UPDATE"
        assert after[0]["fact_text"]=="I love to watch flights"
        assert len(after)==1
        assert len(calls)==2

def test_candidate_delete_gate(monkeypatch,tmp_path):
    
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    def find_candidate(conn, new_text_fact): return rows
    def decide_action(new_text_fact, candidates): return {"action" : "DELETE", "target_id" : seed_id}
    monkeypatch.setattr(storage, "find_candidate", find_candidate)
    monkeypatch.setattr(storage, "decide_action", decide_action)
    with get_db() as conn:
        seed_id= add_fact(conn,"I love to watch trains")
        rows=list_active_facts(conn)
        result=process_fact(conn,"I don't love to watch trains")
        after=list_active_facts(conn)
        assert result=="DELETE"
        assert len(calls)==1

def test_candidate_delete_gate(monkeypatch,tmp_path):
    
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    def find_candidate(conn, new_text_fact): return rows
    def decide_action(new_text_fact, candidates): return {"action" : "DELETE", "target_id" : seed_id}
    monkeypatch.setattr(storage, "find_candidate", find_candidate)
    monkeypatch.setattr(storage, "decide_action", decide_action)
    with get_db() as conn:
        seed_id= add_fact(conn,"I love to watch trains")
        rows=list_active_facts(conn)
        result=process_fact(conn,"I don't love to watch trains")
        after=list_active_facts(conn)
        assert result=="DELETE"
        assert len(calls)==1
        assert len(after)==0

def test_candidate_none_gate(monkeypatch,tmp_path):
    
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    def find_candidate(conn, new_text_fact): return rows
    def decide_action(new_text_fact, candidates): return {"action" : "DELETE", "target_id" : seed_id}
    monkeypatch.setattr(storage, "find_candidate", find_candidate)
    monkeypatch.setattr(storage, "decide_action", decide_action)
    with get_db() as conn:
        seed_id= add_fact(conn,"I love to watch trains")
        rows=list_active_facts(conn)
        result=process_fact(conn,"I don't love to watch trains")
        after=list_active_facts(conn)
        assert result=="DELETE"
        assert len(calls)==1

def test_candidate_none_gate(monkeypatch,tmp_path):
    
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    def find_candidate(conn, new_text_fact): return rows
    def decide_action(new_text_fact, candidates): return {"action" : "NONE", "target_id" : None}
    monkeypatch.setattr(storage, "find_candidate", find_candidate)
    monkeypatch.setattr(storage, "decide_action", decide_action)
    with get_db() as conn:
        seed_id= add_fact(conn,"I love to watch trains")
        rows=list_active_facts(conn)
        result=process_fact(conn,"I enjoy watching trains")
        after=list_active_facts(conn)
        assert result=="NONE"
        assert len(after)==1
        assert after[0]["fact_text"]=="I love to watch trains"
        assert len(calls)==1

def test_gate_string_target_id(monkeypatch,tmp_path):
    
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    def find_candidate(conn, new_text_fact): return rows
    def decide_action(new_text_fact, candidates): return {"action" : "UPDATE", "target_id" : str(seed_id)}
    monkeypatch.setattr(storage, "find_candidate", find_candidate)
    monkeypatch.setattr(storage, "decide_action", decide_action)
    with get_db() as conn:
        seed_id=add_fact(conn,"I like to have tea from assam")
        rows=list_active_facts(conn)
        new_fact="I love to drink filtered coffee"
        fact=process_fact(conn,new_fact)
        after=list_active_facts(conn)
        assert after[0]["fact_text"]==new_fact
        assert fact=="UPDATE"

def test_gate_unknown_target_id(monkeypatch,tmp_path):

    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    calls=[]
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): calls.append(embedding_id)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    def find_candidate(conn, new_text_fact): return rows
    def decide_action(new_text_fact, candidates): return {"action" : "UPDATE", "target_id" : 999}
    monkeypatch.setattr(storage, "find_candidate", find_candidate)
    monkeypatch.setattr(storage, "decide_action", decide_action)
    with get_db() as conn:
        seed_id=add_fact(conn,"I like to have tea from assam")
        rows=list_active_facts(conn)
        new_fact="I love to drink filtered coffee"
        fact=process_fact(conn,new_fact)
        after=list_active_facts(conn)
        assert fact=="ADD"
        len(after)==2
        texts=[r["fact_text"]for r in after]
        print(texts)

def test_extract_facts(monkeypatch):

    def fake_chat(**kwargs): return {"message" : {"content": '{"facts":["I study chess"]}'}}
    monkeypatch.setattr(neo_ollama._ollama,"chat",fake_chat)
    result=extract_facts("any text")
    assert result==["I study chess"]

def test_extract_facts_bad_json(monkeypatch):
    def fake_chat(**kwargs): return {"message": {"content": "hello"}}
    monkeypatch.setattr(neo_ollama._ollama, "chat", fake_chat)
    result = extract_facts("any text")
    assert result == []


def test_extract_facts_strips_blanks(monkeypatch):
    def fake_chat(**kwargs): return {"message": {"content": '{"facts":["  I love to watch trains  ", "", "   "]}'}}
    monkeypatch.setattr(neo_ollama._ollama, "chat", fake_chat)
    result = extract_facts("any text")
    assert result == ["I love to watch trains"]


def test_extract_facts_no_facts_key(monkeypatch):
    def fake_chat(**kwargs): return {"message": {"content": '{"other": 1}'}}
    monkeypatch.setattr(neo_ollama._ollama, "chat", fake_chat)
    result = extract_facts("any text")
    assert result == []


def test_extract_facts_not_a_list(monkeypatch):
    def fake_chat(**kwargs): return {"message": {"content": '{"facts": "I love to read books"}'}}
    monkeypatch.setattr(neo_ollama._ollama, "chat", fake_chat)
    result = extract_facts("any text")
    assert result == []


def test_extract_facts_non_strings(monkeypatch):
    def fake_chat(**kwargs): return {"message": {"content": '{"facts": ["I live in Norway", 5, null]}'}}
    monkeypatch.setattr(neo_ollama._ollama, "chat", fake_chat)
    result = extract_facts("any text")
    assert result == ["I live in Norway"]


def test_extract_facts_exception(monkeypatch):
    def fake_chat(**kwargs): raise RuntimeError("boom")
    monkeypatch.setattr(neo_ollama._ollama, "chat", fake_chat)
    result = extract_facts("any text")
    assert result == []

def test_remember_facts_empty(monkeypatch):

    calls=[]
    def fake_extract(prompt): return []
    def fake_process(conn,fact): calls.append(fact)
    monkeypatch.setattr(neo,"extract_facts",fake_extract)
    monkeypatch.setattr(neo,"process_fact", fake_process)
    neo.remember_facts(None, "what do I study")
    assert calls==[]

def test_remember_facts_two_facts(monkeypatch,tmp_path):
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path/"test.db"))
    def fake_embedding(text): return [0.1,0.2,0.3]
    def fake_insert(embedding_id, vector): pass
    def fake_find(conn, new_text_fact): return []
    def fake_extract(prompt): return ["I love to watch trains", "I live in Norway"]
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "find_candidate", fake_find)
    monkeypatch.setattr(neo,"extract_facts", fake_extract)
    with get_db() as conn:
        neo.remember_facts(conn, "any text")
        texts=[r["fact_text"] for r in list_active_facts(conn)]
    assert len(texts)==2
    assert "I love to watch trains" in texts
    assert "I live in Norway"in texts

def test_remember_facts_one_bad_fact(monkeypatch):
    saved=[]
    def fake_extract(prompt): return ["bad fact", "good fact"]
    def fake_process(conn, fact):
        if fact == "bad fact":
            raise RuntimeError("boom")
        saved.append(fact)
    monkeypatch.setattr(neo, "extract_facts", fake_extract)
    monkeypatch.setattr(neo, "process_fact", fake_process)
    neo.remember_facts(None, "any text")
    assert saved == ["good fact"]

def test_update_target_identical_gate(monkeypatch, tmp_path):
    monkeypatch.setattr(storage, "_DB_PATH", str(tmp_path / "test.db"))
    def fake_embedding(text): return [0.1, 0.2, 0.3]
    def fake_insert(embedding_id, vector): pass
    def fake_find(conn, new_text_fact): return rows
    def fake_decide(new_text_fact, candidates): return {"action": "UPDATE", "target_id": seed_id}
    monkeypatch.setattr(storage, "get_embedding", fake_embedding)
    monkeypatch.setattr(storage, "insert_vector", fake_insert)
    monkeypatch.setattr(storage, "find_candidate", fake_find)
    monkeypatch.setattr(storage, "decide_action", fake_decide)
    with get_db() as conn:
        seed_id = add_fact(conn, "I stopped loving chess")
        rows = list_active_facts(conn)
        result = process_fact(conn, "I stopped loving chess")
        after = list_active_facts(conn)
        assert result == "NONE"
        assert len(after) == 1
        assert after[0]["id"] == seed_id

def test_delete_memory_yes(monkeypatch):
    
    seed()
    assert count("ai_history")>0
    assert count("facts")>0

    answers=iter(["yes","yes"])
    monkeypatch.setattr("builtins.input", lambda _="": next(answers))
    calls=[]
    monkeypatch.setattr("lancedb_store.clear_all_vectors",lambda: calls.append(1))
    delete_memory()
    # conn.execute("DELETE FROM facts")
    assert count("ai_history")==0
    assert count("facts")==0
    assert len(calls)==1

@pytest.fixture(autouse=True)
def temp_db(monkeypatch, tmp_path):
    monkeypatch.setattr("storage._DB_PATH", str(tmp_path / "test.db"))  # your real name

def seed():
    with get_db() as conn:
        conn.execute(
            "INSERT INTO ai_history (prompt, response, timestamp) VALUES (?, ?, ?)",
            ("hello", "hi", "2026-01-01T00:00:00"),
        )
        conn.execute(
            "INSERT INTO ai_history (prompt, response, timestamp) VALUES (?, ?, ?)",
            ("bye", "see you", "2026-01-01T00:00:01"),
        )
        conn.execute(
            "INSERT INTO facts (fact_text, created_at, status) VALUES (?, ?, ?)",
            ("I like chess", "2026-01-01T00:00:00", "active"),
        )
        conn.execute(
            "INSERT INTO facts (fact_text, created_at, status) VALUES (?, ?, ?)",
            ("I study robotics", "2026-01-01T00:00:00", "active"),
        )


def count(table):
    with get_db() as conn:
        return conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

def test_search_similarity_drops_far_results(monkeypatch):

    monkeypatch.setattr(lancedb_store,"get_lance_table", lambda:FakeTable())
    result=lancedb_store.search_similarity([0,0])
    assert result==["a"]

def make_fake_chat(saved):
    def fake_chat(model, messages, **kwargs):
        saved.append(messages)
        return {"message": {"content": "ok"}}
    return fake_chat


def test_chat_includes_web_results(monkeypatch):
    saved = []
    monkeypatch.setattr("neo_ollama._ollama.chat", make_fake_chat(saved))
    web = [{"title": "Moscow - Wikipedia", "snippet": "Moscow is the capital of Russia."}]

    chat("Manoj", "capital of Russia?", [], [], web)

    system_text = saved[0][0]["content"]
    assert "Web results" in system_text
    assert "Moscow is the capital of Russia" in system_text
    assert "offline" not in system_text


def test_chat_without_web_results_stays_offline(monkeypatch):
    saved = []
    monkeypatch.setattr("neo_ollama._ollama.chat", make_fake_chat(saved))

    chat("Manoj", "hi", [], [], [])

    system_text = saved[0][0]["content"]
    assert "Web results" not in system_text
    assert "offline" in system_text
