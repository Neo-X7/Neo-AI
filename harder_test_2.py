import storage, lancedb_store
storage._DB_PATH = "/tmp/hard.db"
lancedb_store.DB_PATH = "/tmp/hard_lance"

from storage import get_db, retrieve_similar

with get_db() as conn:
    rows = conn.execute("SELECT fact_text, status FROM facts ORDER BY id").fetchall()
active = [r["fact_text"] for r in rows if r["status"] == "active"]
retracted = [r["fact_text"] for r in rows if r["status"] == "retracted"]

print("--- ACTIVE ---")
for t in active: print(" ", t)
print("--- RETRACTED ---")
for t in retracted: print(" ", t)

def has(texts, kws):
    return any(all(k in t.lower() for k in kws) for t in texts)

MUST_ACTIVE = [("born", "bengaluru"), ("school", "bengaluru"), ("first job",),
               ("zurich",), ("mother",), ("father",), ("ted chiang",),
               ("diwali",), ("left-handed",), ("jazz",), ("guitar",),
               ("python",), ("kannada",)]
MUST_RETRACTED = [("startup",), ("germany",), ("liechistan",), ("france",),
                  ("vegetarian",), ("playing chess",), ("own", "bicycle"),
                  ("dog",)]

print("--- STATE CHECKS ---")
for kws in MUST_ACTIVE:
    print("PASS" if has(active, kws) else "FAIL", "| active:", kws)
for kws in MUST_RETRACTED:
    print("PASS" if has(retracted, kws) else "FAIL", "| retracted:", kws)

QUESTIONS = [
    ("where was I born?", ("born",)),
    ("where did I go to school?", ("school",)),
    ("where do I work these days?", ("zurich",)),
    ("what instruments do I play?", ("guitar",)),
    ("what languages do I speak?", ("kannada",)),
    ("what do I do outdoors on weekends?", ("trail",)),
    ("who is my favourite writer?", ("ted chiang",)),
    ("which festival do I celebrate?", ("diwali",)),
    ("which programming languages do I know?", ("python",)),
    ("what is my dad up to?", ("father",)),
    ("what's the capital of France?", None),
    ("explain how a transformer works", None),
]

print("--- RETRIEVAL ---")
leaks = 0
for q, kws in QUESTIONS:
    res = [r["fact_text"] for r in retrieve_similar(q)]
    leaks += sum(1 for t in res if t in retracted and t not in active)
    if kws is None:
        ok = res == []
    else:
        ok = any(all(k in res[0].lower() for k in kws)for t in res)
    print("PASS" if ok else "FAIL", "|", q, "->", res)
print("retracted facts leaked into retrieval:", leaks)