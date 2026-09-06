# bench_latency.py
import time
from db import get_db
from config import get_user
from entity_memory import get_active_attrs, get_primary_entity
from extraction import extract_keywords
from retrieval import retrieve_similar
from neo_ollama import chat
from query_classifier import init_query_classifier
from web_db import init_web
from neo_search_pipeline import search_and_learn

init_web()
init_query_classifier()

username, _ = get_user()

# ---------------------------------------------------------------------------
# Benchmark 1: search_and_learn() latency — pure search round trip
# ---------------------------------------------------------------------------
search_query = "who won the 2026 F1 constructors championship"

with get_db() as conn:
    primary_entity = get_primary_entity(conn)

t0 = time.perf_counter()
with get_db() as conn:
    search_results = search_and_learn(conn, search_query, primary_entity)
t1 = time.perf_counter()
print(f"[BENCH 1] search_and_learn(): {t1 - t0:.2f}s, {len(search_results)} results")

# ---------------------------------------------------------------------------
# Benchmark 2: full chat turn latency — ask_neo()'s real work, without the
# interactive input() loop. Two variants: with search, without search.
# ---------------------------------------------------------------------------

def timed_turn(prompt, use_search_results=None, label=""):
    with get_db() as conn:
        primary_entity = get_primary_entity(conn)
        active_attrs = get_active_attrs(conn, primary_entity) if primary_entity else {}

    compressed = extract_keywords(prompt)

    t0 = time.time()
    similar = retrieve_similar(compressed, raw_query=prompt) if compressed else []
    t1 = time.time()

    response = chat(
        username, prompt, history=[], similar=similar, active_attrs=active_attrs,
        has_search_results=bool(use_search_results), search_results=use_search_results
    )
    t2 = time.time()

    print(f"[BENCH 2 - {label}] retrieve_similar: {t1-t0:.2f}s | chat(): {t2-t1:.2f}s | total: {t2-t0:.2f}s")
    print(f"  response preview: {response[:100]!r}")

# no-search turn (pure memory/chat path)
timed_turn("what is my age", label="no-search")

# search-grounded turn (reuses search_results from benchmark 1)
timed_turn(search_query, use_search_results=search_results, label="with-search")