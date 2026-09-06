import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from storage import extract_keywords
from retrieval import retrieve_similar

# Every entry here has matching seed data in seed_eval_data.py — run that
# script first, or these will legitimately score 0.0 (no fixture data to
# find, not a retrieval bug). __NONE__ entries are negative controls:
# retrieval should find nothing relevant for them.

TEST_SET = [
    # --- direct hits, plain phrasing ---
    {"query": "where do I live", "expect": ["bengaluru", "india"]},
    {"query": "what is my age", "expect": ["24"]},
    {"query": "where do I work", "expect": ["stark industries"]},
    {"query": "who created you", "expect": ["rin"]},
    {"query": "what do you know about edith", "expect": ["edith"]},
    {"query": "what do you know about jarvis", "expect": ["jarvis"]},

    # --- paraphrase robustness ---
    {"query": "what city am I based in", "expect": ["bengaluru"]},
    {"query": "what country am I in", "expect": ["india"]},
    {"query": "how old am I", "expect": ["24"]},
    {"query": "which company do I work for", "expect": ["stark industries"]},
    {"query": "who built you", "expect": ["rin"]},
    {"query": "who made you", "expect": ["rin"]},

    # --- entity collision / near-duplicate ---
    {"query": "difference between jarvis and edith", "expect": ["jarvis", "edith"]},
    {"query": "which stark AI came first, jarvis or edith", "expect": ["jarvis", "edith"]},

    # --- multi-fact compound ---
    {"query": "tell me my age and where I live", "expect": ["24", "bengaluru"]},
    {"query": "what's my job and my location", "expect": ["stark industries", "bengaluru"]},

    # --- negative controls: should retrieve nothing relevant ---
    {"query": "what's the capital of france", "expect": ["__NONE__"]},
    {"query": "recommend a good sci-fi book", "expect": ["__NONE__"]},
    {"query": "how do I fix a flat tire", "expect": ["__NONE__"]},
    {"query": "what's 2 plus 2", "expect": ["__NONE__"]},
    {"query": "translate hello to spanish", "expect": ["__NONE__"]},
    {"query": "did I ever mention my salary", "expect": ["__NONE__"]},
    {"query": "tell me about quantum physics", "expect": ["__NONE__"]},
    {"query": "random unrelated query about cooking pasta", "expect": ["__NONE__"]},
]


def is_hit(row, expects):
    """Checks response only — NOT prompt. row['prompt'] can equal the
    current query text verbatim (the resolve_entity_query fast path in
    retrieval.py returns {'prompt': embed_input, ...}), so matching
    against prompt risks a tautological hit: the query matching itself,
    not evidence retrieval found anything real. response is always the
    actual retrieved/generated content, so it's the only field worth
    scoring against."""
    if expects == ["__NONE__"]:
        return False
    text = (row["response"] or "").lower()
    return any(e.lower() in text for e in expects)


def eval_query(item, k=4):
    kw = extract_keywords(item["query"])
    results = retrieve_similar(kw, top_k_final=k, raw_query=item["query"])
    hits = [is_hit(r, item["expect"]) for r in results]
    recall = 1.0 if any(hits) else 0.0
    rr = 0.0
    for i, h in enumerate(hits):
        if h:
            rr = 1.0 / (i + 1)
            break
    return recall, rr, len(results)


def main():
    recalls, rrs = [], []
    none_false_positives = 0
    for item in TEST_SET:
        r, rr, n = eval_query(item)
        is_none_case = item["expect"] == ["__NONE__"]
        if not is_none_case:
            recalls.append(r)
            rrs.append(rr)
        else:
            # A __NONE__ case "hitting" isn't possible via is_hit (it always
            # returns False for __NONE__), so instead flag if retrieval
            # returned anything at all with real content — worth a manual
            # look even though it's not scored as a hard failure.
            if n > 0:
                none_false_positives += 1
        print(f"Q: {item['query']!r:55s} recall={r:.0f} rr={rr:.2f} n={n}")

    print(f"\nRecall@k: {sum(recalls)/len(recalls):.3f}")
    print(f"MRR:      {sum(rrs)/len(rrs):.3f}")
    print(f"__NONE__ cases that returned non-empty results (n>0): {none_false_positives} "
          f"(not auto-failed, but worth eyeballing what they returned)")


if __name__ == "__main__":
    main()