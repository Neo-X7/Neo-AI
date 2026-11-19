import os
import tempfile
import time
import lancedb_store
import storage


SEED_FACTS = [
    "I live in Bengaluru",
    "I study robotics and automation",
    "I like playing chess",
    "I love drinking Assam tea",
    "My sister lives in Delhi",
    "I own a laptop with an RTX 4050",
    "I use Python for backend projects",
    "I enjoy watching trains",
    "My favorite food is masala dosa",
    "I am preparing for software internships",
]

RETRIEVAL_CASES = [
    ("Where do I stay?", "I live in Bengaluru"),
    ("What is my college major?", "I study robotics and automation"),
    ("What board game do I enjoy?", "I like playing chess"),
    ("What hot beverage do I prefer?", "I love drinking Assam tea"),
    ("Where is my sibling based?", "My sister lives in Delhi"),
    ("What GPU does my machine have?", "I own a laptop with an RTX 4050"),
    ("Which programming language do I use?", "I use Python for backend projects"),
    ("What do I like to watch?", "I enjoy watching trains"),
    ("What do I like to eat?", "My favorite food is masala dosa"),
    ("What career step am I preparing for?", "I am preparing for software internships"),
]


def setup_isolated_store(tmp_dir: str) -> None:
    """Point storage and lancedb_store at a clean temporary directory."""
    storage._DB_PATH = os.path.join(tmp_dir, "eval.db")
    lancedb_store.DB_PATH = os.path.join(tmp_dir, "eval_lance")
    lancedb_store._db = None
    lancedb_store._table = None


def eval_retrieval(cases: list[tuple[str, str]]) -> dict:
    """Compute Recall@1, Recall@3, Recall@5, MRR, and mean retrieval latency."""
    hits_at_1 = 0
    hits_at_3 = 0
    hits_at_5 = 0
    rr_sum = 0.0
    latencies = []

    for query, expected_fact in cases:
        t0 = time.perf_counter()
        results = storage.retrieve_similar(query)
        latencies.append(time.perf_counter() - t0)

        returned = [r["fact_text"] for r in results]
        if expected_fact in returned[:1]:
            hits_at_1 += 1
        if expected_fact in returned[:3]:
            hits_at_3 += 1
        if expected_fact in returned[:5]:
            hits_at_5 += 1
        if expected_fact in returned:
            rank = returned.index(expected_fact) + 1
            rr_sum += 1.0 / rank

    n = len(cases)
    return {
        "n": n,
        "recall@1": hits_at_1 / n,
        "recall@3": hits_at_3 / n,
        "recall@5": hits_at_5 / n,
        "mrr": rr_sum / n,
        "avg_ms": (sum(latencies) / n) * 1000,
    }


def eval_write_gate(conn) -> dict:
    """Run live write-gate cases and check action + active-fact state."""
    gate_cases = [
        {
            "name": "unrelated_add",
            "new_fact": "I have a golden retriever named Bruno",
            "expected_actions": {"ADD"},
            "must_contain": "I have a golden retriever named Bruno",
            "must_not_contain": None,
        },
        {
            "name": "location_update",
            "new_fact": "I moved to Pune",
            "expected_actions": {"UPDATE"},
            "must_contain": "I moved to Pune",
            "must_not_contain": "I live in Bengaluru",
        },
        {
            "name": "hobby_reversal_update",
            "new_fact": "I stopped playing chess",
            "expected_actions": {"UPDATE"},
            "must_contain": "I stopped playing chess",
            "must_not_contain": "I like playing chess",
        },
        {
            "name": "identical_none",
            "new_fact": "I study robotics and automation",
            "expected_actions": {"NONE"},
            "must_contain": "I study robotics and automation",
            "must_not_contain": None,
        },
        {
            "name": "explicit_forget_delete",
            "new_fact": "Forget that I enjoy watching trains",
            "expected_actions": {"DELETE"},
            "must_contain": None,
            "must_not_contain": "I enjoy watching trains",
        },
    ]

    passed = 0
    latencies = []
    details = []

    for case in gate_cases:
        t0 = time.perf_counter()
        action = storage.process_fact(conn, case["new_fact"])
        conn.commit()
        dt_ms = (time.perf_counter() - t0) * 1000
        latencies.append(dt_ms)

        active = [r["fact_text"] for r in storage.list_active_facts(conn)]
        ok = action in case["expected_actions"]
        if case["must_contain"] and case["must_contain"] not in active:
            ok = False
        if case["must_not_contain"] and case["must_not_contain"] in active:
            ok = False

        if ok:
            passed += 1
        details.append((case["name"], action, ok, dt_ms))

    return {
        "passed": passed,
        "total": len(gate_cases),
        "accuracy": passed / len(gate_cases),
        "avg_ms": sum(latencies) / len(latencies),
        "details": details,
    }


def eval_retraction_leakage() -> dict:
    """Verify retracted facts never reappear via retrieve_similar."""
    checks = [
        ("Where do I live?", "I live in Bengaluru"),
        ("Do I play chess?", "I like playing chess"),
        ("Do I like trains?", "I enjoy watching trains"),
    ]
    leaked = 0
    for query, retracted_fact in checks:
        returned = [r["fact_text"] for r in storage.retrieve_similar(query)]
        if retracted_fact in returned:
            leaked += 1
    return {"leaked": leaked, "total": len(checks)}


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="neo_eval_") as tmp_dir:
        setup_isolated_store(tmp_dir)

        with storage.get_db() as conn:
            for fact in SEED_FACTS:
                storage.add_fact(conn, fact)

        ret = eval_retrieval(RETRIEVAL_CASES)
        print("=== Retrieval Metrics ===")
        print(f"Queries   : {ret['n']}")
        print(f"Recall@1  : {ret['recall@1']:.2f}")
        print(f"Recall@3  : {ret['recall@3']:.2f}")
        print(f"Recall@5  : {ret['recall@5']:.2f}")
        print(f"MRR       : {ret['mrr']:.2f}")
        print(f"Avg lat   : {ret['avg_ms']:.1f} ms\n")

        with storage.get_db() as conn:
            gate = eval_write_gate(conn)

        print("=== Write-Gate Metrics ===")
        for name, action, ok, dt_ms in gate["details"]:
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {name:<24} -> {action:<7} ({dt_ms:.0f} ms)")
        print(f"Accuracy  : {gate['passed']}/{gate['total']} ({gate['accuracy']:.0%})")
        print(f"Avg lat   : {gate['avg_ms']:.1f} ms\n")

        leak = eval_retraction_leakage()
        print("=== Retracted-Fact Leakage ===")
        print(f"Leaked    : {leak['leaked']}/{leak['total']}")


if __name__ == "__main__":
    main()