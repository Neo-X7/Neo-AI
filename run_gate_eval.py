import json, os
from collections import Counter
import storage as m
import lancedb_store as ls

here = os.path.dirname(os.path.abspath(__file__))
m._DB_PATH = os.path.join(here, "test_gate.db")
ls.DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lance_test")
ls._db = None
ls._table = None

RUNS = 3
SKIP = {21, 30}                              # forget-intent commands, not supported yet
ALSO_OK = {5: ["DELETE", "UPDATE"], 16: ["DELETE", "UPDATE"], 23: ["DELETE", "UPDATE"]}

def reset():
    if os.path.exists(m._DB_PATH):
        os.remove(m._DB_PATH)
    ls.clear_all_vectors()

cases = [c for c in json.load(open(os.path.join(here, "gate_eval.json"), encoding="utf-8")) if c["id"] not in SKIP]
correct, by_cat = 0, {}

for case in cases:
    allowed = ALSO_OK.get(case["id"], [case["expected"]])
    results = []
    for _ in range(RUNS):
        reset()
        with m.get_db() as conn:
            for fact in case["existing"]:
                m.add_fact(conn, fact)
            results.append(m.process_fact(conn, case["new"]))
    predicted = Counter(results).most_common(1)[0][0]
    ok = predicted in allowed
    correct += ok
    by_cat.setdefault(case["cat"], []).append(ok)
    if not ok:
        print(f'WRONG #{case["id"]}: {case["new"]} | expected {allowed} got {results}')

print(f"\nAccuracy: {correct}/{len(cases)} = {correct/len(cases):.1%}")
for cat, v in sorted(by_cat.items()):
    print(f"  {cat:<14} {sum(v)}/{len(v)}")