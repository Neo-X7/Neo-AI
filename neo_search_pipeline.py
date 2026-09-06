import time
from query_classifier import web_search_multi


def search_and_learn(conn, query: str, primary_entity: str = None, max_results: int = 5, log_timing: bool = False) -> list[dict]:
    """Runs a live web search and returns the raw results. No verification
    step — search_verify.py was removed. Grounding against fabrication is
    handled entirely by search_completion()'s prompt-level constraints;
    this does not protect against a bad/wrong source being cited as
    reliable.
    """
    t0 = time.time()
    results = web_search_multi(query, max_results=max_results)
    if log_timing:
        print(f"[TIMING] web_search_multi: {time.time()-t0:.1f}s, {len(results)} results")
    return results