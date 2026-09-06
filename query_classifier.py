import numpy as np
from model_config import get_shared_embed_model
from concurrent.futures import ThreadPoolExecutor, as_completed
from web_search import web_search
from web_db import store_web_result
from logger import log_info

# Hand-picked examples of queries that clearly need live/current info.
# Not matched literally — an incoming query is compared against these by
# embedding similarity (see is_time_sensitive).
TIME_SENSITIVE_EXAMPLES = [
    "current price of gold",
    "who won the match today",
    "latest iPhone release",
    "who is the current CEO of the company",
    "today's exchange rate",
    "latest news on the election",
    "current score of the game",
    "who is the president right now",
    "stock price today",
    "is the ceasefire still holding",
    "current status of the war",
    "latest update on the conflict",
    "recent update on the situation",
]

_reference_embeddings = None

SEARCH_REQUEST_EXAMPLES = [
    "search for this online",
    "can you search about current war"
    "search the internet for this",
    "look this up online",
    "look up the latest on this",
    "check the internet for this",
    "find current information about this",
    "what's happening with this right now",
    "search about this topic",
    "can you search about this",
    "can you look this up for me",
    "search about the current status of this",
    "go search for information on this",
    "please search the web for this",
    "what's the current exchange rate",
    "what's the current price of this",
    "what's the current value of this",
]

_search_request_embeddings = None


def init_query_classifier():
    """Call once at startup. Embeds TIME_SENSITIVE_EXAMPLES and
    SEARCH_REQUEST_EXAMPLES so is_time_sensitive and wants_search
    have something to compare against."""
    global _reference_embeddings, _search_request_embeddings
    _reference_embeddings = get_shared_embed_model().encode(TIME_SENSITIVE_EXAMPLES, convert_to_numpy=True)
    _search_request_embeddings = get_shared_embed_model().encode(SEARCH_REQUEST_EXAMPLES, convert_to_numpy=True)


EXPLICIT_SEARCH_TRIGGERS = (
    "search for", "search about", "search the web", "search the internet",
    "look this up", "look that up", "look up", "go search",
)


def wants_search(query: str, threshold: float = 0.48) -> bool:
    """True if the query is similar enough to an explicit search request
    that Neo should trigger a live web search. Separate from
    is_time_sensitive — this decides WHETHER to search at all, that one
    decides HOW BROADLY once search is already triggered.

    Checks literal trigger phrases first (see EXPLICIT_SEARCH_TRIGGERS) —
    this is a fast path to True only, it never overrides a genuine search
    request with a False. Falls back to embedding similarity for requests
    that want search without using one of those exact phrases."""
    query_lower = query.lower()
    if any(trigger in query_lower for trigger in EXPLICIT_SEARCH_TRIGGERS):
        return True

    if _search_request_embeddings is None:
        raise RuntimeError("call init_query_classifier() at startup before using this")

    query_vec = get_shared_embed_model().encode(query, convert_to_numpy=True)
    best_score = max(_cosine_sim(query_vec, ref) for ref in _search_request_embeddings)
    return best_score >= threshold


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def is_time_sensitive(query: str, threshold: float = 0.45) -> bool:
    """True if the query is similar enough to one of the time-sensitive
    example queries that it likely needs a live search rather than an
    answer from memory alone."""
    if _reference_embeddings is None:
        raise RuntimeError("call init_query_classifier() at startup before using this")

    query_vec = get_shared_embed_model().encode(query, convert_to_numpy=True)
    best_score = max(_cosine_sim(query_vec, ref) for ref in _reference_embeddings)
    return best_score >= threshold


def reword_query(query: str, variant: int) -> str:
    """Generates a small set of alternate phrasings of the same query,
    used to catch more relevant results when fanning out to multiple
    searches. variant 0 is always the original, unmodified query."""
    if variant == 0:
        return query
    elif variant == 1:
        return f"{query} latest"
    elif variant == 2:
        return f"{query} 2026"
    return query


def web_search_multi(query: str, max_results: int = 5) -> list[dict]:
    """Runs one search for ordinary queries, or three reworded variants
    for time-sensitive ones (more likely to need broader coverage).
    Variants run concurrently (each is a separate network call to
    SearXNG) rather than sequentially. Results are deduped by URL
    across all variants and capped at max_results total."""
    should_multi = is_time_sensitive(query)
    num_queries = 3 if should_multi else 1
    variant_queries = [reword_query(query, i) for i in range(num_queries)]

    seen_urls = set()
    all_results = []

    def _run_variant(variant_query):
        try:
            results = web_search(variant_query, max_results)
            if results:
                store_web_result(variant_query, results)
            return results
        except Exception as e:
            log_info(f"Search variant failed: '{variant_query}' — {e}", level="WARNING", module="WEB_SEARCH")
            print(f"[DEBUG] variant='{variant_query}' EXCEPTION: {e}")
            return []

    with ThreadPoolExecutor(max_workers=num_queries) as executor:
        futures = {executor.submit(_run_variant, vq): vq for vq in variant_queries}
        for future in as_completed(futures):
            try:
                results = future.result()
            except Exception:
                results = []
            for r in results or []:
                url = r.get("url")
                if url not in seen_urls:
                    seen_urls.add(url)
                    all_results.append(r)
    return all_results[:max_results]