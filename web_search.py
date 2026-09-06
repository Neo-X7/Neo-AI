import os
import json
import requests
from abc import ABC, abstractmethod
from rich.console import Console
from rich.text import Text
from logger import ai_log_info, log_info

console = Console()

SEARCH_CONTAINER_URLS = os.environ.get(
    "NEO_SEARCH_URLS", "http://localhost:8888,http://localhost:8889,http://localhost:8890"
).split(",")
CONFIG_PATH = os.path.expanduser("~/.neo/search_config.json")

_searxng_round_robin_idx = 0


class SearchBackend(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        ...


class SearxngBackend(SearchBackend):
    """Local SearXNG Docker container(s). No API key needed. Round-robins
    across multiple instances if NEO_SEARCH_URLS lists more than one,
    for throughput and basic redundancy."""
    def search(self, query: str, max_results: int = 5) -> list[dict]:
        global _searxng_round_robin_idx
        url = SEARCH_CONTAINER_URLS[_searxng_round_robin_idx % len(SEARCH_CONTAINER_URLS)]
        _searxng_round_robin_idx += 1

        resp = requests.get(
            f"{url}/search",
            params={"q": query, "format": "json", "time_range": "month"},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        results = data.get("results", [])[:max_results]
        return [
            {"title": r.get("title", ""), "url": r.get("url", ""),
             "snippet": r.get("content", ""), "date": r.get("publishedDate", "")}
            for r in results
        ]


class BraveBackend(SearchBackend):
    """Brave Search API. Requires an API key."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        resp = requests.get(
            "https://api.search.brave.com/res/v1/web/search",
            headers={"X-Subscription-Token": self.api_key},
            params={"q": query, "count": max_results},
            timeout=15,
        )
        if resp.status_code == 429:
            raise RuntimeError("brave_rate_limited: try again later")
        if resp.status_code == 401:
            raise RuntimeError("brave_auth_failed: check api key")
        resp.raise_for_status()
        data = resp.json()
        return [
            {"title": r["title"], "url": r["url"], "snippet": r.get("description", "")}
            for r in data.get("web", {}).get("results", [])[:max_results]
        ]

    def validate(self) -> bool:
        try:
            self.search("test", max_results=1)
            return True
        except Exception:
            return False


class TavilyBackend(SearchBackend):
    """Tavily Search API. Requires an API key."""
    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, max_results: int = 5) -> list[dict]:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={"api_key": self.api_key, "query": query, "max_results": max_results},
            timeout=15,
        )
        if resp.status_code == 429:
            raise RuntimeError("tavily_rate_limited: try again later")
        if resp.status_code == 401:
            raise RuntimeError("tavily_auth_failed: check api key")
        resp.raise_for_status()
        data = resp.json()
        return [
            {"title": r["title"], "url": r["url"], "snippet": r.get("content", "")}
            for r in data.get("results", [])[:max_results]
        ]

    def validate(self) -> bool:
        try:
            self.search("test", max_results=1)
            return True
        except Exception:
            return False


_BACKENDS = {"searxng": SearxngBackend, "brave": BraveBackend, "tavily": TavilyBackend}
_active_backend: SearchBackend = SearxngBackend()
_active_backend_name = "searxng"


def register_backend(name: str, backend_cls: type) -> None:
    """Add a new search backend without editing this file. backend_cls
    must subclass SearchBackend and implement .search(query, max_results).

    Example, from anywhere in your own code, before calling
    set_search_backend:
        from web_search import register_backend, SearchBackend

        class MyBackend(SearchBackend):
            def search(self, query, max_results=5):
                ...
                return [{"title": ..., "url": ..., "snippet": ...}, ...]

        register_backend("mybackend", MyBackend)
        set_search_backend("mybackend", api_key="...")
    """
    if not (isinstance(backend_cls, type) and issubclass(backend_cls, SearchBackend)):
        raise TypeError(f"{backend_cls} must be a subclass of SearchBackend")
    _BACKENDS[name] = backend_cls


def _load_config() -> dict:
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def _save_config(name: str, api_key: str | None):
    os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
    with open(CONFIG_PATH, "w") as f:
        json.dump({"backend": name, "api_key": api_key}, f)


def load_saved_backend():
    """Call once at Neo startup to restore the last-used backend.
    Defaults to searxng (the module-level default) if nothing was saved,
    or if the saved backend fails to load — but unlike before, a failure
    is now actually logged and shown, not swallowed silently."""
    cfg = _load_config()
    name = cfg.get("backend", "searxng")
    key = cfg.get("api_key")

    result = set_search_backend(name, key, persist=False)
    if result.startswith("error"):
        ai_log_info(f"Saved search backend '{name}' failed to load: {result}", level="WARNING", module="WEB_SEARCH")
        console.print(Text(f"Saved search backend '{name}' failed to load, falling back to searxng: {result}", style="yellow"))


def set_search_backend(name: str, api_key: str | None = None, persist: bool = True) -> str:
    """Switch the active search backend. Validates paid backends with a
    real test search before switching, so a bad key doesn't silently
    become the active backend. Returns a status string starting with
    'error' on failure — callers must check this, it does not raise."""
    global _active_backend, _active_backend_name
    if name not in _BACKENDS:
        return f"error: unknown backend '{name}', choose from {list(_BACKENDS)}"

    if name == "searxng":
        _active_backend = SearxngBackend()
    else:
        if not api_key:
            return f"error: backend '{name}' requires an api_key"
        candidate = _BACKENDS[name](api_key)
        if not candidate.validate():
            return f"error: '{name}' key rejected or unreachable, backend NOT switched"
        _active_backend = candidate

    _active_backend_name = name
    if persist:
        _save_config(name, api_key)
    return f"search backend switched to '{name}'"


def web_search(query: str, max_results: int = 5) -> list[dict]:
    """The actual search entry point used elsewhere in Neo. Always returns
    a list (empty on failure) rather than raising, so callers don't need
    try/except around every call."""
    try:
        return _active_backend.search(query, max_results)
    except requests.exceptions.Timeout:
        log_info(f"Timeout on backend '{_active_backend_name}'", level="WARNING", module="WEB_SEARCH")
        return []
    except requests.exceptions.ConnectionError:
        log_info(f"Connection failed, backend '{_active_backend_name}' unreachable", level="ERROR", module="WEB_SEARCH")
        return []
    except Exception as e:
        log_info(f"Failed on backend '{_active_backend_name}': {e}", level="ERROR", module="WEB_SEARCH")
        return []