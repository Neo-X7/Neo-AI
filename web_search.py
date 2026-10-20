import requests
from logger import log_info
SEARCH_URL="http://localhost:8888"

def web_search(query : str , max_results : int =3)->list[dict]:
    """Query the local SearXNG container and return up to max_results dicts with title, url and snippet.

    Never raises: on any failure it logs to error.log and returns [].
    """

    try:
        resp=requests.get(f"{SEARCH_URL}/search",params={"q" : query, "format" : "json"},timeout=15)
        resp.raise_for_status()
        results=resp.json().get("results",[])[:max_results]
        clean=[]
        for r in results:
            clean.append({"title" : r.get("title",""), "url" : r.get("url",""), "snippet" : r.get("content","")})
        return clean
    except Exception as e:
        log_info(f"ERROR : {e}", level="ERROR", module="WEB_SEARCH")
        return []