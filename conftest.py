"""
Stubs every heavy third-party dependency that entity_memory.py's import
chain pulls in transitively (via neo_ollama, extraction, write_gate),
so the tests can run without spaCy, sentence-transformers, Ollama,
yake, pycountry, geonamescache, lancedb, or rich actually installed.

This is intentional: the two functions under test (resolve_entity_query,
_flag_retracted_history) are pure DB/string logic. They don't need real
embeddings or a real NLP model to be correct — they need a real SQLite
connection with real rows in it. Everything ML/network-shaped gets a
fake that returns predictable, injectable values instead.
"""
import sys
import types
import numpy as np


def _install_stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# --- ollama -----------------------------------------------------------
class _FakeOllamaClient:
    def chat(self, model=None, messages=None, options=None):
        return {"message": {"content": "YES"}}


_install_stub("ollama", Client=lambda *a, **k: _FakeOllamaClient(), chat=lambda *a, **k: {"message": {"content": "YES"}})


# --- sentence_transformers ---------------------------------------------
class _FakeSentenceTransformer:
    def __init__(self, *a, **k):
        pass

    def encode(self, text, convert_to_numpy=False):
        # Deterministic pseudo-embedding: hash the text into a fixed-size
        # vector so identical strings always embed identically, and
        # different strings differ, without needing a real model.
        if isinstance(text, list):
            return np.array([self._encode_one(t) for t in text])
        return self._encode_one(text)

    def _encode_one(self, t):
        h = abs(hash(t)) % (10 ** 8)
        rng = np.random.RandomState(h)
        return rng.rand(16)


_install_stub("sentence_transformers", SentenceTransformer=_FakeSentenceTransformer)

# --- yake ---------------------------------------------------------------
class _FakeKeywordExtractor:
    def __init__(self, *a, **k):
        pass

    def extract_keywords(self, text):
        words = text.split()[:5]
        return [(w, 0.1) for w in words]


_install_stub("yake", KeywordExtractor=_FakeKeywordExtractor)

# --- spacy (extraction.py falls back to None NLP if load fails, which
# is exactly the code path we want exercised — so we let spacy "import"
# but make load() raise OSError, same as a real missing-model failure) --
def _spacy_load(*a, **k):
    raise OSError("stub: no spacy model in sandbox")


_install_stub("spacy", load=_spacy_load)

# --- pycountry ------------------------------------------------------------
class _FakeCountry:
    def __init__(self, name):
        self.name = name


_install_stub("pycountry", countries=[_FakeCountry("India"), _FakeCountry("United States")])

# --- geonamescache --------------------------------------------------------
class _FakeGeonamesCache:
    def get_countries(self):
        return {"IN": {"name": "India"}, "US": {"name": "United States"}}

    def get_cities(self):
        return {"1": {"name": "Bengaluru"}, "2": {"name": "Delhi"}, "3": {"name": "New Delhi"}}


_install_stub("geonamescache", GeonamesCache=_FakeGeonamesCache)

# --- lancedb / pyarrow (only imported by lancedb_store.py, not on the
# import path of the functions under test, but stub defensively in case
# something transitively pulls it in) ---
_install_stub("lancedb", connect=lambda *a, **k: None)
_install_stub("pyarrow", schema=lambda *a, **k: None, field=lambda *a, **k: None, utf8=lambda: None,
              list_=lambda *a, **k: None, float32=lambda: None, int64=lambda: None)

# --- rich (console/text/table/panel — used widely for printing only) ---
class _FakeConsole:
    def print(self, *a, **k):
        pass


_install_stub("rich", **{})
_install_stub("rich.console", Console=_FakeConsole)
_install_stub("rich.text", Text=lambda *a, **k: (a[0] if a else ""))
_install_stub("rich.table", Table=object)
_install_stub("rich.panel", Panel=object)

# --- psutil (only used by hardware_check.py, not on the import path
# under test, stub defensively) ---
class _FakeVirtualMemory:
    total = 16 * (1024 ** 3)


_install_stub("psutil", virtual_memory=lambda: _FakeVirtualMemory())