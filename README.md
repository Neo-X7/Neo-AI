# Neo

**A fully local AI assistant that remembers what's true about you, and notices when it changes.**

No cloud. No API keys. Your memory never leaves your machine.

[![Python](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/powered%20by-Ollama-black)](https://ollama.com/)

<!-- Re-record on v1.9.5, then replace this comment with the demo GIF:
![Neo demo](screenshots/neo_demo.gif)
-->

Neo pulls facts out of what you say, decides whether each one is **new**, a **change**, or a **repeat**, and keeps a clean list of what is currently true. Nothing is overwritten: a changed fact is retracted, and the new one is added. Live web search is optional, runs through your own SearXNG container, and is never written into memory.

> Neo runs a local 8B model. It can make mistakes or hallucinate. Don't rely on it for critical decisions.

---

## See it work

```
You:  I live in Munich.
      → fact added: "lives in Munich"

You:  I moved to Zurich last month.
      → "lives in Munich" retracted
      → fact added: "lives in Zurich"

You:  Where am I from?        (born in Bengaluru, stated earlier)
Neo:  answers from active facts only. Munich never comes back.
```

This is the behaviour the [stress test](#stress-test) checks over 65 turns, including after a restart.

---

## Quick start

**Requirements:** Python 3.12+, [Ollama](https://ollama.com/), and optionally [Docker](https://www.docker.com/) for web search.

```bash
git clone https://github.com/Neo-X7/neo-AI.git
cd neo-AI
pip install -r requirements.txt
ollama pull llama3.1
ollama pull nomic-embed-text
python main.py
```

Neo starts the Ollama server for you if it isn't running.

**Web search (optional).** Run one SearXNG container with JSON output enabled (`search.formats` in its `settings.yml`):

```bash
docker run -d --name searxng -p 8888:8080 searxng/searxng
```

Without it, Neo still works. Web search returns nothing and the error goes to `error.log`.

Tests: `pytest -q` from the project folder.

---

## How it works

### Chat turn

```mermaid
flowchart TD
    A["User message"] --> B{"/search, or needs_search says yes?"}
    B -- yes --> W["web_search (local SearXNG)"]
    B -- no --> R
    W --> R["retrieve_similar: top 5 active facts within the distance cutoff"]
    R --> C["chat: system message with facts and web results, plus last 5 turns"]
    C --> Z["Reply"]
    Z --> S["save_message"]
    S --> Q{"Did a web search run?"}
    Q -- no --> E["extract_facts"]
    Q -- yes --> X["Skip fact extraction"]
    E --> G["process_fact (write gate)"]
```

### Write gate

For every new fact, Neo finds similar active facts (cosine similarity of 0.5 or more). The LLM answers ADD, UPDATE, DELETE or NONE, and plain code validates the answer before anything is written.

```mermaid
flowchart TD
    F["New fact"] --> K["find_candidate: active facts with cosine similarity of 0.5 or more"]
    K -- none --> ADD["ADD"]
    K -- some --> D["decide_action: LLM returns action and target_id"]
    D --> V["apply_decision: validate the answer"]
    V --> ADD
    V --> UPD["UPDATE: retract the old fact, add the new one"]
    V --> DEL["DELETE: retract the old fact"]
    V --> NON["NONE: do nothing"]
```

`apply_decision` falls back to ADD if the LLM names an unknown action or a target that isn't among the candidates. An UPDATE onto an identical fact becomes NONE.

### Storage

| Store | Role |
|---|---|
| SQLite (`ai_history.db`) | Source of truth: chat history (`ai_history`) and facts (`facts`) |
| LanceDB (`lance_store/`) | Vector store: one embedding per fact and per chat turn |

Web results are never stored.

---

## Stress test

A 65-turn script (`harder_test_1.py`, then `harder_test_2.py` as a fresh process) feeds Neo 50 facts and 15 chat-only turns, including a location history that ends in Zurich.

- Each move retracted the previous current location. Born, schooled and first job stayed active.
- Facts about other people (mother, father, sister) were not retracted by the moves.
- Explicit "forget" requests, and stopped or sold items, retracted the old fact.
- None of the 15 chat-only turns created a fact.
- No retracted fact came back through retrieval, including after a restart.

---

## Known issues

Full list in [`KNOWN_ISSUES.md`](KNOWN_ISSUES.md).

- **Comparative preferences aren't extracted.** "I like X more than Y" produces no fact.
- **Some sentences split into fragments.** "I want to work in machine learning in Europe" can become two facts, and the second can retract the first.
- **The retrieval cutoff is loose at larger fact counts.** It was tuned on 8 facts. At about 50, an unrelated question can pull in a loosely related fact.
- **Follow-up web searches lose context.** The raw message is the search query.
- **Quality depends on the model.** Extraction and the gate run on `llama3.1` 8B and can misjudge unusual wording.

---

## Why a rebuild (v1.9 → v1.9.5)

v1.9.5 is a rebuild from v1.8.5, not a patch on v1.9. In testing, v1.9's entity attributes were hard to keep consistent: retractions on workplace didn't hold, and education was dropped after a correction. The goal was the same memory behaviour with far fewer moving parts.

| | v1.9 | v1.9.5 |
|---|---|---|
| Memory model | Entity tables (`entity_attrs`, `entity_mentions`, `semantic_memory`) | One `facts` table plus `ai_history` |
| Write gate | Gazetteer, LLM plausibility, cosine contradiction at 0.8 | Candidate lookup at 0.5, one LLM decision, validated in code |
| Fact extraction | spaCy and YAKE | LLM (`extract_facts`) |
| Retrieval | Jaccard, entity and vector fusion, HNSW index | Vector search on the full prompt, top 5 within a distance cutoff |
| Embeddings | `all-mpnet-base-v2` (local) | `nomic-embed-text` through Ollama |
| Web search | SearXNG, Brave or Tavily, classifier, cache | One local SearXNG URL, LLM gate, no cache |
| Extras | Hardware-aware context sizing, consolidation, backfill | Dropped on purpose |
| Tests | `tests/test_fixes.py` | `test_facts.py`, 26 tests |

v1.9 reported Recall@4 of 0.94 and MRR of 0.88 on its retrieval eval. v1.9.5 has no equivalent benchmark; its retrieval was tuned and checked with the stress test above.

---

## Commands

**Main menu:** `/chat` · `/delete` (wipes chat history, facts and vectors) · `/status` · `/logs` · `/help` · `/exit`

**Inside `/chat`:** `/search <query>` forces a web search (Neo also searches on its own when needed) · `/exit`

**Inside `/logs`:** `/all [error|warning]` · `/ai [error|warning]` · `/clear` · `/clear-ai` · `/help` · `/exit`

---

## Project structure

```
neo-AI/
├── main.py            # Entry point, command loop, Ollama auto-start
├── neo.py             # Chat loop, delete memory, fact saving
├── neo_ollama.py      # LLM calls: chat, embeddings, write-gate decision, extraction, search check
├── storage.py         # SQLite, facts, candidate lookup, write gate, retrieval
├── lancedb_store.py   # LanceDB insert, similarity search, clear
├── web_search.py      # SearXNG client
├── config.py          # First-run username setup
├── logger.py          # Dual logger (system + memory), log viewer CLI
├── file_data.py       # System status checker
└── test_facts.py      # pytest suite
```

<details>
<summary>Database schema</summary>

```sql
CREATE TABLE ai_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt              TEXT NOT NULL,
    response            TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    compressed_keywords TEXT,
    embeddings_id       TEXT
);

CREATE TABLE facts (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    fact_text    TEXT NOT NULL,
    created_at   TEXT NOT NULL,
    status       TEXT NOT NULL,      -- 'active' or 'retracted'
    embedding_id TEXT,               -- UUID of the vector in LanceDB
    retracted_at TEXT
);
```

</details>

---

## Version history

| Version | Changes |
|---|---|
| v1.0 – v1.7 | Initial CLI build, JSON to SQLite migration, logging, pytest suite |
| v1.8 | Local LLM via Ollama, LanceDB vector store, semantic memory retrieval |
| v1.8.5 | Bug fixes across logging, delete logic, embedding guards |
| v1.9 | Entity memory, three-layer write gate, fusion retrieval, multi-backend web search |
| **v1.9.5** | Rebuild from v1.8.5: flat facts table, LLM extraction, one-step write gate, distance-cutoff retrieval, on-demand SearXNG search, 26 tests, 65-turn stress test |

## Project status

v1.9.5 is the final release of this repository. Neo's design assumes a machine that can run `llama3.1` 8B locally, which isn't a realistic baseline for a packaged, download-and-run product, so no further versions are planned here.

## Built with Claude

Neo was co-engineered with Claude (Anthropic): design, code review, debugging, tests and documentation. Claude also wrote some pieces directly, such as the docstrings, the stress-test script and the search-gate function. v1.9.5 was rebuilt step by step from v1.8.5, and the v1.9 it replaces was largely written with Claude.