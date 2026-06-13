# Neo AI — v1.8.5

[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/powered%20by-Ollama-black)](https://ollama.com/)

Every time you close ChatGPT, it forgets you. Neo is built to fix that — locally, permanently, privately. No cloud. No API keys. No data leaves your machine.

> **Disclaimer:** Neo runs a local LLM. Like all language models, it can make mistakes, hallucinate, or produce incorrect information. Do not rely on it for critical decisions.

---

## Known Issues

- **Ollama cold start latency** — if Ollama isn't already running, startup may take up to 10 seconds
- **Short input keyword extraction** — YAKE fails on single words or numbers (e.g. "18", "1945"); no embedding stored for those turns, so they won't appear in semantic retrieval
- **High response latency** — dependent on local hardware; slower machines will see noticeable delays
- **Model hallucination** — the underlying LLM may hallucinate on ambiguous or very short inputs; model-layer limitation, not a Neo bug

---

## How It Works

```
User prompt
    │
    ▼
YAKE keyword extraction
    │
    ▼
nomic-embed-text (Ollama) ──► LanceDB ANN search
                                        │
                                        ▼
                              Match embedding IDs → SQLite JOIN
                                        │
                                        ▼
                              Top 3 similar past conversations
                                        │
                ┌───────────────────────┘
                ▼
    Build context window:
    [system prompt]
    + [similar past exchanges]     ← semantic memory
    + [last 5 conversations]       ← recent history
    + [current prompt]
                │
                ▼
        llama3.1 (Ollama) → response
                │
                ▼
    Save to SQLite + embed keywords → LanceDB
```

Two stores, one purpose:

| Store | Role |
|---|---|
| SQLite (`ai_history.db`) | Source of truth — prompt, response, timestamp, keywords, vector reference ID |
| LanceDB (`lance_store/`) | Vector store — embeddings keyed to SQLite rows, used for ANN search only |

---

## Project Structure

```
neo-AI/
├── main.py            # Entry point, command loop, Ollama auto-start
├── neo.py             # Chat loop, memory delete
├── neo_ollama.py      # Ollama LLM chat + embedding calls
├── storage.py         # SQLite ops, YAKE keyword extraction, retrieval
├── lancedb_store.py   # LanceDB insert, ANN search, clear
├── config.py          # First-run username setup, base64 persistence
├── logger.py          # Dual logger (system + memory), log viewer CLI
└── file_data.py       # System status checker
```

---

## Screenshots and Demos

### Demos
![Neo AI Main Demo](demo.gif/neo_ai_demo.gif)
![Neo AI Retrieval Demo](demo.gif/neo_ai_retrieval.gif)

### Chat
![Chat](screenshots/chat.png)

### System Status
![Status](screenshots/status.png)

### Logs
![Logs](screenshots/logs.png)

### Memory Delete
![Delete](screenshots/delete.png)

## Database Schema

```sql
CREATE TABLE ai_history (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt              TEXT NOT NULL,
    response            TEXT NOT NULL,
    timestamp           TEXT NOT NULL,
    compressed_keywords TEXT,
    embeddings_id       TEXT
);
```

`embeddings_id` is a UUID4 that maps to the corresponding vector row in LanceDB.

---

## Setup

**Requirements:** Python 3.10+, [Ollama](https://ollama.com/) installed

```bash
git clone https://github.com/Neo-X7/neo-AI.git
cd neo-AI
pip install ollama lancedb pyarrow yake rich httpx
```

Pull required models:

```bash
ollama pull llama3.1
ollama pull nomic-embed-text
```

Run:

```bash
python main.py
```

Neo will attempt to start the Ollama server automatically if it isn't already running.

---

## Commands

**Main menu:**

```
/chat      start a conversation
/delete    wipe all memory (SQLite + LanceDB)
/status    system status
/logs      open log viewer
/help      show commands
/exit      exit from neo
```

**Inside `/chat`:**

```
/exit      return to main menu
```

**Inside `/logs`:**

```
/all       system logs
/ai        memory event logs
/clear     wipe system logs
/back      return to main menu
```

---

## Version History

| Version | Changes |
|---|---|
| v1.0 | Initial release, basic CRUD, JSON storage |
| v1.1 | Logic fixes, UUID generation, JSON serialization |
| v1.2 | Username indexing, UI effects |
| v1.3 | Validation fixes, separate delete operations |
| v1.4 | Atomic saves, file verification, case-insensitive search |
| v1.5 | Duplicate fix, partial search, standard logging |
| v1.6 | Full SQLite migration, context manager pattern |
| v1.7 | Type hints, pytest suite, rich formatting, schema hardening |
| v1.8 | Local LLM via Ollama, LanceDB vector store, semantic memory retrieval, keyword extraction, WAL mode, config persistence |
| v1.8.5 | Bug fixes: NameError on embedding success log, delete_memory logic inversion, ai_log_info() no-args crash, console.print style misplacement, redundant conn.commit() in save_message, duplicate logger imports — error log now shown in /status — short input embedding guard added |

---

## Roadmap

| Version | Planned |
|---|---|
| v1.9 | CLI command (`/forget`), entity extraction column, privacy search (DuckDuckGo, SearXNG) |
| v2.0 | Final public release|