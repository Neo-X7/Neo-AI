
# Neo AI — v1.8
 
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/powered%20by-Ollama-black)](https://ollama.com/)
 
A fully offline personal AI assistant. Runs a local LLM, stores conversations in SQLite, and retrieves semantically relevant past context before every response. No cloud. No API keys. No data leaves your machine.
 
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
/quit      exit
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
 
---
 
## Roadmap
 
| Version | Planned |
|---|---|
| v1.9 | CLI commands (`/ask`, `/forget`, `/status`), entity extraction column, privacy search (DuckDuckGo, SearXNG) |
| v2.0 | Final public release — repo goes private after release |