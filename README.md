# Multi-Agent System Base (MASB)

A FastAPI application that orchestrates a **multi-agent AI system** to answer user questions using a combination of live web search and a FAISS-backed retrieval-augmented generation (RAG) knowledge base. All agents are powered by a local or cloud LLM via [smolagents](https://github.com/huggingface/smolagents).

---

## Architecture

```
User (HTTP / SSE)
       │
       ▼
┌─────────────────────────────────────────────┐
│              FastAPI  (app/main.py)          │
│   POST /api/v1/chat/stream   (SSE)           │
│   GET  /api/v1/sessions/{id}/history         │
│   DELETE /api/v1/sessions/{id}               │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│         Runner  (app/agent_system/runner.py) │
│  • Per-session CodeAgent cache               │
│  • Background thread + asyncio.Queue stream  │
└─────────────────┬───────────────────────────┘
                  │
                  ▼
┌─────────────────────────────────────────────┐
│     Manager Agent  (CodeAgent)               │
│  • Delegates to sub-agents                   │
│  • Synthesises final answer                  │
└────────┬────────────────┬────────────────────┘
         │                │
         ▼                ▼
┌─────────────┐  ┌──────────────────────────────┐
│  Web Agent  │  │       Retriever Agent         │
│ (DuckDuckGo │  │ (FAISS + sentence-transformers│
│  search)    │  │  over HuggingFace docs)        │
└─────────────┘  └──────────────────────────────┘
```

### Component overview

| Component                | File                                         | Purpose                                            |
| ------------------------ | -------------------------------------------- | -------------------------------------------------- |
| **FastAPI app**          | `app/main.py`                                | Entry point; pre-warms FAISS on startup            |
| **Chat router**          | `app/routers/chat.py`                        | SSE streaming endpoint + session history/delete    |
| **Health router**        | `app/routers/health.py`                      | `GET /api/v1/health` liveness check                |
| **Manager agent**        | `app/agent_system/orchestrator.py`           | Top-level `CodeAgent` that plans and delegates     |
| **Runner**               | `app/agent_system/runner.py`                 | Async bridge; one `CodeAgent` per session          |
| **LLM model**            | `app/agent_system/model.py`                  | Shared `OpenAIServerModel` (local Ollama or cloud) |
| **Web agent**            | `app/agent_system/agents/web_agent.py`       | `ToolCallingAgent` with DuckDuckGo search          |
| **Retriever agent**      | `app/agent_system/agents/retriever_agent.py` | `ToolCallingAgent` backed by FAISS                 |
| **Vector store**         | `app/vectore_store/`                         | FAISS build, load, embed via `thenlper/gte-small`  |
| **Knowledge base**       | `knowledge_base/sources.py`                  | Loads `m-ric/huggingface_doc` HuggingFace dataset  |
| **DB sessions/messages** | `app/repositories/`                          | PostgreSQL via SQLAlchemy async + Alembic          |

---

## API Endpoints

### `POST /api/v1/chat/stream`

Stream an agent reply via **Server-Sent Events (SSE)**.

**Request body**

```json
{
  "session_id": "<uuid>",
  "user_id": "alice",
  "message": "What is PEFT and how does it work?"
}
```

**SSE event stream**

| Event                   | Payload                 | When                  |
| ----------------------- | ----------------------- | --------------------- |
| `agent.message.delta`   | `{"text": "..."}`       | Each streamed chunk   |
| `agent.message.done`    | `{"session_id": "..."}` | Reply fully persisted |
| `agent.workflow.failed` | `{"error": "..."}`      | Unhandled error       |
| `heartbeat`             | `{}`                    | Every 15 s while open |

**Example**

```bash
SESSION_ID=$(python3 -c 'import uuid; print(uuid.uuid4())')

curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -N \
  -d "{\"session_id\": \"$SESSION_ID\", \"user_id\": \"alice\", \"message\": \"What is PEFT?\"}"
```

---

### `GET /api/v1/sessions/{session_id}/history?user_id=alice`

Retrieve the full message history for a session.

**Response**

```json
{
  "session_id": "<uuid>",
  "messages": [
    { "role": "user", "content": "...", "created_at": "..." },
    { "role": "assistant", "content": "...", "created_at": "..." }
  ]
}
```

---

### `DELETE /api/v1/sessions/{session_id}?user_id=alice`

Delete a session and all its messages. Also evicts the cached agent from memory.

---

## Prerequisites

| Requirement | Version                     |
| ----------- | --------------------------- |
| Python      | 3.11+                       |
| PostgreSQL  | 15+ (or use Docker)         |
| Ollama      | Latest (for local LLM mode) |

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/nguyenhoang2001/Multiple-Agentic-System-Base.git
cd Multiple-Agentic-System-Base
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment

Copy `.env` and fill in the values:

```bash
cp .env.example .env   # or create .env manually
```

**.env variables**

```env
# Database
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/mars

# LLM mode: "local" (Ollama) or "cloud" (Ollama Cloud API)
OLLAMA_MODE=local
LLM_MODEL_ID=qwen3:1.7b
LLAMA_SERVER_URL=http://localhost:11434/v1
OLLAMA_API_KEY=                          # only needed for cloud mode

# Vector store
EMBEDDING_MODEL_NAME=thenlper/gte-small
FAISS_INDEX_PATH=./faiss_index
CHUNK_SIZE=200
CHUNK_OVERLAP=20

# HuggingFace (for downloading the knowledge base dataset)
HF_TOKEN=hf_...
```

### 4. Start PostgreSQL

```bash
docker compose up postgres -d
```

Or point `DATABASE_URL` at any running PostgreSQL instance.

### 5. Run database migrations

```bash
alembic upgrade head
```

### 6. Start the local LLM (local mode only)

```bash
ollama pull qwen3:1.7b
ollama serve
```

### 7. Build the vector store (first time only)

```bash
python -c "from app.vectore_store.builder import build_and_save; build_and_save()"
```

This downloads the `m-ric/huggingface_doc` dataset, splits it into chunks, embeds them with `thenlper/gte-small`, and saves the FAISS index to `faiss_index/`.

### 8. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## Docker (full stack)

Run the entire stack (PostgreSQL + app) with a single command:

```bash
docker compose up --build
```

This will:

1. Start PostgreSQL on port `5432`
2. Build the app image
3. Run `alembic upgrade head`
4. Start the API on port `8000`

> **Note:** You still need a running Ollama instance accessible from within Docker, or switch to `OLLAMA_MODE=cloud` in your `.env`.

---

## LLM Modes

### Local Ollama (default)

Runs models locally via [Ollama](https://ollama.com/).

```env
OLLAMA_MODE=local
LLM_MODEL_ID=qwen3:1.7b
LLAMA_SERVER_URL=http://localhost:11434/v1
```

Recommended local models:

- `qwen3:1.7b` — fast, lightweight (1.4 GB)
- `qwen3:latest` — better quality (5.2 GB)
- `gemma3:4b` — Google's Gemma (3.3 GB)

### Ollama Cloud API

Uses large cloud-hosted models without local download.

```env
OLLAMA_MODE=cloud
LLM_MODEL_ID=gpt-oss:20b-cloud
OLLAMA_API_KEY=your_key_from_https://ollama.com/settings/keys
```

Available cloud models: https://ollama.com/search?c=cloud

---

## Agent System

### Manager Agent (`CodeAgent`)

The top-level orchestrator. It:

- Receives the user query
- Plans which sub-agent(s) to invoke
- Delegates via tool calls to sub-agents
- Synthesises all results into a final answer via `final_answer(...)`

### Web Agent (`ToolCallingAgent`)

- **Tool:** DuckDuckGo search
- **Use case:** Real-time facts, news, locations, anything not in the knowledge base
- **Name:** `search_agent`

### Retriever Agent (`ToolCallingAgent`)

- **Tool:** FAISS semantic search over the knowledge base
- **Knowledge base:** HuggingFace documentation (`m-ric/huggingface_doc`)
- **Embeddings:** `thenlper/gte-small` (sentence-transformers)
- **Use case:** HuggingFace ecosystem questions, PEFT, transformers, datasets, etc.
- **Name:** `retriever_agent`

### Session Memory

Each session gets its own `CodeAgent` instance cached in memory. Multi-turn conversations are supported via smolagents' `reset=False` pattern — the agent retains `memory.steps` across turns within a session.

---

## Extending the Knowledge Base

Add new document sources in `knowledge_base/sources.py`:

```python
from langchain_community.document_loaders import DirectoryLoader, TextLoader

def load_documents():
    hf_docs = _load_hf_dataset()

    # Add local text files
    dir_loader = DirectoryLoader("knowledge_base/files/", glob="**/*.txt",
                                 loader_cls=TextLoader)
    local_docs = dir_loader.load()

    return hf_docs + local_docs
```

Then rebuild the index:

```bash
python -c "from app.vectore_store.builder import build_and_save; build_and_save()"
```

---

## Running Tests

```bash
pytest
```

---

## Project Structure

```
├── app/
│   ├── main.py                        # FastAPI app entry point
│   ├── agent_system/
│   │   ├── model.py                   # Shared LLM (local Ollama or cloud)
│   │   ├── orchestrator.py            # Manager CodeAgent + prompt instructions
│   │   ├── runner.py                  # Async streaming bridge (per-session agents)
│   │   ├── agents/
│   │   │   ├── web_agent.py           # DuckDuckGo web search agent
│   │   │   └── retriever_agent.py     # FAISS retriever agent
│   │   └── tools/
│   │       ├── web_tools.py           # DuckDuckGo + VisitWebpage tools
│   │       └── retriever_tools.py     # FAISS semantic search tool
│   ├── routers/
│   │   ├── chat.py                    # SSE chat endpoint + history/delete
│   │   └── health.py                  # Liveness check
│   ├── vectore_store/
│   │   ├── builder.py                 # Build & persist FAISS index
│   │   ├── embeddings.py              # sentence-transformers embeddings
│   │   ├── loader.py                  # Load persisted FAISS index
│   │   └── store.py                   # Lazy singleton vector store
│   ├── db/                            # SQLAlchemy engine + async session
│   ├── models/                        # ORM models (ChatSession, ChatMessage)
│   └── repositories/                  # DB access layer (session, message)
├── knowledge_base/
│   ├── sources.py                     # Document sources for the knowledge base
│   └── files/                         # Drop local files here for indexing
├── alembic/                           # DB migrations
├── faiss_index/                       # Persisted FAISS index (generated)
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── pytest.ini
```

---

## Tech Stack

| Layer            | Technology                                   |
| ---------------- | -------------------------------------------- |
| API framework    | FastAPI + uvicorn                            |
| Agent framework  | smolagents (HuggingFace)                     |
| LLM              | Ollama (local) or Ollama Cloud               |
| Vector search    | FAISS + LangChain                            |
| Embeddings       | `thenlper/gte-small` (sentence-transformers) |
| Knowledge base   | `m-ric/huggingface_doc` HuggingFace dataset  |
| Web search       | DuckDuckGo (via smolagents)                  |
| Database         | PostgreSQL + SQLAlchemy async + Alembic      |
| Streaming        | Server-Sent Events (SSE) via `sse-starlette` |
| Containerisation | Docker + Docker Compose                      |
