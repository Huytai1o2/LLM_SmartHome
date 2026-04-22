# IoT Smart Home Agent System

A comprehensive multi-agent AI system designed to intelligently control and query smart home IoT devices via the CoreIoT (Thingsboard) API. The system consists of a FastAPI backend powered by local/cloud LLMs via [smolagents](https://github.com/huggingface/smolagents), a Next.js web frontend, and a PlatformIO hardware testing suite.

> **Note:** The base agentic system architecture is forked from [nguyenhoang2001/Multiple-Agentic-System-Base](https://github.com/nguyenhoang2001/Multiple-Agentic-System-Base).

---

## 🧠 Memory & Data Architecture

The system implements a sophisticated 3-tier memory architecture to handle context seamlessly:

1. **Database (Permanent Storage):** PostgreSQL with SQLAlchemy & Alembic stores every chat session and message permanently.
2. **Vector Store / RAG (Long-term Context):** FAISS asynchronously embeds conversational history (`faiss_index/conversation_memory`) and IoT knowledge bases (`knowledge_base/`). It allows agents to semantically recall distant context or automation rules.
3. **Buffer Window Memory (Hot Cache):** A sliding window of the most recent device actions is kept dynamically in `memories/sessions/*.jsonl`. This allows the Orchestrator to resolve missing entities (like "turn on the light" -> figuring out *which* room based on recent actions) without re-querying the user or the database.

---

## 🏗️ System Architecture & Execution Flow

To ensure high reliability with small local LLMs (like `gemma4:e2b` or `qwen3:1.7b`) and strict token economy, the system avoids generating generic code to parse large configurations. Instead, it uses a deterministic pipeline with **Structured Output (JSON)** and **Token-Saving YAML Embeddings**:

```text
   Next.js Frontend (Web UI)
             │
             ▼ (HTTP / SSE)
┌────────────────────────────────────────────────────────┐
│                   FastAPI Backend                      │
│        (Routing, DB Sessions, FAISS pre-warming)       │
└─────────────────────────┬──────────────────────────────┘
                          │ (Runner & Streaming Bridge)
                          ▼
┌────────────────────────────────────────────────────────┐
│           Orchestrator Agent (Manager Pipeline)        │
│                                                        │
│ 1. Parse Intent:  LLM -> Structured JSON Array         │
│ 2. Check Buffer:  Fill missing intents (Room/Device)   │
│ 3. Clarify:       Ask User if intent still missing     │
│ 4. Filter config: JSON -> Convert YAML                 │
│ 5. Select Device: YAML Prompt -> Structured JSON       │
│ 6. Execute:       CodeAgent -> CoreIoT API             │
│ 7. Reply:         LLM natural response generation      │
└───────┬──────────────────┬────────────────┬────────────┘
        │                  │                │
        ▼                  ▼                ▼
┌──────────────┐  ┌─────────────────┐  ┌─────────────────┐
│Clarification │  │ JSON Registry   │  │ IoT Action Agent│
│    Agent     │  │ (Source of Truth)  │ (CoreIoT APIs)  │
└──────────────┘  └─────────────────┘  └────────┬────────┘
                                                │
                                                ▼
                                      ┌──────────────────┐
                                      │   CoreIoT API    │
                                      │  (Thingsboard)   │
                                      └──────────────────┘
```

### ⚙️ The Token-Saving Pipeline (Deep Dive)

1. **Extract Intent (JSON):** The user's query is passed to an LLM strictly formatted via Pydantic to extract an array of intents: `[{"room_name": "...", "type_device": "..."}]`.
2. **Buffer Lookup:** If `room_name` or `type_device` is missing (e.g. "turn *it* off"), the Orchestrator checks the `Buffer Window Memory` to resolve the immediate context.
3. **Clarify:** If ambiguity persists, the `Clarification Agent` intervenes by returning a question to the user.
4. **JSON → Convert YAML (Save Tokens):** `iterate_smart_home_yaml` filters the large master `smart_home_configuration.json` device registry using the extracted intent keywords. It isolates only the matching device node (subtree) and dynamically converts it into a `YAML` string.
5. **Embedded Prompt → Structured Output:** This minimal `YAML` string is embedded into the prompt for the Retreiver/Device Selector. Because YAML is extremely token-efficient compared to JSON, the LLM processes it cheaply and responds with a strict, validated `DeviceAction` **JSON list** (containing exact tokens and API IDs).
6. **Execute (CodeAgent):** The `IoT Action Agent` (the only actual `CodeAgent` in the pipeline) absorbs the validated JSON list, categorizes read vs. write actions, and securely hits the CoreIoT API.
7. **Update Buffer, Save History & Reply:** Successful actions are appended to the session's Buffer Window Memory. A final LLM call transforms raw API output into a friendly, natural language response. Finally, the entire turn (user query and system result) is asynchronously embedded into the FAISS Vector Store for long-term semantic recall.

---

## 🚀 Quick Start

### Method 1: All-in-One Start Script (Recommended)
This script sets up the virtual environment, starts PostgreSQL via Docker, runs migrations, and launches both the backend and frontend.

```bash
# 1. Provide environment variables (First time only)
cp .env.example .env
# Edit .env with your specific tokens, DB URLs, and LLM configurations

# 2. Pull local LLM (If OLLAMA_MODE=local)
ollama pull gemma4:e2b && ollama serve

# 3. Make executable and run
chmod +x start.sh
./start.sh
```
* **Web Chat (Frontend):** `http://localhost:3000`
* **Backend API:** `http://localhost:8000`
*(Press `Ctrl + C` to gracefully stop all services).*

---

### Method 2: Manual Setup

**1. Backend:**
```bash
python -m venv .IotAgent_venv
source .IotAgent_venv/bin/activate
pip install -r requirements.txt

cp .env.example .env

# Start Database
docker-compose up postgres -d
alembic upgrade head

# Start API
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**2. Frontend:**
```bash
# In a new terminal
cd frontend
npm install
npm run dev
```

---

## 📂 Project Structure

| Path                    | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| `app/agent_system/`     | 8-step Deterministic Orchestrator, memory buffers, CodeAgents |
| `app/vectore_store/`    | FAISS indexing logic for Conversation & Knowledge Base        |
| `app/` (rest)           | FastAPI routing, Database SQLAlchemy models, repo layer       |
| `knowledge_base/`       | Automation rules, examples, and the master `json` registry    |
| `memories/sessions/`    | Per-session `.jsonl` sliding buffer window logs               |
| `frontend/`             | Next.js web interface for interacting with the agents         |
| `HardwareTest_CoreIoT/` | PlatformIO C++ firmware to simulate physical ESP32/IoT nodes  |
| `alembic/`              | Database migration versions                                   |

---

## 🤖 LLM Strategy

The system easily toggles between local and cloud models via the `.env` file. By adhering strictly to Structured JSON Outputs and constrained CodeAgents, the system achieves very high reliability on smaller local models.

**Local Mode (Ollama - Default)**
```env
OLLAMA_MODE=local
LLM_MODEL_ID=gemma4:e2b
```

**Cloud Mode (API)**
```env
OLLAMA_MODE=cloud
LLM_MODEL_ID=gpt-oss:20b-cloud
OLLAMA_API_KEY=your_api_key_here
```

---

## 🛠 Hardware Testing

For physical device simulation or testing against real hardware, check the `HardwareTest_CoreIoT/` folder. It uses **PlatformIO** and enables flashing mock endpoints to ESP32/Yolo Uno boards via `platformio.ini`.

## 📸 System in Action & Usecases

Here are some real-world examples demonstrating how the AI Agents parse natural language, resolve missing contexts, and execute Home commands on the hardware:

![11-29-03](pictures/Screenshot%20from%202026-04-17%2011-29-03.png)
![11-29-11](pictures/Screenshot%20from%202026-04-17%2011-29-11.png)
![11-29-19](pictures/Screenshot%20from%202026-04-17%2011-29-19.png)
![11-29-34](pictures/Screenshot%20from%202026-04-17%2011-29-34.png)
![11-29-40](pictures/Screenshot%20from%202026-04-17%2011-29-40.png)
![11-29-48](pictures/Screenshot%20from%202026-04-17%2011-29-48.png)
![11-29-54](pictures/Screenshot%20from%202026-04-17%2011-29-54.png)
![11-30-01](pictures/Screenshot%20from%202026-04-17%2011-30-01.png)
![11-30-06](pictures/Screenshot%20from%202026-04-17%2011-30-06.png)
![11-30-12](pictures/Screenshot%20from%202026-04-17%2011-30-12.png)
![11-30-19](pictures/Screenshot%20from%202026-04-17%2011-30-19.png)
![11-30-22](pictures/Screenshot%20from%202026-04-17%2011-30-22.png)
