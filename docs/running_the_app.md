# Running the App

This guide covers how to start the server and interact with the multi-agent system via the HTTP API.

---

## Prerequisites

- Dependencies installed: `pip install -r requirements.txt`
- PostgreSQL running (see `docker-compose.yml`)
- `.env` file configured with required environment variables
- FAISS index built (see [vector_store_and_knowledge_base.md](vector_store_and_knowledge_base.md))

Start the database:

```bash
docker compose up -d
```

---

## Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

| Flag             | Purpose                                                       |
| ---------------- | ------------------------------------------------------------- |
| `--host 0.0.0.0` | Accept connections from any interface                         |
| `--port 8000`    | Port to listen on                                             |
| `--reload`       | Auto-restart on code changes (dev only, remove in production) |

Once running, the interactive API docs are available at:

- Swagger UI → http://localhost:8000/docs
- ReDoc → http://localhost:8000/redoc

---

## API endpoints

| Method   | Path                                    | Description                       |
| -------- | --------------------------------------- | --------------------------------- |
| `POST`   | `/api/v1/chat/stream`                   | Stream an agent reply via SSE     |
| `GET`    | `/api/v1/sessions/{session_id}/history` | Fetch full message history        |
| `DELETE` | `/api/v1/sessions/{session_id}`         | Delete a session and its messages |

---

## POST /api/v1/chat/stream

Sends a message to the agent and streams the response back as
[Server-Sent Events (SSE)](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events).

### Request body

```json
{
  "session_id": "<uuid-v4>",
  "user_id": "alice",
  "message": "Your question here"
}
```

| Field        | Type    | Description                                                             |
| ------------ | ------- | ----------------------------------------------------------------------- |
| `session_id` | UUID v4 | Unique conversation identifier — generate one and reuse it across turns |
| `user_id`    | string  | Owner of the session — used to scope history and access control         |
| `message`    | string  | The message to send to the agent                                        |

### SSE event types

| Event                   | Payload                 | Description                                             |
| ----------------------- | ----------------------- | ------------------------------------------------------- |
| `agent.message.delta`   | `{"text": "..."}`       | One chunk of the agent's reply — arrives multiple times |
| `agent.message.done`    | `{"session_id": "..."}` | Reply is complete and persisted to the database         |
| `agent.workflow.failed` | `{"error": "..."}`      | An unhandled error occurred                             |
| `heartbeat`             | `{}`                    | Sent every 15 s to keep the connection alive            |

---

## curl examples

### Ask the agent a question (stream)

```bash
SESSION_ID="$(python3 -c 'import uuid; print(uuid.uuid4())')"

curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -N \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"user_id\": \"alice\",
    \"message\": \"What is PEFT and how does it work?\"
  }"
```

> `-N` disables buffering so SSE chunks are printed as they arrive.

**Example output:**

```
event: agent.message.delta
data: {"text": "PEFT stands for Parameter-Efficient Fine-Tuning"}

event: agent.message.delta
data: {"text": ". It allows you to adapt large language models..."}

event: agent.message.done
data: {"session_id": "3f2a1b4c-..."}
```

### Continue the same conversation (multi-turn)

Reuse the same `SESSION_ID` — prior messages are loaded from the database automatically:

```bash
curl -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -H "Accept: text/event-stream" \
  -N \
  -d "{
    \"session_id\": \"$SESSION_ID\",
    \"user_id\": \"alice\",
    \"message\": \"Can you give me a code example?\"
  }"
```

### Get conversation history

```bash
curl "http://localhost:8000/api/v1/sessions/$SESSION_ID/history?user_id=alice"
```

**Example response:**

```json
{
  "session_id": "3f2a1b4c-...",
  "messages": [
    {
      "role": "user",
      "content": "What is PEFT?",
      "created_at": "2026-03-21T10:00:00"
    },
    {
      "role": "assistant",
      "content": "PEFT stands for ...",
      "created_at": "2026-03-21T10:00:05"
    }
  ]
}
```

### Delete a session

```bash
curl -X DELETE "http://localhost:8000/api/v1/sessions/$SESSION_ID?user_id=alice"
```

Returns `204 No Content` on success.

---

## Health check

```bash
curl http://localhost:8000/health
```
