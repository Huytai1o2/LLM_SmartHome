import asyncio
import json
import uuid

from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse, ServerSentEvent

from app.db.session import AsyncSessionLocal
from app.models.message import MessageRole
from app.repositories.message_repo import get_history, insert_message
from app.repositories.session_repo import (
    delete_session,
    get_or_create_session,
    get_session,
)
from app.agent_system.runner import clear_session, stream_response

router = APIRouter(prefix="/api/v1")

HEARTBEAT_INTERVAL = 15  # seconds


# ── Request schema ────────────────────────────────────────────────────────────


class ChatRequest(BaseModel):
    session_id: uuid.UUID
    user_id: str
    message: str


# ── SSE helpers ───────────────────────────────────────────────────────────────


def _delta_event(text: str) -> ServerSentEvent:
    return ServerSentEvent(
        event="agent.message.delta",
        data=json.dumps({"text": text}),
    )


def _done_event(session_id: uuid.UUID) -> ServerSentEvent:
    return ServerSentEvent(
        event="agent.message.done",
        data=json.dumps({"session_id": str(session_id)}),
    )


def _failed_event(error: str) -> ServerSentEvent:
    return ServerSentEvent(
        event="agent.workflow.failed",
        data=json.dumps({"error": error}),
    )


def _heartbeat_event() -> ServerSentEvent:
    return ServerSentEvent(event="heartbeat", data=json.dumps({}))


# ── Endpoint ──────────────────────────────────────────────────────────────────


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    """
    Stream an agent response via Server-Sent Events.

    Event flow:
        agent.message.delta  – one per text chunk from the model
        agent.message.done   – when the full reply has been streamed & persisted
        agent.workflow.failed – on any unhandled error
        heartbeat            – every 15 s while the stream is open
    """
    queue: asyncio.Queue[ServerSentEvent | None] = asyncio.Queue()

    async def producer() -> None:
        """Runs the full chat logic and pushes SSE events to the queue."""
        async with AsyncSessionLocal() as db:
            try:
                # 1. Get or create the session (scoped to user_id)
                await get_or_create_session(db, body.session_id, body.user_id)

                # 2. Load prior history for agent context
                history_rows = await get_history(db, body.session_id)
                history = [
                    {"role": row.role.value, "content": row.content}
                    for row in history_rows
                ]

                # 3. Persist the user message BEFORE running the agent
                await insert_message(
                    db, body.session_id, MessageRole.user, body.message
                )
                await db.commit()

                # 4. Stream the agent and emit delta events
                reply_chunks: list[str] = []
                async for chunk in stream_response(
                    body.message, history, body.session_id
                ):
                    reply_chunks.append(chunk)
                    await queue.put(_delta_event(chunk))

                # 5. Persist the full assistant reply AFTER streaming completes
                full_reply = "".join(reply_chunks)
                await insert_message(
                    db, body.session_id, MessageRole.assistant, full_reply
                )
                await db.commit()

                # 6. Signal completion
                await queue.put(_done_event(body.session_id))

            except Exception as exc:  # noqa: BLE001
                await queue.put(_failed_event(str(exc)))

            finally:
                await queue.put(None)  # sentinel — tells the consumer to stop

    async def heartbeat() -> None:
        """Sends a heartbeat event every HEARTBEAT_INTERVAL seconds."""
        while True:
            await asyncio.sleep(HEARTBEAT_INTERVAL)
            await queue.put(_heartbeat_event())

    async def event_generator():
        producer_task = asyncio.create_task(producer())
        heartbeat_task = asyncio.create_task(heartbeat())
        try:
            while True:
                event = await queue.get()
                if event is None:  # sentinel from producer
                    break
                yield event
        finally:
            producer_task.cancel()
            heartbeat_task.cancel()
            # Await cancellations to suppress CancelledError noise
            await asyncio.gather(producer_task, heartbeat_task, return_exceptions=True)

    return EventSourceResponse(event_generator())


# ── Response schemas ──────────────────────────────────────────────────────────


class MessageOut(BaseModel):
    role: str
    content: str
    created_at: datetime


class HistoryResponse(BaseModel):
    session_id: uuid.UUID
    messages: list[MessageOut]


# ── GET /api/v1/sessions/{session_id}/history ─────────────────────────────────


@router.get(
    "/sessions/{session_id}/history",
    response_model=HistoryResponse,
)
async def get_session_history(
    session_id: uuid.UUID,
    user_id: str = Query(..., description="Owner of the session"),
) -> HistoryResponse:
    """Return full message history for a session, scoped to user_id."""
    async with AsyncSessionLocal() as db:
        session = await get_session(db, session_id, user_id)
        if session is None:
            raise HTTPException(status_code=404, detail="Session not found")

        messages = await get_history(db, session_id)

    return HistoryResponse(
        session_id=session_id,
        messages=[
            MessageOut(
                role=m.role.value,
                content=m.content,
                created_at=m.created_at,
            )
            for m in messages
        ],
    )


# ── DELETE /api/v1/sessions/{session_id} ──────────────────────────────────────


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session_endpoint(
    session_id: uuid.UUID,
    user_id: str = Query(..., description="Owner of the session"),
) -> None:
    """Delete a session and all its messages, scoped to user_id."""
    async with AsyncSessionLocal() as db:
        deleted = await delete_session(db, session_id, user_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    # Remove the session's cached agent from memory
    clear_session(str(session_id))
