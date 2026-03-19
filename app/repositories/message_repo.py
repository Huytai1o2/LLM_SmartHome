import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.message import ChatMessage, MessageRole


async def insert_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: MessageRole,
    content: str,
) -> ChatMessage:
    """Persist a single chat message and flush it to the DB."""
    message = ChatMessage(
        session_id=session_id,
        role=role,
        content=content,
    )
    db.add(message)
    await db.flush()
    return message


async def get_history(
    db: AsyncSession,
    session_id: uuid.UUID,
) -> list[ChatMessage]:
    """Return all messages for a session ordered by creation time."""
    result = await db.execute(
        select(ChatMessage)
        .where(ChatMessage.session_id == session_id)
        .order_by(ChatMessage.created_at)
    )
    return list(result.scalars().all())
