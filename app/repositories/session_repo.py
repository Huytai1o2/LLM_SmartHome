import uuid
from datetime import datetime, timezone

from sqlalchemy import select, delete
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.session import ChatSession


async def get_or_create_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: str,
) -> ChatSession:
    """Return an existing session that belongs to user_id, or create it."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    session = result.scalar_one_or_none()

    if session is None:
        session = ChatSession(
            id=session_id,
            user_id=user_id,
        )
        db.add(session)
        await db.flush()  # assign PK without committing the transaction

    return session


async def get_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: str,
) -> ChatSession | None:
    """Fetch a session only if it belongs to user_id."""
    result = await db.execute(
        select(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    return result.scalar_one_or_none()


async def delete_session(
    db: AsyncSession,
    session_id: uuid.UUID,
    user_id: str,
) -> bool:
    """Delete a session and all its messages (cascade). Returns True if deleted."""
    result = await db.execute(
        delete(ChatSession).where(
            ChatSession.id == session_id,
            ChatSession.user_id == user_id,
        )
    )
    await db.commit()
    return result.rowcount > 0
