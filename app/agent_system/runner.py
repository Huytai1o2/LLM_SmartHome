"""
Runner – async interface between the FastAPI layer and the smolagents Manager Agent.

smolagents' CodeAgent.run() is synchronous, so we execute it in a background
thread and stream chunks back to the caller via an asyncio.Queue so the event
loop stays unblocked.

--- Official smolagents chat-history pattern ---
From the smolagents docs (GradioUI source + memory tutorial):

    # First turn – reset=True clears memory (default behaviour)
    agent.run(user_request, reset=True)

    # Subsequent turns in the same conversation – reset=False keeps memory
    agent.run(user_request, reset=False)

When reset=False, the agent preserves agent.memory.steps (TaskStep, ActionStep,
PlanningStep objects) so the model sees the full conversation history naturally.

Because multiple sessions can be active simultaneously we maintain a
per-session agent cache (_SESSION_AGENTS). Each session owns its own CodeAgent
instance so their memories never bleed into one another.

--- Streaming strategy ---
1. A step callback pushes each step's observations to the queue for live UX.
2. Once the agent finishes, the final answer is the last yielded chunk.
3. A sentinel None signals the generator to stop.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from collections.abc import AsyncGenerator
from threading import Thread

from smolagents import CodeAgent
from smolagents.memory import ActionStep

from app.agent_system.orchestrator import manager_agent, _INSTRUCTIONS

logger = logging.getLogger(__name__)

# Per-session agent cache  {session_id_str -> CodeAgent}
_SESSION_AGENTS: dict[str, CodeAgent] = {}


def _make_agent() -> CodeAgent:
    """Create a fresh per-session CodeAgent mirroring the manager_agent config."""
    agent = CodeAgent(
        tools=list(manager_agent.tools.values()),
        model=manager_agent.model,
        managed_agents=list(manager_agent.managed_agents.values()),
        max_steps=manager_agent.max_steps,
        additional_authorized_imports=manager_agent.authorized_imports,
        verbosity_level=manager_agent.logger.level,
        instructions=_INSTRUCTIONS,
    )

    # Attach per-request routing state (set before each run, cleared after).
    agent._current_queue: asyncio.Queue | None = None
    agent._current_loop: asyncio.AbstractEventLoop | None = None

    # Register a single persistent callback using the CallbackRegistry API.
    def _dispatch(step_log: ActionStep) -> None:
        q = agent._current_queue
        lp = agent._current_loop
        if q is None or lp is None:
            return
        obs = getattr(step_log, "observations", None)
        if obs and isinstance(obs, str) and obs.strip():
            lp.call_soon_threadsafe(q.put_nowait, obs.strip() + "\n")

    agent.step_callbacks.register(ActionStep, _dispatch)
    return agent


def _get_agent(session_id: str) -> tuple[CodeAgent, bool]:
    """
    Return (agent, is_new) for the given session.
    Creates a new agent the first time a session is seen.
    """
    if session_id not in _SESSION_AGENTS:
        _SESSION_AGENTS[session_id] = _make_agent()
        return _SESSION_AGENTS[session_id], True
    return _SESSION_AGENTS[session_id], False


def clear_session(session_id: str) -> None:
    """Remove a session's agent from the cache (call on session delete)."""
    _SESSION_AGENTS.pop(session_id, None)


async def stream_response(
    message: str,
    history: list[dict],
    session_id: str | uuid.UUID,
) -> AsyncGenerator[str, None]:
    """
    Async generator that yields text chunks as the agent processes the query.

    The official smolagents pattern for multi-turn conversation:
      - First turn  → agent.run(task, reset=True)   clears memory
      - Follow-ups  → agent.run(task, reset=False)  keeps memory

    Args:
        message:    The latest user message.
        history:    Prior turns as [{"role": "user"|"assistant", "content": str}].
                    Used only to decide whether this is a first turn (reset=True)
                    or a follow-up (reset=False). The actual history lives in
                    agent.memory.steps on the cached agent instance.
        session_id: Unique session identifier used to look up the cached agent.

    Yields:
        str – each text chunk (step observation or final answer).
    """
    sid = str(session_id)
    agent, is_new = _get_agent(sid)

    # reset=True on the first turn of a session; reset=False afterwards
    reset = is_new or not history

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[str | None] = asyncio.Queue()

    # ------------------------------------------------------------------
    # Background thread: run the synchronous CodeAgent.run().
    # The persistent callback registered in _make_agent() routes
    # observations to agent._current_queue / agent._current_loop.
    # ------------------------------------------------------------------
    def _run() -> None:
        agent._current_queue = queue
        agent._current_loop = loop
        try:
            # Official pattern: reset=False preserves agent.memory.steps
            result = agent.run(message, reset=reset)
            loop.call_soon_threadsafe(queue.put_nowait, str(result))
        except Exception as exc:  # noqa: BLE001
            logger.exception("Agent raised an exception (session=%s)", sid)
            loop.call_soon_threadsafe(queue.put_nowait, f"Agent error: {exc}")
        finally:
            agent._current_queue = None
            agent._current_loop = None
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

    thread = Thread(target=_run, daemon=True)
    thread.start()

    while True:
        chunk = await queue.get()
        if chunk is None:
            break
        yield chunk
