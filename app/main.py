from dotenv import load_dotenv

load_dotenv()  # must run before any SDK/OpenAI imports read env vars

from contextlib import asynccontextmanager  # noqa: E402

from fastapi import FastAPI  # noqa: E402

from app.routers import chat, health  # noqa: E402
from app.vectore_store.store import get_vector_store  # noqa: E402
import logging  # noqa: E402

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-warm the FAISS vector store on startup so the first request is fast."""
    import asyncio

    loop = asyncio.get_running_loop()
    logger.info("Pre-warming FAISS vector store...")
    await loop.run_in_executor(None, get_vector_store)
    logger.info("FAISS vector store ready.")
    yield


app = FastAPI(
    title="Multi Agent Research Supervisor Assistance",
    version="1.0.0",
    description="A FastAPI application that provides assistance for multi-agent research supervision tasks, leveraging LLM's capabilities to enhance productivity and efficiency in research management.",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(chat.router)
