from dotenv import load_dotenv

load_dotenv()  # must run before any SDK/OpenAI imports read env vars

from fastapi import FastAPI  # noqa: E402

from app.routers import chat, health  # noqa: E402

app = FastAPI(
    title="Multi Agent Research Supervisor Assistance",
    version="1.0.0",
    description="A FastAPI application that provides assistance for multi-agent research supervision tasks, leveraging LLM's capabilities to enhance productivity and efficiency in research management.",
)

app.include_router(health.router)
app.include_router(chat.router)
