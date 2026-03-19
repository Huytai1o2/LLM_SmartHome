from fastapi import APIRouter
from pydantic import BaseModel


router = APIRouter(prefix="/api/v1")


class HealthResponse(BaseModel):
    status: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Returns 200 OK to confirm the service is running."""
    return HealthResponse(status="ok")
