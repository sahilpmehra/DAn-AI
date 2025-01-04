from fastapi import APIRouter
from app.api.v1.endpoints import upload, analysis

api_router = APIRouter()

# Include routers from endpoints
api_router.include_router(
    upload.router,
    prefix="/upload",
    tags=["upload"]
)

api_router.include_router(
    analysis.router,
    prefix="/analyze",
    tags=["analyze"]
)

# Health check endpoint
@api_router.get("/health")
async def health_check():
    return {"status": "healthy"} 