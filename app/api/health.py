"""
Health Check Router
"""
from fastapi import APIRouter
from datetime import datetime

router = APIRouter()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "LexiAI",
        "timestamp": datetime.utcnow().isoformat()
    }


@router.get("/ready")
async def readiness_check():
    """Readiness check - verify models are loaded."""
    # TODO: Check if models are actually loaded
    return {
        "ready": True,
        "models_loaded": True,
        "timestamp": datetime.utcnow().isoformat()
    }
