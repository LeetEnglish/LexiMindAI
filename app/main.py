"""
LexiAI - FastAPI Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import get_settings
from app.api import health, document, scoring, chat

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup: Load models
    print("🚀 Starting LexiAI Service...")
    print(f"📦 Document Model: {settings.hf_document_model}")
    print(f"📦 Grammar Model: {settings.hf_grammar_model}")
    print(f"📦 Chat Model: {settings.hf_chat_model}")
    yield
    # Shutdown
    print("👋 Shutting down LexiAI Service...")


app = FastAPI(
    title="LexiAI",
    description="AI Microservice for English Learning Platform",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routers
app.include_router(health.router, tags=["Health"])
app.include_router(document.router, prefix=f"{settings.api_prefix}/document", tags=["Document"])
app.include_router(scoring.router, prefix=f"{settings.api_prefix}/scoring", tags=["Scoring"])
app.include_router(chat.router, prefix=f"{settings.api_prefix}/chat", tags=["Chat"])


@app.get("/")
async def root():
    return {"service": "LexiAI", "status": "running"}
