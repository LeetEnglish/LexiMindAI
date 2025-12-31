"""
Chat Completion Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class ChatMessage(BaseModel):
    """Single chat message."""
    role: MessageRole
    content: str


class ChatCompletionRequest(BaseModel):
    """Request for chat completion."""
    messages: List[ChatMessage] = Field(..., min_length=1)
    max_tokens: int = Field(256, ge=1, le=2048)
    temperature: float = Field(0.7, ge=0, le=2)
    system_prompt: Optional[str] = Field(
        "You are an English language tutor. Help users improve their English skills.",
        description="System prompt for the AI"
    )


class ChatCompletionResponse(BaseModel):
    """Response from chat completion."""
    message: ChatMessage
    finish_reason: str = "stop"
    tokens_used: int
    processing_time_ms: int
