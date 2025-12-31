"""
Chat Completion API Routes
"""
from fastapi import APIRouter, HTTPException

from app.schemas.chat import (
    ChatCompletionRequest, ChatCompletionResponse, ChatMessage, MessageRole
)
from app.models.chat_model import get_chat_model_instance

router = APIRouter()


@router.post("/complete", response_model=ChatCompletionResponse)
async def chat_complete(request: ChatCompletionRequest):
    """
    Generate a chat completion for tutoring conversation.
    
    The AI acts as an English language tutor.
    """
    try:
        model = get_chat_model_instance()
        
        # Convert pydantic models to dicts
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]
        
        result = model.complete(
            messages=messages,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            system_prompt=request.system_prompt
        )
        
        return ChatCompletionResponse(
            message=ChatMessage(
                role=MessageRole.ASSISTANT,
                content=result["message"]["content"]
            ),
            finish_reason=result["finish_reason"],
            tokens_used=result["tokens_used"],
            processing_time_ms=result["processing_time_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chat completion failed: {str(e)}")
