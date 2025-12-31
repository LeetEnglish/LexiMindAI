"""
Document Parsing API Routes
"""
from fastapi import APIRouter, HTTPException
from celery.result import AsyncResult

from app.schemas.document import (
    DocumentParseRequest, DocumentParseResponse,
    AsyncParseRequest, AsyncParseResponse, TaskStatusResponse,
    Lesson, Flashcard, Exercise
)
from app.models.document_parser import get_document_parser
from app.tasks.document_tasks import parse_document_async
from app.core.celery_app import celery_app

router = APIRouter()


@router.post("/parse", response_model=DocumentParseResponse)
async def parse_document(request: DocumentParseRequest):
    """
    Synchronously parse a document and generate lessons/flashcards.
    
    For large documents, use /parse-async instead.
    """
    try:
        parser = get_document_parser()
        result = parser.parse_document(
            content=request.content,
            document_type=request.document_type.value,
            filename=request.filename,
            generate_lessons=request.generate_lessons,
            generate_flashcards=request.generate_flashcards
        )
        
        # Convert to response model
        lessons = [
            Lesson(
                title=l["title"],
                content=l["content"],
                order_index=l["order_index"],
                flashcards=[Flashcard(**f) for f in l.get("flashcards", [])],
                exercises=[Exercise(**e) for e in l.get("exercises", [])]
            )
            for l in result["lessons"]
        ]
        
        return DocumentParseResponse(
            success=result["success"],
            title=result["title"],
            summary=result.get("summary"),
            lessons=lessons,
            total_flashcards=result["total_flashcards"],
            processing_time_ms=result["processing_time_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Parsing failed: {str(e)}")


@router.post("/parse-async", response_model=AsyncParseResponse)
async def parse_document_async_endpoint(request: AsyncParseRequest):
    """
    Asynchronously parse a document using Celery background task.
    
    Returns a task ID that can be polled for status.
    """
    try:
        task = parse_document_async.delay(
            content=request.content,
            document_type=request.document_type.value,
            filename=request.filename,
            generate_lessons=request.generate_lessons,
            generate_flashcards=request.generate_flashcards,
            callback_url=request.callback_url
        )
        
        return AsyncParseResponse(
            task_id=task.id,
            status="PENDING",
            message="Document parsing started in background"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start task: {str(e)}")


@router.get("/task/{task_id}", response_model=TaskStatusResponse)
async def get_task_status(task_id: str):
    """
    Get the status of an async parsing task.
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        
        response = TaskStatusResponse(
            task_id=task_id,
            status=result.status,
            progress=0
        )
        
        if result.state == "PENDING":
            response.progress = 0
        elif result.state == "PROCESSING":
            meta = result.info or {}
            response.progress = meta.get("progress", 50)
        elif result.state == "SUCCESS":
            response.status = "COMPLETED"
            response.progress = 100
            # Convert result to response model
            data = result.result
            lessons = [
                Lesson(
                    title=l["title"],
                    content=l["content"],
                    order_index=l["order_index"],
                    flashcards=[Flashcard(**f) for f in l.get("flashcards", [])],
                    exercises=[Exercise(**e) for e in l.get("exercises", [])]
                )
                for l in data.get("lessons", [])
            ]
            response.result = DocumentParseResponse(
                success=data["success"],
                title=data["title"],
                summary=data.get("summary"),
                lessons=lessons,
                total_flashcards=data["total_flashcards"],
                processing_time_ms=data["processing_time_ms"]
            )
        elif result.state == "FAILED":
            response.error = str(result.info) if result.info else "Unknown error"
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")
