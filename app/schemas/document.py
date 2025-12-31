"""
Document Parsing Schemas
"""
from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


# ==================== Request Schemas ====================

class DocumentParseRequest(BaseModel):
    """Request to parse a document."""
    content: str = Field(..., description="Base64 encoded document content or raw text")
    document_type: DocumentType = Field(..., description="Type of document")
    filename: Optional[str] = Field(None, description="Original filename")
    generate_lessons: bool = Field(True, description="Generate lesson structure")
    generate_flashcards: bool = Field(True, description="Generate flashcards")


class AsyncParseRequest(DocumentParseRequest):
    """Request for async document parsing."""
    callback_url: Optional[str] = Field(None, description="URL to POST results when complete")


# ==================== Response Schemas ====================

class Flashcard(BaseModel):
    """Generated flashcard."""
    front: str
    back: str
    card_type: str = "VOCABULARY"
    phonetic: Optional[str] = None
    example: Optional[str] = None


class Exercise(BaseModel):
    """Generated exercise."""
    question: str
    question_type: str = "MULTIPLE_CHOICE"
    options: Optional[List[str]] = None
    correct_answer: str
    explanation: Optional[str] = None


class Lesson(BaseModel):
    """Generated lesson from document."""
    title: str
    content: str
    order_index: int
    flashcards: List[Flashcard] = []
    exercises: List[Exercise] = []


class DocumentParseResponse(BaseModel):
    """Response from document parsing."""
    success: bool
    title: str
    summary: Optional[str] = None
    lessons: List[Lesson] = []
    total_flashcards: int = 0
    processing_time_ms: int = 0


class AsyncParseResponse(BaseModel):
    """Response for async parse request."""
    task_id: str
    status: str = "PENDING"
    message: str = "Document parsing started"


class TaskStatusResponse(BaseModel):
    """Status of async task."""
    task_id: str
    status: str  # PENDING, PROCESSING, COMPLETED, FAILED
    progress: int = 0  # 0-100
    result: Optional[DocumentParseResponse] = None
    error: Optional[str] = None
