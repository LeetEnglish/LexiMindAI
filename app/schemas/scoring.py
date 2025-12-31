"""
Scoring Schemas for Writing and Speaking
"""
from pydantic import BaseModel, Field
from typing import Optional


# ==================== Writing Scoring ====================

class WritingScoringRequest(BaseModel):
    """Request to score a writing response."""
    text: str = Field(..., min_length=10, description="The essay/writing to score")
    prompt: Optional[str] = Field(None, description="Original question/prompt")
    max_score: float = Field(10.0, description="Maximum possible score")


class WritingScoreDetail(BaseModel):
    """Detailed writing scores."""
    grammar_score: float = Field(..., ge=0, le=10)
    vocabulary_score: float = Field(..., ge=0, le=10)
    coherence_score: float = Field(..., ge=0, le=10)
    task_achievement_score: float = Field(..., ge=0, le=10)


class WritingScoringResponse(BaseModel):
    """Response from writing scoring."""
    overall_score: float
    max_score: float
    percentage: float
    details: WritingScoreDetail
    feedback: str
    word_count: int
    processing_time_ms: int


# ==================== Speaking Scoring ====================

class SpeakingScoringRequest(BaseModel):
    """Request to score a speaking response."""
    transcript: str = Field(..., min_length=5, description="Transcribed speech text")
    audio_url: Optional[str] = Field(None, description="URL to audio file for analysis")
    prompt: Optional[str] = Field(None, description="Original speaking prompt")
    max_score: float = Field(10.0, description="Maximum possible score")


class SpeakingScoreDetail(BaseModel):
    """Detailed speaking scores."""
    pronunciation_score: float = Field(..., ge=0, le=10)
    fluency_score: float = Field(..., ge=0, le=10)
    vocabulary_score: float = Field(..., ge=0, le=10)
    grammar_score: float = Field(..., ge=0, le=10)


class SpeakingScoringResponse(BaseModel):
    """Response from speaking scoring."""
    overall_score: float
    max_score: float
    percentage: float
    ielts_band_equivalent: float
    details: SpeakingScoreDetail
    feedback: str
    word_count: int
    processing_time_ms: int
