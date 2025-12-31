"""
Scoring API Routes
"""
from fastapi import APIRouter, HTTPException
from app.schemas.scoring import (
    WritingScoringRequest, WritingScoringResponse, WritingScoreDetail,
    SpeakingScoringRequest, SpeakingScoringResponse, SpeakingScoreDetail
)
from app.models.scoring_model import get_scoring_model

router = APIRouter()


@router.post("/writing", response_model=WritingScoringResponse)
async def score_writing(request: WritingScoringRequest):
    """
    Score a writing response using AI evaluation.
    
    Evaluates on 4 dimensions:
    - Grammar & Mechanics
    - Vocabulary & Word Choice
    - Coherence & Cohesion
    - Task Achievement
    """
    try:
        model = get_scoring_model()
        result = model.score_writing(request.text, request.prompt)
        
        # Convert to response model
        percentage = (result["overall_score"] / 10) * 100
        scaled_score = (result["overall_score"] / 10) * request.max_score
        
        return WritingScoringResponse(
            overall_score=round(scaled_score, 2),
            max_score=request.max_score,
            percentage=round(percentage, 2),
            details=WritingScoreDetail(**result["details"]),
            feedback=result["feedback"],
            word_count=result["word_count"],
            processing_time_ms=result["processing_time_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")


@router.post("/speaking", response_model=SpeakingScoringResponse)
async def score_speaking(request: SpeakingScoringRequest):
    """
    Score a speaking response using AI evaluation.
    
    Evaluates on 4 dimensions:
    - Pronunciation
    - Fluency & Coherence
    - Lexical Resource (Vocabulary)
    - Grammatical Range & Accuracy
    """
    try:
        model = get_scoring_model()
        result = model.score_speaking(request.transcript, request.audio_url)
        
        # Convert to response model
        percentage = (result["overall_score"] / 10) * 100
        scaled_score = (result["overall_score"] / 10) * request.max_score
        
        return SpeakingScoringResponse(
            overall_score=round(scaled_score, 2),
            max_score=request.max_score,
            percentage=round(percentage, 2),
            ielts_band_equivalent=result["ielts_band_equivalent"],
            details=SpeakingScoreDetail(**result["details"]),
            feedback=result["feedback"],
            word_count=result["word_count"],
            processing_time_ms=result["processing_time_ms"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scoring failed: {str(e)}")
