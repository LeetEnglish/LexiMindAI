"""
Document Parsing Celery Tasks
"""
from app.core.celery_app import celery_app
from app.models.document_parser import get_document_parser


@celery_app.task(bind=True, name="parse_document_async")
def parse_document_async(self, content: str, document_type: str, 
                         filename: str = None, 
                         generate_lessons: bool = True,
                         generate_flashcards: bool = True,
                         callback_url: str = None):
    """
    Async task for parsing documents.
    
    Args:
        content: Base64 encoded document or raw text
        document_type: pdf, docx, or txt
        filename: Original filename
        generate_lessons: Whether to generate lessons
        generate_flashcards: Whether to generate flashcards
        callback_url: Optional URL to POST results
    """
    try:
        # Update state to processing
        self.update_state(state="PROCESSING", meta={"progress": 10})
        
        parser = get_document_parser()
        
        self.update_state(state="PROCESSING", meta={"progress": 30})
        
        result = parser.parse_document(
            content=content,
            document_type=document_type,
            filename=filename,
            generate_lessons=generate_lessons,
            generate_flashcards=generate_flashcards
        )
        
        self.update_state(state="PROCESSING", meta={"progress": 90})
        
        # Callback if specified
        if callback_url:
            try:
                import httpx
                httpx.post(callback_url, json=result, timeout=10)
            except Exception as e:
                print(f"Callback failed: {e}")
        
        return result
        
    except Exception as e:
        self.update_state(state="FAILED", meta={"error": str(e)})
        raise
