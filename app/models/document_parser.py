"""
Document Parsing Model
Uses Hugging Face for summarization and content extraction.
"""
import time
import base64
from typing import List, Dict, Optional
from functools import lru_cache

from app.core.config import get_settings

settings = get_settings()

# Lazy load model
_summarizer = None


def get_summarizer():
    """Lazy load summarization model."""
    global _summarizer
    if _summarizer is None:
        try:
            from transformers import pipeline
            _summarizer = pipeline("summarization", model=settings.hf_document_model)
            print(f"✅ Loaded summarization model: {settings.hf_document_model}")
        except Exception as e:
            print(f"⚠️ Failed to load summarization model: {e}")
            _summarizer = "mock"
    return _summarizer


class DocumentParser:
    """Document parsing and lesson generation."""
    
    def parse_document(self, content: str, document_type: str, 
                       filename: str = None,
                       generate_lessons: bool = True,
                       generate_flashcards: bool = True) -> Dict:
        """
        Parse document content and generate lessons/flashcards.
        
        Args:
            content: Base64 encoded content or raw text
            document_type: pdf, docx, or txt
            filename: Original filename
            generate_lessons: Whether to generate lesson structure
            generate_flashcards: Whether to generate flashcards
        """
        start_time = time.time()
        
        # Extract text based on document type
        text = self._extract_text(content, document_type)
        
        # Generate title
        title = filename.rsplit('.', 1)[0] if filename else "Untitled Document"
        
        # Generate summary
        summary = self._generate_summary(text)
        
        # Generate lessons
        lessons = []
        if generate_lessons:
            lessons = self._generate_lessons(text, generate_flashcards)
        
        total_flashcards = sum(len(l.get("flashcards", [])) for l in lessons)
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "success": True,
            "title": title,
            "summary": summary,
            "lessons": lessons,
            "total_flashcards": total_flashcards,
            "processing_time_ms": processing_time
        }
    
    def _extract_text(self, content: str, doc_type: str) -> str:
        """Extract text from document content."""
        # Check if base64 encoded
        try:
            if doc_type == "txt":
                # Try to decode base64, fallback to raw text
                try:
                    return base64.b64decode(content).decode('utf-8')
                except Exception:
                    return content
            elif doc_type == "pdf":
                return self._parse_pdf(content)
            elif doc_type == "docx":
                return self._parse_docx(content)
            else:
                return content
        except Exception as e:
            print(f"Text extraction error: {e}")
            return content
    
    def _parse_pdf(self, content: str) -> str:
        """Parse PDF content."""
        try:
            import io
            from PyPDF2 import PdfReader
            
            pdf_bytes = base64.b64decode(content)
            reader = PdfReader(io.BytesIO(pdf_bytes))
            
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"PDF parsing error: {e}")
            return content
    
    def _parse_docx(self, content: str) -> str:
        """Parse DOCX content."""
        try:
            import io
            from docx import Document
            
            docx_bytes = base64.b64decode(content)
            doc = Document(io.BytesIO(docx_bytes))
            
            text = ""
            for para in doc.paragraphs:
                text += para.text + "\n"
            
            return text.strip()
        except Exception as e:
            print(f"DOCX parsing error: {e}")
            return content
    
    def _generate_summary(self, text: str) -> str:
        """Generate document summary using AI."""
        summarizer = get_summarizer()
        
        if summarizer != "mock":
            try:
                # Truncate for model limits
                truncated = text[:1024]
                result = summarizer(truncated, max_length=150, min_length=30, do_sample=False)
                return result[0]["summary_text"]
            except Exception as e:
                print(f"Summarization error: {e}")
        
        # Fallback: first 2 sentences
        sentences = text.replace('!', '.').replace('?', '.').split('.')[:2]
        return '. '.join(s.strip() for s in sentences if s.strip()) + '.'
    
    def _generate_lessons(self, text: str, generate_flashcards: bool) -> List[Dict]:
        """Generate lesson structure from text."""
        # Split into paragraphs or sections
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip() and len(p.strip()) > 50]
        
        if not paragraphs:
            paragraphs = [text]
        
        lessons = []
        for i, para in enumerate(paragraphs[:5]):  # Max 5 lessons
            lesson = {
                "title": f"Lesson {i + 1}: {self._extract_title(para)}",
                "content": para,
                "order_index": i,
                "flashcards": [],
                "exercises": []
            }
            
            if generate_flashcards:
                lesson["flashcards"] = self._generate_flashcards(para)
                lesson["exercises"] = self._generate_exercises(para)
            
            lessons.append(lesson)
        
        return lessons
    
    def _extract_title(self, text: str) -> str:
        """Extract a title from text."""
        words = text.split()[:6]
        title = ' '.join(words)
        if len(title) > 40:
            title = title[:40] + "..."
        return title
    
    def _generate_flashcards(self, text: str) -> List[Dict]:
        """Generate vocabulary flashcards from text."""
        # Extract key terms (simplified - in production use NER or keyword extraction)
        words = text.split()
        
        # Find longer, potentially important words
        key_words = [w.strip('.,!?()[]"\'') for w in words 
                     if len(w) > 7 and w[0].islower()]
        key_words = list(set(key_words))[:3]  # Max 3 per lesson
        
        flashcards = []
        for word in key_words:
            flashcards.append({
                "front": word.capitalize(),
                "back": f"A term found in the lesson context. Review the lesson for full understanding.",
                "card_type": "VOCABULARY",
                "example": f"Context: ...{word}..."
            })
        
        return flashcards
    
    def _generate_exercises(self, text: str) -> List[Dict]:
        """Generate exercises from text."""
        # Generate a simple comprehension question
        sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20]
        
        exercises = []
        if sentences:
            exercises.append({
                "question": f"What is the main idea of this passage?",
                "question_type": "SHORT_ANSWER",
                "correct_answer": sentences[0] if sentences else "See lesson content",
                "explanation": "This tests reading comprehension."
            })
        
        return exercises


@lru_cache()
def get_document_parser() -> DocumentParser:
    """Get cached document parser instance."""
    return DocumentParser()
