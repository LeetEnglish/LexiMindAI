"""
Scoring Model - Writing and Speaking Evaluation
Uses Hugging Face models for grammar, coherence analysis.
"""
import time
from typing import Tuple, Dict
from functools import lru_cache

from app.core.config import get_settings

settings = get_settings()

# Lazy load to avoid startup delay
_grammar_model = None
_tokenizer = None


def get_grammar_model():
    """Lazy load grammar model."""
    global _grammar_model, _tokenizer
    if _grammar_model is None:
        try:
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            model_name = settings.hf_grammar_model
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _grammar_model = AutoModelForSequenceClassification.from_pretrained(model_name)
            print(f"✅ Loaded grammar model: {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to load grammar model: {e}")
            _grammar_model = "mock"
    return _grammar_model, _tokenizer


class ScoringModel:
    """Writing and Speaking scoring using AI models."""
    
    # Transition words for coherence detection
    TRANSITIONS = ["firstly", "secondly", "finally", "in addition", "for example",
                   "in conclusion", "on the other hand", "as a result", "however",
                   "moreover", "furthermore", "consequently", "therefore", "similarly"]
    
    # Advanced vocabulary indicators
    ADVANCED_VOCAB = ["significant", "demonstrate", "illustrate", "analyze", "evaluate",
                      "perspective", "comprehensive", "fundamental", "substantial",
                      "consequently", "nevertheless", "furthermore", "predominantly"]
    
    def score_writing(self, text: str, prompt: str = None) -> Dict:
        """
        Score writing on 4 dimensions:
        - Grammar (0-10)
        - Vocabulary (0-10)
        - Coherence (0-10)
        - Task Achievement (0-10)
        """
        start_time = time.time()
        
        words = text.split()
        word_count = len(words)
        sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]
        
        # Grammar score (using model if available, else heuristics)
        grammar_score = self._evaluate_grammar(text)
        
        # Vocabulary score
        vocabulary_score = self._evaluate_vocabulary(text, words)
        
        # Coherence score
        coherence_score = self._evaluate_coherence(text, sentences)
        
        # Task achievement
        task_score = self._evaluate_task_achievement(text, prompt, word_count)
        
        # Overall
        overall = (grammar_score + vocabulary_score + coherence_score + task_score) / 4
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "overall_score": round(overall, 2),
            "details": {
                "grammar_score": round(grammar_score, 2),
                "vocabulary_score": round(vocabulary_score, 2),
                "coherence_score": round(coherence_score, 2),
                "task_achievement_score": round(task_score, 2),
            },
            "word_count": word_count,
            "processing_time_ms": processing_time,
            "feedback": self._generate_writing_feedback(grammar_score, vocabulary_score, 
                                                         coherence_score, task_score)
        }
    
    def score_speaking(self, transcript: str, audio_url: str = None) -> Dict:
        """
        Score speaking on 4 dimensions:
        - Pronunciation (0-10)
        - Fluency (0-10)
        - Vocabulary (0-10)
        - Grammar (0-10)
        """
        start_time = time.time()
        
        words = transcript.split()
        word_count = len(words)
        
        # Pronunciation (would use audio analysis in production)
        pronunciation_score = self._evaluate_pronunciation(transcript, audio_url)
        
        # Fluency
        fluency_score = self._evaluate_fluency(transcript, words)
        
        # Vocabulary
        vocabulary_score = self._evaluate_vocabulary(transcript, words)
        
        # Grammar
        grammar_score = self._evaluate_grammar(transcript)
        
        # Overall
        overall = (pronunciation_score + fluency_score + vocabulary_score + grammar_score) / 4
        ielts_band = (overall / 10) * 9  # Convert to IELTS 0-9 scale
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "overall_score": round(overall, 2),
            "ielts_band_equivalent": round(ielts_band, 1),
            "details": {
                "pronunciation_score": round(pronunciation_score, 2),
                "fluency_score": round(fluency_score, 2),
                "vocabulary_score": round(vocabulary_score, 2),
                "grammar_score": round(grammar_score, 2),
            },
            "word_count": word_count,
            "processing_time_ms": processing_time,
            "feedback": self._generate_speaking_feedback(pronunciation_score, fluency_score,
                                                          vocabulary_score, grammar_score)
        }
    
    def _evaluate_grammar(self, text: str) -> float:
        """Evaluate grammar using model or heuristics."""
        model, tokenizer = get_grammar_model()
        
        if model != "mock" and tokenizer:
            try:
                import torch
                inputs = tokenizer(text[:512], return_tensors="pt", truncation=True)
                with torch.no_grad():
                    outputs = model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=1)
                    # Higher probability of "acceptable" = better grammar
                    score = probs[0][1].item() * 10
                    return min(10.0, max(0.0, score))
            except Exception:
                pass
        
        # Fallback heuristics
        score = 6.0
        if text[0].isupper():
            score += 0.5
        if text.rstrip()[-1] in '.!?':
            score += 0.5
        if '  ' not in text:
            score += 0.5
        sentences = text.split('.')
        if len(sentences) >= 3:
            score += 1.0
        avg_len = len(text.split()) / max(1, len(sentences))
        if 10 <= avg_len <= 25:
            score += 1.5
        return min(10.0, score)
    
    def _evaluate_vocabulary(self, text: str, words: list) -> float:
        """Evaluate vocabulary diversity and sophistication."""
        unique = len(set(w.lower() for w in words))
        diversity = unique / max(1, len(words))
        
        score = 4.0 + (diversity * 4)
        
        # Check for advanced vocabulary
        text_lower = text.lower()
        for word in self.ADVANCED_VOCAB:
            if word in text_lower:
                score += 0.3
        
        return min(10.0, score)
    
    def _evaluate_coherence(self, text: str, sentences: list) -> float:
        """Evaluate coherence and cohesion."""
        score = 5.0
        
        text_lower = text.lower()
        for trans in self.TRANSITIONS:
            if trans in text_lower:
                score += 0.5
        
        # Check paragraph structure
        if '\n\n' in text or len(text) > 500:
            score += 1.5
        
        return min(10.0, score)
    
    def _evaluate_task_achievement(self, text: str, prompt: str, word_count: int) -> float:
        """Evaluate how well the response addresses the task."""
        score = 5.0
        
        if word_count >= 250:
            score += 2.0
        elif word_count >= 150:
            score += 1.0
        
        if prompt:
            prompt_words = [w.lower() for w in prompt.split() if len(w) > 4]
            text_lower = text.lower()
            matches = sum(1 for w in prompt_words if w in text_lower)
            if matches >= 3:
                score += 2.0
            elif matches >= 1:
                score += 1.0
        
        return min(10.0, score)
    
    def _evaluate_pronunciation(self, transcript: str, audio_url: str) -> float:
        """Evaluate pronunciation (would use audio in production)."""
        score = 6.0
        if audio_url:
            score += 1.0  # Bonus for providing audio
        
        # Check for challenging words
        challenging = ["particularly", "characteristic", "pronunciation",
                       "circumstances", "vocabulary", "communication"]
        for word in challenging:
            if word in transcript.lower():
                score += 0.3
        
        return min(10.0, score)
    
    def _evaluate_fluency(self, transcript: str, words: list) -> float:
        """Evaluate fluency based on transcript."""
        score = 5.0
        word_count = len(words)
        
        if word_count >= 150:
            score += 2.0
        elif word_count >= 100:
            score += 1.0
        
        # Check for filler words (too many = disfluency)
        fillers = ["um", "uh", "like", "you know", "basically"]
        filler_count = sum(1 for f in fillers if f in transcript.lower())
        
        if filler_count <= 2:
            score += 1.5
        elif filler_count >= 5:
            score -= 1.0
        
        # Coherence markers
        markers = ["firstly", "secondly", "in my opinion", "for example"]
        for m in markers:
            if m in transcript.lower():
                score += 0.4
        
        return min(10.0, max(0.0, score))
    
    def _generate_writing_feedback(self, g: float, v: float, c: float, t: float) -> str:
        """Generate writing feedback."""
        avg = (g + v + c + t) / 4
        
        feedback = []
        if avg >= 8:
            feedback.append("Excellent writing! Strong command of language.")
        elif avg >= 6:
            feedback.append("Good writing with room for improvement.")
        else:
            feedback.append("Keep practicing. Focus on the areas below.")
        
        if g < 7:
            feedback.append("Grammar: Focus on sentence structure and punctuation.")
        if v < 7:
            feedback.append("Vocabulary: Try using more varied and advanced words.")
        if c < 7:
            feedback.append("Coherence: Use more transition words for flow.")
        if t < 7:
            feedback.append("Task: Ensure you fully address the prompt.")
        
        return " ".join(feedback)
    
    def _generate_speaking_feedback(self, p: float, f: float, v: float, g: float) -> str:
        """Generate speaking feedback."""
        avg = (p + f + v + g) / 4
        ielts = (avg / 10) * 9
        
        feedback = [f"Estimated IELTS Band: {ielts:.1f}"]
        
        if avg >= 8:
            feedback.append("Outstanding speaking ability!")
        elif avg >= 6:
            feedback.append("Good communication with some areas to develop.")
        else:
            feedback.append("Continue practicing the skills below.")
        
        if f < 7:
            feedback.append("Fluency: Reduce hesitations and filler words.")
        if v < 7:
            feedback.append("Vocabulary: Expand your range of expressions.")
        if g < 7:
            feedback.append("Grammar: Use more complex sentence structures.")
        
        return " ".join(feedback)


@lru_cache()
def get_scoring_model() -> ScoringModel:
    """Get cached scoring model instance."""
    return ScoringModel()
