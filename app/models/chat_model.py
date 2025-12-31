"""
Chat Model - Conversational AI for tutoring
"""
import time
from typing import List, Dict
from functools import lru_cache

from app.core.config import get_settings

settings = get_settings()

# Lazy load model
_chat_model = None
_tokenizer = None


def get_chat_model():
    """Lazy load chat model."""
    global _chat_model, _tokenizer
    if _chat_model is None:
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            model_name = settings.hf_chat_model
            _tokenizer = AutoTokenizer.from_pretrained(model_name)
            _chat_model = AutoModelForCausalLM.from_pretrained(model_name)
            if _tokenizer.pad_token is None:
                _tokenizer.pad_token = _tokenizer.eos_token
            print(f"✅ Loaded chat model: {model_name}")
        except Exception as e:
            print(f"⚠️ Failed to load chat model: {e}")
            _chat_model = "mock"
    return _chat_model, _tokenizer


class ChatModel:
    """Chat completion using Hugging Face models."""
    
    SYSTEM_PROMPT = """You are an English language tutor. Help users improve their English skills.
Provide clear explanations, correct grammar mistakes, and suggest better vocabulary when appropriate.
Be encouraging and supportive."""
    
    def complete(self, messages: List[Dict], max_tokens: int = 256,
                 temperature: float = 0.7, system_prompt: str = None) -> Dict:
        """
        Generate a chat completion.
        
        Args:
            messages: List of {"role": "user"|"assistant", "content": "..."}
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            system_prompt: Optional custom system prompt
        """
        start_time = time.time()
        
        model, tokenizer = get_chat_model()
        
        # Build conversation context
        context = system_prompt or self.SYSTEM_PROMPT
        
        if model != "mock" and tokenizer:
            try:
                response_text = self._generate_with_model(
                    model, tokenizer, messages, max_tokens, context
                )
            except Exception as e:
                print(f"Model generation error: {e}")
                response_text = self._generate_mock_response(messages)
        else:
            response_text = self._generate_mock_response(messages)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            "message": {
                "role": "assistant",
                "content": response_text
            },
            "finish_reason": "stop",
            "tokens_used": len(response_text.split()),
            "processing_time_ms": processing_time
        }
    
    def _generate_with_model(self, model, tokenizer, messages: List[Dict],
                              max_tokens: int, context: str) -> str:
        """Generate response using actual model."""
        import torch
        
        # Format conversation for DialoGPT-style models
        conversation = ""
        for msg in messages[-5:]:  # Last 5 messages for context
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "user":
                conversation += f"User: {content}\n"
            else:
                conversation += f"Assistant: {content}\n"
        
        conversation += "Assistant:"
        
        # Tokenize
        inputs = tokenizer.encode(conversation, return_tensors="pt", truncation=True, max_length=512)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                inputs,
                max_new_tokens=max_tokens,
                temperature=temperature if temperature > 0 else 1.0,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract assistant's response
        if "Assistant:" in response:
            parts = response.split("Assistant:")
            response = parts[-1].strip()
        
        # Clean up
        response = response.split("User:")[0].strip()
        
        return response or "I'm here to help you learn English. What would you like to practice today?"
    
    def _generate_mock_response(self, messages: List[Dict]) -> str:
        """Generate a mock response when model is unavailable."""
        user_msg = messages[-1].get("content", "") if messages else ""
        
        # Simple pattern matching for common queries
        user_lower = user_msg.lower()
        
        if "grammar" in user_lower:
            return "Grammar is essential for clear communication. What specific grammar topic would you like to practice? I can help with tenses, articles, prepositions, and more."
        elif "vocabulary" in user_lower:
            return "Building vocabulary is key to fluency! Let's work on new words. Would you like to learn academic vocabulary, everyday expressions, or idioms?"
        elif "speaking" in user_lower or "pronunciation" in user_lower:
            return "Great focus on speaking! Practice makes perfect. Try reading aloud and recording yourself. Would you like some speaking exercises?"
        elif "writing" in user_lower:
            return "Writing skills are valuable. I can help you with essays, emails, or creative writing. What would you like to write about?"
        elif "hello" in user_lower or "hi" in user_lower:
            return "Hello! I'm your English tutor. How can I help you improve your English skills today?"
        else:
            return f"That's a great question about English learning. Let me help you with that. Could you tell me more about what specific aspect you'd like to focus on?"


@lru_cache()
def get_chat_model_instance() -> ChatModel:
    """Get cached chat model instance."""
    return ChatModel()
