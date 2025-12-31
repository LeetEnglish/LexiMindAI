"""
LexiAI Configuration
"""
from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # App
    app_name: str = "LexiAI"
    debug: bool = False
    api_prefix: str = "/api/v1"
    
    # Hugging Face Models
    hf_document_model: str = "facebook/bart-large-cnn"
    hf_grammar_model: str = "textattack/roberta-base-CoLA"
    hf_chat_model: str = "microsoft/DialoGPT-medium"
    
    # Redis/Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # Timeouts
    model_load_timeout: int = 120
    inference_timeout: int = 60
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    """Cached settings instance."""
    return Settings()
