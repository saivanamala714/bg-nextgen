from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Supabase Configuration
    supabase_url: str
    supabase_service_role_key: str
    
    # API Keys
    groq_api_key: str
    openai_api_key: str
    
    # Optional: Allow test mode
    test_mode: bool = False
    
    # Application Settings
    app_name: str = "PDF Chat RAG Backend"
    debug: bool = False
    log_level: str = "INFO"
    
    # RAG Settings
    embedding_model: str = "text-embedding-3-small"
    llm_model: str = "llama-3.3-70b-versatile"
    max_context_chunks: int = 8
    max_history_messages: int = 8
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # Vector Settings (optional)
    embedding_dimension: int = 1024
    similarity_threshold: float = 0.1  # Lowered from 0.7 to find more matches
    
    # PDF Processing
    chunk_size: int = 800
    chunk_overlap: int = 150
    
    class Config:
        env_file = ".env"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
