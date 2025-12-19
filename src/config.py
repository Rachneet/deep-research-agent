"""Application configuration loaded from .env and environment."""

from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Load and validate environment variables with type hints and defaults."""
    
    # LLM APIs
    openai_api_key: Optional[str] = None
    groq_api_key: Optional[str] = None
    google_api_key: Optional[str] = None
    deepseek_api_key: Optional[str] = None
    huggingface_api_key: Optional[str] = None
    
    # Modal deployment
    token_id: Optional[str] = None
    token_secret: Optional[str] = None
    
    # Notifications
    pushover_user: Optional[str] = None
    pushover_token: Optional[str] = None
    
    # Email
    sendgrid_api_key: Optional[str] = None
    
    # Search & APIs
    serp_api_key: Optional[str] = None
    serper_api_key: Optional[str] = None
    brave_api_key: Optional[str] = None
    polygon_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    
    # LangSmith
    langsmith_tracing: bool = True
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "deep-research-agent"
    
    # Crew AI
    crewai_tracing_enabled: bool = True
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False  # Allow TAVILY_API_KEY or tavily_api_key


# Global settings instance (load once at startup)
settings = Settings()