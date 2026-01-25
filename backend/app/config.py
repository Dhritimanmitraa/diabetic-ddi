"""Application configuration settings."""
from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    """Application settings."""
    
    APP_NAME: str = "Drug-Drug Interaction Predictor"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./drug_interactions.db"
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # API Keys (optional - for enhanced data sources)
    OPENFDA_API_KEY: str = ""
    UMLS_API_KEY: str = ""  # Free registration at https://uts.nlm.nih.gov/uts/signup-login
    API_KEY: str = ""
    
    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MIN: int = 60
    
    # API Reliability Settings
    API_RETRY_ATTEMPTS: int = 3
    API_BACKOFF_BASE: float = 1.0
    API_CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    # OCR Settings
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows default
    
    # Prescription RAG Settings
    GOOGLE_API_KEY: str = ""  # Set in .env file
    GEMINI_API_KEY: str = ""  # Set in .env file
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    OLLAMA_VISION_MODEL: str = "llava"  # Vision-capable model for OCR
    
    # LLM Settings (Ollama)
    OLLAMA_MODEL: str = "gpt-oss:120b-cloud"  # Main LLM model
    OLLAMA_DRUG_CHECK_MODEL: str = "gpt-oss:120b-cloud"  # Drug checker model
    OLLAMA_HOST: str = "http://localhost:11434"  # Ollama server URL
    LLM_FALLBACK_TO_TEMPLATES: bool = True  # Use templates if LLM unavailable
    
    # File paths
    DATA_DIR: str = "./data"
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"  # Ignore extra fields in .env


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

