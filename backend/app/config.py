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
    API_KEY: str = ""
    
    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MIN: int = 60
    
    # OCR Settings
    TESSERACT_CMD: str = r"C:\Program Files\Tesseract-OCR\tesseract.exe"  # Windows default
    
    # File paths
    DATA_DIR: str = "./data"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()

