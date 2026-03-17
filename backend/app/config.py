"""Application configuration settings."""
import os
from functools import lru_cache

from pydantic import ConfigDict
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    model_config = ConfigDict(
        env_file=(".env", ".env.local"),
        case_sensitive=True,
        extra="ignore",
    )
    
    APP_NAME: str = "Drug-Drug Interaction Predictor"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    APP_ENV: str = "development"
    STRICT_STARTUP_VALIDATION: bool = False
    LOG_LEVEL: str = "INFO"

    # Database
    DATABASE_URL: str = "sqlite+aiosqlite:///./drug_interactions.db"
    DB_AUTO_CREATE: bool = True
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_RECYCLE_SECONDS: int = 1800
    REDIS_URL: str = "redis://localhost:6379/0"
    
    # API Keys (optional - for enhanced data sources)
    OPENFDA_API_KEY: str = ""
    UMLS_API_KEY: str = ""  # Free registration at https://uts.nlm.nih.gov/uts/signup-login
    API_KEY: str = ""
    REQUIRE_API_KEY_FOR_ADMIN: bool = True
    REQUIRE_GEMINI_KEY: bool = False
    ENABLE_CLOUD_SPEECH: bool = False
    JWT_SECRET: str = "change-me-in-production"
    JWT_ISSUER: str = "drugguard"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    JWT_REFRESH_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7
    METRICS_ENABLED: bool = True
    
    # Rate limiting
    RATE_LIMIT_REQUESTS_PER_MIN: int = 60
    HEAVY_RATE_LIMIT_REQUESTS_PER_MIN: int = 10

    # CORS
    CORS_ORIGINS: str = "http://localhost:5173,http://localhost:3000,http://localhost:3001"
    
    # API Reliability Settings
    API_RETRY_ATTEMPTS: int = 3
    API_BACKOFF_BASE: float = 1.0
    API_CACHE_TTL_SECONDS: int = 3600  # 1 hour
    
    # OCR Settings
    TESSERACT_CMD: str = os.getenv("TESSERACT_PATH", "tesseract")
    
    # Prescription RAG Settings
    GOOGLE_API_KEY: str = ""  # Set in .env file
    GEMINI_API_KEY: str = ""  # Set in .env file
    CLOUD_SPEECH_API_KEY: str = ""  # Optional Google Cloud Speech key
    CHROMA_PERSIST_DIR: str = "./chroma_db"
    OLLAMA_VISION_MODEL: str = "llava"  # Vision-capable model for OCR
    
    # NVIDIA Cosmos Vision Model (Cloud API)
    NVIDIA_API_KEY: str = ""  # Set in .env file — NGC/NIM API key
    NVIDIA_NIM_BASE_URL: str = "https://integrate.api.nvidia.com/v1"  # Cloud endpoint
    NVIDIA_COSMOS_MODEL: str = "nvidia/llama-3.1-nemotron-nano-vl-8b-v1"  # Cloud VLM (or cosmos-reason2-8b for local NIM)
    
    # LLM Settings (Ollama)
    OLLAMA_MODEL: str = "gpt-oss:120b-cloud"  # Main LLM model
    OLLAMA_DRUG_CHECK_MODEL: str = "gpt-oss:120b-cloud"  # Drug checker model
    OLLAMA_HOST: str = "http://localhost:11434"  # Ollama server URL
    LLM_FALLBACK_TO_TEMPLATES: bool = True  # Use templates if LLM unavailable
    
    # File paths
    DATA_DIR: str = "./data"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
