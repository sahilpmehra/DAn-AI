from pydantic_settings import BaseSettings
from typing import Optional
from functools import lru_cache

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "AI Data Analyst"
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: list[str] = [
        "http://localhost:5173",  # Svelte dev server
        "http://localhost:4173"   # Svelte preview
    ]
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    REDIS_PASSWORD: Optional[str] = None
    
    # MongoDB Configuration
    MONGODB_URL: str = "mongodb://localhost:27017/"
    MONGODB_DB_NAME: str = "ai_analyst"
    
    # PostgreSQL Configuration
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "ai_analyst"
    POSTGRES_PORT: str = "5432"
    
    # AI Configuration
    OPENAI_API_KEY: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    
    # File Upload Configuration
    MAX_UPLOAD_SIZE: int = 50_000_000  # 50MB in bytes
    ALLOWED_EXTENSIONS: set[str] = {".csv", ".xlsx"}
    
    # Analysis Configuration
    DEFAULT_ANALYSIS_TIMEOUT: int = 300  # 5 minutes
    MAX_ROWS_PREVIEW: int = 1000
    CACHE_EXPIRY: int = 3600  # 1 hour
    
    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """
    Get settings with caching to avoid re-reading the environment every time.
    """
    return Settings()

# Create a global settings object
settings = get_settings() 