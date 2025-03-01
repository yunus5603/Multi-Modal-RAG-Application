from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
from pydantic import field_validator

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: str
    HF_TOKEN: Optional[str] = None
    
    # AstraDB Settings (Required)
    ASTRA_DB_API_ENDPOINT: str
    ASTRA_DB_APPLICATION_TOKEN: str
    ASTRA_DB_KEYSPACE: str
    COLLECTION_NAME: str
    
    # Chunking Settings
    CHUNK_SIZE: int = 1000  # Default value if not in .env
    CHUNK_OVERLAP: int = 200  # Default value if not in .env
    
    # Model Settings
    EMBEDDING_TYPE: str
    EMBEDDING_MODEL: str
    LLM_MODEL: str
    TEMPERATURE: float
    
    # LangSmith Settings
    LANGCHAIN_TRACING_V2: bool
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str
    
    # Updated configuration using Pydantic v2 style
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True
    )
    
    @field_validator('GROQ_API_KEY', 'ASTRA_DB_API_ENDPOINT', 'ASTRA_DB_APPLICATION_TOKEN', 'ASTRA_DB_KEYSPACE')
    @classmethod
    def validate_required_fields(cls, v: str, info) -> str:
        if not v:
            raise ValueError(f"{info.field_name} is required")
        return v
    
    @field_validator('EMBEDDING_TYPE')
    @classmethod
    def validate_embedding_type(cls, v: str) -> str:
        if v.lower() not in ["openai", "huggingface"]:
            raise ValueError("EMBEDDING_TYPE must be either 'openai' or 'huggingface'")
        return v.lower()