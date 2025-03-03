from pydantic_settings import BaseSettings, SettingsConfigDict
<<<<<<< HEAD
import os

class Settings(BaseSettings):
    # Model Settings
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    HF_TOKEN: str  # Required for HuggingFace
    
    # LLM Settings
    GROQ_API_KEY: str
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    TEMPERATURE: float = 0.7
    
    # ChromaDB Settings
    CHROMA_PERSIST_DIRECTORY: str = "chroma_db"
    CHROMA_COLLECTION_NAME: str = "pdf_documents"
    
    # Document Processing Settings
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
=======
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
>>>>>>> a3f48f137da3ee5a880eeef5c6f61be0d0864499
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True
<<<<<<< HEAD
    )
=======
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
>>>>>>> a3f48f137da3ee5a880eeef5c6f61be0d0864499
