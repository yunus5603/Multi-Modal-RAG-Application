from pydantic_settings import BaseSettings, SettingsConfigDict
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
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        validate_default=True
    )