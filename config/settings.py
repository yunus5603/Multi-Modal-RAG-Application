from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    GROQ_API_KEY: Optional[str] = None
    HF_TOKEN: Optional[str] = None
    
    # Vector DB Settings
    VECTOR_STORE_TYPE: str = "astradb"  # "chroma" or "astradb"
    VECTOR_DB_PATH: str = "MultiModal_RAG_App"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    
    # AstraDB Settings (if using AstraDB)
    ASTRA_DB_API_ENDPOINT: Optional[str] = None
    ASTRA_DB_TOKEN: Optional[str] = None
    ASTRA_DB_KEYSPACE: Optional[str] = None
    COLLECTION_NAME: str = "pdf_rag_collection"
    
    # Model Settings
    EMBEDDING_TYPE: str = "huggingface"  # "openai" or "huggingface"
    EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"  # or "BAAI/bge-base-en-v1.5" for HuggingFace
    LLM_MODEL: str = "llama-3.3-70b-versatile"
    TEMPERATURE: float = 0.7
    
    # LangSmith Settings
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "pdf-rag-app"
    
    class Config:
        env_file = ".env"