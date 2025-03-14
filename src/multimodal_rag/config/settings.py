from dotenv import load_dotenv
import os

load_dotenv()

class Settings:
    """Configuration settings for the multimodal RAG system."""
    
    # API Keys
    HUGGINGFACE_API_KEY = os.getenv("HF_TOKEN")
    if not HUGGINGFACE_API_KEY:
        raise ValueError("HUGGINGFACE_API_KEY not found in environment variables")
        
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY not found in environment variables")
        
    LANGCHAIN_API_KEY = os.getenv("LANGCHAIN_API_KEY")
    LANGCHAIN_TRACING_V2 = os.getenv("LANGCHAIN_TRACING_V2", "true")

    # Model Settings
    DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"  # Using Mixtral for all operations
    MAX_TOKENS = 1000
    
    # PDF Processing Settings
    MAX_CHARACTERS = 10000
    COMBINE_CHARS = 2000
    NEW_CHARS = 6000

    # Rate Limiting Settings
    BATCH_SIZE = 2
    BATCH_DELAY = 1.0  # seconds
    MAX_RETRIES = 5
    INITIAL_RETRY_DELAY = 1.0 