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
    MAX_TOKENS = 800
    
    # PDF Processing Settings
    MAX_CHARACTERS = 8000  # Reduced from 10000
    COMBINE_CHARS = 1500  # Reduced from 2000
    NEW_CHARS = 4000     # Reduced from 6000

    # Rate Limiting Settings
    BATCH_SIZE = 1       # Reduced from 2 to avoid rate limits
    BATCH_DELAY = 2.0    # Increased from 1.0 to add more delay
    MAX_RETRIES = 3      # Reduced from 5 to fail faster
    INITIAL_RETRY_DELAY = 2.0  # Increased from 1.0 