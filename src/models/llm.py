from langchain_groq import ChatGroq
import logging

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self, config):
        self.config = config
        logger.info(f"Initializing Groq LLM with model: {self.config.LLM_MODEL}")
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        try:
            return ChatGroq(
                model=self.config.LLM_MODEL,
                temperature=self.config.TEMPERATURE,
                groq_api_key=self.config.GROQ_API_KEY
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def get_llm(self):
        return self.llm 