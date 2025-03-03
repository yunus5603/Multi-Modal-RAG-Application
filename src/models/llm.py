from langchain_groq import ChatGroq
import logging

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self, config):
        self.config = config
<<<<<<< HEAD
        logger.info(f"Initializing Groq LLM with model: {self.config.LLM_MODEL}")
=======
>>>>>>> a3f48f137da3ee5a880eeef5c6f61be0d0864499
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        try:
<<<<<<< HEAD
            return ChatGroq(
                model=self.config.LLM_MODEL,
                temperature=self.config.TEMPERATURE,
                groq_api_key=self.config.GROQ_API_KEY
            )
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
=======
            if self.config.GROQ_API_KEY:
                return ChatGroq(
                    model=self.config.LLM_MODEL,
                    temperature=self.config.TEMPERATURE,
                    groq_api_key=self.config.GROQ_API_KEY
                )
            elif self.config.OPENAI_API_KEY:
                return ChatOpenAI(
                    model=self.config.LLM_MODEL,
                    temperature=self.config.TEMPERATURE,
                    openai_api_key=self.config.OPENAI_API_KEY
                )
            else:
                raise ValueError("Neither GROQ_API_KEY nor OPENAI_API_KEY is provided")
        except Exception as e:
            raise Exception(f"Failed to initialize LLM: {str(e)}")
>>>>>>> a3f48f137da3ee5a880eeef5c6f61be0d0864499
    
    def get_llm(self):
        return self.llm 