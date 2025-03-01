from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

class LLMManager:
    def __init__(self, config):
        self.config = config
        self.llm = self._initialize_llm()
    
    def _initialize_llm(self):
        """Initialize the LLM based on configuration."""
        try:
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
    
    def get_llm(self):
        return self.llm 