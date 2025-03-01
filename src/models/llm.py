from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq

class LLMManager:
    def __init__(self, config):
        if config.GROQ_API_KEY:
            self.llm = ChatGroq(
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                groq_api_key=config.GROQ_API_KEY
            )
        else:
            self.llm = ChatOpenAI(
                model=config.LLM_MODEL,
                temperature=config.TEMPERATURE,
                openai_api_key=config.OPENAI_API_KEY
            )
    
    def get_llm(self):
        return self.llm 