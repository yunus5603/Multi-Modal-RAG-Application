from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import Optional

class EmbeddingManager:
    def __init__(self, config):
        self.config = config
        self.embedding = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize the embedding model based on configuration."""
        if self.config.EMBEDDING_TYPE.lower() == "openai":
            return OpenAIEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                openai_api_key=self.config.OPENAI_API_KEY
            )
        elif self.config.EMBEDDING_TYPE.lower() == "huggingface":
            return HuggingFaceInferenceAPIEmbeddings(
                api_key=self.config.HF_TOKEN,
                model_name=self.config.EMBEDDING_MODEL
            )
        else:
            raise ValueError(f"Unsupported embedding type: {self.config.EMBEDDING_TYPE}")
    
    def get_embeddings(self):
        return self.embedding 