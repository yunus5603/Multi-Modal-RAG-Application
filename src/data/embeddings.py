from langchain_huggingface import HuggingFaceEmbeddings

class EmbeddingManager:
    def __init__(self, config):
        self.config = config
        self.embedding = self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize HuggingFace embeddings."""
        try:
            return HuggingFaceEmbeddings(
                model_name=self.config.EMBEDDING_MODEL,
                model_kwargs={'device': 'cpu'}
            )
        except Exception as e:
            raise Exception(f"Failed to initialize embeddings: {str(e)}")
    
    def get_embeddings(self):
        return self.embedding 