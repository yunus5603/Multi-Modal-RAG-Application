from langchain_community.vectorstores import Chroma
from langchain_astradb import AstraDBVectorStore
from langchain.schema import Document
from typing import List, Optional, Tuple

class VectorStore:
    def __init__(self, embedding_function, config):
        self.config = config
        self.embedding = embedding_function
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the vector store based on configuration."""
        if self.config.VECTOR_STORE_TYPE.lower() == "chroma":
            return Chroma(
                persist_directory=self.config.VECTOR_DB_PATH,
                embedding_function=self.embedding
            )
        elif self.config.VECTOR_STORE_TYPE.lower() == "astradb":
            return AstraDBVectorStore(
                embedding=self.embedding,
                collection_name=self.config.COLLECTION_NAME,
                api_endpoint=self.config.ASTRA_DB_API_ENDPOINT,
                token=self.config.ASTRA_DB_TOKEN,
                namespace=self.config.ASTRA_DB_KEYSPACE
            )
        else:
            raise ValueError(f"Unsupported vector store type: {self.config.VECTOR_STORE_TYPE}")
    
    def add_documents(self, documents: List[Document]) -> Optional[List[str]]:
        """Add documents to vector store and return document IDs if available."""
        if isinstance(self.vector_store, AstraDBVectorStore):
            return self.vector_store.add_documents(documents)
        else:
            self.vector_store.add_documents(documents)
            self.vector_store.persist()
            return None
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve similar documents for a query."""
        return self.vector_store.similarity_search(query, k=k) 