from langchain_astradb import AstraDBVectorStore
from langchain.schema import Document
from typing import List
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_function, config):
        self.config = config
        self.embedding = embedding_function
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the AstraDB vector store."""
        try:
            logger.info("Initializing AstraDB connection...")
            logger.info(f"API Endpoint: {self.config.ASTRA_DB_API_ENDPOINT}")
            logger.info(f"Collection Name: {self.config.COLLECTION_NAME}")
            logger.info(f"Keyspace: {self.config.ASTRA_DB_KEYSPACE}")
            
            # Verify AstraDB credentials
            if not all([
                self.config.ASTRA_DB_API_ENDPOINT,
                self.config.ASTRA_DB_APPLICATION_TOKEN,
                self.config.ASTRA_DB_KEYSPACE
            ]):
                raise ValueError("Missing required AstraDB credentials")
            
            vector_store = AstraDBVectorStore(
                embedding=self.embedding,
                collection_name=self.config.COLLECTION_NAME,
                api_endpoint=self.config.ASTRA_DB_API_ENDPOINT,
                token=self.config.ASTRA_DB_APPLICATION_TOKEN,
                namespace=self.config.ASTRA_DB_KEYSPACE,
                batch_size=20  # Added for better performance
            )
            
            logger.info("Successfully connected to AstraDB")
            return vector_store
            
        except Exception as e:
            logger.error(f"Failed to initialize AstraDB: {str(e)}")
            raise Exception(f"AstraDB initialization failed: {str(e)}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store and return document IDs."""
        try:
            logger.info(f"Adding {len(documents)} documents to AstraDB")
            
            # Log document types being added
            content_types = {}
            for doc in documents:
                content_type = doc.metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            logger.info(f"Content types being added: {content_types}")
            
            # Add documents
            ids = self.vector_store.add_documents(documents)
            logger.info(f"Successfully added {len(ids)} documents to AstraDB")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise Exception(f"Failed to add documents to AstraDB: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve similar documents for a query."""
        try:
            logger.info(f"Performing similarity search for query: {query}")
            results = self.vector_store.similarity_search(query, k=k)
            logger.info(f"Found {len(results)} matching documents")
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise Exception(f"Failed to perform similarity search: {str(e)}") 