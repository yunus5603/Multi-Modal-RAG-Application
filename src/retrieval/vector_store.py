from qdrant_client import QdrantClient
from langchain.schema import Document
from typing import List
import logging

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_function, config):
        """Initialize Qdrant client and create collection if it doesn't exist."""
        self.config = config
        self.embedding = embedding_function
        self.vector_store = self._initialize_vector_store()
    
    def _initialize_vector_store(self):
        """Initialize the Qdrant vector store."""
        try:
            logger.info("Initializing Qdrant connection...")
            logger.info(f"Qdrant URL: {self.config.QDRANT_URL}")
            logger.info(f"Collection Name: {self.config.COLLECTION_NAME}")
            
            # Verify Qdrant credentials
            if not self.config.QDRANT_URL:
                raise ValueError("Missing Qdrant URL")
            
            client = QdrantClient(
                url=self.config.QDRANT_URL,
                api_key=self.config.QDRANT_API_KEY
            )
            
            # Ensure the collection exists
            if not client.get_collection(self.config.COLLECTION_NAME):
                client.create_collection(
                    collection_name=self.config.COLLECTION_NAME,
                    vector_size=self.embedding.vector_size,
                    distance="Cosine"
                )
            
            logger.info("Successfully connected to Qdrant")
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize Qdrant: {str(e)}")
            raise Exception(f"Qdrant initialization failed: {str(e)}")
    
    def add_documents(self, documents: List[Document]) -> List[str]:
        """Add documents to vector store and return document IDs."""
        try:
            logger.info(f"Adding {len(documents)} documents to Qdrant")
            
            # Log document types being added
            content_types = {}
            for doc in documents:
                content_type = doc.metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            logger.info(f"Content types being added: {content_types}")
            
            # Add documents
            ids = self.vector_store.upload_collection(
                collection_name=self.config.COLLECTION_NAME,
                vectors=[self.embedding.embed_documents([doc.page_content])[0] for doc in documents],
                payloads=[doc.metadata for doc in documents]
            )
            logger.info(f"Successfully added {len(ids)} documents to Qdrant")
            return ids
            
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise Exception(f"Failed to add documents to Qdrant: {str(e)}")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Retrieve similar documents for a query."""
        try:
            logger.info(f"Performing similarity search for query: {query}")
            results = self.vector_store.search(
                collection_name=self.config.COLLECTION_NAME,
                query_vector=self.embedding.embed_query(query),
                limit=k
            )
            logger.info(f"Found {len(results)} matching documents")
            return [Document(page_content=res.payload['page_content'], metadata=res.payload) for res in results]
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise Exception(f"Failed to perform similarity search: {str(e)}")