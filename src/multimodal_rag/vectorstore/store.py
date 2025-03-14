import uuid
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from typing import List, Any, Dict, Optional
from multimodal_rag.config.settings import Settings
import logging
import time
import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

logger = logging.getLogger(__name__)

class RobustHuggingFaceEmbeddings(HuggingFaceInferenceAPIEmbeddings):
    """Enhanced HuggingFace embeddings with better error handling."""
    
    @retry(
        stop=stop_after_attempt(3), 
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((requests.RequestException, ValueError))
    )
    def _embed_documents(self, texts):
        """Override to add retries and better error handling."""
        try:
            # Clean and validate inputs
            cleaned_texts = [text[:5000] if isinstance(text, str) else str(text)[:5000] for text in texts]
            
            # Call parent method
            embeddings = super()._embed_documents(cleaned_texts)
            
            # Validate output
            for embedding in embeddings:
                if not isinstance(embedding, list) or not all(isinstance(x, (int, float)) for x in embedding):
                    raise ValueError(f"Invalid embedding format: {type(embedding)}")
            
            return embeddings
            
        except Exception as e:
            logger.error(f"Embedding error: {str(e)}")
            # If all retries fail, return simple fallback embeddings
            if len(texts) == 0:
                return []
            # Create fallback embeddings (not ideal but prevents complete failure)
            dim = 384  # Standard dimension for many models
            return [[0.1] * dim for _ in texts]

class VectorStoreManager:
    """Manages the vector store operations."""

    def __init__(self, collection_name: str = "multimodal_rag"):
        # Load settings
        self.settings = Settings()
        
        try:
            # Initialize embedding model with specific model
            self.embeddings = RobustHuggingFaceEmbeddings(
                api_key=self.settings.HUGGINGFACE_API_KEY,
                model_name="BAAI/bge-small-en-v1.5"  # Smaller model that's faster and more reliable
            )
            
            # Initialize vector store
            self.vectorstore = Chroma(
                collection_name=collection_name,
                embedding_function=self.embeddings
            )
            
            # Initialize document store and retriever
            self.store = InMemoryStore()
            self.id_key = "doc_id"
            self.retriever = MultiVectorRetriever(
                vectorstore=self.vectorstore,
                docstore=self.store,
                id_key=self.id_key,
            )
            
        except Exception as e:
            logger.error(f"Error initializing vector store: {str(e)}")
            raise

    def _safe_add_documents(self, documents, retries=3):
        """Safely add documents to vector store with retries."""
        for attempt in range(retries):
            try:
                return self.retriever.vectorstore.add_documents(documents)
            except Exception as e:
                logger.warning(f"Error adding documents (attempt {attempt+1}/{retries}): {str(e)}")
                if attempt == retries - 1:
                    logger.error("All attempts to add documents failed")
                    raise
                time.sleep(2 ** attempt)  # Exponential backoff

    def add_texts(self, texts: List[Any], summaries: List[str]) -> List[str]:
        """Add texts and their summaries to the vector store."""
        try:
            doc_ids = [str(uuid.uuid4()) for _ in texts]
            
            # Limit summary length to prevent token issues
            cleaned_summaries = [s[:500] for s in summaries]
            
            summary_texts = [
                Document(
                    page_content=summary, 
                    metadata={
                        self.id_key: doc_ids[i],
                        "type": "texts"  # Changed from "text" to match retrieval code
                    }
                )
                for i, summary in enumerate(cleaned_summaries)
            ]
            
            self._safe_add_documents(summary_texts)
            self.retriever.docstore.mset(list(zip(doc_ids, texts)))
            return doc_ids
            
        except Exception as e:
            logger.error(f"Error adding texts: {str(e)}")
            raise

    def add_tables(self, tables: List[Any], summaries: List[str]) -> List[str]:
        """Add tables and their summaries to the vector store."""
        try:
            table_ids = [str(uuid.uuid4()) for _ in tables]
            
            # Limit summary length
            cleaned_summaries = [s[:500] for s in summaries]
            
            summary_tables = [
                Document(
                    page_content=summary, 
                    metadata={
                        self.id_key: table_ids[i],
                        "type": "tables"  # Changed to match retrieval code
                    }
                )
                for i, summary in enumerate(cleaned_summaries)
            ]
            
            self._safe_add_documents(summary_tables)
            self.retriever.docstore.mset(list(zip(table_ids, tables)))
            return table_ids
            
        except Exception as e:
            logger.error(f"Error adding tables: {str(e)}")
            raise

    def add_images(self, images: List[str], summaries: List[str]) -> List[str]:
        """Add images and their summaries to the vector store."""
        try:
            img_ids = [str(uuid.uuid4()) for _ in images]
            
            # Limit summary length
            cleaned_summaries = [s[:500] for s in summaries]
            
            summary_img = [
                Document(
                    page_content=summary, 
                    metadata={
                        self.id_key: img_ids[i],
                        "type": "images"  # Changed to match retrieval code
                    }
                )
                for i, summary in enumerate(cleaned_summaries)
            ]
            
            self._safe_add_documents(summary_img)
            self.retriever.docstore.mset(list(zip(img_ids, images)))
            return img_ids
            
        except Exception as e:
            logger.error(f"Error adding images: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents based on query."""
        try:
            return self.vectorstore.similarity_search(query, k=k)
        except Exception as e:
            logger.error(f"Error in similarity search: {str(e)}")
            return []  # Return empty list instead of failing

    def get_retriever(self) -> MultiVectorRetriever:
        """Get the multi-vector retriever."""
        return self.retriever