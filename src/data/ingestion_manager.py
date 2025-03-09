import logging
from typing import Optional, Tuple, List
from langchain.schema import Document
import os
from src.data.document_processor import DocumentProcessor
from src.data.embeddings import EmbeddingManager
from src.retrieval.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DataIngestionManager:
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        self.document_processor = DocumentProcessor(config)
        self.embedding_manager = EmbeddingManager(config)
        self.vector_store = VectorStore(
            self.embedding_manager.get_embeddings(),
            config
        )
    
    def ingest_pdf(self, file_path: str) -> Tuple[object, Optional[List[str]]]:
        """
        Process and ingest a PDF file into the vector store.
        Returns the vector store instance and list of document IDs.
        """
        try:
            logger.info(f"Starting ingestion process for: {file_path}")
            
            # Process PDF
            documents = self.document_processor.process_pdf(file_path)
            logger.info(f"Processed PDF into {len(documents)} documents")
            
            # Log document types
            content_types = {}
            for doc in documents:
                content_type = doc.metadata.get('content_type', 'unknown')
                content_types[content_type] = content_types.get(content_type, 0) + 1
            logger.info(f"Document types extracted: {content_types}")
            
            # Add to vector store
            if documents:
                doc_ids = self.vector_store.add_documents(documents)
                logger.info(f"Successfully added {len(doc_ids)} documents to vector store")
                return self.vector_store, doc_ids
            else:
                logger.warning("No documents were extracted from the PDF")
                return self.vector_store, None
                
        except Exception as e:
            logger.error(f"Ingestion failed: {str(e)}")
            raise Exception(f"Failed to ingest PDF: {str(e)}")
        finally:
            # Cleanup temporary files if needed
            if os.path.exists(file_path) and file_path.startswith('temp_'):
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
    
    def search_similar_content(self, query: str, k: int = 4) -> List[Document]:
        logger.info(f"Searching for similar content to: {query}")
        return self.vector_store.similarity_search(query, k=k) 