from typing import Optional, Tuple, List
from langchain.schema import Document
from src.data.document_processor import DocumentProcessor
from src.data.embeddings import EmbeddingManager
from src.retrieval.vector_store import VectorStore

class DataIngestionManager:
    def __init__(self, config):
        self.config = config
        
        # Initialize components
        self.doc_processor = DocumentProcessor(config)
        self.embedding_manager = EmbeddingManager(config)
        self.vector_store = VectorStore(
            self.embedding_manager.get_embeddings(),
            config
        )
    
    def ingest_pdf(self, file_path: str) -> Tuple[VectorStore, Optional[List[str]]]:
        """
        Process and ingest PDF into vector store.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            Tuple containing vector store and optionally inserted document IDs
        """
        # Process PDF into documents
        documents = self.doc_processor.process_pdf(file_path)
        
        # Add to vector store
        insert_ids = self.vector_store.add_documents(documents)
        
        return self.vector_store, insert_ids
    
    def search_similar_content(self, query: str, k: int = 4) -> List[Document]:
        """
        Search for similar content based on query.
        
        Args:
            query: Search query string
            k: Number of results to return
            
        Returns:
            List of similar documents
        """
        return self.vector_store.similarity_search(query, k=k) 