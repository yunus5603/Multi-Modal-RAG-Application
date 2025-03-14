import uuid
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from typing import List, Any
from multimodal_rag.config.settings import Settings
import logging

logger = logging.getLogger(__name__)

class VectorStoreManager:
    """Manages the vector store operations."""

    def __init__(self, collection_name: str = "multi_modal_rag"):
        self.settings = Settings()
        
        # Initialize HuggingFace embeddings
        if not self.settings.HUGGINGFACE_API_KEY:
            raise ValueError("HuggingFace API key is required but not found in settings")
            
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",  # Good balance of speed and performance
            model_kwargs={'device': 'cpu'},  # Use CPU for inference
            encode_kwargs={'normalize_embeddings': True},  # Normalize embeddings for better similarity search
            huggingface_api_key=self.settings.HUGGINGFACE_API_KEY
        )
        
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self.embeddings
        )
        
        self.store = InMemoryStore()
        self.id_key = "doc_id"
        self.retriever = MultiVectorRetriever(
            vectorstore=self.vectorstore,
            docstore=self.store,
            id_key=self.id_key,
        )

    def add_texts(self, texts: List[Any], summaries: List[str]):
        """Add texts and their summaries to the vector store."""
        doc_ids = [str(uuid.uuid4()) for _ in texts]
        summary_texts = [
            Document(page_content=summary, metadata={self.id_key: doc_ids[i]})
            for i, summary in enumerate(summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_texts)
        self.retriever.docstore.mset(list(zip(doc_ids, texts)))

    def add_tables(self, tables: List[Any], summaries: List[str]):
        """Add tables and their summaries to the vector store."""
        table_ids = [str(uuid.uuid4()) for _ in tables]
        summary_tables = [
            Document(page_content=summary, metadata={self.id_key: table_ids[i]})
            for i, summary in enumerate(summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_tables)
        self.retriever.docstore.mset(list(zip(table_ids, tables)))

    def add_images(self, images: List[str], summaries: List[str]):
        """Add images and their summaries to the vector store."""
        img_ids = [str(uuid.uuid4()) for _ in images]
        summary_img = [
            Document(page_content=summary, metadata={self.id_key: img_ids[i]})
            for i, summary in enumerate(summaries)
        ]
        self.retriever.vectorstore.add_documents(summary_img)
        self.retriever.docstore.mset(list(zip(img_ids, images)))

    def get_retriever(self) -> MultiVectorRetriever:
        """Get the multi-vector retriever."""
        return self.retriever