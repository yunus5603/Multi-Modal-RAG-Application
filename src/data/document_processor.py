from typing import List, Dict
from unstructured.partition.pdf import partition_pdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
    
    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF and return list of Document objects"""
        # Extract content from PDF
        raw_chunks = partition_pdf(
            filename=file_path,
            strategy="fast"
        )
        
        # Convert to text and create documents
        documents = []
        for chunk in raw_chunks:
            doc = Document(
                page_content=str(chunk),
                metadata={"source": file_path}
            )
            documents.append(doc)
        
        # Split into smaller chunks
        return self.text_splitter.split_documents(documents) 