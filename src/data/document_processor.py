from typing import List
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import (
    Table, Text, Image, ElementMetadata
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import base64
import logging
import os
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Configure PDF processing settings
        self.pdf_settings = {
            "strategy": "fast",  # Use fast strategy instead of hi_res
            "include_metadata": True,
            "include_page_breaks": True,
            "extract_images_in_pdf": True,
            "extract_tables": True,
            "detect_tables": True,
            "pdf_image_dpi": 200,  # Lower DPI for better performance
        }
        
        # Add Poppler path if on Windows
        if os.name == 'nt':  # Windows
            poppler_path = os.getenv('POPPLER_PATH')
            if (poppler_path and os.path.exists(poppler_path)):
                logger.info(f"Using Poppler from: {poppler_path}")
                os.environ['PATH'] = f"{poppler_path};{os.environ['PATH']}"
            else:
                logger.warning("Poppler path not found or invalid. Some PDF features might be limited.")

    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF and extract text, tables, and images."""
        try:
            logger.info(f"Starting to process PDF: {file_path}")
            
            # Verify file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"PDF file not found: {file_path}")
            
            try:
                # First attempt with full settings
                elements = partition_pdf(
                    filename=file_path,
                    **self.pdf_settings
                )
                # Debug: Log the types of elements found
                element_types = {}
                for elem in elements:
                    elem_type = type(elem).__name__
                    element_types[elem_type] = element_types.get(elem_type, 0) + 1
                logger.info(f"Elements found in PDF: {element_types}")
                
            except Exception as e:
                logger.warning(f"Full PDF processing failed, falling back to basic mode: {e}")
                elements = partition_pdf(
                    filename=file_path,
                    strategy="fast",
                    include_metadata=True
                )
            
            documents = []
            for element in elements:
                try:
                    if isinstance(element, Table):
                        content = str(element)
                        metadata = {
                            "source": file_path,
                            "content_type": "table",
                            "page_number": getattr(element, "metadata", ElementMetadata()).page_number
                        }
                    elif isinstance(element, Image):
                        logger.info(f"Found image element. Available attributes: {dir(element)}")  # Debug image attributes
                        try:
                            image_data = None
                            if hasattr(element, 'image'):
                                logger.info("Trying to extract image via 'image' attribute")
                                image_data = element.image
                            elif hasattr(element, 'raw'):
                                logger.info("Trying to extract image via 'raw' attribute")
                                image_data = element.raw
                            elif hasattr(element, 'image_path'):
                                logger.info(f"Trying to extract image from path: {element.image_path}")
                                with open(element.image_path, 'rb') as img_file:
                                    image_data = img_file.read()
                            elif hasattr(element, 'data'):
                                logger.info("Trying to extract image via 'data' attribute")
                                image_data = element.data
                            
                            if image_data:
                                image_base64 = base64.b64encode(image_data).decode()
                                content = "[Image extracted from PDF]"
                                metadata = {
                                    "source": file_path,
                                    "content_type": "image",
                                    "image_data": image_base64,
                                    "page_number": getattr(element, "metadata", ElementMetadata()).page_number
                                }
                                logger.info("Successfully processed image")
                            else:
                                logger.warning("No image data found in any attribute")
                                continue
                        except Exception as img_err:
                            logger.warning(f"Failed to process image: {img_err}", exc_info=True)
                            continue
                    else:
                        content = str(element)
                        if not content.strip():
                            continue
                        metadata = {
                            "source": file_path,
                            "content_type": "text",
                            "page_number": getattr(element, "metadata", ElementMetadata()).page_number
                        }

                    doc = Document(
                        page_content=content,
                        metadata=metadata
                    )
                    documents.append(doc)
                
                except Exception as elem_err:
                    logger.warning(f"Failed to process element: {elem_err}", exc_info=True)
                    continue
            
            # Split text documents, keep others as is
            split_documents = []
            for doc in documents:
                if doc.metadata["content_type"] == "text":
                    splits = self.text_splitter.split_documents([doc])
                    split_documents.extend(splits)
                else:
                    split_documents.append(doc)
            
            logger.info(f"Successfully processed PDF: {len(split_documents)} elements extracted")
            logger.info(f"Content types: Text: {sum(1 for d in split_documents if d.metadata['content_type'] == 'text')}, "
                       f"Tables: {sum(1 for d in split_documents if d.metadata['content_type'] == 'table')}, "
                       f"Images: {sum(1 for d in split_documents if d.metadata['content_type'] == 'image')}")
            
            return split_documents

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
            raise Exception(f"Error processing PDF: {str(e)}")

    def _get_element_text(self, element) -> str:
        """Extract text content from an element safely."""
        try:
            return str(element)
        except Exception as e:
            logger.warning(f"Failed to extract text from element: {e}")
            return ""