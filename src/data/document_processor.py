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
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def is_poppler_installed():
    return shutil.which("pdfinfo") is not None

class DocumentProcessor:
    def __init__(self, config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", " ", ""]
        )
        
        # Enhanced PDF processing settings for images
        self.pdf_settings = {
            "strategy": "hi_res",  # Better for images
            "extract_images_in_pdf": True,
            "extract_tables": True,
            "infer_table_structure": True,
            "include_metadata": True,
            "max_image_size": (400, 400),  # Control image size
            "image_output_dir_path": "temp_images",  # Temporary image storage
        }
        
        # Configure Poppler path if available
        self.poppler_path = os.getenv('POPPLER_PATH')
        if self.poppler_path and os.path.exists(self.poppler_path):
            logger.info(f"Using Poppler from: {self.poppler_path}")
            self.pdf_settings["poppler_path"] = self.poppler_path
        else:
            logger.warning("Poppler path not configured properly. Image extraction may be limited.")

    def process_pdf(self, file_path: str) -> List[Document]:
        """Process PDF and extract text, tables, and images."""

        if not is_poppler_installed():
            logger.error("Poppler is not installed or not in PATH. Please install Poppler to process PDF files.")
            return

        try:
            logger.info(f"Starting to process PDF: {file_path}")
            
            # Create image output directory if needed
            os.makedirs(self.pdf_settings["image_output_dir_path"], exist_ok=True)
            
            # Extract elements with enhanced settings
            elements = partition_pdf(
                filename=file_path,
                **self.pdf_settings
            )
            
            documents = []
            for idx, element in enumerate(elements):
                try:
                    # Handle different element types
                    if isinstance(element, Table):
                        content = self._process_table(element)
                        metadata = self._create_metadata(file_path, "table", element)
                    elif isinstance(element, Image):
                        content, metadata = self._process_image(element, file_path, idx)
                        if not content:
                            continue
                    else:
                        content, metadata = self._process_text(element, file_path)
                        if not content:
                            continue

                    documents.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))
                
                except Exception as elem_err:
                    logger.warning(f"Failed to process element {idx}: {elem_err}")
                    continue
            
            # Split text documents
            split_documents = []
            for doc in documents:
                if doc.metadata["content_type"] == "text":
                    splits = self.text_splitter.split_documents([doc])
                    split_documents.extend(splits)
                else:
                    split_documents.append(doc)
            
            logger.info(f"Successfully processed PDF: {len(split_documents)} elements")
            return split_documents

        except Exception as e:
            logger.error(f"PDF processing failed: {str(e)}", exc_info=True)
            raise Exception(f"PDF processing error: {str(e)}")
        finally:
            # Cleanup temporary images
            if os.path.exists(self.pdf_settings["image_output_dir_path"]):
                for f in os.listdir(self.pdf_settings["image_output_dir_path"]):
                    os.remove(os.path.join(self.pdf_settings["image_output_dir_path"], f))
                os.rmdir(self.pdf_settings["image_output_dir_path"])

    def _process_table(self, element: Table) -> str:
        """Process table element into markdown format."""
        try:
            return element.metadata.text_as_html  # Preserve table structure
        except Exception as e:
            logger.warning(f"Failed to process table: {e}")
            return str(element)

    def _process_image(self, element: Image, file_path: str, idx: int) -> tuple:
        """Process image element into base64 encoded string."""
        try:
            # Get image data
            image_path = element.metadata.image_path
            if not image_path or not os.path.exists(image_path):
                raise ValueError("Image file not found")
            
            with open(image_path, "rb") as image_file:
                image_data = base64.b64encode(image_file.read()).decode()
            
            return (
                f"![Image {idx}]",  # Alt text
                {
                    "source": file_path,
                    "content_type": "image",
                    "image_data": image_data,
                    "page_number": element.metadata.page_number,
                    "coordinates": element.metadata.coordinates,
                    "image_format": os.path.splitext(image_path)[1][1:].upper()
                }
            )
        except Exception as e:
            logger.warning(f"Failed to process image {idx}: {e}")
            return None, None

    def _process_text(self, element, file_path: str) -> tuple:
        """Process text element."""
        content = str(element).strip()
        if not content:
            return None, None
            
        return (
            content,
            {
                "source": file_path,
                "content_type": "text",
                "page_number": element.metadata.page_number,
                "coordinates": element.metadata.coordinates
            }
        )

    def _create_metadata(self, file_path: str, content_type: str, element) -> dict:
        """Create standardized metadata."""
        return {
            "source": file_path,
            "content_type": content_type,
            "page_number": element.metadata.page_number,
            "coordinates": element.metadata.coordinates
        }

    def _get_element_text(self, element) -> str:
        """Extract text content from an element safely."""
        try:
            return str(element)
        except Exception as e:
            logger.warning(f"Failed to extract text from element: {e}")
            return ""