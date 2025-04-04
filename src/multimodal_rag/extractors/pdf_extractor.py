from unstructured.partition.pdf import partition_pdf
from typing import List, Dict, Any
from multimodal_rag.config.settings import Settings
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class PDFExtractor:
    """Handles extraction of content from PDF files."""

    def __init__(self, output_path: str = "./content/"):
        """Initialize PDFExtractor with settings from config."""
        self.output_path = output_path
        self.settings = Settings()
        # Create output directory if it doesn't exist
        Path(output_path).mkdir(parents=True, exist_ok=True)

    def extract_elements(self, file_path: str) -> List[Any]:
        """Extract elements from a PDF file using configured settings."""
        try:
            chunks = partition_pdf(
                filename=file_path,
                infer_table_structure=True,
                strategy="hi_res",
                extract_image_block_types=["Image"],
                extract_image_block_to_payload=True,
                chunking_strategy="by_title",
                max_characters=self.settings.MAX_CHARACTERS,
                combine_text_under_n_chars=self.settings.COMBINE_CHARS,
                new_after_n_chars=self.settings.NEW_CHARS,
                table_extraction_mode="lines",  # Use line detection for better table recognition
                table_extraction_confidence_threshold=0.5,  # Lower threshold to catch more tables
                table_extraction_include_headers=True,  # Ensure headers are captured
                table_extraction_include_footers=True,  # Ensure footers are captured
            )
            
            # Save chunks locally
            pdf_name = Path(file_path).stem
            chunks_dir = Path(self.output_path) / pdf_name
            chunks_dir.mkdir(exist_ok=True)
            
            # Save each chunk with its type and metadata
            for i, chunk in enumerate(chunks):
                chunk_type = str(type(chunk))
                chunk_data = {
                    "content": str(chunk),
                    "type": chunk_type,
                    "metadata": {
                        "page_number": chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else "unknown"
                    }
                }
                
                # Add type-specific metadata
                if "Table" in chunk_type:
                    chunk_data["metadata"]["text_as_html"] = chunk.metadata.text_as_html if hasattr(chunk.metadata, 'text_as_html') else str(chunk)
                elif "Image" in chunk_type:
                    chunk_data["metadata"]["image_base64"] = chunk.metadata.image_base64 if hasattr(chunk.metadata, 'image_base64') else None
                
                # Save chunk to JSON file
                chunk_file = chunks_dir / f"chunk_{i}.json"
                with open(chunk_file, 'w', encoding='utf-8') as f:
                    json.dump(chunk_data, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Successfully extracted and saved {len(chunks)} chunks from PDF")
            return chunks
        except Exception as e:
            logger.error(f"Error extracting elements from PDF: {str(e)}")
            raise

    def separate_elements(self, chunks: List[Any]) -> Dict[str, List]:
        """Separate PDF elements into tables, texts, and images with enhanced metadata."""
        tables = []
        texts = []
        images = []

        for chunk in chunks:
            chunk_type = str(type(chunk))
            
            # Enhanced table detection
            if "Table" in chunk_type or "TableChunk" in chunk_type:
                try:
                    # Add table with metadata
                    table_data = {
                        "content": chunk,
                        "metadata": {
                            "type": "table",
                            "text_as_html": chunk.metadata.text_as_html if hasattr(chunk.metadata, 'text_as_html') else str(chunk),
                            "page_number": chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else "unknown"
                        }
                    }
                    tables.append(table_data)
                    logger.debug(f"Extracted table from page {table_data['metadata']['page_number']}")
                except Exception as e:
                    logger.warning(f"Error processing table chunk: {str(e)}")
                    # Fallback to text if table processing fails
                    texts.append({
                        "content": chunk,
                        "metadata": {
                            "type": "text",
                            "page_number": chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else "unknown"
                        }
                    })
            
            elif "CompositeElement" in chunk_type:
                # Add text with metadata
                texts.append({
                    "content": chunk,
                    "metadata": {
                        "type": "text",
                        "page_number": chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else "unknown"
                    }
                })
                
                # Extract images from composite elements
                if hasattr(chunk.metadata, 'orig_elements'):
                    chunk_els = chunk.metadata.orig_elements
                    for el in chunk_els:
                        if "Image" in str(type(el)):
                            try:
                                images.append({
                                    "content": el.metadata.image_base64,
                                    "metadata": {
                                        "type": "image",
                                        "page_number": chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else "unknown"
                                    }
                                })
                            except Exception as e:
                                logger.warning(f"Error processing image: {str(e)}")

        logger.info(f"Separated elements: {len(texts)} texts, {len(tables)} tables, {len(images)} images")
        return {
            "tables": tables,
            "texts": texts,
            "images": images
        } 