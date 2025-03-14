from unstructured.partition.pdf import partition_pdf
from typing import List, Dict, Any
from multimodal_rag.config.settings import Settings

class PDFExtractor:
    """Handles extraction of content from PDF files."""

    def __init__(self, output_path: str = "./content/"):
        """Initialize PDFExtractor with settings from config."""
        self.output_path = output_path
        self.settings = Settings()

    def extract_elements(self, file_path: str) -> List[Any]:
        """Extract elements from a PDF file using configured settings."""
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
        )
        return chunks

    def separate_elements(self, chunks: List[Any]) -> Dict[str, List]:
        """Separate PDF elements into tables, texts, and images."""
        tables = []
        texts = []
        images = []

        for chunk in chunks:
            if "Table" in str(type(chunk)):
                tables.append(chunk)
            if "CompositeElement" in str(type(chunk)):
                texts.append(chunk)
                # Extract images from composite elements
                chunk_els = chunk.metadata.orig_elements
                for el in chunk_els:
                    if "Image" in str(type(el)):
                        images.append(el.metadata.image_base64)

        return {
            "tables": tables,
            "texts": texts,
            "images": images
        } 