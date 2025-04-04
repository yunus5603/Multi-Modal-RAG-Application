from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Any, Dict
import asyncio
from multimodal_rag.utils.rate_limiter import RateLimiter
from multimodal_rag.utils.image_processor import ImageProcessor
from multimodal_rag.config.settings import Settings
import logging

logger = logging.getLogger(__name__)

class ContentSummarizer:
    """Handles summarization of different content types with rate limiting."""

    def __init__(self):
        self.settings = Settings()
        self.rate_limiter = RateLimiter(
            max_retries=self.settings.MAX_RETRIES,
            initial_delay=self.settings.INITIAL_RETRY_DELAY
        )
        self.image_processor = ImageProcessor(
            max_width=256,  # Reduced from 512 to save more tokens
            max_height=256,
            quality=60,     # Lower quality to reduce token count
            format="JPEG"
        )
        self.text_chain = self._create_text_chain()
        self.image_chain = self._create_image_chain()

    def _create_groq_chat(self) -> ChatGroq:
        """Creates a ChatGroq instance with proper configuration."""
        # Only pass recognized parameters to avoid warnings
        return ChatGroq(
            model_name=self.settings.DEFAULT_GROQ_MODEL,
            temperature=0.3,
            max_tokens=self.settings.MAX_TOKENS
        )

    def _create_text_chain(self):
        """Creates the chain for summarizing text and tables."""
        prompt_text = """
        Summarize the following content concisely and professionally:
        
        Content: {element}
        
        Summary:
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        model = self._create_groq_chat()
        return {"element": lambda x: x} | prompt | model | StrOutputParser()

    def _create_image_chain(self):
        """Creates the chain for summarizing images."""
        prompt_text = """
        Describe this image concisely:
        
        Image (base64): {content}
        
        Description:
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        model = self._create_groq_chat()
        return prompt | model | StrOutputParser()

    async def _process_with_rate_limit(self, items: List[Any], chain, batch_size: int = None):
        """Process items with rate limiting."""
        if batch_size is None:
            batch_size = self.settings.BATCH_SIZE

        results = []
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            try:
                # Use the rate limiter for processing
                batch_results = await self.rate_limiter.process_batch_async(
                    batch,
                    lambda x: chain.batch(x, {"max_concurrency": 1}),
                    batch_size=1,  # Process one at a time within each batch
                    delay=self.settings.BATCH_DELAY
                )
                results.extend(batch_results)
                
                # Log progress
                logger.info(f"Processed batch {i//batch_size + 1}/{(len(items) + batch_size - 1)//batch_size}")
                
            except Exception as e:
                logger.error(f"Error processing batch: {str(e)}")
                raise

        return results

    async def summarize_texts(self, texts: List[Any]) -> List[str]:
        """Summarize text elements with rate limiting."""
        logger.info(f"Summarizing {len(texts)} text elements")
        return await self._process_with_rate_limit(texts, self.text_chain)

    async def summarize_tables(self, tables: List[Any]) -> List[str]:
        """Summarize table elements with rate limiting."""
        logger.info(f"Summarizing {len(tables)} table elements")
        tables_html = [table.metadata.text_as_html for table in tables]
        return await self._process_with_rate_limit(tables_html, self.text_chain)

    async def summarize_images(self, images: List[Dict]) -> List[str]:
        """Summarize image elements with rate limiting."""
        logger.info(f"Summarizing {len(images)} image elements")
        
        # Extract base64 content from image dictionaries
        base64_images = [img['content'] for img in images]
        
        # Process images to reduce size before summarizing
        processed_images = self.image_processor.process_images_batch(base64_images)
        logger.info(f"Processed {len(processed_images)} images to reduce size")
        
        # Create input dictionaries for the chain
        chain_inputs = [{'content': img} for img in processed_images]
        
        # Use smaller batch size for image processing
        return await self._process_with_rate_limit(
            chain_inputs, 
            self.image_chain,
            batch_size=1  # Process only one image at a time
        ) 