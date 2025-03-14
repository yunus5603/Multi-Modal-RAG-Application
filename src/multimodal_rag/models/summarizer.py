from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Any
import asyncio
from multimodal_rag.utils.rate_limiter import RateLimiter
from multimodal_rag.config.settings import Settings
import logging
import base64

logger = logging.getLogger(__name__)

class ContentSummarizer:
    """Handles summarization of different content types with rate limiting."""

    def __init__(self):
        self.settings = Settings()
        self.rate_limiter = RateLimiter()
        self.text_chain = self._create_text_chain()
        self.image_chain = self._create_image_chain()

    def _create_text_chain(self):
        """Creates the chain for summarizing text and tables with rate limiting."""
        prompt_text = """
        Summarize the following content concisely:
        {element}
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        
        model = ChatGroq(
            temperature=0.5,
            model=self.settings.DEFAULT_GROQ_MODEL,
            max_tokens=self.settings.MAX_TOKENS,
            retry_on_rate_limit=True,
            max_retries=3
        )
        
        return {"element": lambda x: x} | prompt | model | StrOutputParser()

    def _create_image_chain(self):
        """Creates the chain for summarizing images with rate limiting."""
        prompt_text = """
        You are an expert at analyzing images. I will give you a base64 encoded image.
        Please describe what you see in the image concisely, focusing on key details.
        
        Image (base64): {image}
        """
        prompt = ChatPromptTemplate.from_template(prompt_text)
        
        # Use the same Groq model for image descriptions
        model = ChatGroq(
            temperature=0.5,
            model=self.settings.DEFAULT_GROQ_MODEL,
            max_tokens=self.settings.MAX_TOKENS,
            retry_on_rate_limit=True,
            max_retries=3
        )
        
        return prompt | model | StrOutputParser()

    async def summarize_texts(self, texts: List[Any]) -> List[str]:
        """Summarize text elements with rate limiting."""
        return await self.rate_limiter.process_batch_async(
            texts,
            lambda batch: self.text_chain.batch(batch, {"max_concurrency": 1}),
            batch_size=self.settings.BATCH_SIZE,
            delay=self.settings.BATCH_DELAY
        )

    async def summarize_tables(self, tables: List[Any]) -> List[str]:
        """Summarize table elements with rate limiting."""
        tables_html = [table.metadata.text_as_html for table in tables]
        return await self.rate_limiter.process_batch_async(
            tables_html,
            lambda batch: self.text_chain.batch(batch, {"max_concurrency": 1}),
            batch_size=self.settings.BATCH_SIZE,
            delay=self.settings.BATCH_DELAY
        )

    async def summarize_images(self, images: List[str]) -> List[str]:
        """Summarize image elements with rate limiting."""
        return await self.rate_limiter.process_batch_async(
            images,
            lambda batch: self.image_chain.batch(batch, {"max_concurrency": 1}),
            batch_size=self.settings.BATCH_SIZE,
            delay=self.settings.BATCH_DELAY
        ) 