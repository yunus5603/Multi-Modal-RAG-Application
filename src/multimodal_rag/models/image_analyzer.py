import logging
from typing import List, Optional
import base64
import io
from PIL import Image
import asyncio
import time
import torch
from multimodal_rag.config.settings import Settings
from multimodal_rag.utils.error_handler import safe_execute
from multimodal_rag.utils.image_processor import ImageProcessor
from transformers import pipeline

logger = logging.getLogger(__name__)

class QwenVLImageAnalyzer:
    """Image analyzer using Qwen2.5-VL-72B model through HuggingFace Transformers."""
    
    def __init__(self):
        self.settings = Settings()
        self.image_processor = ImageProcessor(
            max_width=768,
            max_height=768,
            quality=90,
            format="JPEG"
        )
        
        # Load the pipeline only once
        try:
            logger.info("Initializing Qwen2.5-VL-72B pipeline...")
            # Check for GPU availability and set device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Initialize the pipeline with appropriate parameters
            self.pipeline = pipeline(
                "image-text-to-text", 
                model="Qwen/Qwen2.5-VL-72B-Instruct",
                device=self.device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            logger.info("Qwen2.5-VL-72B pipeline initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing Qwen2.5-VL pipeline: {str(e)}")
            raise
    
    @safe_execute(max_retries=2)
    def _analyze_single_image(self, base64_image: str, prompt: str = None) -> str:
        """
        Analyze a single image using Qwen2.5-VL-72B-Instruct.
        
        Args:
            base64_image: Base64-encoded image
            prompt: Optional custom prompt for analysis
            
        Returns:
            Analysis text
        """
        try:
            # Process image to optimal size
            processed_image = self.image_processor.process_base64_image(base64_image)
            
            # Convert base64 to PIL Image
            image_data = base64.b64decode(processed_image)
            image = Image.open(io.BytesIO(image_data))
            
            # Default prompt if none provided
            if not prompt:
                prompt = "Please analyze this image in detail. Describe what you see."
            
            # Format messages for the pipeline
            messages = [
                {"role": "user", "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image", "image": image}
                ]}
            ]
            
            # Run inference
            start_time = time.time()
            result = self.pipeline(messages)
            end_time = time.time()
            
            logger.info(f"Image analysis completed in {end_time - start_time:.2f} seconds")
            
            # Extract the generated text
            if isinstance(result, dict):
                return result.get("generated_text", "").strip()
            elif isinstance(result, list) and result:
                return result[0].get("generated_text", "").strip()
            else:
                return str(result).strip()
                
        except Exception as e:
            logger.error(f"Error in image analysis: {str(e)}")
            return f"Error analyzing image: {str(e)}"
            
    async def analyze_images_batch(self, images: List[str], batch_size: int = 1) -> List[str]:
        """
        Analyze a batch of images.
        
        Args:
            images: List of base64-encoded images
            batch_size: How many to process at once (always 1 for this model)
            
        Returns:
            List of image descriptions
        """
        results = []
        total_images = len(images)
        
        for i, img in enumerate(images):
            try:
                logger.info(f"Processing image {i+1}/{total_images}")
                
                # Run the analysis in a separate thread to avoid blocking
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(None, lambda: self._analyze_single_image(img))
                
                results.append(result)
                logger.info(f"Completed image {i+1}/{total_images}")
                
            except Exception as e:
                logger.error(f"Error analyzing image {i+1}: {str(e)}")
                results.append("Unable to analyze this image due to technical issues.")
                
            # Brief pause between images even when running locally
            if i < total_images - 1:
                await asyncio.sleep(0.5)
                
        return results 