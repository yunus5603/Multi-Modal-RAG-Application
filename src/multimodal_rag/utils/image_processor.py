import base64
import io
from PIL import Image
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Handles image processing to reduce token sizes for LLMs."""
    
    def __init__(self, max_width=512, max_height=512, quality=70, format="JPEG"):
        """
        Initialize image processor with size and quality settings.
        
        Args:
            max_width: Maximum width of processed images
            max_height: Maximum height of processed images
            quality: JPEG compression quality (0-100)
            format: Output image format (JPEG, PNG)
        """
        self.max_width = max_width
        self.max_height = max_height
        self.quality = quality
        self.format = format
    
    def process_base64_image(self, base64_string):
        """
        Process a base64 encoded image to reduce its size.
        
        Args:
            base64_string: Base64-encoded image string
            
        Returns:
            Processed base64-encoded image string
        """
        try:
            # Decode base64 string to image
            image_data = base64.b64decode(base64_string)
            image = Image.open(io.BytesIO(image_data))
            
            # Resize image while maintaining aspect ratio
            image.thumbnail((self.max_width, self.max_height), Image.LANCZOS)
            
            # Convert to RGB if needed (in case of RGBA, etc.)
            if image.mode != 'RGB' and self.format == 'JPEG':
                image = image.convert('RGB')
            
            # Save to buffer with compression
            buffer = io.BytesIO()
            image.save(buffer, format=self.format, quality=self.quality, optimize=True)
            
            # Convert back to base64
            buffer.seek(0)
            compressed_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
            
            # Log compression stats
            original_size = len(base64_string)
            new_size = len(compressed_base64)
            reduction = (1 - new_size / original_size) * 100
            logger.info(f"Image compressed: {original_size} → {new_size} bytes ({reduction:.1f}% reduction)")
            
            return compressed_base64
            
        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            # Return a smaller version or placeholder if processing fails
            return base64_string
    
    def process_images_batch(self, base64_strings):
        """Process a batch of base64-encoded images."""
        return [self.process_base64_image(img) for img in base64_strings] 