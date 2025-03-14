from typing import List, Any
from IPython.display import Image, display
import base64

def display_base64_image(base64_code: str) -> None:
    """Display an image from its base64 representation."""
    image_data = base64.b64decode(base64_code)
    display(Image(data=image_data))

def get_images_base64(chunks: List[Any]) -> List[str]:
    """Extract base64-encoded images from chunks."""
    images_b64 = []
    for chunk in chunks:
        if "CompositeElement" in str(type(chunk)):
            chunk_els = chunk.metadata.orig_elements
            for el in chunk_els:
                if "Image" in str(type(el)):
                    images_b64.append(el.metadata.image_base64)
    return images_b64 