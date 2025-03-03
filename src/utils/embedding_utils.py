import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel, CLIPProcessor, CLIPModel
from PIL import Image as PILImage
import pandas as pd
import io
import logging

logger = logging.getLogger(__name__)

class EmbeddingGenerator:
    def __init__(self, text_model_name="BAAI/bge-large-en", image_model_name="openai/clip-vit-base-patch32"):
        # Initialize text embedding model
        self.text_model_name = text_model_name
        self.text_tokenizer = AutoTokenizer.from_pretrained(text_model_name)
        self.text_model = AutoModel.from_pretrained(text_model_name)
        
        # Initialize CLIP for image embeddings
        self.image_model_name = image_model_name
        self.clip_processor = CLIPProcessor.from_pretrained(image_model_name)
        self.clip_model = CLIPModel.from_pretrained(image_model_name)
        
        self.target_size = 1024
        logger.info(f"Initialized embedding generator with models: {text_model_name}, {image_model_name}")

    def adjust_embedding_size(self, embedding, content_type=None):
        """Adjust the size of the embedding to match the target size."""
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.numpy()
        if isinstance(embedding, np.ndarray):
            embedding = embedding.flatten().tolist()
            
        current_size = len(embedding)
        logger.debug(f"Adjusting embedding size from {current_size} to {self.target_size} for {content_type}")
        
        if current_size < self.target_size:
            # Pad with zeros
            return np.pad(embedding, (0, self.target_size - current_size), 'constant').tolist()
        elif current_size > self.target_size:
            # Use dimensionality reduction to maintain information
            embedding_array = np.array(embedding).reshape(1, -1)
            from sklearn.decomposition import PCA
            pca = PCA(n_components=self.target_size)
            reduced_embedding = pca.fit_transform(embedding_array)[0]
            return reduced_embedding.tolist()
        return embedding

    def generate_text_embedding(self, text):
        """Generate embedding for text content."""
        try:
            inputs = self.text_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.text_model(**inputs)
            embedding = outputs.last_hidden_state.mean(dim=1).squeeze()
            return self.adjust_embedding_size(embedding, "text")
        except Exception as e:
            logger.error(f"Error generating text embedding: {str(e)}")
            return None

    def generate_table_embedding(self, table_content):
        """Generate embedding for table content."""
        try:
            # Convert table to a structured format
            if isinstance(table_content, str):
                # Try to parse the table string into a DataFrame
                try:
                    # First, try to parse as markdown
                    df = pd.read_csv(io.StringIO(table_content), sep='|', skipinitialspace=True)
                except:
                    # If that fails, try to parse as CSV
                    df = pd.read_csv(io.StringIO(table_content))
            elif hasattr(table_content, 'to_string'):
                # If it's already a DataFrame or similar
                df = table_content
            else:
                df = pd.DataFrame([table_content])

            # Convert DataFrame to a structured string representation
            table_str = (
                f"Table Headers: {', '.join(df.columns.astype(str))}\n"
                f"Table Content: {df.to_string(index=False)}"
            )
            
            # Generate embedding for the structured table representation
            return self.generate_text_embedding(table_str)
        except Exception as e:
            logger.error(f"Error generating table embedding: {str(e)}")
            return None

    def generate_image_embedding(self, image_data):
        """Generate embedding for image content."""
        try:
            # Convert image data to PIL Image
            if isinstance(image_data, str) and image_data.startswith('data:image'):
                # Handle base64 encoded images
                import base64
                image_data = image_data.split(',')[1]
                image_bytes = base64.b64decode(image_data)
                image = PILImage.open(io.BytesIO(image_bytes))
            elif isinstance(image_data, bytes):
                # Handle raw bytes
                image = PILImage.open(io.BytesIO(image_data))
            elif isinstance(image_data, str):
                # Handle file paths
                image = PILImage.open(image_data)
            else:
                raise ValueError("Unsupported image data format")

            # Process image with CLIP
            inputs = self.clip_processor(images=image, return_tensors="pt")
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
            
            # Get embedding from image features
            embedding = image_features.squeeze()
            return self.adjust_embedding_size(embedding, "image")
        except Exception as e:
            logger.error(f"Error generating image embedding: {str(e)}")
            return None

    def combine_embeddings(self, embeddings_list):
        """Combine multiple embeddings into a single embedding."""
        if not embeddings_list:
            return None
        
        # Stack embeddings and take the mean
        combined = np.mean([np.array(emb) for emb in embeddings_list if emb is not None], axis=0)
        return self.adjust_embedding_size(combined) 