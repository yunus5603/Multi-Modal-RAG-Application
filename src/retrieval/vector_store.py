import weaviate
from weaviate.auth import AuthApiKey
from urllib.parse import urlparse  # new import
import logging
from typing import List, Optional, Dict, Any
import uuid

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self, embedding_function, config):
        """Initialize Weaviate client and create collection if it doesn't exist."""
        self.config = config
        self.embedding = embedding_function
        
        try:
            # Parse the full URL into scheme and host
            parsed = urlparse(config.WCS_URL)
            scheme = parsed.scheme if parsed.scheme else "https"
            host = parsed.netloc if parsed.netloc else config.WCS_URL

            # Initialize Weaviate client using v4
            self.client = weaviate.WeaviateClient(
                scheme=scheme,
                host=host,
                auth_client_secret=AuthApiKey(api_key=config.WCS_API_KEY)
            )
            
            # Check if Weaviate is ready
            if not self.client.is_ready():
                raise Exception("Weaviate is not ready!")
            
            # Initialize schema using the updated schema API
            self._initialize_schema()
            logger.info(f"Successfully connected to Weaviate at {config.WCS_URL}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Weaviate client: {str(e)}")
            raise

    def _initialize_schema(self):
        """Create schema if it doesn't exist using the latest Weaviate API."""
        try:
            schema = self.client.schema.get()
            class_exists = any(
                cls.get("class") == self.config.COLLECTION_NAME 
                for cls in schema.get('classes', [])
            )

            if not class_exists:
                new_class = {
                    "class": self.config.COLLECTION_NAME,
                    "vectorizer": "none",  # using custom embedding vectors
                    "properties": [
                        {
                            "name": "content",
                            "dataType": ["text"],
                            "description": "The content of the document"
                        },
                        {
                            "name": "content_type",
                            "dataType": ["text"],
                            "description": "Type of content (text, image, table)"
                        },
                        {
                            "name": "source",
                            "dataType": ["text"],
                            "description": "Source file path"
                        },
                        {
                            "name": "page_number",
                            "dataType": ["int"],
                            "description": "Page number in the document"
                        },
                        {
                            "name": "image_data",
                            "dataType": ["text"],
                            "description": "Base64 encoded image data"
                        }
                    ]
                }
                self.client.schema.create_class(new_class)
                logger.info(f"Created new collection (class): {self.config.COLLECTION_NAME}")
            else:
                logger.info(f"Using existing collection (class): {self.config.COLLECTION_NAME}")

            # Optionally, store the class name for future use.
            self.class_name = self.config.COLLECTION_NAME

        except Exception as e:
            logger.error(f"Error initializing Weaviate schema: {str(e)}")
            raise

    def add_documents(self, documents: List[Any]) -> List[str]:
        """Add documents to the vector store."""
        try:
            doc_ids = []
            
            # Use collection's batch interface
            with self.collection.batch.dynamic() as batch:
                for doc in documents:
                    doc_id = str(uuid.uuid4())
                    doc_ids.append(doc_id)

                    # Extract embedding and metadata
                    embedding = doc.metadata.get("embedding")
                    if embedding is None:
                        logger.warning(f"No embedding found for document {doc_id}")
                        continue

                    # Prepare properties
                    properties = {
                        "content": doc.page_content,
                        "content_type": doc.metadata.get("content_type", "text"),
                        "source": doc.metadata.get("source", ""),
                        "page_number": doc.metadata.get("page_number", 0)
                    }

                    # Add image data if present
                    if doc.metadata.get("content_type") == "image" and doc.metadata.get("image_data"):
                        properties["image_data"] = doc.metadata["image_data"]

                    # Add object to batch
                    batch.add_object(
                        properties=properties,
                        vector=embedding,
                        uuid=doc_id
                    )

            logger.info(f"Successfully added {len(doc_ids)} documents to Weaviate")
            return doc_ids

        except Exception as e:
            logger.error(f"Error adding documents to Weaviate: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 4, filter_dict: Optional[Dict] = None) -> List[Any]:
        """Perform similarity search."""
        try:
            # Generate query embedding
            query_embedding = self.embedding(query)

            # Build query using collection's query interface
            query_builder = (
                self.collection.query
                .near_vector(
                    vector=query_embedding,
                    limit=k
                )
            )

            # Add filters if provided
            if filter_dict:
                filter_conditions = []
                for key, value in filter_dict.items():
                    filter_conditions.append(
                        self.collection.query.filter.by_property(key).equal(value)
                    )
                if filter_conditions:
                    query_builder = query_builder.with_where(
                        self.collection.query.filter.and_(*filter_conditions)
                    )

            # Execute query
            response = query_builder.objects.get()

            # Process results
            documents = []
            for obj in response.objects: # Access objects using .objects
                doc = {
                    "content": obj.properties.get("content"),
                    "content_type": obj.properties.get("content_type"),
                    "source": obj.properties.get("source"),
                    "page_number": obj.properties.get("page_number")
                }

                if obj.properties.get("image_data"):
                    doc["image_data"] = obj.properties.get("image_data")

                documents.append(doc)

            return documents

        except Exception as e:
            logger.error(f"Error performing similarity search: {str(e)}")
            raise

    def __del__(self):
        """Cleanup when the object is deleted."""
        if hasattr(self, 'client'):
            self.client.close()