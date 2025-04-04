from langchain_groq import ChatGroq
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import Dict, Any, List
from multimodal_rag.config.settings import Settings
import logging
import base64
from PIL import Image
import io

logger = logging.getLogger(__name__)

class RAGChain:
    """Handles RAG (Retrieval Augmented Generation) operations."""

    def __init__(self, retriever):
        self.settings = Settings()
        self.retriever = retriever
        self.chain = self._create_chain()

    def _create_groq_chat(self) -> ChatGroq:
        """Creates a ChatGroq instance with proper configuration."""
        return ChatGroq(
            model_name=self.settings.DEFAULT_GROQ_MODEL,
            temperature=0.3,
            max_tokens=self.settings.MAX_TOKENS
        )

    def _create_chain(self):
        """Creates the RAG chain with proper prompting."""
        # Format retrieved documents
        def format_docs(docs):
            formatted_texts = []
            for i, doc in enumerate(docs):
                # Format content based on type and metadata
                content_type = getattr(doc.metadata, "type", "text")
                page_number = getattr(doc.metadata, "page_number", "unknown")
                
                if content_type == "text":
                    formatted_texts.append(
                        f"[TEXT from page {page_number}]: {getattr(doc.metadata, 'original_content', doc.page_content)}"
                    )
                elif content_type == "table":
                    formatted_texts.append(
                        f"[TABLE from page {page_number}]: {getattr(doc.metadata, 'html_content', doc.page_content)}"
                    )
                elif content_type == "image":
                    formatted_texts.append(
                        f"[IMAGE from page {page_number}]: {doc.page_content}"
                    )
            
            return "\n\n".join(formatted_texts)

        # Retrieve relevant context
        def get_context(query):
            try:
                docs = self.retriever.get_relevant_documents(query)
                logger.info(f"Retrieved {len(docs)} relevant documents")
                return {"docs": docs, "query": query}
            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                return {"docs": [], "query": query}

        # Create prompt template
        template = """You are a helpful assistant answering questions about a document.
        Use only the following context to answer the question. If you don't know the answer
        based on the context, say "I don't have enough information to answer this question."

        Context:
        {context}

        Question: {query}

        Answer the question concisely and professionally based only on the provided context.
        Include page numbers when referencing specific content.
        """

        prompt = ChatPromptTemplate.from_template(template)
        
        # Create the chain with error handling
        def safe_invoke(model, inputs):
            try:
                return model.invoke(inputs)
            except Exception as e:
                logger.error(f"Error invoking model: {str(e)}")
                return "I encountered an error processing your request. Please try again with a simpler question."
        
        chain = (
            RunnableLambda(get_context)
            | {
                "context": lambda x: format_docs(x["docs"]),
                "query": lambda x: x["query"],
                "raw_context": lambda x: x["docs"]
            }
            | {
                "response": RunnableLambda(lambda x: safe_invoke(
                    prompt | self._create_groq_chat() | StrOutputParser(),
                    {"context": x["context"], "query": x["query"]}
                )),
                "context": lambda x: x["raw_context"]
            }
        )

        return chain

    def query_with_sources(self, query: str) -> Dict[str, Any]:
        """
        Query the RAG system and return response with sources.
        
        Args:
            query: The question to ask
            
        Returns:
            Dict containing response and context
        """
        try:
            logger.info(f"Processing query: {query}")
            result = self.chain.invoke(query)
            
            # Organize context by type with enhanced metadata
            context = {
                "texts": [],
                "tables": [],
                "images": []
            }
            
            if "context" in result:
                for doc in result["context"]:
                    doc_type = getattr(doc.metadata, "type", "text")
                    if doc_type == "text":
                        context["texts"].append({
                            "content": getattr(doc.metadata, "original_content", doc.page_content),
                            "page_number": getattr(doc.metadata, "page_number", "unknown")
                        })
                    elif doc_type == "table":
                        context["tables"].append({
                            "content": getattr(doc.metadata, "html_content", doc.page_content),
                            "page_number": getattr(doc.metadata, "page_number", "unknown")
                        })
                    elif doc_type == "image":
                        context["images"].append({
                            "content": doc.page_content,
                            "page_number": getattr(doc.metadata, "page_number", "unknown")
                        })
            
            return {
                "response": result["response"],
                "context": context
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {str(e)}")
            return {
                "response": "Sorry, I encountered an error while processing your query.",
                "context": {"texts": [], "tables": [], "images": []}
            }