from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .prompt_templates import RAG_PROMPT
import logging
from typing import List, Dict

logger = logging.getLogger(__name__)

class RAGChain:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.chain = self._build_chain()
    
    def _build_chain(self):
        """Build the RAG chain using LangChain components."""
        def retrieve_docs(query: str) -> str:
            docs = self.vector_store.similarity_search(query)
            return "\n\n".join(doc["content"] for doc in docs)
        
        chain = (
            {
                "context": retrieve_docs,
                "question": RunnablePassthrough()
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    async def run(self, query: str) -> str:
        try:
            # Use the chain's ainvoke method for async execution
            response = await self.chain.ainvoke(query)
            return response
        except Exception as e:
            logger.error(f"Error in RAG chain: {str(e)}", exc_info=True)
            raise 