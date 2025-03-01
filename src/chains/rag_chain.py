from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from .prompt_templates import RAG_PROMPT
from langsmith import Client

class RAGChain:
    def __init__(self, vector_store, llm):
        self.vector_store = vector_store
        self.llm = llm
        self.client = Client()  # Initialize LangSmith client
        self.chain = self._build_chain()
    
    def _build_chain(self):
        # Define retrieval function
        def retrieve_docs(query):
            docs = self.vector_store.similarity_search(query)
            return "\n\n".join([doc.page_content for doc in docs])
        
        # Build the chain
        chain = (
            {
                "context": lambda x: retrieve_docs(x),
                "question": RunnablePassthrough()
            }
            | RAG_PROMPT
            | self.llm
            | StrOutputParser()
        )
        
        return chain
    
    async def run(self, query: str) -> str:
        """Run the RAG chain on a query with LangSmith tracking"""
        # The chain will automatically be tracked by LangSmith
        # because we've set LANGCHAIN_TRACING_V2=true in our environment
        return await self.chain.ainvoke(query) 