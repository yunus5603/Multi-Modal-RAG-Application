from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from base64 import b64decode
from typing import Dict, Any, List
import uuid
from langchain_community.vectorstores import Chroma
from langchain.storage import InMemoryStore
from langchain.schema.document import Document
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever

class RAGChain:
    """Implements the RAG (Retrieval Augmented Generation) chain."""

    def __init__(self, retriever: Any, model_name: str = "gpt-4o-mini"):
        self.retriever = retriever
        self.model_name = model_name
        self.chain = self._build_chain()
        self.chain_with_sources = self._build_chain_with_sources()

    def _parse_docs(self, docs: List[Any]) -> Dict[str, List]:
        """Parse documents into images and texts."""
        b64 = []
        text = []
        for doc in docs:
            try:
                b64decode(doc)
                b64.append(doc)
            except Exception:
                text.append(doc)
        return {"images": b64, "texts": text}

    def _build_prompt(self, kwargs: Dict[str, Any]) -> ChatPromptTemplate:
        """Build the prompt for the RAG chain."""
        docs_by_type = kwargs["context"]
        user_question = kwargs["question"]

        context_text = ""
        if len(docs_by_type["texts"]) > 0:
            for text_element in docs_by_type["texts"]:
                context_text += text_element.text

        prompt_template = f"""
        Answer the question based only on the following context, which can include text, tables, and the below image.
        Context: {context_text}
        Question: {user_question}
        """

        prompt_content = [{"type": "text", "text": prompt_template}]

        if len(docs_by_type["images"]) > 0:
            for image in docs_by_type["images"]:
                prompt_content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image}"},
                    }
                )

        return ChatPromptTemplate.from_messages(
            [HumanMessage(content=prompt_content)]
        )

    def _build_chain(self):
        """Build the basic RAG chain."""
        return (
            {
                "context": self.retriever | RunnableLambda(self._parse_docs),
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(self._build_prompt)
            | ChatOpenAI(model=self.model_name)
            | StrOutputParser()
        )

    def _build_chain_with_sources(self):
        """Build the RAG chain that includes sources."""
        return {
            "context": self.retriever | RunnableLambda(self._parse_docs),
            "question": RunnablePassthrough(),
        } | RunnablePassthrough().assign(
            response=(
                RunnableLambda(self._build_prompt)
                | ChatOpenAI(model=self.model_name)
                | StrOutputParser()
            )
        )

    def query(self, question: str) -> str:
        """Query the RAG chain."""
        return self.chain.invoke(question)

    def query_with_sources(self, question: str) -> Dict[str, Any]:
        """Query the RAG chain and return sources."""
        return self.chain_with_sources.invoke(question)