from langchain_core.prompts import ChatPromptTemplate

RAG_PROMPT = ChatPromptTemplate.from_template("""
Answer the question based on the following context. If you cannot find 
the answer in the context, say "I don't have enough information to answer this question."

Context: {context}

Question: {question}

Answer: Let me help you with that.
""") 