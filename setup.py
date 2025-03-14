from setuptools import setup, find_packages

setup(
    name="multimodal_rag",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "streamlit>=1.31.0",
        "Pillow>=10.0.0",
        "langchain>=0.1.0",
        "langchain-community>=0.0.10",
        "langchain-core>=0.1.10",
        "langchain-openai>=0.0.5",
        "langchain-groq>=0.0.1",
        "python-dotenv>=1.0.0",
        "unstructured>=0.10.30",
        "chromadb>=0.4.22",
        "tiktoken>=0.5.2",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "nest-asyncio>=1.5.8",
        "asyncio>=3.4.3",
    ],
) 