import streamlit as st
from pathlib import Path
import tempfile
import logging
import sys
import asyncio
import nest_asyncio
import importlib

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

# Safely import modules that might cause path errors
def safe_import(module_name):
    try:
        return importlib.import_module(module_name)
    except RuntimeError as e:
        if "Tried to instantiate class '__path__._path'" in str(e):
            logger.warning(f"Known PyTorch path issue encountered: {str(e)}")
            # Try again with a workaround for torch path issues
            return importlib.import_module(module_name)
        else:
            raise

# Safely import our modules
try:
    from multimodal_rag.extractors.pdf_extractor import PDFExtractor
    from multimodal_rag.models.summarizer import ContentSummarizer
    from multimodal_rag.vectorstore.store import VectorStoreManager
    from multimodal_rag.models.rag_chain import RAGChain
    from multimodal_rag.utils.helpers import display_base64_image
except RuntimeError as e:
    if "Tried to instantiate class '__path__._path'" in str(e):
        logger.warning(f"Known PyTorch path issue detected. Attempting recovery...")
        # Ensure torch is loaded first to avoid path issues
        safe_import('torch')
        # Now retry imports
        from multimodal_rag.extractors.pdf_extractor import PDFExtractor
        from multimodal_rag.models.summarizer import ContentSummarizer
        from multimodal_rag.vectorstore.store import VectorStoreManager
        from multimodal_rag.models.rag_chain import RAGChain
        from multimodal_rag.utils.helpers import display_base64_image
    else:
        logger.error(f"Failed to import required modules: {str(e)}")
        raise

import base64
from PIL import Image
import io

def run_async(coro):
    """Helper function to run async code in Streamlit."""
    try:
        # Try getting the current event loop
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # If no loop exists, create a new one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(coro)

def display_image(base64_image):
    """Display an image from base64 string."""
    try:
        image_data = base64.b64decode(base64_image)
        image = Image.open(io.BytesIO(image_data))
        st.image(image, use_column_width=True)
    except Exception as e:
        st.error(f"Could not display image: {str(e)}")

async def process_pdf_async(uploaded_file):
    """Process uploaded PDF and initialize RAG system asynchronously."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Extract elements from PDF
        with st.spinner("📄 Extracting content from PDF..."):
            extractor = PDFExtractor()
            chunks = extractor.extract_elements(tmp_path)
            elements = extractor.separate_elements(chunks)
        
        # Initialize summarizer
        summarizer = ContentSummarizer()
        
        # Process text elements
        with st.spinner("📝 Summarizing text content..."):
            text_summaries = await summarizer.summarize_texts(elements['texts'])
        
        # Process table elements
        with st.spinner("📊 Summarizing tables..."):
            table_summaries = await summarizer.summarize_tables(elements['tables'])
        
        # Process image elements
        with st.spinner("🖼️ Analyzing images..."):
            image_summaries = await summarizer.summarize_images(elements['images'])
        
        # Initialize vector store
        with st.spinner("🔄 Building knowledge base..."):
            vector_store = VectorStoreManager()
            vector_store.add_texts(elements['texts'], text_summaries)
            vector_store.add_tables(elements['tables'], table_summaries)
            vector_store.add_images(elements['images'], image_summaries)
        
        # Initialize RAG chain
        with st.spinner("🧠 Initializing RAG system..."):
            rag_chain = RAGChain(vector_store.get_retriever())
        
        # Update session state
        st.session_state.vector_store = vector_store
        st.session_state.rag_chain = rag_chain
        st.session_state.pdf_processed = True
        st.session_state.elements = elements
        
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}", exc_info=True)
        raise e
    finally:
        # Clean up temp file
        Path(tmp_path).unlink()

def main():
    st.set_page_config(
        page_title="Multimodal RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    # Add title and description
    st.title("📚 Multimodal RAG Document Assistant")
    st.markdown("""
    This intelligent system can analyze documents containing text, tables, and images.
    Upload a PDF document and ask questions about its contents!
    """)
    
    # Initialize session state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("📤 Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file and not st.session_state.pdf_processed:
            st.info("Processing document. This may take a moment...")
            with st.spinner("🔄 Processing PDF..."):
                try:
                    run_async(process_pdf_async(uploaded_file))
                    st.success("✅ PDF processed successfully!")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    logger.error(f"PDF processing error: {str(e)}", exc_info=True)
    
    # Main content area
    if not st.session_state.pdf_processed:
        # Show placeholder when no document is loaded
        st.info("👆 Please upload a PDF document to begin.")
        
        # Add sample questions
        with st.expander("💡 Sample questions you can ask once a document is loaded"):
            st.markdown("""
            - What are the main topics covered in this document?
            - Summarize the key findings from the research.
            - What do the images in this document show?
            - Explain the data shown in the tables.
            - How do the charts relate to the text content?
            """)
        return
    
    # Display document information
    st.subheader("📄 Document Overview")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Text Chunks", len(st.session_state.elements['texts']))
    with col2:
        st.metric("Tables", len(st.session_state.elements['tables']))
    with col3:
        st.metric("Images", len(st.session_state.elements['images']))
    
    # Display extracted elements
    with st.expander("🔍 View Extracted Content", expanded=False):
        tab1, tab2, tab3 = st.tabs(["📝 Text", "📊 Tables", "🖼️ Images"])
        
        with tab1:
            if st.session_state.elements['texts']:
                for i, text in enumerate(st.session_state.elements['texts']):
                    with st.expander(f"Text Chunk {i+1}"):
                        st.write(text.text[:500] + ("..." if len(text.text) > 500 else ""))
            else:
                st.info("No text content extracted from the document.")
        
        with tab2:
            if st.session_state.elements['tables']:
                for i, table in enumerate(st.session_state.elements['tables']):
                    with st.expander(f"Table {i+1}"):
                        st.markdown(table.metadata.text_as_html, unsafe_allow_html=True)
            else:
                st.info("No tables extracted from the document.")
        
        with tab3:
            if st.session_state.elements['images']:
                for i, image in enumerate(st.session_state.elements['images']):
                    with st.expander(f"Image {i+1}"):
                        display_image(image)
            else:
                st.info("No images extracted from the document.")
    
    # Query interface
    st.subheader("💬 Ask Questions About Your Document")
    query = st.text_input("What would you like to know about this document?")
    
    if query:
        with st.spinner("🧠 Thinking..."):
            try:
                response = st.session_state.rag_chain.query_with_sources(query)
                
                st.markdown("### 📝 Answer:")
                st.write(response['response'])
                
                with st.expander("🔍 View Sources Used", expanded=False):
                    if response['context']['texts']:
                        st.subheader("Text Sources:")
                        for i, doc in enumerate(response['context']['texts']):
                            st.markdown(f"**Source {i+1}:**")
                            st.write(doc.page_content[:300] + "...")
                    
                    if response['context']['tables']:
                        st.subheader("Table Sources:")
                        for i, table in enumerate(response['context']['tables']):
                            st.markdown(f"**Table {i+1}:**")
                            st.markdown(table.metadata.text_as_html, unsafe_allow_html=True)
                    
                    if response['context']['images']:
                        st.subheader("Image Sources:")
                        for i, img in enumerate(response['context']['images']):
                            st.markdown(f"**Image {i+1}:**")
                            display_image(img)
            except Exception as e:
                st.error(f"❌ Error generating response: {str(e)}")
                logger.error(f"Query error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 