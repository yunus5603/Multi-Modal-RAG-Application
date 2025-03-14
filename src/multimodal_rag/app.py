import streamlit as st
from pathlib import Path
import tempfile
from multimodal_rag.extractors.pdf_extractor import PDFExtractor
from multimodal_rag.models.summarizer import ContentSummarizer
from multimodal_rag.vectorstore.store import VectorStoreManager
from multimodal_rag.models.rag_chain import RAGChain
from multimodal_rag.utils.helpers import display_base64_image
import base64
from PIL import Image
import io
import asyncio
import nest_asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Apply nest_asyncio to allow nested event loops
nest_asyncio.apply()

def init_event_loop():
    """Initialize event loop for async operations."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop

async def process_pdf_async(uploaded_file):
    """Process uploaded PDF and initialize RAG system asynchronously."""
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_path = tmp_file.name

    try:
        # Extract elements from PDF
        extractor = PDFExtractor()
        chunks = extractor.extract_elements(tmp_path)
        elements = extractor.separate_elements(chunks)
        
        # Initialize summarizer and create summaries
        summarizer = ContentSummarizer()
        text_summaries = await summarizer.summarize_texts(elements['texts'])
        table_summaries = await summarizer.summarize_tables(elements['tables'])
        image_summaries = await summarizer.summarize_images(elements['images'])
        
        # Initialize vector store
        vector_store = VectorStoreManager()
        vector_store.add_texts(elements['texts'], text_summaries)
        vector_store.add_tables(elements['tables'], table_summaries)
        vector_store.add_images(elements['images'], image_summaries)
        
        # Initialize RAG chain
        rag_chain = RAGChain(vector_store.get_retriever())
        
        # Update session state
        st.session_state.vector_store = vector_store
        st.session_state.rag_chain = rag_chain
        st.session_state.pdf_processed = True
        st.session_state.elements = elements
        
    finally:
        # Clean up temp file
        Path(tmp_path).unlink()

def display_image(base64_str):
    """Display base64 encoded image in Streamlit."""
    image_bytes = base64.b64decode(base64_str)
    image = Image.open(io.BytesIO(image_bytes))
    st.image(image, use_column_width=True)

def main():
    st.set_page_config(
        page_title="Multimodal RAG System",
        page_icon="🤖",
        layout="wide"
    )
    
    # Initialize session state
    if 'pdf_processed' not in st.session_state:
        st.session_state.pdf_processed = False
    
    # Sidebar for PDF upload
    with st.sidebar:
        st.header("Upload Document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
        
        if uploaded_file and not st.session_state.pdf_processed:
            with st.spinner("Processing PDF..."):
                # Initialize event loop and run async processing
                loop = init_event_loop()
                try:
                    loop.run_until_complete(process_pdf_async(uploaded_file))
                    st.success("PDF processed successfully!")
                except Exception as e:
                    st.error(f"Error processing PDF: {str(e)}")
                    logger.error(f"PDF processing error: {str(e)}", exc_info=True)
    
    # Main content area
    if not st.session_state.pdf_processed:
        st.info("Please upload a PDF document to begin.")
        return
    
    # Display extracted elements
    with st.expander("📄 Extracted Elements", expanded=False):
        tab1, tab2, tab3 = st.tabs(["Texts", "Tables", "Images"])
        
        with tab1:
            for text in st.session_state.elements['texts']:
                st.text(text.text[:500] + "...")
        
        with tab2:
            for table in st.session_state.elements['tables']:
                st.markdown(table.metadata.text_as_html, unsafe_allow_html=True)
        
        with tab3:
            for image in st.session_state.elements['images']:
                display_image(image)
    
    # Query interface
    st.header("🔍 Ask Questions")
    query = st.text_input("Enter your question:")
    
    if query:
        with st.spinner("Generating response..."):
            try:
                response = st.session_state.rag_chain.query_with_sources(query)
                st.markdown("### Answer:")
                st.write(response['response'])
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                logger.error(f"Query error: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main() 