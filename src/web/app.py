from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os
from pathlib import Path
import logging
from config.settings import Settings
from src.data.ingestion_manager import DataIngestionManager
from src.models.llm import LLMManager
from src.chains.rag_chain import RAGChain

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="PDF RAG Application")

# Get the absolute path to the directories
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

# Setup templates and static files
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Initialize components
config = Settings()
manager = DataIngestionManager(config)
llm_manager = LLMManager(config)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    logger.info("Accessing index page")
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile):
    logger.info(f"Received file upload: {file.filename}")
    temp_path = f"temp_{file.filename}"
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        vector_store, insert_ids = manager.ingest_pdf(temp_path)
        
        if insert_ids:
            logger.info(f"Successfully processed PDF. Documents added: {len(insert_ids)}")
            return {"message": f"PDF processed successfully. {len(insert_ids)} documents added.", "status": "success"}
        else:
            logger.warning("No documents were extracted from the PDF")
            return {"message": "PDF processed but no content was extracted.", "status": "warning"}
            
    except Exception as e:
        logger.error(f"Error processing upload: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)
            logger.info(f"Cleaned up temporary file: {temp_path}")

@app.post("/query")
async def query(prompt: str = Form(...)):
    logger.info(f"Received query: {prompt}")
    try:
        rag_chain = RAGChain(
            vector_store=manager.vector_store,
            llm=llm_manager.get_llm()  # This now returns the ChatGroq instance
        )
        response = await rag_chain.run(prompt)
        logger.info("Query processed successfully")
        return {"response": response, "status": "success"}
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}", exc_info=True)
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/health")
async def health_check():
    return {"status": "healthy"} 