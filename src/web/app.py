from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
import os
from pathlib import Path
import logging
from config.settings import Settings
from src.data.ingestion_manager import DataIngestionManager
from src.chains.rag_chain import RAGChain
from src.models.llm import LLMManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(title="PDF RAG Application")

# Get the absolute path to the directories
BASE_DIR = Path(__file__).resolve().parent
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"  # Updated path

# Verify directories exist
logger.info(f"Templates directory: {TEMPLATES_DIR}")
logger.info(f"Static directory: {STATIC_DIR}")

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
    return templates.TemplateResponse(
        "index.html", 
        {"request": request}
    )

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/upload")
async def upload_file(file: UploadFile):
    temp_path = f"temp_{file.filename}"
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        vector_store, insert_ids = manager.ingest_pdf(temp_path)
        
        return {
            "message": f"PDF processed successfully. {len(insert_ids) if insert_ids else 0} documents added.",
            "status": "success"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
async def query(prompt: str = Form(...)):
    try:
        rag_chain = RAGChain(manager.vector_store, llm_manager.get_llm())
        response = await rag_chain.run(prompt)
        
        return {
            "response": response,
            "status": "success"
        }
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        ) 