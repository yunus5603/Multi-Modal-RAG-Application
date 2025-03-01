from fastapi import FastAPI, UploadFile, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import os
from config.settings import Settings
from src.data.ingestion_manager import DataIngestionManager
from src.chains.rag_chain import RAGChain
from src.models.llm import LLMManager

app = FastAPI()

# Setup templates and static files
templates = Jinja2Templates(directory="src/web/templates")
app.mount("/static", StaticFiles(directory="src/web/templates/static"), name="static")

# Initialize components
config = Settings()
manager = DataIngestionManager(config)
llm_manager = LLMManager(config)

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/upload")
async def upload_file(file: UploadFile):
    # Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    try:
        contents = await file.read()
        with open(temp_path, "wb") as f:
            f.write(contents)
        
        # Process and ingest PDF
        vector_store, insert_ids = manager.ingest_pdf(temp_path)
        
        return {
            "message": "PDF processed successfully",
            "documents_added": len(insert_ids) if insert_ids else "Unknown"
        }
    finally:
        # Cleanup temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query")
async def query(prompt: str = Form(...)):
    try:
        # Initialize RAG chain
        rag_chain = RAGChain(manager.vector_store, llm_manager.get_llm())
        
        # Get response (will be automatically tracked by LangSmith)
        response = await rag_chain.run(prompt)
        return {"response": response}
    except Exception as e:
        return {"error": str(e)} 