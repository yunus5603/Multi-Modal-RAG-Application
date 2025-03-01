import uvicorn
from src.web.app import app
from config.settings import Settings
from src.data.ingestion_manager import DataIngestionManager

def main():
    # Initialize components
    config = Settings()
    manager = DataIngestionManager(config)
    
    # Run the FastAPI application
    uvicorn.run(app, host="0.0.0.0", port=8000)

if __name__ == "__main__":
    main()