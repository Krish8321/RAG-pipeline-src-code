from pathlib import Path

class Settings:
    # Configuration
    
    # Project Information
    # APP_NAME = "ARIA"
    # APP_VERSION = "0.1.0"
    
    # Project Paths
    PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent 
    
    DATA_DIR = PROJECT_ROOT / "data"
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    
    CHROMA_DB_DIR = DATA_DIR / "chroma_db"
    
    MODELS_DIR = PROJECT_ROOT / "models"
    
    LOGS_DIR = PROJECT_ROOT / "logs"
    
    
    # Embedding Configuration
    
    EMBEDDING_MODEL = "BAAI/bge-base-en-v1.5"
    BATCH_SIZE = 64
    
    # Retrieval configuration
    
    TOP_K = 5
    
    # LLM Configuration
    
    LLM_MODEL = "qwen3:8b"
    
    TEMPERATURE = 0.0
    
    LOG_LEVEL = "INFO"
    
settings = Settings()