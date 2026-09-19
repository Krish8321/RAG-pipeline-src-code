import logging
from pathlib import Path

from src.config.settings import Settings

def setup_logger() -> logging.Logger:
    
    
    Settings.LOGS_DIR.mkdir(parents=True, exist_ok=True)\
    
    log_file = Settings.LOGS_DIR / "aria.log"
    
    logging.basicConfig(
        level = Settings.LOG_LEVEL,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ],
    )
    
    # Silence third-party libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("huggingface_hub").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    
    return logging.getLogger("ARIA")