from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Database Directory
CHROMA_DB_PATH = PROJECT_ROOT / "data" / "chroma_db"

# Collection Name
COLLECTION_NAME = "aria_knowledge_base"