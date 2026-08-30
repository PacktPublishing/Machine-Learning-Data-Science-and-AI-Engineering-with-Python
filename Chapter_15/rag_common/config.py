from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIRECTORY = PROJECT_ROOT / "rag_common" / "data"
CHROMA_DIRECTORY = PROJECT_ROOT / "storage" / "chroma_db"
COLLECTION_NAME = "pdf-knowledge"

