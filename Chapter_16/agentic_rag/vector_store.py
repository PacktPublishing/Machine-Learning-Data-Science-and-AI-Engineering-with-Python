import logging

from langchain_chroma import Chroma

from rag_common.config import CHROMA_DIRECTORY, COLLECTION_NAME
from shared.embeddings import create_embeddings

logger = logging.getLogger("agentic-rag")


def load_vector_store() -> Chroma:
    if not CHROMA_DIRECTORY.exists():
        raise RuntimeError(
            "The vector index does not exist. Run ingestion first: "
            "uv run python -m rag_common.ingest"
        )

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(CHROMA_DIRECTORY),
        collection_metadata={
            "hnsw:space": "cosine",
        },
    )

    logger.info("Loaded vector index from %s", CHROMA_DIRECTORY)
    return vector_store
