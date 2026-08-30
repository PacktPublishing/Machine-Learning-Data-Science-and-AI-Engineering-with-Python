import os

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings


def create_embeddings() -> Embeddings:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set."
        )

    return OpenAIEmbeddings(
        model=os.getenv(
            "OPENAI_EMBEDDING_MODEL",
            "text-embedding-3-small",
        ),
        api_key=api_key,
    )
