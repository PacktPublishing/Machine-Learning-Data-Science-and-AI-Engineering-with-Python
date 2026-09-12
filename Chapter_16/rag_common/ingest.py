import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from rag_common.config import (
    CHROMA_DIRECTORY,
    COLLECTION_NAME,
    DATA_DIRECTORY,
)
from shared.embeddings import create_embeddings


load_dotenv(override=True)


def load_documents(directory: Path) -> list[Document]:
    documents: list[Document] = []

    for file_path in sorted(directory.glob("*.pdf")):
        reader = PdfReader(str(file_path))

        for page_number, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()

            if not text:
                continue

            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": file_path.name,
                        "path": str(file_path),
                        "page": page_number,
                    },
                )
            )

    if not documents:
        raise RuntimeError(
            f"No readable PDF pages found in {directory}."
        )

    return documents


def split_documents(documents: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=350,
        chunk_overlap=60,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = index

    return chunks


def create_chunk_ids(chunks: list[Document]) -> list[str]:
    return [
        (
            f"{chunk.metadata['source']}"
            f":{chunk.metadata['page']}"
            f":{chunk.metadata['chunk_id']}"
        )
        for chunk in chunks
    ]


def ingest(reset: bool = False) -> None:
    if reset and CHROMA_DIRECTORY.exists():
        shutil.rmtree(CHROMA_DIRECTORY)

    documents = load_documents(DATA_DIRECTORY)
    chunks = split_documents(documents)
    chunk_ids = create_chunk_ids(chunks)

    vector_store = Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(CHROMA_DIRECTORY),
        collection_metadata={"hnsw:space": "cosine"},
    )

    vector_store.add_documents(
        documents=chunks,
        ids=chunk_ids,
    )

    print(f"Loaded {len(documents)} PDF pages.")
    print(f"Indexed {len(chunks)} chunks.")
    print(f"Index: {CHROMA_DIRECTORY}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the existing Chroma index before ingestion.",
    )
    args = parser.parse_args()
    ingest(reset=args.reset)
