from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from rag_common.config import (
    CHROMA_DIRECTORY,
    COLLECTION_NAME,
)
from shared.embeddings import create_embeddings
from shared.models import create_model


load_dotenv(override=True)


class SourceReference(BaseModel):
    source: str = Field(description="Supporting PDF filename")
    page: int = Field(description="Supporting PDF page number")
    chunk_id: int = Field(description="Supporting chunk identifier")


class RagResponse(BaseModel):
    answer: str
    grounded: bool
    sources: list[SourceReference]
    missing_information: list[str]


RAG_SYSTEM_PROMPT = """
You answer questions using only the retrieved PDF context.

Rules:
- Do not use outside knowledge.
- Do not invent information.
- If the context is insufficient, say so clearly.
- Set grounded to true only when the answer is supported by the context.
- Include only sources that support the answer.
- Include missing information when relevant.
- Keep the answer concise.
""".strip()


def load_vector_store() -> Chroma:
    if not CHROMA_DIRECTORY.exists():
        raise RuntimeError(
            "The Chroma index does not exist. "
            "Run: python -m rag_common.ingest --reset"
        )

    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=create_embeddings(),
        persist_directory=str(CHROMA_DIRECTORY),
    )


def retrieve_documents(
    vector_store: Chroma,
    question: str,
    number_of_results: int = 3,
) -> list[Document]:
    return vector_store.similarity_search(
        query=question,
        k=number_of_results,
    )


def format_retrieved_context(
    documents: list[Document],
) -> str:
    sections: list[str] = []

    for rank, document in enumerate(documents, start=1):
        sections.append(
            "\n".join(
                [
                    f"[Retrieved chunk {rank}]",
                    f"Source: {document.metadata.get('source', 'unknown')}",
                    f"Page: {document.metadata.get('page', -1)}",
                    f"Chunk ID: {document.metadata.get('chunk_id', -1)}",
                    "Content:",
                    document.page_content,
                ]
            )
        )

    return "\n\n---\n\n".join(sections)


def create_rag_model():
    return create_model().with_structured_output(RagResponse)


def generate_answer(
    model,
    question: str,
    retrieved_documents: list[Document],
) -> RagResponse:
    context = format_retrieved_context(retrieved_documents)
    prompt = f"""
Retrieved PDF context:

{context}

Question:

{question}
""".strip()

    response = model.invoke(
        [
            SystemMessage(content=RAG_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]
    )

    if not isinstance(response, RagResponse):
        raise RuntimeError("The model returned an invalid RagResponse.")

    return response


def answer_question(
    vector_store: Chroma,
    rag_model,
    question: str,
    number_of_results: int = 3,
) -> tuple[RagResponse, list[Document]]:
    documents = retrieve_documents(
        vector_store=vector_store,
        question=question,
        number_of_results=number_of_results,
    )

    response = generate_answer(
        model=rag_model,
        question=question,
        retrieved_documents=documents,
    )

    return response, documents


def main() -> None:
    vector_store = load_vector_store()
    rag_model = create_rag_model()

    print("PDF question answering. Type 'exit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        response, _ = answer_question(
            vector_store=vector_store,
            rag_model=rag_model,
            question=question,
        )

        print(f"\nAnswer:\n{response.answer}")
        print(f"\nGrounded: {'yes' if response.grounded else 'no'}")

        if response.sources:
            print("\nSources:")
            for source in response.sources:
                print(
                    f"- {source.source}, page {source.page}, "
                    f"chunk {source.chunk_id}"
                )

        if response.missing_information:
            print("\nMissing information:")
            for item in response.missing_information:
                print(f"- {item}")


if __name__ == "__main__":
    main()
