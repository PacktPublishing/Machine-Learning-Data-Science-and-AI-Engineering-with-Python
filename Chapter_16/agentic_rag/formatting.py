from langchain_core.documents import Document


def format_retrieved_context(documents: list[Document]) -> str:
    sections: list[str] = []

    for rank, document in enumerate(documents, start=1):
        source = document.metadata.get("source", "unknown")
        chunk_id = document.metadata.get("chunk_id", -1)

        sections.append(
            "\n".join(
                [
                    f"[Retrieved chunk {rank}]",
                    f"Source: {source}",
                    f"Chunk ID: {chunk_id}",
                    "Content:",
                    document.page_content,
                ]
            )
        )

    return "\n\n---\n\n".join(sections)
