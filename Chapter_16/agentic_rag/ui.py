from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from agentic_rag.schemas import RagResponse

console = Console()


NODE_TITLES = {
    "validate_query": "Query guardrail",
    "analyze_query": "Query analysis",
    "rewrite_query": "Query rewrite",
    "retrieve": "Retriever",
    "grade_retrieval": "Evidence guardrail",
    "generate": "Answer generation",
    "reflect": "Reflection",
    "finalize": "Finalization",
    "reject_query": "Query rejected",
    "abstain": "Abstention",
}


def render_header() -> None:
    console.print(
        Panel(
            "[bold]Agentic Banking RAG[/]\n"
            "validate → analyze → retrieve → grade → generate → reflect",
            border_style="blue",
        )
    )


def render_node_update(node_name: str, update: dict[str, Any]) -> None:
    title = NODE_TITLES.get(node_name, node_name)
    status = update.get("status", "updated")
    events = update.get("events", [])

    body_lines = [f"[bold]Status:[/] {status}"]
    body_lines.extend(f"• {event}" for event in events)

    if node_name == "retrieve":
        documents = update.get("retrieved_documents", [])
        scores = update.get("retrieval_scores", [])

        for index, (document, score) in enumerate(
            zip(documents, scores, strict=False),
            start=1,
        ):
            source = document.metadata.get("source", "unknown")
            chunk_id = document.metadata.get("chunk_id", "unknown")
            body_lines.append(
                f"{index}. {source} / chunk {chunk_id} / score {score:.3f}"
            )

    console.print(
        Panel(
            "\n".join(body_lines),
            title=title,
            border_style="cyan",
        )
    )


def render_final_response(response: RagResponse) -> None:
    console.print(
        Panel(
            response.answer,
            title="[bold green]Final answer[/]",
            border_style="green" if response.grounded else "yellow",
        )
    )

    console.print(
        "[bold green]Grounded: Yes[/]"
        if response.grounded
        else "[bold yellow]Grounded: No[/]"
    )

    if response.sources:
        table = Table(title="Sources", show_lines=True)
        table.add_column("Source")
        table.add_column("Chunk ID", justify="right")

        for source in response.sources:
            table.add_row(source.source, str(source.chunk_id))

        console.print(table)

    if response.missing_information:
        console.print(
            Panel(
                "\n".join(
                    f"• {item}" for item in response.missing_information
                ),
                title="[bold yellow]Missing information[/]",
                border_style="yellow",
            )
        )
