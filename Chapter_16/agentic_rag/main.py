import logging
from typing import Any

from dotenv import load_dotenv
from rich.logging import RichHandler

from agentic_rag.config import GRAPH_RECURSION_LIMIT
from agentic_rag.graph import build_graph
from agentic_rag.schemas import RagResponse
from agentic_rag.state import AgenticRagState
from agentic_rag.ui import (
    console,
    render_final_response,
    render_header,
    render_node_update,
)
from agentic_rag.vector_store import load_vector_store
from shared.models import create_model

load_dotenv(override=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[
        RichHandler(
            rich_tracebacks=True,
            show_time=True,
            show_path=False,
        )
    ],
)

logger = logging.getLogger("agentic-rag")


def create_initial_state(question: str) -> AgenticRagState:
    return {
        "original_question": question,
        "query_validation": None,
        "query_analysis": None,
        "retrieval_query": "",
        "rewrite_reason": None,
        "rewrite_attempts": 0,
        "retrieved_documents": [],
        "retrieval_scores": [],
        "retrieval_evaluation": None,
        "draft_response": None,
        "answer_evaluation": None,
        "final_response": None,
        "status": "started",
        "events": [],
    }


def run_question(graph: Any, question: str) -> RagResponse:
    final_response: RagResponse | None = None

    for event in graph.stream(
        create_initial_state(question),
        config={"recursion_limit": GRAPH_RECURSION_LIMIT},
        stream_mode="updates",
    ):
        for node_name, update in event.items():
            render_node_update(node_name, update)

            candidate = update.get("final_response")
            if isinstance(candidate, RagResponse):
                final_response = candidate

    if final_response is None:
        raise RuntimeError("Graph completed without a final response.")

    return final_response


def main() -> None:
    render_header()

    logger.info("Loading persistent vector store.")
    vector_store = load_vector_store()

    logger.info(
        "Chroma collection configuration: %s",
        vector_store._collection.configuration,
    )

    logger.info("Creating raw chat model.")
    model = create_model()
    logger.info("Model type: %s", type(model).__name__)

    logger.info("Building and compiling graph.")
    graph = build_graph(
        vector_store=vector_store,
        model=model,
    )

    logger.info("Graph initialized successfully.")
    console.print("\n[dim]Type 'exit' to stop.[/]")

    while True:
        question = console.input("\n[bold cyan]Question:[/] ").strip()

        if question.lower() in {"exit", "quit"}:
            break

        if not question:
            continue

        try:
            response = run_question(graph, question)
            render_final_response(response)
        except Exception:
            logger.exception("Agentic RAG execution failed.")
            console.print("[bold red]Agentic RAG execution failed.[/]")


if __name__ == "__main__":
    main()
