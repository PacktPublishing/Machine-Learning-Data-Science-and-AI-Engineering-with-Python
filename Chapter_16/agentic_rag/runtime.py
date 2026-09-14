from collections.abc import Generator
from typing import Any

from langgraph.graph.state import CompiledStateGraph

from agentic_rag.config import (
    GRAPH_RECURSION_LIMIT,
)
from agentic_rag.errors import (
    PlatformGuardrailError,
)
from agentic_rag.schemas import RagResponse
from agentic_rag.state import AgenticRagState


def create_initial_state(
    question: str,
) -> AgenticRagState:
    return {
        "original_question": question,
        "query_validation": None,
        "query_analysis": None,
        "retrieval_query": question,
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


def create_platform_guardrail_response(
    exception: PlatformGuardrailError,
) -> RagResponse:
    return RagResponse(
        answer=(
            "This request could not be processed because "
            "the AI platform blocked it under its content "
            "management policy."
        ),
        grounded=False,
        sources=[],
        missing_information=[
            (
                f"{exception.provider} blocked the prompt "
                f"before model inference."
            )
        ],
    )


def stream_question(
    graph: CompiledStateGraph,
    question: str,
) -> Generator[
    tuple[str, dict[str, Any]],
    None,
    None,
]:
    try:
        for event in graph.stream(
            create_initial_state(question),
            config={
                "recursion_limit": (
                    GRAPH_RECURSION_LIMIT
                ),
            },
            stream_mode="updates",
        ):
            for node_name, update in event.items():
                yield node_name, update

    except PlatformGuardrailError as exception:
        final_response = (
            create_platform_guardrail_response(
                exception
            )
        )

        yield (
            "platform_guardrail",
            {
                "status": "platform_blocked",
                "events": [
                    (
                        f"{exception.provider} blocked "
                        "the prompt before inference."
                    ),
                    (
                        "The graph execution was stopped "
                        "without generating an answer."
                    ),
                ],
                "final_response": final_response,
            },
        )