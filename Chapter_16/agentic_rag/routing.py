from typing import Literal

from agentic_rag.config import MAX_REWRITE_ATTEMPTS
from agentic_rag.state import AgenticRagState


def route_after_query_validation(
    state: AgenticRagState,
) -> Literal["analyze_query", "reject_query"]:
    validation = state["query_validation"]
    return "analyze_query" if validation and validation.valid else "reject_query"


def route_after_retrieval(
    state: AgenticRagState,
) -> Literal["generate", "rewrite_query", "abstain"]:
    evaluation = state["retrieval_evaluation"]

    if evaluation is not None and evaluation.sufficient:
        return "generate"

    if state["rewrite_attempts"] < MAX_REWRITE_ATTEMPTS:
        return "rewrite_query"

    return "abstain"


def route_after_reflection(
    state: AgenticRagState,
) -> Literal["finalize", "rewrite_query", "abstain"]:
    evaluation = state["answer_evaluation"]

    if evaluation is None:
        return "abstain"

    if evaluation.acceptable and evaluation.recommended_action == "accept":
        return "finalize"

    if (
        evaluation.recommended_action == "rewrite"
        and state["rewrite_attempts"] < MAX_REWRITE_ATTEMPTS
    ):
        return "rewrite_query"

    return "abstain"
