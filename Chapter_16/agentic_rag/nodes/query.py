from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.prompts import (
    QUERY_ANALYSIS_SYSTEM_PROMPT,
    QUERY_REWRITE_SYSTEM_PROMPT,
    QUERY_VALIDATION_SYSTEM_PROMPT,
)
from agentic_rag.schemas import (
    QueryAnalysis,
    QueryValidation,
    RewrittenQuery,
)
from agentic_rag.state import AgenticRagState
from agentic_rag.errors import invoke_model_safely

def create_validate_query_node(
    model: BaseChatModel,
) -> Callable[[AgenticRagState], dict]:
    validator = model.with_structured_output(QueryValidation)

    def validate_query(state: AgenticRagState) -> dict:
        result = invoke_model_safely(
            validator,
            [
                SystemMessage(content=QUERY_VALIDATION_SYSTEM_PROMPT),
                HumanMessage(content=state["original_question"]),
            ]
        )

        if not isinstance(result, QueryValidation):
            raise RuntimeError("Query validator returned an invalid result.")

        return {
            "query_validation": result,
            "status": "query_validated" if result.valid else "query_rejected",
            "events": [f"Query validation: {result.reason}"],
        }

    return validate_query


def create_analyze_query_node(
    model: BaseChatModel,
) -> Callable[[AgenticRagState], dict]:
    analyzer = model.with_structured_output(QueryAnalysis)

    def analyze_query(state: AgenticRagState) -> dict:
        result = invoke_model_safely(
            analyzer,
            [
                SystemMessage(content=QUERY_ANALYSIS_SYSTEM_PROMPT),
                HumanMessage(content=state["original_question"]),
            ]
        )

        if not isinstance(result, QueryAnalysis):
            raise RuntimeError("Query analyzer returned an invalid result.")

        return {
            "query_analysis": result,
            "retrieval_query": result.retrieval_query,
            "status": "query_analyzed",
            "events": [
                f"Intent: {result.intent}",
                f"Retrieval query: {result.retrieval_query}",
            ],
        }

    return analyze_query


def create_rewrite_query_node(
    model: BaseChatModel,
) -> Callable[[AgenticRagState], dict]:
    rewriter = model.with_structured_output(RewrittenQuery)

    def rewrite_query(state: AgenticRagState) -> dict:
        feedback: list[str] = []

        retrieval_evaluation = state.get("retrieval_evaluation")
        if retrieval_evaluation is not None:
            feedback.append(retrieval_evaluation.reason)

        answer_evaluation = state.get("answer_evaluation")
        if answer_evaluation is not None:
            feedback.append(answer_evaluation.reason)

        prompt = f"""
Original question:
{state['original_question']}

Previous retrieval query:
{state['retrieval_query']}

Failure feedback:
{'; '.join(feedback) or 'No previous feedback.'}
""".strip()

        result = invoke_model_safely(
            rewriter,
            [
                SystemMessage(content=QUERY_REWRITE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        if not isinstance(result, RewrittenQuery):
            raise RuntimeError("Query rewriter returned an invalid result.")

        attempt = state["rewrite_attempts"] + 1

        return {
            "retrieval_query": result.query,
            "rewrite_reason": result.reason,
            "rewrite_attempts": attempt,
            "retrieved_documents": [],
            "retrieval_scores": [],
            "retrieval_evaluation": None,
            "draft_response": None,
            "answer_evaluation": None,
            "status": "query_rewritten",
            "events": [
                f"Rewrite attempt {attempt}: {result.query}",
                f"Rewrite reason: {result.reason}",
            ],
        }

    return rewrite_query
