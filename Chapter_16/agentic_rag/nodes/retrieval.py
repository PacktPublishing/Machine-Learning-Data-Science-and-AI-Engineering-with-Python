from collections.abc import Callable

from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.config import (
    MIN_ACCEPTABLE_CHUNKS,
    MIN_BEST_SCORE,
    MIN_CHUNK_SCORE,
    TOP_K,
)
from agentic_rag.formatting import format_retrieved_context
from agentic_rag.prompts import RETRIEVAL_GRADING_SYSTEM_PROMPT
from agentic_rag.schemas import RetrievalEvaluation
from agentic_rag.state import AgenticRagState
from agentic_rag.errors import invoke_model_safely


def create_retrieve_node(
    vector_store: Chroma,
) -> Callable[[AgenticRagState], dict]:
    def retrieve(state: AgenticRagState) -> dict:
        query = state["retrieval_query"]

        results = vector_store.similarity_search_with_relevance_scores(
            query=query,
            k=TOP_K,
        )

        documents = [document for document, _ in results]
        scores = [float(score) for _, score in results]

        score_text = ", ".join(f"{score:.3f}" for score in scores)

        return {
            "retrieved_documents": documents,
            "retrieval_scores": scores,
            "status": "documents_retrieved",
            "events": [
                f"Retrieved {len(documents)} chunks for: {query}",
                f"Relevance scores: {score_text or 'none'}",
            ],
        }

    return retrieve


def _deterministic_retrieval_check(
    scores: list[float],
) -> tuple[bool, str]:
    if not scores:
        return False, "No documents were retrieved."

    best_score = max(scores)
    qualifying_count = sum(score >= MIN_CHUNK_SCORE for score in scores)

    if best_score < MIN_BEST_SCORE:
        return False, (
            f"Best relevance score {best_score:.3f} is below "
            f"the threshold {MIN_BEST_SCORE:.3f}."
        )

    if qualifying_count < MIN_ACCEPTABLE_CHUNKS:
        return False, (
            f"Only {qualifying_count} chunks passed the chunk threshold; "
            f"{MIN_ACCEPTABLE_CHUNKS} required."
        )

    return True, "Deterministic relevance thresholds passed."


def create_grade_retrieval_node(
    model: BaseChatModel,
) -> Callable[[AgenticRagState], dict]:
    grader = model.with_structured_output(RetrievalEvaluation)

    def grade_retrieval(state: AgenticRagState) -> dict:
        documents = state["retrieved_documents"]
        scores = state["retrieval_scores"]
        passed, reason = _deterministic_retrieval_check(scores)

        best_score = max(scores) if scores else None
        average_score = sum(scores) / len(scores) if scores else None

        if not passed:
            evaluation = RetrievalEvaluation(
                sufficient=False,
                reason=reason,
                best_score=best_score,
                average_score=average_score,
                relevant_chunk_ids=[],
            )
        else:
            context = format_retrieved_context(documents)
            prompt = f"""
Question:
{state['original_question']}

Retrieved context:
{context}

The deterministic score gate passed.
Decide whether the excerpts contain enough direct evidence to answer.
""".strip()

            model_result = invoke_model_safely(
                grader,
                [
                    SystemMessage(content=RETRIEVAL_GRADING_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )

            if not isinstance(model_result, RetrievalEvaluation):
                raise RuntimeError("Retrieval grader returned an invalid result.")

            evaluation = model_result.model_copy(
                update={
                    "best_score": best_score,
                    "average_score": average_score,
                }
            )

        return {
            "retrieval_evaluation": evaluation,
            "status": (
                "retrieval_accepted"
                if evaluation.sufficient
                else "retrieval_rejected"
            ),
            "events": [f"Retrieval evaluation: {evaluation.reason}"],
        }

    return grade_retrieval
