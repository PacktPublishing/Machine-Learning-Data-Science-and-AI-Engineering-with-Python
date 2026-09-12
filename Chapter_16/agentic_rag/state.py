import operator
from typing import Annotated, TypedDict

from langchain_core.documents import Document

from agentic_rag.schemas import (
    AnswerEvaluation,
    QueryAnalysis,
    QueryValidation,
    RagResponse,
    RetrievalEvaluation,
)


class AgenticRagState(TypedDict):
    original_question: str

    query_validation: QueryValidation | None
    query_analysis: QueryAnalysis | None

    retrieval_query: str
    rewrite_reason: str | None
    rewrite_attempts: int

    retrieved_documents: list[Document]
    retrieval_scores: list[float]
    retrieval_evaluation: RetrievalEvaluation | None

    draft_response: RagResponse | None
    answer_evaluation: AnswerEvaluation | None
    final_response: RagResponse | None

    status: str
    events: Annotated[list[str], operator.add]
