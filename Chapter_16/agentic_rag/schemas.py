from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class SourceReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(
        description="The source filename supporting the answer."
    )
    chunk_id: int = Field(
        description="The identifier of the supporting chunk."
    )


class RagResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str = Field(
        description="A concise answer based only on retrieved context."
    )
    grounded: bool = Field(
        description="Whether the answer is directly supported by context."
    )
    sources: list[SourceReference] = Field(
        description="Only chunks that directly support the answer."
    )
    missing_information: list[str] = Field(
        description="Information needed but absent from context."
    )


class QueryValidation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    reason: str
    category: Literal[
        "banking",
        "ambiguous",
        "nonsense",
        "unsafe",
        "unsupported",
    ]


class QueryAnalysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    intent: str
    entities: list[str]
    required_information: list[str]
    retrieval_query: str


class RewrittenQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str
    reason: str


class RetrievalEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sufficient: bool
    reason: str
    best_score: float | None
    average_score: float | None
    relevant_chunk_ids: list[int]


class AnswerEvaluation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    acceptable: bool
    grounded: bool
    complete: bool
    reason: str
    recommended_action: Literal[
        "accept",
        "rewrite",
        "abstain",
    ]
