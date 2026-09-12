from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import streamlit as st
from langchain_core.documents import Document

from agentic_rag.schemas import RagResponse


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
    "platform_guardrail": "Azure platform guardrail",
}

NODE_ORDER = [
    "validate_query",
    "analyze_query",
    "rewrite_query",
    "retrieve",
    "grade_retrieval",
    "generate",
    "reflect",
    "finalize",
    "reject_query",
    "abstain",
    "platform_guardrail",
]


@dataclass
class ExecutionView:
    updates: list[tuple[str, dict[str, Any]]] = field(
        default_factory=list
    )
    final_response: RagResponse | None = None

    def add(
        self,
        node_name: str,
        update: dict[str, Any],
    ) -> None:
        self.updates.append((node_name, update))

        candidate = update.get("final_response")
        if isinstance(candidate, RagResponse):
            self.final_response = candidate


def configure_page() -> None:
    st.set_page_config(
        page_title="Agentic Banking RAG",
        page_icon="🏦",
        layout="wide",
    )

    st.title("🏦 Agentic Banking RAG")
    st.caption(
        "Query validation → analysis → retrieval → evidence grading → "
        "generation → reflection"
    )


def render_sidebar() -> None:
    with st.sidebar:
        st.header("Graph behavior")
        st.markdown(
            """
- Invalid or unsupported questions stop before retrieval.
- Weak evidence triggers a bounded query rewrite.
- Generation is skipped when evidence remains insufficient.
- Reflection either accepts, retries, or abstains.
- Azure content filtering is shown as a controlled graph outcome.
            """
        )

        st.divider()

        st.caption(
            "The UI displays structured decisions and state updates, "
            "not private model reasoning."
        )


def render_progress(
    view: ExecutionView,
    placeholder: Any,
) -> None:
    completed = [name for name, _ in view.updates]
    current = completed[-1] if completed else None

    with placeholder.container():
        st.subheader("Execution path")
        columns = st.columns(4)

        display_nodes = [
            "validate_query",
            "analyze_query",
            "retrieve",
            "grade_retrieval",
            "generate",
            "reflect",
            "finalize",
            "abstain",
            "platform_guardrail",
        ]

        for index, node_name in enumerate(display_nodes):
            if node_name == current:
                icon = "🔄"
            elif node_name in completed:
                icon = (
                    "⚠️"
                    if node_name == "platform_guardrail"
                    else "✅"
                )
            else:
                icon = "⚪"

            columns[index % 4].markdown(
                f"{icon} **{NODE_TITLES.get(node_name, node_name)}**"
            )


def _render_documents(
    documents: list[Document],
    scores: list[float],
) -> None:
    rows: list[dict[str, Any]] = []

    for rank, (document, score) in enumerate(
        zip(documents, scores, strict=False),
        start=1,
    ):
        rows.append(
            {
                "rank": rank,
                "source": document.metadata.get(
                    "source",
                    "unknown",
                ),
                "chunk_id": document.metadata.get(
                    "chunk_id",
                    "unknown",
                ),
                "relevance": round(float(score), 4),
                "preview": document.page_content[:300],
            }
        )

    if rows:
        st.dataframe(
            rows,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No documents were returned by the retriever.")


def _render_events(events: list[Any]) -> None:
    for event in events:
        st.write(f"• {event}")


def _render_platform_guardrail(
    update: dict[str, Any],
) -> None:
    st.error(
        "Azure AI blocked the prompt before the model executed."
    )

    _render_events(
        list(update.get("events", []))
    )

    st.caption(
        "This is a platform-level safety decision. "
        "The graph stopped without generating or grounding an answer."
    )


def render_update(
    node_name: str,
    update: dict[str, Any],
    container: Any,
) -> None:
    title = NODE_TITLES.get(
        node_name,
        node_name.replace("_", " ").title(),
    )
    status = str(update.get("status", "updated"))
    events = list(update.get("events", []))

    with container:
        with st.expander(
            f"{title} · {status}",
            expanded=node_name in {
                "validate_query",
                "retrieve",
                "grade_retrieval",
                "abstain",
                "finalize",
                "platform_guardrail",
            },
        ):
            if node_name == "platform_guardrail":
                _render_platform_guardrail(update)
                return

            _render_events(events)

            if node_name == "analyze_query":
                analysis = update.get("query_analysis")
                if analysis is not None:
                    st.json(analysis.model_dump())

            elif node_name == "rewrite_query":
                st.code(
                    update.get("retrieval_query", ""),
                    language=None,
                )

                rewrite_reason = update.get("rewrite_reason")
                if rewrite_reason:
                    st.caption(str(rewrite_reason))

            elif node_name == "retrieve":
                _render_documents(
                    update.get("retrieved_documents", []),
                    update.get("retrieval_scores", []),
                )

            elif node_name == "grade_retrieval":
                evaluation = update.get(
                    "retrieval_evaluation"
                )
                if evaluation is not None:
                    st.json(evaluation.model_dump())

            elif node_name == "generate":
                draft = update.get("draft_response")
                if isinstance(draft, RagResponse):
                    st.info(draft.answer)

            elif node_name == "reflect":
                evaluation = update.get(
                    "answer_evaluation"
                )
                if evaluation is not None:
                    st.json(evaluation.model_dump())

            elif node_name in {
                "reject_query",
                "abstain",
            }:
                response = update.get("final_response")
                if isinstance(response, RagResponse):
                    st.warning(response.answer)


def render_final_response(
    response: RagResponse,
) -> None:
    st.subheader("Final answer")

    if response.grounded:
        st.success(response.answer)
    else:
        st.warning(response.answer)

    left, right = st.columns(2)
    left.metric(
        "Grounded",
        "Yes" if response.grounded else "No",
    )
    right.metric(
        "Sources",
        len(response.sources),
    )

    if response.sources:
        st.markdown("#### Sources")
        st.dataframe(
            [
                source.model_dump()
                for source in response.sources
            ],
            use_container_width=True,
            hide_index=True,
        )

    if response.missing_information:
        st.markdown("#### Missing information")
        for item in response.missing_information:
            st.write(f"- {item}")
