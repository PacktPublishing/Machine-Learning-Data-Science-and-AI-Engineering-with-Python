from langchain_chroma import Chroma
from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from agentic_rag.nodes.answer import (
    abstain_node,
    create_generate_node,
    create_reflect_node,
    finalize_node,
    reject_query_node,
)
from agentic_rag.nodes.query import (
    create_analyze_query_node,
    create_rewrite_query_node,
    create_validate_query_node,
)
from agentic_rag.nodes.retrieval import (
    create_grade_retrieval_node,
    create_retrieve_node,
)
from agentic_rag.routing import (
    route_after_query_validation,
    route_after_reflection,
    route_after_retrieval,
)
from agentic_rag.state import AgenticRagState


def build_graph(
    vector_store: Chroma,
    model: BaseChatModel,
) -> CompiledStateGraph:
    builder = StateGraph(AgenticRagState)

    builder.add_node("validate_query", create_validate_query_node(model))
    builder.add_node("analyze_query", create_analyze_query_node(model))
    builder.add_node("rewrite_query", create_rewrite_query_node(model))
    builder.add_node("retrieve", create_retrieve_node(vector_store))
    builder.add_node("grade_retrieval", create_grade_retrieval_node(model))
    builder.add_node("generate", create_generate_node(model))
    builder.add_node("reflect", create_reflect_node(model))
    builder.add_node("finalize", finalize_node)
    builder.add_node("reject_query", reject_query_node)
    builder.add_node("abstain", abstain_node)

    builder.add_edge(START, "validate_query")
    builder.add_conditional_edges(
        "validate_query",
        route_after_query_validation,
    )

    builder.add_edge("analyze_query", "retrieve")
    builder.add_edge("rewrite_query", "retrieve")
    builder.add_edge("retrieve", "grade_retrieval")

    builder.add_conditional_edges(
        "grade_retrieval",
        route_after_retrieval,
    )

    builder.add_edge("generate", "reflect")
    builder.add_conditional_edges(
        "reflect",
        route_after_reflection,
    )

    builder.add_edge("finalize", END)
    builder.add_edge("reject_query", END)
    builder.add_edge("abstain", END)

    return builder.compile()
