import logging
from typing import Any

import streamlit as st
from dotenv import load_dotenv

from agentic_rag.graph import build_graph
from agentic_rag.runtime import stream_question
from agentic_rag.schemas import RagResponse
from agentic_rag.streamlit_ui import (
    ExecutionView,
    configure_page,
    render_final_response,
    render_progress,
    render_sidebar,
    render_update,
)
from agentic_rag.vector_store import load_vector_store
from shared.models import create_model

load_dotenv(override=False)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("agentic-rag-streamlit")


@st.cache_resource(show_spinner="Loading banking knowledge base...")
def create_graph() -> Any:
    vector_store = load_vector_store()
    model = create_model()
    return build_graph(
        vector_store=vector_store,
        model=model,
    )


def initialize_session() -> None:
    if "history" not in st.session_state:
        st.session_state.history = []


def run_question(graph: Any, question: str) -> RagResponse:
    view = ExecutionView()
    progress_placeholder = st.empty()
    updates_container = st.container()

    render_progress(view, progress_placeholder)

    for node_name, update in stream_question(graph, question):
        view.add(node_name, update)
        render_progress(view, progress_placeholder)
        render_update(node_name, update, updates_container)

    if view.final_response is None:
        raise RuntimeError("Graph completed without a final response.")

    return view.final_response


def main() -> None:
    configure_page()
    render_sidebar()
    initialize_session()

    try:
        graph = create_graph()
    except Exception as exc:
        logger.exception("Application initialization failed")
        st.error(
            "The application could not initialize. Check the model, "
            "embedding, Chroma, and Azure credential configuration."
        )
        with st.expander("Technical details"):
            st.exception(exc)
        st.stop()

    for item in st.session_state.history:
        with st.chat_message("user"):
            st.write(item["question"])
        with st.chat_message("assistant"):
            render_final_response(item["response"])

    question = st.chat_input(
        "Ask a question about banking products or procedures"
    )

    if not question:
        return

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        try:
            response = run_question(graph, question)
            render_final_response(response)
            st.session_state.history.append(
                {
                    "question": question,
                    "response": response,
                }
            )
        except Exception as exc:
            logger.exception("Graph execution failed")
            st.error("Agentic RAG execution failed.")
            with st.expander("Technical details"):
                st.exception(exc)


if __name__ == "__main__":
    main()
