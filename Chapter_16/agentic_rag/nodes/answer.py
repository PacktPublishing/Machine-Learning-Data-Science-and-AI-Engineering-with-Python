from collections.abc import Callable

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from agentic_rag.formatting import format_retrieved_context
from agentic_rag.prompts import RAG_SYSTEM_PROMPT, REFLECTION_SYSTEM_PROMPT
from agentic_rag.schemas import AnswerEvaluation, RagResponse
from agentic_rag.state import AgenticRagState
from agentic_rag.errors import invoke_model_safely

def create_generate_node(
    model: BaseChatModel,
) -> Callable[[AgenticRagState], dict]:
    generator = model.with_structured_output(RagResponse)

    def generate(state: AgenticRagState) -> dict:
        context = format_retrieved_context(state["retrieved_documents"])
        prompt = f"""
RETRIEVED CONTEXT

{context}

CUSTOMER QUESTION

{state['original_question']}
""".strip()

        response = invoke_model_safely(
            generator,
            [
                SystemMessage(content=RAG_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
        )

        if not isinstance(response, RagResponse):
            raise RuntimeError("Generator returned an invalid RagResponse.")

        return {
            "draft_response": response,
            "status": "draft_generated",
            "events": [
                f"Draft generated; grounded={response.grounded}."
            ],
        }

    return generate


def create_reflect_node(
    model: BaseChatModel,
) -> Callable[[AgenticRagState], dict]:
    reflector = model.with_structured_output(AnswerEvaluation)

    def reflect(state: AgenticRagState) -> dict:
        draft = state["draft_response"]

        if draft is None:
            evaluation = AnswerEvaluation(
                acceptable=False,
                grounded=False,
                complete=False,
                reason="Generation produced no draft response.",
                recommended_action="abstain",
            )
        else:
            context = format_retrieved_context(state["retrieved_documents"])
            prompt = f"""
Original question:
{state['original_question']}

Draft response:
{draft.model_dump_json(indent=2)}

Retrieved evidence:
{context}
""".strip()

            model_result = invoke_model_safely(
                reflector,
                [
                    SystemMessage(content=REFLECTION_SYSTEM_PROMPT),
                    HumanMessage(content=prompt),
                ]
            )

            if not isinstance(model_result, AnswerEvaluation):
                raise RuntimeError("Reflector returned an invalid result.")

            evaluation = model_result

        return {
            "answer_evaluation": evaluation,
            "status": "answer_reflected",
            "events": [f"Answer evaluation: {evaluation.reason}"],
        }

    return reflect


def finalize_node(state: AgenticRagState) -> dict:
    draft = state["draft_response"]
    if draft is None:
        raise RuntimeError("Cannot finalize without a draft response.")

    return {
        "final_response": draft,
        "status": "completed",
        "events": ["Answer accepted and finalized."],
    }


def reject_query_node(state: AgenticRagState) -> dict:
    validation = state["query_validation"]
    reason = validation.reason if validation is not None else "Invalid query."

    response = RagResponse(
        answer=(
            "I cannot process that question as a supported banking "
            "knowledge request."
        ),
        grounded=False,
        sources=[],
        missing_information=[reason],
    )

    return {
        "final_response": response,
        "status": "query_rejected",
        "events": [f"Query rejected: {reason}"],
    }


def abstain_node(state: AgenticRagState) -> dict:
    answer_evaluation = state.get("answer_evaluation")
    retrieval_evaluation = state.get("retrieval_evaluation")

    if answer_evaluation is not None:
        reason = answer_evaluation.reason
    elif retrieval_evaluation is not None:
        reason = retrieval_evaluation.reason
    else:
        reason = "The available knowledge is insufficient."

    response = RagResponse(
        answer=(
            "I could not find sufficiently reliable information in the "
            "banking knowledge base to answer this question."
        ),
        grounded=False,
        sources=[],
        missing_information=[reason],
    )

    return {
        "final_response": response,
        "status": "abstained",
        "events": [f"Answer skipped: {reason}"],
    }
