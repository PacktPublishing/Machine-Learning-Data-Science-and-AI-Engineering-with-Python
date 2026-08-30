from dotenv import load_dotenv
from deepeval import evaluate
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
)
from deepeval.test_case import LLMTestCase

from evaluation.test_cases import EVALUATION_CASES
from simple_rag.main import (
    answer_question,
    create_rag_model,
    load_vector_store,
)


load_dotenv(override=True)


def build_test_cases() -> list[LLMTestCase]:
    vector_store = load_vector_store()
    rag_model = create_rag_model()
    test_cases: list[LLMTestCase] = []

    for item in EVALUATION_CASES:
        response, documents = answer_question(
            vector_store=vector_store,
            rag_model=rag_model,
            question=item["question"],
        )

        test_cases.append(
            LLMTestCase(
                input=item["question"],
                actual_output=response.answer,
                expected_output=item["expected_output"],
                retrieval_context=[
                    document.page_content
                    for document in documents
                ],
            )
        )

    return test_cases


def main() -> None:
    metrics = [
        ContextualPrecisionMetric(
            threshold=0.7,
            model="gpt-4.1",
            include_reason=True,
        ),
        ContextualRecallMetric(
            threshold=0.7,
            model="gpt-4.1",
            include_reason=True,
        ),
        AnswerRelevancyMetric(
            threshold=0.7,
            model="gpt-4.1",
            include_reason=True,
        ),
        FaithfulnessMetric(
            threshold=0.7,
            model="gpt-4.1",
            include_reason=True,
        ),
    ]

    evaluate(
        test_cases=build_test_cases(),
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
