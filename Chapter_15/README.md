# PDF Question Answering with RAG

This project implements the Chapter 15 example:

```text
PDF files -> text chunks -> OpenAI embeddings -> Chroma
question -> similarity search -> retrieved context -> generated answer
```

## Setup

```bash
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
cp .env.example .env
```

Set `OPENAI_API_KEY` in `.env`. Add PDF files to `rag_common/data/`.

## Build the Chroma index

```bash
python -m rag_common.ingest
```

To rebuild the index after changing the documents or chunking settings:

```bash
python -m rag_common.ingest --reset
```

## Ask questions

```bash
python -m simple_rag.main
```

Type `exit` to stop.

## Run the DeepEval evaluation

Edit `evaluation/test_cases.py` and add questions and reference answers for the PDFs in `rag_common/data/`. Then run:

```bash
python -m evaluation.evaluate_rag
```

The evaluation uses DeepEval's `ContextualPrecisionMetric`, `ContextualRecallMetric`, `AnswerRelevancyMetric`, and `FaithfulnessMetric`.
