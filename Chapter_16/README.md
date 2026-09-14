# PDF Question Answering with RAG

This project implements the Chapter 16 example:

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
uv run python -m rag_common.ingest --reset
```

## Ask questions

```bash
uv run python -m agentic_rag.main
```

## Test with Streamlit UI

```bash
streamlit run streamlit_app.py
```

