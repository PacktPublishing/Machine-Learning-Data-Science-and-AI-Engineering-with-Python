# Agentic RAG with LangGraph

This project demonstrates how to build a production-inspired **Agentic Retrieval-Augmented Generation (RAG)** workflow using **LangGraph**, **Azure AI Foundry/OpenAI**, **Chroma**, and **Streamlit**.

Unlike the `graph_rag` example, this implementation does not simply retrieve documents and generate an answer. Instead, it introduces multiple reasoning and validation stages that decide whether the workflow should continue, retry, or abstain.

---

# Objectives

This example demonstrates:

- multi-step agent orchestration using LangGraph
- structured outputs with Pydantic
- Retrieval-Augmented Generation (RAG)
- query understanding
- query rewriting
- retrieval quality evaluation
- answer reflection
- guardrails
- streaming graph execution
- interactive Streamlit visualization

---

# Workflow

```text
User Question
        │
        ▼
Query Guardrail
        │
        ▼
Query Analysis
        │
        ▼
Retriever
        │
        ▼
Evidence Guardrail
        │
        ├───────────────┐
        ▼               │
Generate Answer         │
        │               │
        ▼               │
Reflection              │
        │               │
        ├──── Retry ────┘
        ▼
Final Answer

```

The workflow may terminate early if:

- the initial question is invalid
- Azure AI blocks the prompt
- the retrieved evidence is insufficient
- the reflection stage determines that the answer is unreliable

---

# Project Structure

```
agentic_rag/
│
├── main.py
├── graph.py
├── runtime.py
├── routing.py
├── state.py
├── schemas.py
├── prompts.py
├── config.py
├── formatting.py
├── vector_store.py
├── ui.py
├── streamlit_ui.py
├── errors.py
│
└── nodes/
    ├── query.py
    ├── retrieval.py
    └── answer.py
```

### Responsibilities

## graph.py

Builds the LangGraph workflow.

Responsible for:

- registering nodes
- defining edges
- conditional routing
- compiling the graph

---

## state.py

Defines the LangGraph shared state.

Contains:

- original question
- rewritten query
- retrieved chunks
- retrieval scores
- evaluations
- final response
- execution events

---

## schemas.py

Contains all structured Pydantic models.

Examples:

- QueryValidation
- QueryAnalysis
- RetrievalEvaluation
- AnswerEvaluation
- RagResponse

---

## nodes/

Contains the implementation of the graph nodes.

### query.py

Responsible for

- query validation
- query analysis
- query rewriting

---

### retrieval.py

Responsible for

- vector search
- retrieval quality evaluation
- similarity score validation

---

### answer.py

Responsible for

- answer generation
- reflection
- abstention
- finalization

---

## routing.py

Contains all routing logic.

Routing functions decide which node executes next.

Examples:

- continue
- rewrite query
- abstain
- finalize

---

## runtime.py

Runs the graph.

Provides

- initial state
- streaming execution
- Azure content-filter handling

---

## ui.py

Rich console interface.

Displays:

- retrieved chunks
- scores
- structured answers

---

## streamlit_ui.py

Interactive Streamlit interface.

Displays:

- graph evolution
- node execution
- partial results
- retrieval scores
- final answer

---

## errors.py

Centralized handling of platform-level exceptions.

Converts Azure AI content filtering into a normal graph outcome.

---

# Guardrails

This implementation demonstrates several independent guardrails.

## 1. Query Guardrail

Rejects:

- unrelated questions
- ambiguous requests
- unsupported operations
- prompt injection attempts

---

## 2. Platform Guardrail

Azure AI may reject prompts before inference.

Instead of crashing, the graph produces a controlled execution result.

---

## 3. Retrieval Guardrail

Checks whether the retrieved documents are sufficiently relevant.

The workflow skips answer generation when evidence is weak.

---

## 4. Reflection

Evaluates the generated answer.

The graph may

- accept
- retry
- abstain

---

# Streaming Execution

The graph is executed using

```python
graph.stream(...)
```

instead of

```python
graph.invoke(...)
```

This allows:

- live execution visualization
- node-by-node updates
- partial retrieval results
- progressive UI updates

---

# Streamlit UI

The Streamlit application visualizes every graph step.

Displayed information includes:

- query validation
- query analysis
- rewritten query
- retrieved chunks
- similarity scores
- retrieval evaluation
- reflection
- final answer

---

# Running

## Build the vector index

```bash
uv run python -m rag_common.ingest
```

---

## Console

```bash
uv run python -m agentic_rag.main
```

---

## Streamlit

```bash
streamlit run streamlit_app.py
```

---

# Azure Deployment

The project can be deployed to Azure.

Recommended option:

- Azure Container Apps

Other supported options:

- Azure App Service
- Azure Kubernetes Service (AKS)

Authentication should use

```python
DefaultAzureCredential()
```

instead of

```python
AzureCliCredential()
```

when running inside Azure.

---

# Learning Goals

This example introduces several production concepts:

- Agentic AI
- LangGraph orchestration
- dependency injection
- structured outputs
- guardrails
- reflection
- retrieval evaluation
- retry loops
- streaming execution
- Azure deployment

Although simplified for teaching purposes, the architecture follows patterns commonly used in production AI applications.