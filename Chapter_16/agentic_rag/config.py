import os


def _get_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value is not None else default


def _get_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value is not None else default


TOP_K = _get_int("AGENTIC_RAG_TOP_K", 5)
MAX_REWRITE_ATTEMPTS = _get_int("AGENTIC_RAG_MAX_REWRITE_ATTEMPTS", 2)
MIN_BEST_SCORE = _get_float("AGENTIC_RAG_MIN_BEST_SCORE", 0.45)
MIN_CHUNK_SCORE = _get_float("AGENTIC_RAG_MIN_CHUNK_SCORE", 0.30)
MIN_ACCEPTABLE_CHUNKS = _get_int("AGENTIC_RAG_MIN_ACCEPTABLE_CHUNKS", 1)
GRAPH_RECURSION_LIMIT = _get_int("AGENTIC_RAG_RECURSION_LIMIT", 20)
