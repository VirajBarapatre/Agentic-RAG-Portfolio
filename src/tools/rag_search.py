"""rag_search tool: search my documents and return the best-matching chunks."""

from src.config import TOP_K
from src.retriever import Retriever

# Created on first use, then reused, so the model loads only once per run.
_retriever: Retriever | None = None


def _get_retriever() -> Retriever:
    """Return the shared Retriever, creating it the first time it's needed."""
    global _retriever
    if _retriever is None:
        _retriever = Retriever()
    return _retriever


def rag_search(query: str, k: int = TOP_K) -> list[dict]:
    """Search the document set. Returns up to k chunks: source, text, score (higher = closer)."""
    query = query.strip()
    if not query:
        return []
    return _get_retriever().search(query, k)


def format_results(results: list[dict]) -> str:
    """Turn search results into readable text: numbered, with source and score."""
    if not results:
        return "No results."
    blocks = []
    for i, r in enumerate(results, start=1):
        blocks.append(f"[{i}] {r['source']} (score {r['score']:.3f})\n{r['text']}")
    return "\n\n".join(blocks)