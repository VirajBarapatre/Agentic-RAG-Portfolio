"""Manual retrieval check.

Run from the project root:
    python -m scripts.try_rag "your question"
    python -m scripts.try_rag "your question" 3
"""

import sys

from src.config import TOP_K
from src.tools.rag_search import format_results, rag_search


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit('Usage: python -m scripts.try_rag "your question" [k]')
    query = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else TOP_K
    print(format_results(rag_search(query, k)))


if __name__ == "__main__":
    main()