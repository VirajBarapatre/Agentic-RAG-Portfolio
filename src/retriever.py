"""Load the saved index and return the chunks most similar to a question."""

import json

import faiss
from sentence_transformers import SentenceTransformer

from src.config import CHUNKS_FILE, EMBEDDING_MODEL_NAME, INDEX_FILE, TOP_K


class Retriever:
    """Loads the index and the embedding model once, then runs many searches."""

    def __init__(self) -> None:
        if not INDEX_FILE.exists() or not CHUNKS_FILE.exists():
            raise FileNotFoundError("Index not found. Build it first: python -m src.ingest")

        data = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
        if data["embedding_model"] != EMBEDDING_MODEL_NAME:
            raise ValueError(
                f"Index was built with {data['embedding_model']!r}, but config says "
                f"{EMBEDDING_MODEL_NAME!r}. Rebuild it: python -m src.ingest"
            )

        self.chunks: list[dict] = data["chunks"]
        self.index = faiss.read_index(str(INDEX_FILE))
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    def search(self, query: str, k: int = TOP_K) -> list[dict]:
        """Return the top-k chunks: source, text, score (cosine similarity, higher = closer)."""
        query_vector = self.model.encode(
            [query], normalize_embeddings=True, convert_to_numpy=True
        )
        scores, ids = self.index.search(query_vector, k)

        results = []
        for score, chunk_id in zip(scores[0], ids[0]):
            if chunk_id == -1:  # FAISS returns -1 when k is larger than the number of vectors
                continue
            chunk = self.chunks[chunk_id]
            results.append(
                {"source": chunk["source"], "text": chunk["text"], "score": float(score)}
            )
        return results