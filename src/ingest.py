"""Build the search index: load docs -> chunk -> embed -> save.

Run from the project root:
    python -m src.ingest
"""

import json
from collections import Counter

import faiss
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CHUNKS_FILE,
    DOCS_DIR,
    EMBEDDING_MODEL_NAME,
    INDEX_DIR,
    INDEX_FILE,
)


def load_documents() -> list[dict]:
    """Read every .md file in data/docs. Each item: {"source": filename, "text": content}."""
    documents = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        documents.append({"source": path.name, "text": text})
    return documents

def document_title(doc: dict) -> str:
    """The document's first '# ' heading, or its file name if it has none."""
    for line in doc["text"].splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return doc["source"]

def chunk_documents(documents: list[dict]) -> list[dict]:
    """Split each document into overlapping chunks, keeping the source filename."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = []
    for doc in documents:
        title = document_title(doc)
        for text in splitter.split_text(doc["text"]):
            # The title gives every chunk its context, e.g. a results table
            # that never names its project on its own.
            chunks.append({"id": len(chunks), "source": doc["source"], "text": f"Document: {title}\n\n{text}"})
    return chunks

def build_index(chunks: list[dict]) -> faiss.IndexFlatIP:
    """Embed every chunk and store the vectors in a FAISS index."""
    model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    vectors = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    index = faiss.IndexFlatIP(vectors.shape[1])  # shape[1] = 384 for this model
    index.add(vectors)
    return index


def save(index: faiss.IndexFlatIP, chunks: list[dict]) -> None:
    """Write the vectors and the chunk text (plus the model name) to data/index."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_FILE))
    data = {"embedding_model": EMBEDDING_MODEL_NAME, "chunks": chunks}
    CHUNKS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> None:
    documents = load_documents()
    if not documents:
        raise SystemExit(f"No .md files found in {DOCS_DIR}")

    chunks = chunk_documents(documents)
    index = build_index(chunks)
    save(index, chunks)

    print(f"Loaded {len(documents)} documents, created {len(chunks)} chunks.")
    print("Chunks per document:")
    for source, count in Counter(c["source"] for c in chunks).most_common():
        print(f"  {count:3d}  {source}")
    print(f"Saved index to {INDEX_DIR}")


if __name__ == "__main__":
    main()