"""Central place for paths and settings. Change values here, not across files."""

import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---------------------------------------------------------------
# Built from this file's location, so they work no matter which folder
# you run commands from.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCS_DIR = PROJECT_ROOT / "data" / "docs"
INDEX_DIR = PROJECT_ROOT / "data" / "index"
INDEX_FILE = INDEX_DIR / "faiss.index"   # the vectors
CHUNKS_FILE = INDEX_DIR / "chunks.json"  # the text + source of each vector
DB_FILE = PROJECT_ROOT / "data" / "projects.db"

QUESTIONS_FILE = PROJECT_ROOT / "eval" / "questions.jsonl"
RESULTS_DIR = PROJECT_ROOT / "results"
# --- Environment -----------------------------------------------------------
# Reads .env from the project root into environment variables.
# If .env is missing, nothing breaks; defaults below are used.
load_dotenv(PROJECT_ROOT / ".env")

# Model name is not a secret, so it has a default.
# The API key is NOT read here; it is only needed from Milestone 2.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
# Seconds to wait for one LLM reply before giving up.
LLM_TIMEOUT_SECONDS = 30

# The one refusal sentence. The agent's prompt and the eval scorer both use it.
REFUSAL_PHRASE = "I don't know based on my documents."
# --- Chunking --------------------------------------------------------------
# Measured in characters. Carried over from Ask My Portfolio: 400/50 split
# dense sections mid-topic, 600/100 kept related lines together.
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100

# --- Embeddings and retrieval -----------------------------------------------
# Small local model, 384-dim vectors, runs on CPU, no API key.
# Ingest and search MUST use the same model, so both read it from here.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# How many chunks a search returns. Revisit after the Step 7 check,
# because the document set is small.
TOP_K = 5