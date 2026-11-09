"""Create data/projects.db with one small table of my projects.

Run from the project root:
    python -m src.build_db

Re-running rebuilds the table from scratch, so the PROJECTS list below is the
single source of truth.
"""

import sqlite3

from src.config import DB_FILE

# Every value comes from a project README or from me directly. Never guess:
# unknown or not applicable = None, which is stored as NULL.
# (name, domain, framework, main_model, year, deployed, team_size, accuracy_pct)
PROJECTS = [
    ("AML Sentinel AI", "anti-money laundering", "scikit-learn", "Isolation Forest", 2026, 0, 1, None),
    ("Ask My Portfolio", "RAG chatbot", "LangChain", "FAISS retrieval + Groq LLM", 2026, 1, 1, None),
    ("Brain Tumor Detection", "medical imaging", "TensorFlow/Keras", "CNN", 2025, 0, 1, 98.43),
    ("CampusEYE", "campus management", "Django", "SmolVLM, YOLOv8", 2025, 0, 4, None),
    ("Dynamic Pricing Simulator", "retail pricing", "scikit-learn", "Random Forest", 2025, 1, 1, None),
    ("Skin Cancer Detection", "medical imaging", "TensorFlow/Keras", "CNN", 2025, 0, 1, 90.0),
    ("Smart Expense Tracker API", "personal finance", "FastAPI", None, 2026, 0, 1, None),
]

SCHEMA = """
CREATE TABLE projects (
    name          TEXT PRIMARY KEY,
    domain        TEXT NOT NULL,
    framework     TEXT NOT NULL,
    main_model    TEXT,          -- NULL if the project has no ML model
    year          INTEGER NOT NULL,
    deployed      INTEGER NOT NULL CHECK (deployed IN (0, 1)),  -- 1 = live demo
    team_size     INTEGER NOT NULL,
    accuracy_pct  REAL           -- headline accuracy from the README; NULL if none
)
"""


def main() -> None:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    try:
        with conn:  # commits if everything succeeds, rolls back on any error
            conn.execute("DROP TABLE IF EXISTS projects")
            conn.execute(SCHEMA)
            conn.executemany(
                "INSERT INTO projects VALUES (?, ?, ?, ?, ?, ?, ?, ?)", PROJECTS
            )
    finally:
        conn.close()
    print(f"Wrote {len(PROJECTS)} rows to {DB_FILE}")


if __name__ == "__main__":
    main()