"""sql_query tool: run one read-only SELECT on the projects database."""

import sqlite3
import time

from src.config import DB_FILE

MAX_ROWS = 50          # never return more rows than this
TIMEOUT_SECONDS = 5    # stop any query that runs longer than this


def _check_is_select(query: str) -> str:
    """Layer 1: allow exactly one statement, and it must be a SELECT."""
    cleaned = query.strip().rstrip(";").strip()
    if ";" in cleaned:
        raise ValueError("Only one SQL statement is allowed.")
    if not cleaned.lower().startswith("select"):
        raise ValueError("Only SELECT queries are allowed.")
    return cleaned


def sql_query(query: str) -> list[dict]:
    """Run a SELECT on data/projects.db and return rows as dicts (max MAX_ROWS)."""
    cleaned = _check_is_select(query)
    if not DB_FILE.exists():
        raise FileNotFoundError("Database not found. Build it first: python -m src.build_db")

    # Layer 2: open the file in read-only mode, so SQLite itself refuses any write.
    conn = sqlite3.connect(f"{DB_FILE.as_uri()}?mode=ro", uri=True)
    try:
        deadline = time.monotonic() + TIMEOUT_SECONDS
        # SQLite calls this every 10,000 steps; returning 1 stops the query.
        conn.set_progress_handler(lambda: int(time.monotonic() > deadline), 10_000)
        cursor = conn.execute(cleaned)
        columns = [col[0] for col in cursor.description]
        rows = cursor.fetchmany(MAX_ROWS)
    finally:
        conn.close()
    return [dict(zip(columns, row)) for row in rows]