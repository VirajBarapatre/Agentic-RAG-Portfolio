"""Tests for the read-only SQL tool, on a temporary database built by build_db."""

import sqlite3

import pytest

import src.build_db as build_db
import src.tools.sql_query as sql_module
from src.tools.sql_query import sql_query


@pytest.fixture
def db_file(tmp_path, monkeypatch):
    """Build the real projects table in a temporary folder and point both modules at it."""
    path = tmp_path / "projects.db"
    monkeypatch.setattr(build_db, "DB_FILE", path)
    monkeypatch.setattr(sql_module, "DB_FILE", path)
    build_db.main()
    return path


def test_select_returns_rows_as_dicts(db_file):
    rows = sql_query("SELECT name FROM projects WHERE deployed = 1 ORDER BY name")
    assert rows == [{"name": "Ask My Portfolio"}, {"name": "Dynamic Pricing Simulator"}]


def test_trailing_semicolon_is_allowed(db_file):
    assert sql_query("SELECT COUNT(*) AS n FROM projects;") == [{"n": 7}]


@pytest.mark.parametrize("query", [
    "DROP TABLE projects",
    "DELETE FROM projects",
    "UPDATE projects SET year = 2000",
    "SELECT 1; DROP TABLE projects",
])
def test_layer1_rejects_non_select_and_multiple_statements(db_file, query):
    with pytest.raises(ValueError):
        sql_query(query)
    assert sql_query("SELECT COUNT(*) AS n FROM projects") == [{"n": 7}]  # table untouched


def test_layer2_read_only_connection_refuses_writes(db_file):
    """Bypass layer 1 completely: the read-only connection must still refuse."""
    conn = sqlite3.connect(f"{db_file.as_uri()}?mode=ro", uri=True)
    try:
        with pytest.raises(sqlite3.OperationalError):
            conn.execute("DELETE FROM projects")
    finally:
        conn.close()


def test_row_limit(db_file, monkeypatch):
    monkeypatch.setattr(sql_module, "MAX_ROWS", 2)
    assert len(sql_query("SELECT name FROM projects")) == 2


def test_missing_database(tmp_path, monkeypatch):
    monkeypatch.setattr(sql_module, "DB_FILE", tmp_path / "missing.db")
    with pytest.raises(FileNotFoundError):
        sql_query("SELECT 1")


def test_unknown_column_raises(db_file):
    with pytest.raises(sqlite3.OperationalError):
        sql_query("SELECT nope FROM projects")