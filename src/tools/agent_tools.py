"""The three tools the agent can call, wrapped for LangChain/LangGraph.

Each wrapper returns plain text for the LLM and never raises: errors come back
as text, so one failing tool cannot crash an agent run.
"""

from langchain_core.tools import tool

from src.tools.calculator import calculate
from src.tools.rag_search import format_results, rag_search
from src.tools.sql_query import MAX_ROWS, sql_query


@tool("rag_search")
def rag_search_tool(query: str) -> str:
    """Search Viraj's project READMEs and return the most relevant text passages.

    Use for questions about what a project does, how it works, its tech stack,
    dataset, results, or design decisions. Each passage shows its source file.
    Not every project has a README; if a project is not found here, check sql_query.
    """
    try:
        return format_results(rag_search(query))
    except Exception as err:
        return f"Error in rag_search: {type(err).__name__}: {err}"


@tool("calculator")
def calculator_tool(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(98.43 - 90) / 90 * 100'.

    Supports numbers, + - * / ** % and parentheses only.
    Use it for any arithmetic instead of calculating in your head.
    """
    try:
        return str(calculate(expression))
    except Exception as err:
        return f"Error in calculator: {type(err).__name__}: {err}"


@tool("sql_query")
def sql_query_tool(query: str) -> str:
    """Run one read-only SQLite SELECT on the table `projects` (one row per project).

    Columns: name TEXT, domain TEXT, framework TEXT,
    main_model TEXT (NULL if the project has no ML model), year INTEGER,
    deployed INTEGER (1 = has a live demo, 0 = not deployed), team_size INTEGER,
    accuracy_pct REAL (headline accuracy in %, NULL if not reported).

    Use for lists, counts, filters and comparisons across projects,
    e.g. which projects are deployed or how many were built in 2025.
    """
    try:
        rows = sql_query(query)
    except Exception as err:
        return f"Error in sql_query: {type(err).__name__}: {err}"
    if not rows:
        return "No rows."
    text = "\n".join(str(row) for row in rows)
    if len(rows) == MAX_ROWS:
        text += f"\n(Showing the first {MAX_ROWS} rows only.)"
    return text


ALL_TOOLS = [rag_search_tool, calculator_tool, sql_query_tool]