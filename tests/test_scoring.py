"""Tests for the eval scoring functions, including real gpt-oss output quirks."""

import pytest

from src.config import REFUSAL_PHRASE
from src.eval.scoring import (
    extract_numbers,
    is_refusal,
    normalize,
    score_answer,
    score_grounded,
    score_keywords,
    score_number,
    score_tools,
)

# A real answer from the baseline run: narrow no-break spaces (\u202f) and
# non-breaking hyphens (\u2011). This once made a correct answer fail.
REAL_ANSWER = (
    "AML\u202fSentinel\u202fAI uses an **Isolation\u202fForest** algorithm for detecting "
    "suspicious transactions. (Source:\u202faml\u2011sentinel\u2011ai.md)"
)


def test_normalize_handles_unicode_spaces_hyphens_quotes_and_markers():
    assert "isolation forest" in normalize(REAL_ANSWER)
    assert "aml-sentinel-ai.md" in normalize(REAL_ANSWER)
    assert normalize("I don\u2019t") == "i don't"
    assert normalize("seven\u30101\u3011") == "seven "


def test_extract_numbers():
    assert extract_numbers("98.43 % and **8.43 %**") == [98.43, 8.43]
    assert extract_numbers("That is 50,000 transactions.") == [50000.0]
    assert extract_numbers("1. Ask My Portfolio (2026)\n2. CampusEYE (2025)") == [2026.0, 2025.0]
    assert extract_numbers("HAM10000 and all-MiniLM-L6-v2") == []  # digits inside names
    assert extract_numbers("seven classes") == []
    assert extract_numbers("seven classes", include_words=True) == [7.0]


def test_score_number_uses_tolerance():
    assert score_number("Difference: 8.430000000000007", 8.43)
    assert score_number("about 94.22%", 94.215)
    assert not score_number("about 94.2%", 94.215)


def test_refusal_detection():
    assert is_refusal(REFUSAL_PHRASE)
    assert is_refusal("I don\u2019t know based on my documents.")
    assert not is_refusal("I'm sorry, but I can't help with that.")


def test_keywords_need_every_group():
    groups = [["dynamic pricing"], ["tailwind", "chart.js"]]
    assert score_keywords("Dynamic Pricing uses Chart.js", groups)
    assert not score_keywords("Dynamic Pricing uses React", groups)
    assert score_keywords(REAL_ANSWER, [["isolation forest"]])


def test_forbidden_string_fails_an_otherwise_correct_refusal():
    question = {"answer_type": "refusal", "forbidden": ["98.43"]}
    assert score_answer(question, REFUSAL_PHRASE)
    assert not score_answer(question, REFUSAL_PHRASE + " The brain tumor model got 98.43%.")


def test_unknown_answer_type():
    with pytest.raises(ValueError):
        score_answer({"answer_type": "essay"}, "text")


def test_tools_compared_as_sets():
    assert score_tools(["sql_query", "calculator", "sql_query"], [["calculator", "sql_query"]])
    assert score_tools([], [[]])
    assert not score_tools(["rag_search"], [[]])
    assert not score_tools(["sql_query", "rag_search"], [["sql_query"]])


def test_groundedness():
    answer = "Brain tumor 98.43 %, skin cancer 90.0 %, difference 8.43 %."
    evidence = ["{'accuracy_pct': 98.43} {'accuracy_pct': 90.0}", "8.430000000000007"]
    assert score_grounded(answer, evidence) is True
    assert score_grounded("HAM10000 has 10015 images.", ["HAM10000 dataset"]) is False
    assert score_grounded(REFUSAL_PHRASE, []) is None  # no numbers: nothing to check