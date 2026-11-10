"""Scoring functions for the evaluation. Pure Python: no LLM, no network, no files."""

import re
import unicodedata
from src.config import REFUSAL_PHRASE

NUMBER_TOLERANCE = 0.01

NUMBER_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19, "twenty": 20,
}


def normalize(text: str) -> str:
    """Make text comparable: plain spaces and hyphens, lowercase, straight quotes, no 【1】 markers."""
    text = unicodedata.normalize("NFKC", text)  # non-breaking spaces -> normal spaces
    text = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u2010", "-").replace("\u2011", "-")  # unicode hyphens -> "-"
    return re.sub(r"\u3010[^\u3011]*\u3011", " ", text)


def extract_numbers(text: str, include_words: bool = False) -> list[float]:
    """Return every number in the text. 'include_words' also reads 'two' as 2."""
    text = normalize(text)
    text = re.sub(r"(?m)^\s*\d+[.)]\s", " ", text)   # numbered-list markers: "1. ", "2) "
    text = re.sub(r"(?<=\d),(?=\d{3}\b)", "", text)    # thousands separators: 50,000 -> 50000
    if include_words:
        for word, value in NUMBER_WORDS.items():
            text = re.sub(rf"\b{word}\b", str(value), text)
    return [float(n) for n in re.findall(r"\b\d+(?:\.\d+)?\b", text)]


def is_refusal(answer: str) -> bool:
    """True if the answer contains the standard refusal phrase."""
    return normalize(REFUSAL_PHRASE).rstrip(".") in normalize(answer)


def score_number(answer: str, expected: float) -> bool:
    """True if any number in the answer is within NUMBER_TOLERANCE of the expected value."""
    numbers = extract_numbers(answer, include_words=True)
    return any(abs(n - expected) <= NUMBER_TOLERANCE for n in numbers)


def score_keywords(answer: str, groups: list[list[str]]) -> bool:
    """True if every group has at least one of its words in the answer."""
    text = normalize(answer)
    return all(any(word in text for word in group) for group in groups)


def contains_forbidden(answer: str, forbidden: list[str]) -> bool:
    """True if the answer contains any forbidden string."""
    text = normalize(answer)
    return any(word.lower() in text for word in forbidden)


def score_answer(question: dict, answer: str) -> bool:
    """Answer correctness for one question, using its answer_type."""
    if contains_forbidden(answer, question.get("forbidden", [])):
        return False
    answer_type = question["answer_type"]
    if answer_type == "refusal":
        return is_refusal(answer)
    if answer_type == "number":
        return score_number(answer, question["expected_number"])
    if answer_type == "keywords":
        return score_keywords(answer, question["keywords"])
    raise ValueError(f"Unknown answer_type: {answer_type}")


def score_tools(tools_used: list[str], expected_tools: list[list[str]]) -> bool:
    """True if the set of tools used exactly matches one of the acceptable sets."""
    used = set(tools_used)
    return any(used == set(option) for option in expected_tools)


def score_grounded(answer: str, evidence: list[str]) -> bool | None:
    """True if every number in the answer also appears in the evidence.

    Evidence = the question text plus tool arguments and outputs.
    Returns None when the answer has no numbers (nothing to check).
    """
    answer_numbers = extract_numbers(answer)
    if not answer_numbers:
        return None
    evidence_numbers = extract_numbers(" ".join(evidence))
    return all(
        any(abs(a - e) <= NUMBER_TOLERANCE for e in evidence_numbers)
        for a in answer_numbers
    )