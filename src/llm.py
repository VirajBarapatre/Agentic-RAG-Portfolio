"""Create the chat model the agent uses."""

import os

from langchain_groq import ChatGroq

from src.config import GROQ_MODEL, LLM_TIMEOUT_SECONDS


def get_llm() -> ChatGroq:
    """Return the Groq chat model. Fails early with a clear message if the key is missing."""
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return ChatGroq(
        model=GROQ_MODEL,
        temperature=0,
        timeout=LLM_TIMEOUT_SECONDS,
        max_retries=2,
    )