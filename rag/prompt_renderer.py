"""Render the constrained support prompt used by the RAG pipeline."""

from pathlib import Path
from typing import Sequence

from poml import poml

SUPPORT_PROMPT = Path(__file__).parent / "prompts" / "support_answer.poml"


def render_support_prompt(*, issue: str, context_chunks: list[dict[str, object]],
                          allowed_categories: Sequence[str]) -> str:
    rendered = poml(
        str(SUPPORT_PROMPT), chat=True, format="langchain",
        context={
            "issue": issue,
            "context_chunks": context_chunks,
            "allowed_categories": list(allowed_categories),
        },
    )
    return rendered["messages"][0]["data"]["content"]
