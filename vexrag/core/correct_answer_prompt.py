from textwrap import dedent

from vexrag.core.contracts import TargetStyle

_STYLE_GUIDE: dict[TargetStyle, str] = {
    "short_fact": "Keep answers short, factual, and compact (1-2 sentences).",
    "paragraph": "Write a single coherent paragraph with 3-6 sentences.",
}


def build_correct_answer_prompt(query: str, target_style: TargetStyle) -> str:
    """Build prompt for generating a reference correct answer."""
    style_requirement = _STYLE_GUIDE[target_style]
    return dedent(
        f"""
        You are generating a high-quality reference answer for a user query.
        Style requirement: {style_requirement}

        Return ONLY valid JSON with this shape:
        {{
          "correct_answer": "..."
        }}

        Query:
        {query}
        """
    ).strip()
