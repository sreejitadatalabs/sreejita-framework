from typing import Optional

from sreejita.narrative.schema import NarrativeInput
from sreejita.narrative.llm import LLMClient


# =====================================================
# v3.5 NARRATIVE COMPOSER
# =====================================================
# This module converts structured decisions into
# human-readable executive language using an LLM.
# =====================================================


SYSTEM_PROMPT = """
You are an AI assistant helping rewrite EXISTING business decisions
into clear, executive-friendly language.

STRICT RULES (DO NOT VIOLATE):
- Do NOT invent insights
- Do NOT invent actions
- Do NOT introduce numbers or metrics
- Do NOT add analysis or reasoning
- Do NOT add recommendations
- Do NOT change priorities or timelines

You may ONLY explain and rephrase the information provided.
If information is insufficient, say so clearly.

You are NOT an analyst.
You are NOT a decision-maker.
You are a narrator.
""".strip()


def build_user_prompt(data: NarrativeInput) -> str:
    """
    Build a controlled user prompt from validated NarrativeInput.
    """

    payload = data.to_prompt_dict()

    return f"""
Rewrite the following structured business findings into
clear, concise executive language.

Context:
- Domain: {payload["domain"]}
- Confidence Level: {payload["confidence_band"]}
- Summary Level: {payload["summary_level"]}

Insights (already approved):
{_format_insights(payload["insights"])}

Recommended Actions (already approved):
{_format_actions(payload["actions"])}

Constraints:
- Do not add new insights
- Do not add new actions
- Do not add numbers
- Do not speculate

Write 2–4 short paragraphs suitable for an executive summary.
""".strip()


def _format_insights(insights) -> str:
    if not insights:
        return "- No significant insights identified."

    lines = []
    for ins in insights:
        lines.append(
            f"- [{ins['level']}] {ins['title']}: {ins['description']}"
        )
    return "\n".join(lines)


def _format_actions(actions) -> str:
    if not actions:
        return "- No immediate actions recommended."

    lines = []
    for act in actions:
        lines.append(
            f"- {act['action']} "
            f"(Priority: {act['priority']}, Timeline: {act['timeline']})"
        )
    return "\n".join(lines)


# -----------------------------------------------------
# PUBLIC API
# -----------------------------------------------------

def generate_narrative(
    narrative_input: NarrativeInput,
    llm_client: LLMClient,
) -> Optional[str]:
    """
    Generate AI-assisted narrative text.

    Returns:
    - Markdown string if enabled
    - None if narrative is disabled
    """

    if not llm_client.enabled:
        return None

    user_prompt = build_user_prompt(narrative_input)

    full_prompt = f"{SYSTEM_PROMPT}\n\n{user_prompt}"

    return llm_client.generate(full_prompt)

