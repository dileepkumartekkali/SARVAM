"""Translation policy: default is direct native-language reasoning.

Per agent_system_prompt.md, translation is a workaround for two concrete
gaps — a downstream tool that only accepts English, or the active LLM not
supporting the detected language — not a default step. Modern LLM providers
handle these 13 languages (and code-mixing) directly; translating by default
would erase code-mixing and cost latency for no reason.
"""

from __future__ import annotations

# Empty by default: no known Phase-3 gap in LLM reasoning for any of the 13
# supported languages. Populate if a specific provider proves unable to
# reason in one of them (e.g. via eval failures).
NOT_NATIVELY_SUPPORTED: set[str] = set()


def decide_translation(
    language: str,
    *,
    tool_requires_english: bool = False,
    not_natively_supported: set[str] = NOT_NATIVELY_SUPPORTED,
) -> bool:
    """True only when translation is actually required — never the default."""
    return tool_requires_english or language in not_natively_supported
