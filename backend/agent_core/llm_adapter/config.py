"""Env-driven provider assembly — no hardcoded order or API keys anywhere else.

To swap the primary provider or reorder fallback: change `LLM_PROVIDER_ORDER`
alone, no code change. Per-provider model/base-url are also env-overridable;
only the API key *env var name* per provider is fixed (not the key value).
"""

from __future__ import annotations

import os

from .base import LLMProvider, LLMRouter
from .providers.anthropic import AnthropicProvider
from .providers.gemini import GeminiProvider
from .providers.openai_compatible import OpenAICompatibleProvider

# Gemini primary (2.5 Flash — fast, cheap, no reasoning-token overhead unlike
# Sarvam's models), everything else as fallback in this order. NOT verified
# against a live Gemini account as of this change — no GEMINI_API_KEY was
# configured yet; see gemini.py's docstring. Confirm live the moment a real
# key is in place, the same way every other provider in this file was.
_DEFAULT_ORDER = "gemini,sarvam,grok,claude,gpt"


def _build_provider(name: str) -> LLMProvider:
    if name == "grok":
        return OpenAICompatibleProvider(
            name="grok",
            base_url=os.environ.get("GROK_BASE_URL", "https://api.x.ai/v1"),
            model=os.environ.get("GROK_MODEL", "grok-4"),
            api_key_env="GROK_API_KEY",
        )
    if name == "gpt":
        return OpenAICompatibleProvider(
            name="gpt",
            base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            api_key_env="OPENAI_API_KEY",
        )
    if name == "sarvam":
        return OpenAICompatibleProvider(
            name="sarvam",
            base_url=os.environ.get("SARVAM_LLM_BASE_URL", "https://api.sarvam.ai/v1"),
            # "sarvam-m" (the old default) is deprecated by Sarvam as of this
            # writing — verified live; the API now rejects it outright.
            model=os.environ.get("SARVAM_LLM_MODEL", "sarvam-30b"),
            api_key_env="SARVAM_API_KEY",
            # Doesn't eliminate the reasoning-token overhead (verified: "low"
            # still spends ~100-400 tokens before real content), but keeps it
            # from defaulting to "high" and reasoning even longer.
            extra_body={"reasoning_effort": os.environ.get("SARVAM_REASONING_EFFORT", "low")},
        )
    if name == "claude":
        return AnthropicProvider(model=os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-5"))
    if name == "gemini":
        return GeminiProvider(model=os.environ.get("GEMINI_MODEL", "gemini-2.5-flash"))
    raise ValueError(f"unknown LLM provider name: {name!r}")


def provider_order() -> list[str]:
    raw = os.environ.get("LLM_PROVIDER_ORDER", _DEFAULT_ORDER)
    return [p.strip() for p in raw.split(",") if p.strip()]


def build_router_from_env() -> LLMRouter:
    """Primary = first entry in `LLM_PROVIDER_ORDER`."""
    return LLMRouter([_build_provider(name) for name in provider_order()])
