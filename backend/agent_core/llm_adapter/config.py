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

# Grok primary by explicit product choice (latency over Sarvam's reasoning
# overhead) — accepted tradeoff: Grok (Groq-hosted Llama 3.3) does not
# officially support most of the 13 Indic languages (only Hindi among them,
# per Meta's published list), so non-Hindi Indic-language replies fall
# through to Sarvam (the actually-multilingual model) whenever Grok's own
# call fails. Gemini kept at the end (Google Cloud quota is confirmed dead
# — every model 404s/429s) in case it's ever fixed.
_DEFAULT_ORDER = "grok,sarvam,gemini,claude,gpt"


def _build_provider(name: str) -> LLMProvider:
    if name == "grok":
        # Real bug hit live: this provider is Groq (api.groq.com, hosting
        # Llama models) -- every comment in this codebase and .env.example
        # say so -- but the default base_url/model pointed at xAI's actual
        # Grok API (api.x.ai) instead. A real Groq API key was being sent to
        # the wrong company's endpoint, rejected as "Incorrect API key
        # provided" (a genuine-looking auth error, not a hint at the real
        # cause). Kept the "grok" name (env var GROK_API_KEY, provider
        # registry key) unchanged -- only the wrong DEFAULT endpoint/model
        # were the bug, not the naming.
        return OpenAICompatibleProvider(
            name="grok",
            base_url=os.environ.get("GROK_BASE_URL", "https://api.groq.com/openai/v1"),
            model=os.environ.get("GROK_MODEL", "llama-3.3-70b-versatile"),
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
