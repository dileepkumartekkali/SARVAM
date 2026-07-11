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

# Grok (Groq-hosted Llama) primary — verified live at ~1s per call,
# consistently, all session. Latency complaint traced to two real, additive
# costs of the previous order (gemini,sarvam,...): (1) Gemini's Google Cloud
# project has zero usable quota (confirmed live — every model 404s or
# 429s), so it was failing on EVERY LLM call in a turn (main generation,
# self-check, any correction retry) before ever falling through — pure
# wasted latency, no benefit; (2) Sarvam's models are chain-of-thought
# reasoners with real, verified-live overhead (multiple seconds, sometimes
# hundreds of tokens just to reach a short reply) — fine as a fallback, bad
# as the thing actually serving every request. Gemini is kept at the end
# (not removed) in case its quota is ever fixed — with Grok and Sarvam
# both healthy, it's essentially never reached.
_DEFAULT_ORDER = "grok,sarvam,gemini,claude,gpt"


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
