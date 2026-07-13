"""Env-driven provider order — swapping the primary must take zero code changes."""

from agent_core.llm_adapter.config import build_router_from_env, provider_order


def test_default_order_has_grok_primary_with_all_others_as_fallback(monkeypatch):
    """Grok primary by product choice (latency). Sarvam (the actually
    multilingual model) is the fallback for non-Hindi Indic languages Grok
    doesn't officially support."""
    monkeypatch.delenv("LLM_PROVIDER_ORDER", raising=False)
    assert provider_order() == ["grok", "sarvam", "gemini", "claude", "gpt"]


def test_env_var_swaps_primary_with_no_code_change(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "gemini,claude,grok,gpt,sarvam")
    assert provider_order() == ["gemini", "claude", "grok", "gpt", "sarvam"]

    router = build_router_from_env()

    assert router._providers[0].name == "gemini"
    assert router._providers[1].name == "claude"


def test_unknown_provider_name_rejected(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER_ORDER", "not-a-real-provider")
    try:
        build_router_from_env()
        assert False, "expected ValueError"
    except ValueError:
        pass
