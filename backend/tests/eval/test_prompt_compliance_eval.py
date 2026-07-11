"""Prompt-compliance eval suite (agent_system_prompt.md §4 implementation
note): length compliance, no-markdown-in-voice-mode compliance, and
language-preservation compliance, per language — run in CI before any prompt
template change ships (see .github/workflows/ci.yml's `prompt-eval` job,
path-filtered on `prompts/**`).

Two tiers, both real:

1. **Compliance-checker regression** (always runs, no network/keys needed):
   drives `_self_check` and `detect_language` — the actual functions that
   gate a real model's output in production — against known-compliant and
   known-violating samples, across all 13 languages. This is what actually
   catches a broken prompt template or a broken checker in CI.
2. **Live model eval** (`test_live_model_compliance`, skipped unless
   `MAAV_EVAL_LIVE=true` and at least one provider key is configured): runs
   the real templates against a real configured LLM and checks its actual
   output. Tier 1 cannot substitute for this — it proves the checker works,
   not that a deployed model's real behavior passes it.
"""

import os

import pytest

from agent_core.agents.task_agent import _build_system_prompt, _self_check
from agent_core.llm_adapter import LLMRouter, build_router_from_env
from agent_core.supervisor.state import Mode, SessionState

from .._fakes import ScriptedProvider
from .._language_fixtures import PURE_LANGUAGE_CASES

COMPLIANT_VOICE_REPLY = "Sure, I can help with that today."
VIOLATING_VOICE_REPLY_MARKDOWN = "**Sure!** Here's a list:\n- one\n- two"
VIOLATING_VOICE_REPLY_TOO_LONG = "word " * 100

COMPLIANT_TEXT_REPLY = "**Sure!** Here's a short list:\n- one\n- two"
VIOLATING_TEXT_REPLY_TOO_LONG = "word " * 200


@pytest.mark.parametrize("text,language", PURE_LANGUAGE_CASES)
async def test_voice_mode_no_markdown_and_length_compliance(text, language):
    session = SessionState(
        session_id="s", conversation_id="c", thread_id="t", mode=Mode.SPEECH_TO_SPEECH, response_language=language
    )
    router = LLMRouter([ScriptedProvider(["OK"])])

    ok, _ = await _self_check(COMPLIANT_VOICE_REPLY, session.mode, router)
    assert ok is True

    ok, reason = await _self_check(VIOLATING_VOICE_REPLY_MARKDOWN, session.mode, router)
    assert ok is False and "markdown" in reason

    ok, reason = await _self_check(VIOLATING_VOICE_REPLY_TOO_LONG, session.mode, router)
    assert ok is False and "length" in reason


@pytest.mark.parametrize("text,language", PURE_LANGUAGE_CASES)
async def test_text_mode_length_compliance_markdown_allowed(text, language):
    session = SessionState(
        session_id="s", conversation_id="c", thread_id="t", mode=Mode.TEXT_TO_TEXT, response_language=language
    )
    router = LLMRouter([ScriptedProvider(["OK"])])

    ok, _ = await _self_check(COMPLIANT_TEXT_REPLY, session.mode, router)
    assert ok is True  # markdown present and still compliant — TEXT_MODE allows it

    ok, reason = await _self_check(VIOLATING_TEXT_REPLY_TOO_LONG, session.mode, router)
    assert ok is False and "length" in reason


@pytest.mark.parametrize("text,language", PURE_LANGUAGE_CASES)
def test_language_preservation_system_prompt_reflects_detected_language(text, language):
    """The system prompt actually sent to the model must name the detected
    language — a template regression (e.g. a typo in the substitution key)
    would silently break language preservation for every user."""
    session = SessionState(
        session_id="s", conversation_id="c", thread_id="t", mode=Mode.TEXT_TO_TEXT, response_language=language
    )
    system_prompt, _ = _build_system_prompt(session)
    assert f"is: {language}" in system_prompt


@pytest.mark.skipif(
    os.environ.get("MAAV_EVAL_LIVE", "").lower() != "true",
    reason="MAAV_EVAL_LIVE not set — set it (with a real provider key configured) to eval an actual deployed model",
)
@pytest.mark.parametrize("text,language", PURE_LANGUAGE_CASES)
async def test_live_model_compliance(text, language):
    """Real eval against whatever LLM_PROVIDER_ORDER resolves to. Not run by
    default — this hits a real, billed API. This is the tier that Tier 1
    cannot substitute for."""
    router = build_router_from_env()
    session = SessionState(
        session_id="s", conversation_id="c", thread_id="t", mode=Mode.SPEECH_TO_SPEECH, response_language=language
    )
    system_prompt, _ = _build_system_prompt(session)

    reply = await router.complete_with_fallback([{"role": "user", "content": text}], system=system_prompt)

    ok, reason = await _self_check(reply, session.mode, router)
    assert ok, f"[{language}] live model output failed compliance: {reason}\nreply={reply!r}"
