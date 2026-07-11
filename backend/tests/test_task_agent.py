"""Tool-call budget enforcement, self-check, and TEXT_MODE markdown/length —
all against fake providers (no network, no real API keys needed)."""

from agent_core.agents.task_agent import LLM_UNAVAILABLE_APOLOGY, CLARIFYING_QUESTION, _self_check, run_turn
from agent_core.llm_adapter import LLMRouter
from agent_core.llm_adapter.base import LLMProviderError
from agent_core.supervisor.state import Mode, SessionState

from ._fakes import InfiniteToolCallProvider, ScriptedProvider


def _session(mode=Mode.TEXT_TO_TEXT) -> SessionState:
    return SessionState(session_id="s", conversation_id="c", thread_id="t", mode=mode)


async def noop_tool(**kwargs) -> str:
    return "ok"


async def test_tool_budget_is_enforced_in_code_not_just_prompted():
    """A model that always asks for another tool call must be stopped by the
    budget, not by the prompt asking it nicely."""
    provider = InfiniteToolCallProvider()
    router = LLMRouter([provider])

    result = await run_turn(
        _session(), router, "loop forever", tools={"noop": noop_tool}, max_tool_calls=2
    )

    assert result.text == CLARIFYING_QUESTION
    assert result.tool_call_count == 2  # stopped exactly at budget, no further tool call made


async def test_tool_budget_of_zero_asks_immediately():
    provider = InfiniteToolCallProvider()
    router = LLMRouter([provider])

    result = await run_turn(_session(), router, "hi", tools={"noop": noop_tool}, max_tool_calls=0)

    assert result.text == CLARIFYING_QUESTION
    assert result.tool_call_count == 0


async def test_native_tool_call_with_null_args_does_not_crash():
    """A real bug hit live: a provider's native tool-calling returned `null`
    args for a zero-argument tool (json.loads("null") is a real None, not
    {}), and `tool_fn(**None)` crashed the whole turn with a raw 500."""
    from agent_core.llm_adapter.base import ToolCall

    from ._fakes import NativeToolCallProvider

    provider = NativeToolCallProvider(
        tool_calls_by_step=[[ToolCall(id="call_1", name="noop", args=None)], []],
        final_texts=["", "done"],
    )
    router = LLMRouter([provider])

    result = await run_turn(_session(), router, "what time is it", tools={"noop": noop_tool})

    assert result.tool_call_count == 1
    assert result.text == "done"


async def test_tool_exception_reported_as_tool_result_not_crash():
    async def broken_tool(**kwargs) -> str:
        raise RuntimeError("boom")

    provider = ScriptedProvider(['TOOL_CALL: {"name": "broken", "args": {}}', "Sorry, that failed."])
    router = LLMRouter([provider])

    result = await run_turn(_session(), router, "hi", tools={"broken": broken_tool}, max_tool_calls=3)

    assert result.tool_call_count == 1
    assert "failed" in result.text


async def test_unavailable_tool_reported_as_tool_result_not_crash():
    provider = ScriptedProvider(
        ['TOOL_CALL: {"name": "does_not_exist", "args": {}}', "Sorry, I couldn't find that."]
    )
    router = LLMRouter([provider])

    result = await run_turn(_session(), router, "hi", tools={}, max_tool_calls=3)

    assert result.tool_call_count == 1
    assert "couldn't find" in result.text


async def test_text_mode_allows_markdown_and_respects_length_cap():
    reply = "**Summary:** here's a concise reply with a list:\n- one\n- two"
    provider = ScriptedProvider([reply, "OK"])
    router = LLMRouter([provider])

    result = await run_turn(_session(Mode.TEXT_TO_TEXT), router, "give me a quick list")

    assert result.text == reply
    assert "**" in result.text  # markdown present — TEXT_MODE allows it
    assert len(result.text.split()) <= 160
    assert result.self_check_ok is True


async def test_self_check_flags_text_mode_length_violation():
    long_draft = "word " * 200
    router = LLMRouter([ScriptedProvider(["unused"])])

    ok, reason = await _self_check(long_draft, Mode.TEXT_TO_TEXT, router)

    assert ok is False
    assert "length" in reason


async def test_self_check_flags_voice_mode_markdown_violation():
    markdown_draft = "**This should never appear in speech**"
    router = LLMRouter([ScriptedProvider(["unused"])])

    ok, reason = await _self_check(markdown_draft, Mode.SPEECH_TO_SPEECH, router)

    assert ok is False
    assert "markdown" in reason


async def test_self_check_flags_romanized_script_in_voice_mode():
    """Real bug hit live: even with an explicit "use native script"
    instruction, the model sometimes answers in romanized script anyway (an
    instruction-following gap) — a native-script TTS voice mispronounces and
    drops words fed Latin-transliterated text. Caught deterministically, not
    left to the LLM reviewer, same as the markdown/length checks."""
    romanized_draft = "Python lo variables ante oka pేru, adi oka value ni store cheystundi."
    router = LLMRouter([ScriptedProvider(["unused"])])

    ok, reason = await _self_check(
        romanized_draft, Mode.SPEECH_TO_SPEECH, router, response_language="te", expected_script="native"
    )

    assert ok is False
    assert "script" in reason


async def test_self_check_flags_native_script_when_romanized_was_expected():
    """The symmetric case: a text-mode reply must mirror the user's own
    romanized script back, not silently switch to native script."""
    native_draft = "పైథాన్ లో వేరియబుల్స్ అంటే ఒక పేరు."
    router = LLMRouter([ScriptedProvider(["unused"])])

    ok, reason = await _self_check(
        native_draft, Mode.TEXT_TO_TEXT, router, response_language="te", expected_script="latin"
    )

    assert ok is False
    assert "script" in reason


async def test_self_check_allows_native_script_in_voice_mode():
    native_draft = "పైథాన్‌లో వేరియబుల్స్ అంటే విలువలను నిల్వ చేసే పేర్లు."
    router = LLMRouter([ScriptedProvider(["OK"])])

    ok, reason = await _self_check(native_draft, Mode.SPEECH_TO_SPEECH, router, response_language="te")

    assert ok is True


async def test_romanized_voice_reply_gets_corrected_to_native_script():
    """End-to-end: the deterministic script check feeds into the SAME
    bounded correction retry as any other violation."""
    romanized_draft = "Idi oka romanized reply, kani ikkada undali native script lo."
    native_draft = "ఇది స్థానిక లిపిలో ఒక సమాధానం."
    provider = ScriptedProvider([romanized_draft, native_draft, "OK"])
    router = LLMRouter([provider])

    session = _session(Mode.SPEECH_TO_SPEECH).model_copy(update={"response_language": "te"})
    result = await run_turn(session, router, "cheppu")

    assert result.text == native_draft
    assert result.self_check_ok is True


async def test_native_script_text_reply_gets_corrected_to_romanized():
    """The symmetric case, end-to-end: text mode must mirror the user's own
    romanized "Telugulish" input back — a native-script reply gets corrected
    by the SAME bounded retry, not just flagged and shipped anyway."""
    native_draft = "పైథాన్ లో వేరియబుల్స్ అంటే ఒక పేరు."
    romanized_draft = "Python lo variables ante oka peru."
    provider = ScriptedProvider([native_draft, romanized_draft, "OK"])
    router = LLMRouter([provider])

    session = _session(Mode.SPEECH_TO_TEXT).model_copy(update={"response_language": "te", "is_code_mixed": True})
    result = await run_turn(session, router, "naku python lo variables gurinchi explain cheyi")

    assert result.text == romanized_draft
    assert result.self_check_ok is True


async def test_self_check_passes_compliant_text_mode_draft():
    draft = "Here's a short, compliant answer."
    router = LLMRouter([ScriptedProvider(["OK"])])

    ok, reason = await _self_check(draft, Mode.TEXT_TO_TEXT, router)

    assert ok is True
    assert reason == ""


async def test_language_directive_is_prepended_to_the_first_user_message():
    """A short, explicit language instruction sits right next to the user's
    actual message — not just buried in the system prompt several hundred
    words earlier, which fast/small models follow less reliably."""
    provider = ScriptedProvider(["ఇది (Telugu reply)", "OK"])
    router = LLMRouter([provider])

    session = _session(Mode.TEXT_TO_TEXT).model_copy(
        update={"response_language": "te", "is_code_mixed": True}
    )
    await run_turn(session, router, "naku python gurinchi cheppu")

    first_call_messages = provider.messages_by_call[0]
    user_content = first_call_messages[0]["content"]
    assert "Telugu" in user_content
    assert "naku python gurinchi cheppu" in user_content


async def test_no_language_directive_when_language_unknown():
    provider = ScriptedProvider(["a reply", "OK"])
    router = LLMRouter([provider])

    await run_turn(_session(Mode.TEXT_TO_TEXT), router, "hello")

    first_call_messages = provider.messages_by_call[0]
    assert first_call_messages[0]["content"] == "hello"


async def test_directive_asks_for_romanized_script_when_user_typed_latin_script():
    """"Telugulish" (romanized Telugu-English, like the user typed) must get
    a reply in the SAME script — answering in fluent Telugu is only half
    right if it's in Telugu Unicode script the user never typed in."""
    provider = ScriptedProvider(["reply", "OK"])
    router = LLMRouter([provider])

    session = _session(Mode.TEXT_TO_TEXT).model_copy(update={"response_language": "te", "is_code_mixed": True})
    await run_turn(session, router, "naku python gurinchi cheppu")

    user_content = provider.messages_by_call[0][0]["content"]
    assert "romanized" in user_content
    assert "Telugulish" in user_content


async def test_directive_does_not_mention_script_when_user_typed_native_script():
    """Real Telugu Unicode input needs no script instruction — the model
    already defaults to native script, which is correct here."""
    provider = ScriptedProvider(["reply", "OK"])
    router = LLMRouter([provider])

    session = _session(Mode.TEXT_TO_TEXT).model_copy(update={"response_language": "te", "is_code_mixed": False})
    await run_turn(session, router, "పైథాన్ గురించి చెప్పు")

    user_content = provider.messages_by_call[0][0]["content"]
    assert "romanized" not in user_content


async def test_voice_mode_explicitly_demands_native_script_even_for_latin_input():
    """A real bug hit live TWICE: first, telling the model to answer in
    romanized "Telugulish" script (fine for TEXT, wrong for voice — a
    native-script TTS voice maps phonemes off proper Unicode and produced
    garbled/mispronounced, dropped words when fed Latin-transliterated
    text). Then, simply OMITTING that instruction for voice mode still
    wasn't enough — verified live the model kept defaulting to romanized
    script anyway, matching the user's own input. Voice mode needs an
    EXPLICIT native-script instruction, not just the absence of the
    romanized one."""
    provider = ScriptedProvider(["reply", "OK"])
    router = LLMRouter([provider])

    session = _session(Mode.SPEECH_TO_SPEECH).model_copy(
        update={"response_language": "te", "is_code_mixed": True}
    )
    await run_turn(session, router, "naku python gurinchi cheppu")

    user_content = provider.messages_by_call[0][0]["content"]
    assert "native script" in user_content
    assert "SPOKEN" in user_content
    assert "Telugu" in user_content  # still gets the language instruction, just no script override


async def test_voice_mode_gets_a_brevity_directive_text_mode_does_not():
    provider = ScriptedProvider(["reply", "OK"])
    router = LLMRouter([provider])

    voice_session = _session(Mode.TEXT_TO_SPEECH)
    await run_turn(voice_session, router, "explain something")
    voice_content = provider.messages_by_call[0][0]["content"]
    assert "short" in voice_content and "simple" in voice_content

    provider2 = ScriptedProvider(["reply", "OK"])
    router2 = LLMRouter([provider2])
    text_session = _session(Mode.TEXT_TO_TEXT)
    await run_turn(text_session, router2, "explain something")
    text_content = provider2.messages_by_call[0][0]["content"]
    assert text_content == "explain something"  # no directive at all — plain English, text mode


async def test_self_check_failure_triggers_one_bounded_correction_retry():
    """A detected violation must actually change the shipped answer — not
    just get flagged and ignored (the bug: self_check_ok=False went out
    unchanged for a mismatched-language reply)."""
    violating_draft = "This is a perfectly fine-looking answer in the wrong language."
    corrected_draft = "यह एक सही उत्तर है।"
    provider = ScriptedProvider([violating_draft, "VIOLATION: wrong language", corrected_draft, "OK"])
    router = LLMRouter([provider])

    # Native-script (Devanagari) input, not romanized — keeps this test
    # isolated to the language violation it's about; romanized input would
    # also trigger the separate script-expectation check (its own test).
    session = _session(Mode.TEXT_TO_TEXT).model_copy(update={"response_language": "hi"})
    result = await run_turn(session, router, "हिंदी में जवाब दो")

    assert result.text == corrected_draft
    assert result.self_check_ok is True
    correction_call_messages = provider.messages_by_call[2]
    assert any("wrong language" in str(m.get("content", "")) for m in correction_call_messages)


async def test_self_check_correction_failure_keeps_original_draft():
    """If the correction call itself fails, ship the original (flagged)
    draft rather than crash — one retry attempt, never more."""
    violating_draft = "Still the wrong language, but this is all we have."
    provider = ScriptedProvider([violating_draft, "VIOLATION: wrong language"])
    provider.complete = _fail_on_third_call(provider.complete)
    router = LLMRouter([provider])

    result = await run_turn(_session(Mode.TEXT_TO_TEXT), router, "hi")

    assert result.text == violating_draft
    assert result.self_check_ok is False


def _fail_on_third_call(original):
    calls = {"n": 0}

    async def wrapper(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise LLMProviderError("boom", retriable=False, provider="scripted")
        return await original(*args, **kwargs)

    return wrapper


async def test_llm_provider_error_returns_apology_not_a_crash():
    """Every configured provider failing (e.g. a malformed spontaneous tool
    call the API itself rejects) must degrade to a fixed apology, never an
    uncaught exception surfacing as a raw 500 to the client."""

    class AlwaysFailingProvider:
        name = "broken"

        async def complete(self, *args, **kwargs):
            raise LLMProviderError("boom", retriable=False, provider="broken")

        async def stream(self, *args, **kwargs):
            raise LLMProviderError("boom", retriable=False, provider="broken")
            yield  # pragma: no cover — unreachable, satisfies async generator shape

        async def complete_with_tools(self, *args, **kwargs):
            raise LLMProviderError("boom", retriable=False, provider="broken")

    router = LLMRouter([AlwaysFailingProvider()])
    result = await run_turn(_session(), router, "hello")

    assert result.text == LLM_UNAVAILABLE_APOLOGY
    assert result.self_check_ok is True
