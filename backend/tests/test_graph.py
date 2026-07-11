import pytest

from agent_core.llm_adapter import LLMRouter
from agent_core.supervisor.graph import build_text_graph
from agent_core.supervisor.state import Mode, SessionState

from ._fakes import PoisonProvider, ScriptedProvider
from ._language_fixtures import PURE_LANGUAGE_CASES


async def test_text_graph_end_to_end():
    provider = ScriptedProvider(["Hello! Here's your answer.", "OK"])
    router = LLMRouter([provider])
    graph = build_text_graph(router)
    session = SessionState(session_id="s", conversation_id="c", thread_id="thread-1", mode=Mode.TEXT_TO_TEXT)

    result = await graph.ainvoke(
        {"session": session, "user_message": "hi"},
        config={"configurable": {"thread_id": "thread-1"}},
    )

    assert result["response"] == "Hello! Here's your answer."
    assert result["prompt_version"] == "text_mode_system.v1"
    assert result["tool_call_count"] == 0
    assert result["self_check_ok"] is True
    assert result["session"].response_language == "en"


async def test_low_confidence_language_short_circuits_before_task_agent_llm():
    """Ambiguous input must never reach task_agent's LLM — the poison provider
    raises if it's ever called. No language_router is given either, so
    detection falls through to its deterministic low-confidence default."""
    poison = PoisonProvider()
    graph = build_text_graph(LLMRouter([poison]), language_router=None)
    session = SessionState(session_id="s", conversation_id="c", thread_id="t2", mode=Mode.TEXT_TO_TEXT)

    result = await graph.ainvoke(
        {"session": session, "user_message": "hmm"},
        config={"configurable": {"thread_id": "t2"}},
    )

    assert poison.calls == 0
    assert result["prompt_version"] == "low_confidence_clarify"
    assert "clarify" in result["response"].lower() or "?" in result["response"]


async def test_conversation_history_carries_across_turns_on_same_thread():
    """The agent must remember prior turns of the same thread — without this
    every turn was stateless and it couldn't answer "what did I just ask
    you?". History persists via the graph's checkpointer, keyed by
    thread_id."""
    provider = ScriptedProvider(["Nice to meet you, Ravi!", "OK", "Your name is Ravi.", "OK"])
    graph = build_text_graph(LLMRouter([provider]))
    session = SessionState(session_id="s", conversation_id="c", thread_id="mem-1", mode=Mode.TEXT_TO_TEXT)
    config = {"configurable": {"thread_id": "mem-1"}}

    await graph.ainvoke({"session": session, "user_message": "My name is Ravi, hello there friend"}, config=config)
    await graph.ainvoke({"session": session, "user_message": "What is my name, do you remember it?"}, config=config)

    # Call 2 of the provider (index 2 — 0 is turn 1's generation, 1 its
    # self-check) must have received turn 1's exchange as real messages.
    second_turn_messages = provider.messages_by_call[2]
    contents = [str(m.get("content")) for m in second_turn_messages]
    assert any("My name is Ravi" in c for c in contents)
    assert any("Nice to meet you, Ravi!" in c for c in contents)
    assert any("What is my name" in c for c in contents)


async def test_history_never_bleeds_across_different_threads():
    provider = ScriptedProvider(["Reply one.", "OK", "Reply two.", "OK"])
    graph = build_text_graph(LLMRouter([provider]))
    session_a = SessionState(session_id="s", conversation_id="c", thread_id="iso-a", mode=Mode.TEXT_TO_TEXT)
    session_b = SessionState(session_id="s", conversation_id="c", thread_id="iso-b", mode=Mode.TEXT_TO_TEXT)

    await graph.ainvoke(
        {"session": session_a, "user_message": "secret thing only thread A knows"},
        config={"configurable": {"thread_id": "iso-a"}},
    )
    await graph.ainvoke(
        {"session": session_b, "user_message": "hello from thread B"},
        config={"configurable": {"thread_id": "iso-b"}},
    )

    thread_b_messages = provider.messages_by_call[2]
    contents = [str(m.get("content")) for m in thread_b_messages]
    assert not any("secret thing" in c for c in contents)


@pytest.mark.parametrize("text,expected_lang", PURE_LANGUAGE_CASES)
async def test_graph_threads_detected_language_into_task_agent_system_prompt(text, expected_lang):
    """Proves language_agent's detection reaches task_agent's system prompt for
    all 13 languages — the strongest test possible without a live LLM (see
    tests/test_prompt_injection.py's docstring for the same honest caveat)."""
    provider = ScriptedProvider(["A reasonable reply.", "OK"])
    graph = build_text_graph(LLMRouter([provider]))
    session = SessionState(session_id="s", conversation_id="c", thread_id="t3", mode=Mode.TEXT_TO_TEXT)

    await graph.ainvoke(
        {"session": session, "user_message": text},
        config={"configurable": {"thread_id": "t3"}},
    )

    # call 0 is task_agent's main generation; call 1 is the self-check critique
    # (which gets a different, fixed system prompt) — check the right one.
    assert f"is: {expected_lang}" in provider.systems[0]
