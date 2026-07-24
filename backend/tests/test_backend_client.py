"""speech_gateway.backend_client — the HTTP bridge to the main backend's
/chat. Uses httpx.MockTransport (no real network, no mocking library) by
swapping the module's singleton _http_client for one built on a mock
transport."""

import json

import httpx
import pytest

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider

import agent_core.speech_gateway.backend_client as backend_client_module
from agent_core.speech_gateway.backend_client import BackendChatError, call_backend_chat

# propagate.inject() is a no-op with no active span (an invalid/all-zero
# span context is deliberately never injected) -- the real call site
# (speech_gateway/main.py's think(), wrapped in start_span("converse.think"))
# always has one active, so this test needs the same to exercise the real path.
trace.set_tracer_provider(TracerProvider())
_tracer = trace.get_tracer("test")


def _json_response(status: int, body: dict) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(body).encode())


async def _call(**overrides):
    kwargs = {
        "message": "hello",
        "session_id": "s",
        "conversation_id": "c",
        "thread_id": "t",
        "language": "en",
        "auth_token": "tok",
    }
    kwargs.update(overrides)
    return await call_backend_chat(**kwargs)


async def test_successful_call_returns_the_reply(monkeypatch):
    transport = httpx.MockTransport(lambda request: _json_response(200, {"response": "hi there"}))
    monkeypatch.setattr(backend_client_module, "_http_client", httpx.AsyncClient(transport=transport))

    reply = await _call()

    assert reply.text == "hi there"


async def test_injects_w3c_trace_context_header(monkeypatch):
    """Architecture gap closed: a voice turn's trace used to be split into
    two disconnected traces at this process boundary -- nothing here ever
    told the backend which trace it belonged to."""
    seen = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen["traceparent"] = request.headers.get("traceparent")
        return _json_response(200, {"response": "hi"})

    transport = httpx.MockTransport(_capture)
    monkeypatch.setattr(backend_client_module, "_http_client", httpx.AsyncClient(transport=transport))

    with _tracer.start_as_current_span("test_span"):
        await _call()

    assert seen["traceparent"] is not None
    assert seen["traceparent"].startswith("00-")  # W3C traceparent version byte


async def test_forwards_auth_token_and_response_language(monkeypatch):
    def _capture(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer real-token"
        return _json_response(200, {"response": "hi", "response_language": "te"})

    transport = httpx.MockTransport(_capture)
    monkeypatch.setattr(backend_client_module, "_http_client", httpx.AsyncClient(transport=transport))

    reply = await _call(auth_token="real-token")

    assert reply.response_language == "te"


async def test_retries_once_on_a_5xx_then_succeeds(monkeypatch):
    """Architecture gap closed: a transient backend failure (cold start, a
    momentary blip, a 502/503 during a rolling deploy) used to go straight
    to the fixed apology after the full timeout, no retry at all."""
    calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, content=b"temporarily unavailable")
        return _json_response(200, {"response": "recovered"})

    transport = httpx.MockTransport(_handler)
    monkeypatch.setattr(backend_client_module, "_http_client", httpx.AsyncClient(transport=transport))
    monkeypatch.setattr(backend_client_module, "_RETRY_DELAY_SECONDS", 0)  # don't actually wait in tests

    reply = await _call()

    assert reply.text == "recovered"
    assert calls["n"] == 2


async def test_gives_up_after_max_attempts_on_persistent_5xx(monkeypatch):
    transport = httpx.MockTransport(lambda request: httpx.Response(503, content=b"still down"))
    monkeypatch.setattr(backend_client_module, "_http_client", httpx.AsyncClient(transport=transport))
    monkeypatch.setattr(backend_client_module, "_RETRY_DELAY_SECONDS", 0)

    with pytest.raises(BackendChatError):
        await _call()


async def test_does_not_retry_a_non_retriable_4xx(monkeypatch):
    """A 401/400 is a real rejection, not a transient hiccup -- retrying it
    would just waste time before the same fixed apology anyway."""
    calls = {"n": 0}

    def _handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(401, content=b"unauthorized")

    transport = httpx.MockTransport(_handler)
    monkeypatch.setattr(backend_client_module, "_http_client", httpx.AsyncClient(transport=transport))

    with pytest.raises(BackendChatError):
        await _call()

    assert calls["n"] == 1
