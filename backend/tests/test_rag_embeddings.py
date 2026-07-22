"""agent_core.rag.embeddings -- BAAI/bge-m3 via Hugging Face's Inference API.
Uses httpx.MockTransport (no real network call, no mocking library).
"""

import json

import httpx
import pytest

from agent_core.rag import embeddings


def _json_response(status: int, body) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(body).encode())


async def test_embed_text_returns_flat_vector_as_is(monkeypatch):
    monkeypatch.setenv("HF_API_TOKEN", "test-token")
    transport = httpx.MockTransport(lambda request: _json_response(200, [0.1, 0.2, 0.3]))

    vector = await embeddings.embed_text("hello", transport=transport)

    assert vector == [0.1, 0.2, 0.3]


async def test_embed_text_mean_pools_per_token_vectors(monkeypatch):
    monkeypatch.setenv("HF_API_TOKEN", "test-token")
    # Two "tokens", each a 2-dim vector -- mean-pooled to [2.0, 3.0].
    transport = httpx.MockTransport(lambda request: _json_response(200, [[1.0, 2.0], [3.0, 4.0]]))

    vector = await embeddings.embed_text("hello", transport=transport)

    assert vector == [2.0, 3.0]


async def test_embed_text_sends_bearer_token_and_correct_url(monkeypatch):
    monkeypatch.setenv("HF_API_TOKEN", "test-token")
    seen = {}

    def _capture(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        return _json_response(200, [1.0])

    transport = httpx.MockTransport(_capture)

    await embeddings.embed_text("hello", transport=transport)

    assert seen["auth"] == "Bearer test-token"
    assert seen["url"] == "https://router.huggingface.co/hf-inference/models/BAAI/bge-m3/pipeline/feature-extraction"


async def test_missing_token_is_non_retriable(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)

    with pytest.raises(embeddings.EmbeddingError) as exc_info:
        await embeddings.embed_text("hello")

    assert exc_info.value.retriable is False


async def test_503_and_429_are_retriable_other_errors_are_not(monkeypatch):
    monkeypatch.setenv("HF_API_TOKEN", "test-token")
    transport = httpx.MockTransport(lambda request: httpx.Response(503, content=b"loading"))

    with pytest.raises(embeddings.EmbeddingError) as exc_info:
        await embeddings.embed_text("hello", transport=transport)
    assert exc_info.value.retriable is True

    # Real gap: HF's free-tier rate limit (429) used to be treated as
    # permanent -- genuinely common under real load, not hypothetical.
    transport = httpx.MockTransport(lambda request: httpx.Response(429, content=b"rate limited"))
    with pytest.raises(embeddings.EmbeddingError) as exc_info:
        await embeddings.embed_text("hello", transport=transport)
    assert exc_info.value.retriable is True

    transport = httpx.MockTransport(lambda request: httpx.Response(401, content=b"bad token"))
    with pytest.raises(embeddings.EmbeddingError) as exc_info:
        await embeddings.embed_text("hello", transport=transport)
    assert exc_info.value.retriable is False


def test_is_configured_reflects_env_var(monkeypatch):
    monkeypatch.delenv("HF_API_TOKEN", raising=False)
    assert embeddings.is_configured() is False
    monkeypatch.setenv("HF_API_TOKEN", "test-token")
    assert embeddings.is_configured() is True
