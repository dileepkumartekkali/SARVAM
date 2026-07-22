"""BAAI/bge-m3 embeddings via Hugging Face's hosted Inference API.

Free, cross-lingual (retrieval works directly on a query in any of the 13
supported languages against the English-only knowledge base -- no
translation step; see the session's own live verification against
huggingface.co/BAAI/bge-m3's "feature-extraction" pipeline, which is what
this calls, NOT the "sentence-similarity" pipeline shown in that model
page's default demo widget -- similarity scores against a fixed candidate
list are useless for retrieval, we need the raw vector to store/compare).

Called out-of-process (HF's servers do the actual model inference) rather
than loading the ~2GB of model weights into this backend's own process --
same reasoning as every other LLM provider adapter in this codebase: keep
the backend's memory footprint and crash-risk independent of the model.
"""

from __future__ import annotations

import os

import httpx

_DEFAULT_MODEL = "BAAI/bge-m3"
_DEFAULT_BASE_URL = "https://router.huggingface.co/hf-inference/models"
EMBEDDING_DIM = 1024  # bge-m3's fixed output size -- must match the pgvector column


class EmbeddingError(Exception):
    def __init__(self, message: str, *, retriable: bool = True):
        super().__init__(message)
        self.retriable = retriable


def is_configured() -> bool:
    return bool(os.environ.get("HF_API_TOKEN"))


def _api_token() -> str:
    token = os.environ.get("HF_API_TOKEN")
    if not token:
        raise EmbeddingError("HF_API_TOKEN not set", retriable=False)
    return token


def _pool_if_needed(vector: list) -> list[float]:
    """HF's generic feature-extraction pipeline returns either an already-
    pooled sentence vector (flat list of floats) or per-token vectors (a
    list of lists) depending on how the model repo is configured -- mean-
    pool across tokens in the latter case rather than assume one shape."""
    if not vector:
        raise EmbeddingError("embedding response was empty", retriable=False)
    if isinstance(vector[0], (int, float)):
        return [float(x) for x in vector]
    n_tokens = len(vector)
    dim = len(vector[0])
    return [sum(tok[i] for tok in vector) / n_tokens for i in range(dim)]


async def embed_text(
    text: str,
    *,
    model: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    transport: httpx.BaseTransport | None = None,
) -> list[float]:
    """A single text -> its bge-m3 embedding vector (length EMBEDDING_DIM).
    Used for both ingestion (embedding each knowledge-base chunk once) and
    retrieval (embedding the user's query, in whatever language, every time
    the RAG tool is called)."""
    model = model or os.environ.get("HF_EMBEDDING_MODEL", _DEFAULT_MODEL)
    base_url = (base_url or os.environ.get("HF_EMBEDDING_BASE_URL", _DEFAULT_BASE_URL)).rstrip("/")
    url = f"{base_url}/{model}/pipeline/feature-extraction"
    headers = {"Authorization": f"Bearer {_api_token()}"}

    async with httpx.AsyncClient(timeout=timeout, transport=transport) as client:
        resp = await client.post(url, json={"inputs": text}, headers=headers)
        if resp.status_code != 200:
            # 503 means the model is "loading" on HF's serverless
            # infrastructure -- a transient cold-start. 429 is the free
            # tier's rate limit -- also transient, and genuinely common
            # under real load, not a hypothetical (real gap: this used to
            # mark only 503 as retriable, so a 429 failed permanently with
            # no retry even though waiting a moment would likely succeed).
            # Anything else (401 bad token, 404 model not hosted) is not.
            raise EmbeddingError(
                f"HF embedding request failed: {resp.status_code} {resp.text}",
                retriable=resp.status_code in (503, 429),
            )
        return _pool_if_needed(resp.json())
