"""One-time/manual ingestion: reads the scraper's rag_chunks.json, embeds
each chunk's text via BAAI/bge-m3 (agent_core/rag/embeddings.py), and
upserts into Postgres (agent_core/rag/store.py). Not part of any request
path -- run by hand whenever the scraped site content changes.

Usage (from backend/):
  python scripts/ingest_rag_chunks.py path/to/rag_chunks.json

Requires POSTGRES_DSN and HF_API_TOKEN set (.env or real env vars).
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from _dotenv import load_dotenv  # noqa: E402

load_dotenv()

from agent_core.rag import embeddings, store  # noqa: E402

# The scraper's blog-index-style pages ran to 25k+ chars in one "chunk" --
# low information density (mostly a list of post titles) and a real risk of
# exceeding bge-m3's usable context. Truncating a rare outlier beats writing
# a whole second chunking pass for content the scraper already chunked.
_MAX_CHUNK_CHARS = 6000

# Real gap: this loop used to have NO error handling at all -- a single
# transient failure (HF's free-tier 429 rate limit, a momentary 503 "model
# loading," a network blip) crashed the ENTIRE run, silently leaving only
# whatever chunks happened to finish before it as ingested, with no summary
# of what got skipped. upsert_chunk is idempotent by chunk_id, so re-running
# from scratch was always "safe" but wasteful (re-embeds everything that
# already succeeded). Now: one retry per chunk on a transient failure, then
# skip-and-continue so one bad chunk can't take the whole batch down, with a
# final summary of exactly which chunk_ids need attention.
_MAX_ATTEMPTS_PER_CHUNK = 2
_RETRY_DELAY_SECONDS = 1.0


async def _embed_with_retry(text: str) -> list[float]:
    last_error: embeddings.EmbeddingError | None = None
    for attempt in range(_MAX_ATTEMPTS_PER_CHUNK):
        try:
            return await embeddings.embed_text(text)
        except embeddings.EmbeddingError as e:
            last_error = e
            if not e.retriable or attempt == _MAX_ATTEMPTS_PER_CHUNK - 1:
                raise
            await asyncio.sleep(_RETRY_DELAY_SECONDS)
    raise last_error  # unreachable, satisfies static analysis


async def main(path: str) -> None:
    if not store.is_configured():
        raise SystemExit("POSTGRES_DSN not set -- nothing to ingest into.")
    if not embeddings.is_configured():
        raise SystemExit("HF_API_TOKEN not set -- can't embed anything.")

    chunks = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"Ingesting {len(chunks)} chunks from {path}...")

    failed: list[str] = []
    for i, chunk in enumerate(chunks, start=1):
        text = chunk["text"][:_MAX_CHUNK_CHARS]
        try:
            vector = await _embed_with_retry(text)
            await store.upsert_chunk(chunk["chunk_id"], chunk["page_url"], chunk["page_title"], text, vector)
            print(f"  [{i}/{len(chunks)}] {chunk['chunk_id']}")
        except embeddings.EmbeddingError as e:
            failed.append(chunk["chunk_id"])
            print(f"  [{i}/{len(chunks)}] FAILED: {chunk['chunk_id']} ({e})")

    if failed:
        print(f"\nDone, with {len(failed)} failure(s) -- re-run this script to retry just these (idempotent):")
        for chunk_id in failed:
            print(f"  {chunk_id}")
    else:
        print("\nDone -- all chunks ingested.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: python {sys.argv[0]} path/to/rag_chunks.json")
    asyncio.run(main(sys.argv[1]))
