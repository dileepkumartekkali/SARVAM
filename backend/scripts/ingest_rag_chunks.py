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


async def main(path: str) -> None:
    if not store.is_configured():
        raise SystemExit("POSTGRES_DSN not set -- nothing to ingest into.")
    if not embeddings.is_configured():
        raise SystemExit("HF_API_TOKEN not set -- can't embed anything.")

    chunks = json.loads(Path(path).read_text(encoding="utf-8"))
    print(f"Ingesting {len(chunks)} chunks from {path}...")

    for i, chunk in enumerate(chunks, start=1):
        text = chunk["text"][:_MAX_CHUNK_CHARS]
        vector = await embeddings.embed_text(text)
        await store.upsert_chunk(chunk["chunk_id"], chunk["page_url"], chunk["page_title"], text, vector)
        print(f"  [{i}/{len(chunks)}] {chunk['chunk_id']}")

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit(f"usage: python {sys.argv[0]} path/to/rag_chunks.json")
    asyncio.run(main(sys.argv[1]))
