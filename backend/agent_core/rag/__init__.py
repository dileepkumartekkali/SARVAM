"""Retrieval-Augmented Generation over mTouch Labs' own scraped website
content — lets the assistant answer from real company facts (services,
leadership, awards, etc.) instead of only what the base LLM was trained on.

Pieces, each independently testable:
  embeddings.py  -- BAAI/bge-m3 via Hugging Face's hosted Inference API
                     (free, no self-hosted model weights in this process)
  store.py       -- pgvector on the same Supabase Postgres already used by
                     chat_store.py (no new database/service)
The retrieval tool itself lives in agent_core/tools/rag_tool.py, next to the
other tools it's registered alongside.
"""
