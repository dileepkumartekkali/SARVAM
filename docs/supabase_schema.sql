-- Run once in the Supabase project's SQL Editor.
-- Backing store for agent_core/persistence/chat_store.py (persistent chat
-- history) and TTS/STS audio replay (Supabase Storage, policy below).

create table conversations (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table messages (
  id uuid primary key default gen_random_uuid(),
  conversation_id uuid not null references conversations(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  role text not null check (role in ('user', 'assistant')),
  content text not null,
  audio_path text,              -- Storage object path; null until a reply's audio is saved
  response_language text,
  created_at timestamptz not null default now()
);

alter table conversations enable row level security;
alter table messages enable row level security;

-- The backend connects directly (trusted server-side connection, not
-- through PostgREST) and scopes every query by user_id itself — these
-- policies are defense in depth for any other access path (e.g. if the
-- Supabase REST API is ever used directly from a client).
create policy "own conversations" on conversations for all
  using (auth.uid() = user_id) with check (auth.uid() = user_id);
create policy "own messages" on messages for all
  using (auth.uid() = user_id) with check (auth.uid() = user_id);

-- ---------------------------------------------------------------------
-- Storage: create a private bucket named `tts-audio` in the dashboard
-- (Storage -> New bucket -> uncheck "Public"), then run this policy so
-- each user can only read/write their own folder (path = `{user_id}/...`):
-- ---------------------------------------------------------------------
create policy "own audio" on storage.objects for all
  using (bucket_id = 'tts-audio' and auth.uid()::text = (storage.foldername(name))[1])
  with check (bucket_id = 'tts-audio' and auth.uid()::text = (storage.foldername(name))[1]);

-- ---------------------------------------------------------------------
-- RAG: mTouch Labs website knowledge base (agent_core/rag/store.py).
-- Embeddings are BAAI/bge-m3 (1024 dims) via Hugging Face's hosted
-- Inference API -- see agent_core/rag/embeddings.py. No user_id/RLS here:
-- this is shared company knowledge, not per-user private data.
-- ---------------------------------------------------------------------
create extension if not exists vector;

create table rag_chunks (
  id uuid primary key default gen_random_uuid(),
  chunk_id text not null unique,   -- scraper's own id, e.g. "https://www.mtouchlabs.com#0"
  page_url text not null,
  page_title text not null,
  text text not null,
  embedding vector(1024) not null,
  created_at timestamptz not null default now()
);

-- ivfflat needs rows present to pick good cluster centers -- fine at this
-- table's real size (~100 chunks); re-run `reindex` after a large re-scrape
-- if retrieval quality noticeably drops.
create index rag_chunks_embedding_idx on rag_chunks
  using ivfflat (embedding vector_cosine_ops) with (lists = 10);
