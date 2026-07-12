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
