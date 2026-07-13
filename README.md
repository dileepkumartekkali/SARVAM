# MAAV / Mvoice — multilingual voice agent

A multilingual (13 languages + code-mixing) assistant built on a LangGraph
supervisor, a multi-provider LLM router with fallback, Sarvam AI STT/TTS
(with an Azure fallback for the two languages Sarvam's TTS doesn't cover),
and a real, executable tool-calling loop. Two independently deployable
services — a **backend** (chat reasoning) and a **Speech Gateway** (the only
thing that talks to Sarvam/Azure) — plus a React frontend.

This README is organized by **what the system actually does**, not by build
order. For the phase-by-phase history and the reasoning behind specific
decisions, the code comments and `docs/` are more detailed than this file
tries to be.

---

## Total flow

### 1. Text→Text (fully wired, real end to end)

```
Client                Backend (/chat)              LLM providers
  |  POST /chat            |                             |
  |  (JWT bearer)           |                             |
  |------------------------>|                             |
  |                         | 1. verify JWT (401 if bad)  |
  |                         | 2. language_agent.detect_language(message)
  |                         |    - Unicode-script detection first (free, instant)
  |                         |    - romanized-keyword fallback
  |                         |    - LLM classification, only if nothing else matched
  |                         |                             |
  |                         | 3. confidence < 0.5?        |
  |                         |    -> YES: deterministic clarifying question,
  |                         |            task_agent's LLM is NEVER called
  |                         |    -> NO: continue                            |
  |                         | 4. task_agent.run_turn()                     |
  |                         |    - _build_system_prompt() loads the real   |
  |                         |      prompts/{text,voice}_mode_system.v1.txt |
  |                         |      template + a tool manifest if tools     |
  |                         |      are registered                          |
  |                         |    - loop: LLMRouter.complete_with_fallback()|
  |                         |------------------------------------------->  |
  |                         |      tries providers in LLM_PROVIDER_ORDER,  |
  |                         |      falls through on retriable errors only  |
  |                         |  <-------------------------------------------|
  |                         |    - model asks for a tool (TOOL_CALL: {...})|
  |                         |      -> write-scope tool + voice mode?       |
  |                         |         block, return pending_confirmation   |
  |                         |      -> else: execute the REAL tool          |
  |                         |         (agent_core/tools/), wrap its result |
  |                         |         in <<UNTRUSTED_...>> tags, loop again|
  |                         |    - budget exceeded (default 3 tool calls)? |
  |                         |      -> stop, return a clarifying question   |
  |                         |    - model returns plain text -> draft       |
  |                         | 5. _self_check(draft) — deterministic        |
  |                         |    length/markdown checks, then one more LLM |
  |                         |    call only if ambiguous                    |
  |                         | 6. sanitize_llm_output() — strip HTML/script,|
  |                         |    redact any accidental system-prompt leak  |
  |  <----------------------|                             |
  |  ChatResponse           |                             |
  |  (response, detected                                   |
  |   language, prompt_version,                            |
  |   pending_confirmation?)                                |
```

Every step above is real and covered by tests — see
`backend/tests/test_task_agent*.py`, `test_agentic_loop_real_tools.py`,
`test_language_agent.py`, `test_graph.py`.

### 2. Speech — STT and TTS (two independently real capabilities, not yet one continuous voice loop)

The Speech Gateway exposes two WebSocket routes, and **each one, on its
own, is real and tested**:

```
Client --(WSS, PCM16 audio frames)--> /ws/stt --(validated per-frame,       --> Sarvam STT
                                                  magic-byte/frame-shape)        (WS, retry-with-
                                                                                  backoff, then REST
                                                                                  fallback on drop)
              <-- transcript / VAD events -----------------------------------------|

Client --(WSS, text deltas)--> /ws/tts --(chunk_stream: sentence-boundary   --> Sarvam or Azure TTS
                                           aware, currency-safe, first       (ONE socket for the
                                           chunk capped small for fast TTFB)  whole utterance)
              <-- audio bytes ------------------------------------------------------|
```

**What is NOT wired together, honestly stated rather than implied:** nothing
in the live gateway routes connects "final transcript → send to backend
`/chat` → synthesize the response" into one continuous voice conversation.
The frontend's `useVoiceSession.js` hook drives the STT connection and the
visual state machine, but does not itself call `/chat`. `DuplexSession`
(`agent_core/speech_gateway/duplex_session.py`) — the barge-in state machine
that cancels an in-flight `task_agent` call and the active TTS socket the
instant a new `speech_start` VAD signal arrives mid-playback — is real,
unit-tested (`tests/test_duplex_session.py`, including the rapid-double-
barge-in edge case), and *would* be the right place to orchestrate this, but
**no route in `speech_gateway/main.py` actually instantiates or drives it**.
Building that orchestrating endpoint (STT transcript → `/chat` → chunker →
TTS, wired through `DuplexSession`) is the single biggest remaining gap
between "two solid independent capabilities" and "one real voice
conversation" — see [Known gaps](#known-gaps).

### 3. Auth flow

```
Client -> Supabase Auth (Google OAuth) -> {access_token} (JWT, HS256, sub=user UUID)

Client -> POST /chat, Authorization: Bearer <token> -> Backend
          get_current_principal() verifies signature/expiry/claims/audience (401 if invalid)
```

The backend never issues tokens itself — it only verifies JWTs an IdP signed
(`agent_core/security/auth.py`). Supabase is the IdP: `JWT_SIGNING_SECRET` is
the Supabase project's JWT secret, `JWT_AUDIENCE=authenticated` matches the
`aud` claim Supabase stamps into every token.

### 4. Persistent chat history, multi-conversation switching, audio replay

```
Client -> GET /conversations                       -> list, most-recently-active first
Client -> POST /conversations                       -> {id}  (start a new chat)
Client -> GET /conversations/{id}/messages          -> full history for that chat
Client -> POST /chat {conversation_id, thread_id, ...} -> conversation_id must be owned
                                                          by the caller (404 otherwise);
                                                          thread_id == conversation_id, so
                                                          LangGraph's own short-term memory
                                                          is isolated per conversation too
```

Backed by `agent_core/persistence/chat_store.py` — a dedicated Postgres table
(Supabase), separate from LangGraph's `MemorySaver` checkpointer (which stays
capped/pruned for prompt-size reasons, see `_MAX_HISTORY_MESSAGES`). Every
query is scoped by `user_id`, so one user's conversations are never visible
to another. Entirely optional at runtime: no `POSTGRES_DSN` means no
persistence (every `/chat` call still works statelessly), so local dev/CI
needs no database. Schema: `docs/supabase_schema.sql`.

TTS/STS replies are captured client-side (decoded PCM, re-encoded to one WAV
per turn — `frontend/src/api/ttsPlayback.js`) and uploaded to Supabase
Storage so a message's voice reply can be played again later, not just once
live (`frontend/src/components/MessageBubble.jsx`'s Play button).

### 5. Agentic tool-calling loop

```
User: "What's 17 times 23, and remind me to call mom?"
  -> task_agent sees a tool manifest in its system prompt listing
     get_current_datetime, calculate, convert_units, save_note, delete_note
     (agent_core/tools/builtin.py — real, executable, no mocks)
  -> model replies: TOOL_CALL: {"name": "calculate", "args": {"expression": "17*23"}}
  -> calculate() really evaluates it (safe AST walk, not eval()) -> "391"
  -> result wrapped in <<UNTRUSTED_TOOL_RESULT_CALCULATE>> tags, fed back
  -> model replies: TOOL_CALL: {"name": "save_note", "args": {"text": "call mom"}}
  -> save_note() really persists it (in-memory) -> "Saved as note #1."
  -> model's final answer: "17 times 23 is 391, and I've saved a note to call mom."
```

`delete_note` is marked write-scope — in a voice-originated turn, it's never
executed on first request; the loop returns a `pending_confirmation` token,
and only a matching, single-use, tool+args-scoped token (from
`security/confirmation.py`) resubmitted lets it actually run. This is a code
gate, not a prompt instruction. See `test_agentic_loop_real_tools.py` and
`test_task_agent_confirmation_gate.py`.

---

## Repository layout

```
backend/
  agent_core/
    api/             FastAPI backend: POST /chat, GET/POST /conversations, GET /health, GET /metrics
    agents/           language_agent (detection), task_agent (reasoning loop), cancellation, untrusted
    llm_adapter/      LLMProvider adapters (Grok/GPT/Sarvam/Claude/Gemini) + fallback router
    persistence/      chat_store.py — Postgres-backed conversation/message history (Supabase)
    tools/            real tools (registry.py, builtin.py) + system-prompt manifest generation
    speech/           Sarvam STT/TTS clients, response chunker, audio validation, TTS provider policy
    speech_gateway/   its own FastAPI app (own Dockerfile/container) — the only thing that
                      talks to Sarvam/Azure: /ws/stt, /ws/tts, DuplexSession (barge-in)
    supervisor/       SessionState, session state machine, the LangGraph itself
    security/         JWT auth (Supabase-issued, HS256 or JWKS/ES256)/RBAC, PII masking, output
                      sanitization, rate limiting, voice write-scope confirmation gate
    observability/    structured JSON logging, OpenTelemetry tracing, Prometheus metrics
  scripts/            run_dev_server.py (local dev), load_test.py, chaos_test.py — see below
  tests/              280+ tests (unconditional pass, 13 skip unless a real LLM key + a flag are set)
  prompts/            versioned system-prompt template files (text_mode / voice_mode) — lives
                      inside backend/ (not the repo root) so it's always inside the Docker
                      build context, whatever directory a deployment roots its build at
  Dockerfile, Dockerfile.gateway   multi-stage builds, one per service
frontend/
  src/
    api/              backend/gateway HTTP+WS clients, supabaseClient.js (auth/storage)
    components/       chat UI, voice orb (state-machine visual), login (Google OAuth),
                      language badge, ConversationDrawer (chat switcher), ProfileMenu
    hooks/            useVoiceSession (STT WS + visual state machine, TTS reply capture/upload)
    store/            Zustand store — see its file header for why Zustand, not Redux/Context
infra/
  k8s/                blue-green (backend) + canary-with-connection-draining (gateway) manifests
  terraform/          AWS (EKS/RDS/ElastiCache) — written, never applied (no cloud creds here)
  observability/      Prometheus alert rules
.github/workflows/     CI (tests, eval suite, Docker builds) + deploy (blue-green/canary)
docs/                 see the index below
```

## Quickstart

```bash
# Backend — real code, but /chat needs at least one LLM provider key to
# actually answer (falls through LLM_PROVIDER_ORDER; e.g. GROK_API_KEY), and
# a real Supabase-issued JWT to authenticate (see Auth flow above — the
# backend only verifies tokens, it doesn't issue them, so there's no local
# dev-login shortcut anymore).
cd backend
pip install -e ".[dev]"
python scripts/run_dev_server.py     # :8000
pytest                                # 280+ passed, 13 skipped

# Speech Gateway — needs SARVAM_API_KEY (and AZURE_SPEECH_KEY/_REGION for
# the Assamese/Urdu fallback) to actually reach Sarvam/Azure.
uvicorn agent_core.speech_gateway.main:gateway_app --port 8100

# Frontend
cd frontend
npm install
npm run dev                           # :5173
```

```bash
# A real turn, once a provider key + a Supabase project are set up (sign in
# via the frontend to get a real access_token, or mint a matching HS256 test
# token locally with JWT_SIGNING_SECRET for a quick curl check):
curl -X POST localhost:8000/chat -H "Content-Type: application/json" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -d '{"session_id":"s1","conversation_id":"c1","thread_id":"c1","message":"What is 17 times 23?"}'
```

```bash
# docker-compose (Docker not available in this dev environment — untested,
# but reflects the real service topology: backend + speech-gateway + redis + postgres)
docker-compose up
```

## Testing, eval, load, and chaos

```bash
cd backend
pytest -q                                        # full suite, no network/keys needed
pytest tests/eval/ -v                            # prompt-compliance eval (length, no-markdown-in-
                                                  # voice-mode, language-preservation), all 13 languages

# Load test (start both apps with MAAV_LOAD_TEST_MODE=true first — swaps
# real Sarvam/LLM clients for minimal-latency fakes; see docs/LOAD_TEST_REPORT.md)
python scripts/load_test.py --tiers 100 1000 10000

# Chaos test (starts/kills/restarts the gateway itself mid-session; backend
# must already be running with MAAV_LOAD_TEST_MODE=true)
python scripts/chaos_test.py
```

Metrics + tracing are wired into both apps with zero extra config for local
dev (`ConsoleSpanExporter`, no collector needed):

```bash
curl localhost:8000/metrics   # backend
curl localhost:8100/metrics   # speech gateway
```

## Security summary

JWT auth (real PyJWT signature verification) + RBAC, PII masking before any
transcript hits a log line, output sanitization (strips HTML/script,
redacts accidental system-prompt leakage), per-IP rate limiting on voice
session creation, magic-byte/frame-shape audio validation (never trusts a
client's claimed content type), and a code-level (not prompt-level)
confirmation gate on irreversible voice-triggered tool actions. Full
breakdown of what's mitigated vs. accepted risk: `docs/THREAT_MODEL.md`.

## Docs index

- [`docs/THREAT_MODEL.md`](docs/THREAT_MODEL.md) — security: mitigated vs.
  accepted risk for v1.
- [`docs/SAD.md`](docs/SAD.md) — architecture as built: component/sequence
  diagrams, DR strategy, cost optimization, consolidated known gaps.
- [`docs/LOAD_TEST_REPORT.md`](docs/LOAD_TEST_REPORT.md) — 100/1,000/10,000
  concurrent sessions, actually run. Found and fixed a real WebSocket
  double-close race bug; backend is CPU-bound, gateway is connection-bound
  (measured, not assumed); Sarvam's real ceiling is unmeasurable here.
- [`docs/CHAOS_TEST_REPORT.md`](docs/CHAOS_TEST_REPORT.md) — real process
  kill of the Speech Gateway mid-session; the backend's conversation state
  survives untouched (the gateway is genuinely stateless). Also documents
  what this does NOT prove (a backend/Postgres-checkpointer kill was not
  tested — no Docker/Postgres in this environment).
- [`docs/agent_system_prompt.md`](docs/agent_system_prompt.md) /
  [`docs/speech_to_speech_implementation_plan.md`](docs/speech_to_speech_implementation_plan.md) —
  the binding specs the code is built against.
- [`docs/supabase_schema.sql`](docs/supabase_schema.sql) — the conversations/
  messages tables + RLS policies + Storage bucket policy, run once in the
  Supabase project's SQL Editor.

## Known gaps

Consolidated once, not scattered — if a component depends on one of these,
its own docs say so too.

1. **No route ties STT → `/chat` → TTS into one continuous voice
   conversation.** The two capabilities are independently real; the
   orchestration (and `DuplexSession`'s live barge-in) is not wired into an
   actual endpoint. This is the top functional gap.
2. **LangGraph's own reasoning checkpointer (`MemorySaver`) is still
   in-memory, not Postgres-backed** — a real backend process death loses the
   agent's short-term working context (the model's last few turns of
   context for THIS reasoning session). This is distinct from user-facing
   chat history, which now **is** durable (`agent_core/persistence/
   chat_store.py`, Supabase Postgres) — a restart loses the agent's
   scratch-context, not the conversation the user sees.
3. **Rate limiter and the voice confirmation gate are both single-process.**
   Neither is shared across replicas — a real gap for any multi-replica
   deployment.
4. **Tool-calling is a placeholder text convention**
   (`TOOL_CALL: {...}` in the model's reply), not a provider's native
   function-calling API. `agent_core/tools/` itself is real; the *wire
   format* between model and loop is the still-placeholder part.
5. **CI/CD, Docker, and Terraform are written but unexecuted** — this repo
   had no git history before its production-readiness pass; no GitHub
   remote, Docker, or Terraform CLI exists in this development environment.
6. **Sarvam's real rate/concurrency ceiling is unknown** — not published in
   the docs consulted, and no live account exists here to measure it.
   Confirm directly with Sarvam before finalizing gateway autoscaling
   targets.
7. **Cross-service distributed tracing isn't linked** — spans are correct
   *within* each service; W3C trace-context propagation between the gateway
   and backend isn't wired.
8. **Grok (Groq-hosted Llama 3.3 70B) is not officially multilingual across
   all 13 Indic languages** — Meta's published support list for Llama 3.3
   covers English/French/German/Hindi/Italian/Portuguese/Spanish/Thai, not
   Telugu/Tamil/Kannada/Malayalam/Marathi/Gujarati/Punjabi/Bengali/Odia/
   Assamese/Urdu. Sarvam (purpose-built for all 13 + code-mixing) is the
   primary LLM provider precisely because of this gap; Grok is only reached
   if Sarvam's own LLM call fails, so this risk is rare in practice but real
   — a Sarvam outage could produce a lower-quality or English-language reply
   to a non-Hindi Indic-language question until Sarvam recovers.

# SARVAM
