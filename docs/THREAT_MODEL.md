# MAAV / Mvoice — Threat Model (v1)

Scope: the backend (`agent_core.api`), the Speech Gateway
(`agent_core.speech_gateway`), and the LangGraph reasoning pipeline
(`agent_core.agents`). Frontend and infra hardening (TLS termination, WAF,
network segmentation) are deployment-time concerns noted as accepted risk
below, not solved in this codebase.

## In scope for this pass

- Unauthenticated/unauthorized access to `/chat` and the Speech Gateway's
  WebSocket routes.
- Prompt injection via any input channel (typed text, tool results,
  transcribed speech).
- Irreversible actions (write-scope tools) triggered by voice with no
  confirmation step a user can't accidentally talk past.
- PII in transcripts reaching logs/analytics unmasked.
- Raw audio persisted without consent.
- Cost-exhaustion via unbounded audio session creation.
- Malformed/spoofed audio uploads reaching Sarvam.
- LLM output containing HTML/script content or leaked system-prompt text
  reaching the client.
- Secrets (Sarvam/Azure/LLM provider keys, JWT signing secret) leaking into
  client-shipped code.

## Mitigated

| Risk | Mitigation | Where |
|---|---|---|
| Unauthenticated `/chat` access | JWT bearer auth required; 401 on missing/invalid/expired token | `security/auth.py`, wired into `api/main.py` |
| Write-scope tool called without proper role | `require_role()` RBAC dependency, checked server-side, not implied by a prompt | `security/auth.py` |
| Voice-triggered irreversible action with no real confirmation | Hard code gate: a write-scope tool call from a voice-mode turn is never executed on first request — a single-use, tool+args-scoped confirmation token is required to proceed. A model or fast talker cannot bypass this; it's not a prompt instruction | `security/confirmation.py`, `agents/task_agent.py` (`write_scope_tools`/`confirmation_gate` params) |
| Prompt injection via tool results | Tool output is wrapped in `<<UNTRUSTED_...>>` tags before it can reach the model — structural, not just a system-prompt suggestion | `agents/untrusted.py` |
| Prompt injection via typed OR transcribed user input | `run_turn` treats `user_message` as one untrusted string regardless of source; the system prompt's IDENTITY & SCOPE section instructs the model to disregard embedded instructions in user input generally, not per-channel | `agents/task_agent.py`, re-verified for the transcript channel in `tests/test_voice_transcript_injection.py` |
| System-prompt leakage / injected HTML in model output | Output sanitizer strips script/HTML tags and redacts any text matching real system-prompt section headers before the response is returned | `security/output_validation.py`, applied in `run_turn`'s return path |
| PII in transcripts hitting logs | Transcripts are masked (email/phone/card/ID patterns) before any log line | `security/pii.py`, applied in `speech_gateway/main.py`'s STT event loop |
| Audio persisted without consent | `SessionState.audio_retention_consent` defaults to `False`; a session-level guard raises unless explicitly set. Nothing in this codebase persists raw audio today — this is the gate anything that later wants to must call | `security/retention.py`, `supervisor/state.py` |
| Cost-exhaustion via unbounded voice session creation | Sliding-window rate limiter on `/ws/stt` connection accept, keyed by client IP; rejects with close code 1008 over the limit | `security/rate_limit.py`, `speech_gateway/main.py` |
| Malformed/spoofed audio reaching Sarvam | Magic-byte (WAV) / frame-shape (PCM16) validation on every frame, independent of any client-claimed content type or extension | `speech/audio_validation.py` (Phase 4), reused unchanged here |
| Secrets shipped to the client | Keys are read from env only inside `agent_core.llm_adapter`/`agent_core.speech`/`agent_core.speech_gateway` — never referenced in `frontend/`. CI test greps frontend source for key-shaped literals and known env-var names | `tests/test_frontend_no_secrets.py` |
| Cross-origin request abuse | `CORSMiddleware` on both FastAPI apps, explicit allow-list via `CORS_ALLOWED_ORIGINS` (empty/deny-all by default), not a wildcard | `api/main.py`, `speech_gateway/main.py` |

## Accepted risk for v1 (explicitly not solved here)

- **OAuth2 identity-provider integration.** `security/auth.py` is the
  resource-server side (verifies tokens); it does not implement an
  authorization-code exchange with a real IdP (Auth0/Okta/Cognito/etc.).
  Wiring a specific IdP is a deployment-time configuration exercise
  (issuer/audience/JWKS), not code that should live in this repo — but it is
  **not yet done**, so no real IdP is connected today.
- **Session-to-user binding.** An authenticated principal can currently call
  `/chat` with any `session_id`/`thread_id` — there's no check that the
  session belongs to that principal. Needs a sessions-ownership table, which
  doesn't exist yet.
- **CSRF.** Not implemented, and considered low-priority for v1: this API is
  bearer-token (`Authorization` header) authenticated, not cookie-session
  authenticated, which removes the classic CSRF vector (a forged
  cross-site form/fetch can't attach a bearer token it doesn't have). Revisit
  if a cookie-based session mechanism is ever added.
- **Rate limiter is single-process, in-memory.** Fine for one Speech Gateway
  replica; multiple replicas would each enforce their own independent limit
  (an attacker gets `replicas × limit`). Redis (already in docker-compose) is
  the documented upgrade path — not wired yet.
- **PII masking is regex-based, not NER.** It catches common shapes (email,
  phone, card-like, 12-digit ID numbers) and will miss context-dependent PII
  (names, addresses in free text). This is a deliberate cost/coverage
  tradeoff for v1, not a gap to silently rely on for compliance purposes.
- **No WAF / network-level protections.** DDoS mitigation, TLS termination,
  network segmentation between the backend and Speech Gateway, and secrets
  management (e.g. a real secrets manager vs. plain env vars) are
  deployment/infra concerns — `infra/k8s` and `infra/terraform` are
  explicitly labeled stubs, not production configuration.
- **Output sanitization is defense-in-depth, not primary.** It assumes the
  frontend already escapes rendered text by default (React does). If the
  frontend ever renders raw HTML from the model, re-audit this assumption.
- **No audit log of RBAC/auth decisions.** `require_role()` and
  `get_current_principal` raise on failure but don't yet write to a
  dedicated security audit trail — only whatever the ambient request logs
  capture.

## What could not be verified live in this sandbox

Same constraint as Phases 4-5: no network egress to a real IdP, no live
Sarvam/Azure account. Everything above is verified with real, deterministic
tests (`tests/test_auth.py`, `test_pii.py`, `test_output_validation.py`,
`test_rate_limit.py`, `test_confirmation.py`,
`test_task_agent_confirmation_gate.py`, `test_voice_transcript_injection.py`,
`test_frontend_no_secrets.py`, `test_retention.py`) using real signed JWTs
(via PyJWT) and fake providers/clients — not against a live IdP or a
production traffic pattern. Load-test the rate limiter and pen-test the auth
flow against a real IdP before shipping.
