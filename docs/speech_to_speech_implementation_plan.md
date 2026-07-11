# Speech-to-Speech Implementation Plan

Grounded against Sarvam AI's actual streaming STT/TTS documentation (verified July 2026), not assumed behavior. Where Sarvam's docs impose a constraint, the architecture is designed around that constraint rather than around an idealized API.

## 1. Confirmed Sarvam Platform Facts This Design Depends On

| Fact | Source behavior | Design implication |
|---|---|---|
| STT streaming = `wss://api.sarvam.ai/speech-to-text/ws`, model Saaras v3, modes: transcribe, translate, verbatim, translit, codemix | `mode="codemix"` exists as a first-class STT mode | Use `mode="codemix"` directly for code-mixed utterances instead of building custom code-mix detection on top of plain transcription |
| STT streaming formats: WAV or raw PCM only (pcm_s16le, pcm_l16, pcm_raw); PCM requires explicit `input_audio_codec`; 8kHz now supported (telephony), 16kHz standard | MP3/AAC/OGG etc. are batch/REST-only, not streaming | Client (mic capture) must encode to PCM16/WAV before the WebSocket — a hard client-side requirement, not server negotiation |
| VAD: `high_vad_sensitivity=True` gives ~0.5s silence-based end-of-speech detection; frame = 512 samples (32ms @16kHz) | Endpointing is server-assisted | Use Sarvam's VAD signals (`vad_signals=true`) as primary end-of-turn trigger; don't duplicate VAD client-side unless doing barge-in detection |
| TTS WebSocket has NO server-side cancel/clear message. Only convert, flush, ping, close | Interrupting an in-progress synthesis is impossible in-band | Barge-in is entirely a client responsibility: stop local playback + close socket + open a new one. Biggest architectural constraint on S2S — see §3 |
| TTS models: `bulbul:v2` (pitch/loudness/pace 0.3–3.0, 22050Hz) vs `bulbul:v3` (temperature control instead of pitch/loudness, pace 0.5–2.0, 24000Hz, preprocessing always on) | Different tuning knobs per model | Voice/persona config must be model-aware; don't pass pitch params if `bulbul:v3` is selected — silently ignored or rejected |
| TTS confirmed language list: Hindi, Bengali, Tamil, Telugu, Kannada, Malayalam, Marathi, Gujarati, Punjabi, Odia, English (Indian accent) | Assamese and Urdu NOT confirmed in this list | Risk, not a hand-wave: verify against current Sarvam docs/account before launch. If unsupported, define a fallback TTS provider (Azure/Google) for those two, routed transparently by the Language Agent |
| STT REST = sync, files < 30s. Batch = async, up to 2 hours. Streaming = real-time | Three genuinely different tools | Confirms mode-selection logic in `speech_agent.py` (Phase 1): streaming for live mic, batch for long uploads, REST for short clips |
| WebSockets "will occasionally drop... inspect the close code and reconnect with backoff" — Sarvam's own stated behavior | Disconnects are a normal operating condition | Reconnect-with-backoff is mandatory baseline, not nice-to-have (see §5) |

## 2. End-to-End Pipeline (Speech → Speech)

```
[Mic capture, client]
   → encode PCM16/WAV, 16kHz mono, 32ms frames
   → WebSocket to our Speech Gateway (not directly to Sarvam)

[Speech Gateway service]
   → proxies audio frames to Sarvam STT WS (vad_signals=true, high_vad_sensitivity=true)
   → on vad "speech_start": mark turn ACTIVE, notify client (used for local barge-in)
   → on vad "speech_end" / final transcript: turn closes, transcript handed to graph

[LangGraph supervisor]
   → language_agent: tag language (STT can hint via mode=codemix, but detection
     confidence is still independently scored — don't trust STT's language guess blindly)
   → task_agent: reasoning loop, tool calls, self-check (unchanged from Phase 1)
   → produces final_response_text

[Response chunker]
   → splits final_response_text into TTS-safe chunks (sentence-boundary,
     never mid-clause — see §4) BEFORE calling TTS, not after

[Speech Gateway → Sarvam TTS WS]
   → opens one TTS socket per utterance (not per chunk — reuses socket for
     sequential convert() calls within one reply)
   → streams audio chunks back to client as generated (TTFB optimization)

[Client playback]
   → plays audio chunks as they arrive
   → simultaneously keeps STT mic stream open, listening for speech_start (full-duplex) for barge-in
```

**Key decision:** the client does NOT talk to Sarvam directly. A Speech Gateway sits between for three reasons: (1) API keys never reach the client, (2) natural place to enforce audio validation/rate limits/session auth before spending Sarvam credits, (3) only place that can coordinate the barge-in close/reopen dance without exposing that complexity to every client platform (web, mobile, IVR).

## 3. Barge-In (Interruption Handling) — the hard part

Because Sarvam's TTS WebSocket cannot be told to stop mid-generation, correctness here is a **client + gateway protocol, not a server flag**:

1. Client keeps the STT mic stream always open during agent playback (full-duplex), even while TTS audio is playing.
2. The moment Sarvam STT's VAD emits `speech_start` while mid-playback:
   - Client stops local audio playback immediately and flushes its local audio buffer.
   - Gateway closes the active TTS socket. Chunks already in flight are discarded — expected, not an error.
   - Gateway sends `barge_in_detected` to the LangGraph session, which:
     - Cancels the in-flight LLM stream if the provider supports it (OpenAI/Anthropic/Gemini all do; confirm per-provider in the adapter).
     - Discards any `final_response_text` not yet spoken.
     - Transitions session state back to LISTENING.
3. A new TTS socket is opened for the next reply — sockets are single-utterance-lifetime, because there's no cancel primitive to reset them mid-stream.
4. Chunk TTS replies into short `convert()` calls (sentence-level) as Sarvam recommends — bounds how much generated audio gets thrown away on barge-in, bounding wasted cost and perceived interruption latency.

**Edge case:** rapid double barge-in. Session state machine must support re-entering LISTENING from any state, and the graph checkpointer (MemorySaver/Redis in prod) must not leave orphaned in-flight tool calls — the Task Agent's tool loop needs a cancellation token threaded through, not just "stop reading the stream."

## 4. Response Chunking for TTS (text side)

- Split on sentence boundaries with a language-aware sentence splitter, not naive `.` splitting — Indian-language punctuation and code-mixed sentences don't follow English boundary rules. Fall back to clause-level (comma/conjunction) only if one "sentence" exceeds ~25 words.
- Never split mid-word or mid-number ("₹4,500" must not become "₹4" + ",500").
- First chunk to TTS should be short (5–10 words) to minimize time-to-first-audio — stream the LLM response and chunk as sentence boundaries appear, don't chunk after the full response is generated.
- See `agent_system_prompt.md` for how response length is constrained upstream so chunking has less work.

## 5. Failure Modes & Handling Matrix

| Failure | Detection | Handling |
|---|---|---|
| STT WebSocket drops mid-utterance | Close code / connection error | Reconnect with exponential backoff; if reconnect exceeds ~2s, fall back to REST STT on buffered audio rather than losing the utterance |
| STT returns empty/low-confidence transcript | Confidence below threshold | Agent asks a spoken clarifying question in last-known language, not a silent retry loop |
| TTS socket fails to open / synthesis error | Connect/send error | Retry once with fresh socket; on second failure, fall back to text-only in client UI with a spoken "having trouble with audio" via a cached/pre-recorded clip (don't depend on the failing TTS) |
| LLM provider timeout mid-reasoning | Router's `LLMProviderError(retriable=True)` | `LLMRouter.complete_with_fallback` falls through provider order; if all fail, play a pre-recorded "let me get back to you" clip rather than dead air |
| Network interruption / client reconnects mid-session | WebSocket close on gateway | LangGraph checkpointer resumes from last committed node via `thread_id`; client reconnects with same `session_id`/`conversation_id` and gets in-progress turn state, not a reset |
| Malformed/oversized audio upload | Gateway-side validation before touching Sarvam | Reject before forwarding: MIME/magic-byte check (not just extension), max duration per chunk, max total session audio budget — cost/DoS control |
| User silence mid-session (idle) | No VAD `speech_start` for N seconds | Gateway sends keepalive/no-op per Sarvam idle-timeout guidance; after longer threshold (20–30s), agent may proactively check in once, then close gracefully |
| Overlapping/simultaneous speakers | Not solvable via VAD alone | V1 policy: primary/loudest channel only; document as known limitation, don't fake diarization in v1 |

## 6. Security Additions Specific to Speech-to-Speech

Everything from the Phase 1 text-agent security model applies (tool permission scopes, prompt-injection resistance, output validation). Additional:

- Audio is an untrusted input channel. A transcribed instruction ("ignore previous instructions and...") gets the same system-prompt-level resistance as typed text — transcription doesn't earn a trust upgrade.
- No raw audio persisted beyond the processing window unless the user explicitly consented to recording for QA/training — default ephemeral (buffer, transcribe, discard).
- PII masking applies to transcripts before they hit logs/analytics, same as text.
- Sarvam API keys live only in the Speech Gateway, never shipped to any client.
- Rate-limit audio session creation per user/IP — voice sessions cost more than text turns, so the cost-exhaustion abuse surface is larger.
- Voice spoofing / deepfake: if/when write-scope tools are enabled for voice, require a secondary confirmation (typed/tapped in client UI) before anything irreversible — don't let synthesized/replayed audio alone authorize a write.

## 7. Scalability Notes Specific to Speech

- Voice sessions are long-lived stateful WebSocket connections — changes autoscaling math from stateless HTTP. Scale the Gateway on concurrent active connections, not just CPU/memory, and use connection-draining rollouts (new connections to new pods; existing finish on old) — can't kill pods mid-conversation.
- STT/TTS calls are I/O-bound; use async concurrency (Phase 1's basis), not thread-per-connection, for concurrent-session density.
- TTS sockets are single-utterance-lifetime (§3) → high WebSocket churn to Sarvam at scale. Pool/reuse where allowed, and confirm Sarvam's concurrency/rate limits against projected concurrent sessions before committing to a traffic tier — an external hard ceiling.
- LangGraph checkpointer (Redis/Postgres in prod) must be sized for session-state churn at target concurrency (100/1k/10k/100k) — every turn writes a checkpoint.

## 8. Deferred to Later Phases

- Full diarization / multi-speaker handling.
- On-device/edge STT for offline fallback.
- Voice biometrics for auth (separate from the anti-spoofing confirmation step above).
- SSML-level prosody control beyond Sarvam's pace/pitch/temperature parameters.
