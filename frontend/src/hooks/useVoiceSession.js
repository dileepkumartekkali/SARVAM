import { useCallback, useEffect, useRef } from "react";
import { streamChatMessage } from "../api/chatClient";
import { MicCapture } from "../api/micCapture";
import { supabase } from "../api/supabaseClient";
import { TTSPlayer } from "../api/ttsPlayback";
import { VoiceSocketClient } from "../api/voiceSocket";
import { ConnectionState, Mode, VoiceState, useAppStore } from "../store/useAppStore";

const TTS_AUDIO_BUCKET = "tts-audio";

// The speech-gateway is its own Render service and sleeps independently of
// the backend — a cold start (or any network hiccup) meant the TTS socket's
// "open"/"close" events could simply never fire, and with no timeout
// anywhere, `await`ing them hung the whole turn forever: the composer stayed
// disabled permanently since the promise chain that resets it never
// resolved. Real bug hit live. These bound how long a stuck socket can block.
const TTS_OPEN_TIMEOUT_MS = 8000;
const TTS_CLOSE_TIMEOUT_MS = 20000;

function withTimeout(promise, ms) {
  return Promise.race([promise, new Promise((resolve) => setTimeout(resolve, ms))]);
}

/** Uploads a finished TTS reply's audio so it can be replayed later, and
 * attaches it to the message row (RLS-scoped update — same pattern as the
 * upload path itself, no backend round-trip needed for either). Best-effort:
 * a failed upload should never break the (already-played) voice reply. */
async function saveReplyAudio({ userId, messageServerId, blob, onSaved }) {
  if (!userId || !messageServerId || !blob) return;
  const path = `${userId}/${messageServerId}.wav`;
  const { error } = await supabase.storage.from(TTS_AUDIO_BUCKET).upload(path, blob, { contentType: "audio/wav" });
  if (error) return;
  await supabase.from("messages").update({ audio_path: path }).eq("id", messageServerId);
  onSaved?.(messageServerId, path);
}

// THE one stop rule: 2 seconds of continuous silence ends listening —
// whether the user never spoke at all, or spoke and then went quiet. Every
// voiced frame resets this timer. Deliberately NOT stopping on Sarvam's
// per-segment `speech_end` anymore: that fires on brief mid-sentence pauses
// too, and cutting there ends the question early. Instead, transcript
// segments ACCUMULATE across pauses and everything said gets joined into
// one question when the 2s of real silence finally lands. Applies uniformly
// to Speech-to-Text and Speech-to-Speech — both run through this same hook.
const SILENCE_STOP_MS = 2000;

// After the 3s silence stop, the LAST speech segment's transcript may still
// be in flight from Sarvam (it follows speech_end by up to ~1-2s). The
// socket stays open this much longer to catch it — so what the user said is
// never silently dropped.
const FINAL_TRANSCRIPT_GRACE_MS = 2000;

// Local energy gate deciding "is this frame voice?" — ADAPTIVE, bounded on
// BOTH ends. Two real failures shaped this: (1) a fixed threshold missed a
// quiet mic entirely (no socket ever opened); (2) an absolute "anything
// above 0.02 is voice" fast path meant a noisy room / autoGainControl-
// amplified ambient hum counted EVERY frame as voice, so the 3s silence
// timer reset forever and the turn never concluded — "no response, no
// errors". The threshold now derives only from the session's own measured
// noise floor, clamped into [MIN_VOICE_RMS .. NOISE_FLOOR_CAP x MULTIPLIER]
// = [0.006 .. 0.04]: quiet rooms detect soft speech, noisy rooms don't
// treat their own hum as speech, and speech through autoGainControl
// (0.05-0.3 typical) clears 0.04 comfortably.
const MIN_VOICE_RMS = 0.006; // below this it's noise on ANY mic, never voice
const NOISE_FLOOR_CAP = 0.01; // floor estimate never exceeds this (speaking-from-frame-1 can't inflate it)
const NOISE_FLOOR_MULTIPLIER = 4; // voiced = this many times the (capped) measured floor
const CONSECUTIVE_VOICED_FRAMES_TO_OPEN = 2; // ~64ms — debounces clicks/pops

// Listening can never run unbounded: even if a noisy environment keeps
// resetting the silence timer, the session concludes here — whatever was
// transcribed gets finalized and sent rather than hanging forever.
const MAX_LISTEN_MS = 30000;

// How often the live level line is printed while listening (console.info —
// console.debug is HIDDEN by Chrome's default log level, which made the
// previous detection log invisible exactly when it was needed in the field).
const LEVEL_LOG_INTERVAL_MS = 2000;

// ~0.5s of frames kept locally BEFORE voice is detected, flushed to Sarvam
// once it is — so the start of the first word isn't clipped by the gate.
const PRE_ROLL_MAX_FRAMES = 16;

function frameRMS(arrayBuffer) {
  const view = new DataView(arrayBuffer);
  const sampleCount = arrayBuffer.byteLength / 2;
  if (sampleCount === 0) return 0;
  let sumSquares = 0;
  for (let i = 0; i < sampleCount; i++) {
    const s = view.getInt16(i * 2, true) / 0x8000;
    sumSquares += s * s;
  }
  return Math.sqrt(sumSquares / sampleCount);
}

/**
 * Deliberately NOT a persistent full-duplex session: the STT socket opens
 * only while actively listening for one utterance and closes the instant a
 * final transcript (or an error) arrives; the TTS socket opens only to speak
 * one reply and closes right after. No socket is ever left open idle. This
 * costs one barge-in-while-speaking capability (the mic isn't listening
 * during TTS playback) in exchange for using measurably less of Sarvam's
 * metered STT/TTS streaming time — the explicit tradeoff asked for. Full
 * duplex + barge-in already exists server-side (speech_gateway's
 * `/ws/converse`) if that tradeoff ever needs to flip.
 */
export function useVoiceSession({ token, ids, onUnauthorized }) {
  const micRef = useRef(null);
  const sttSocketRef = useRef(null);
  const ttsPlayerRef = useRef(null);
  const silenceTimerRef = useRef(null);
  const graceTimerRef = useRef(null);
  const maxListenTimerRef = useRef(null);
  // Points at the active listening session's `conclude()` (defined inside
  // `startListening`) so a manual second tap can trigger the exact same
  // "finalize whatever was said" path the silence timeout uses, instead of
  // the raw teardown `stopListening()` below (which discards the transcript
  // — see `toggle()`). `null` whenever nothing is listening.
  const concludeRef = useRef(null);
  // Points at the currently in-flight speak session (see `openSpeakSession`),
  // so a mode change, conversation switch, or unmount can stop its audio —
  // `null` whenever nothing is actively speaking.
  const activeSpeakSessionRef = useRef(null);
  // Aborts the in-flight /chat/stream fetch for the current turn (see
  // `sendMessage`) — `null` whenever no turn is in flight.
  const turnAbortControllerRef = useRef(null);

  const mode = useAppStore((s) => s.mode);
  const setVoiceState = useAppStore((s) => s.setVoiceState);
  const setConnectionState = useAppStore((s) => s.setConnectionState);
  const addMessage = useAppStore((s) => s.addMessage);
  const appendToMessage = useAppStore((s) => s.appendToMessage);
  const finishMessage = useAppStore((s) => s.finishMessage);
  const setLanguageInfo = useAppStore((s) => s.setLanguageInfo);
  const setMessageServerId = useAppStore((s) => s.setMessageServerId);
  const setMessageAudioPath = useAppStore((s) => s.setMessageAudioPath);
  const userId = useAppStore((s) => s.user?.id);

  const stopListening = useCallback(() => {
    clearTimeout(silenceTimerRef.current);
    clearTimeout(graceTimerRef.current);
    clearTimeout(maxListenTimerRef.current);
    micRef.current?.stop();
    micRef.current = null;
    sttSocketRef.current?.closeSTT();
    sttSocketRef.current = null;
    concludeRef.current = null;
  }, []);

  // Opens a TTS socket immediately (before the full reply text is even known)
  // and lets the caller feed it text chunks as they stream in from /chat/stream
  // — `sendChunk`/`finish` both await the socket's "open" event internally, so
  // callers can call `sendChunk` as soon as each delta arrives without
  // worrying about connection timing; order is preserved since awaits on the
  // same promise resolve in the order they were registered.
  const openSpeakSession = useCallback(
    (language) => {
      // Real bug hit live: the single shared TTSPlayer (`??=`, reused
      // forever) had no guard against a second turn opening while a prior
      // turn's session was still mid-utterance -- two voice-output turns
      // fired close together interleaved both replies' audio through one
      // queue, and the saved replay WAV mixed/truncated across turns. A
      // fresh player per turn, plus aborting any still-active previous
      // session first, keeps only ever one turn's audio live at a time.
      activeSpeakSessionRef.current?.abort();
      setVoiceState(VoiceState.SPEAKING);
      const player = new TTSPlayer();
      ttsPlayerRef.current = player;
      // "unknown" is a real value `response_language` can carry (low-confidence
      // detection) — not a valid Sarvam language code. Sending it straight
      // through as-is made synthesis fail outright ("I'm having trouble with
      // audio right now"), even though the text itself (e.g. the English
      // clarifying question) was perfectly speakable in English.
      const ttsLanguage = language && language !== "unknown" ? language : "en";
      const socket = new VoiceSocketClient({
        onAudioChunk: (chunk) => {
          player.playChunk(chunk).catch(() => {}); // one bad chunk shouldn't abort the whole reply
        },
        onEvent: (event) => {
          // Previously dropped silently — a TTS failure looked identical to
          // a normal "finished speaking" close, with no indication why no
          // audio ever played.
          if (event.type === "error" || event.type === "text_only_fallback") {
            addMessage("assistant", event.message || `Voice reply unavailable: ${event.reason || "unknown error"}`);
          }
        },
      });
      // Real bug hit live, twice: every key tried on this account so far
      // is rejected by Sarvam's own server for bulbul:v3 (confirmed
      // directly in production logs, see agent_core/speech/sarvam_tts.py's
      // own note) -- "bulbul:v2" is proven, repeatedly, with real audio
      // bytes returned.
      const ws = socket.connectTTS({ language: ttsLanguage, model: "bulbul:v2" });
      const opened = withTimeout(
        new Promise((resolve) => ws.addEventListener("open", resolve)),
        TTS_OPEN_TIMEOUT_MS
      );
      const closed = withTimeout(
        new Promise((resolve) => {
          ws.addEventListener("close", resolve); // /ws/tts closes once the utterance is fully synthesized
          ws.addEventListener("error", resolve);
        }),
        TTS_CLOSE_TIMEOUT_MS
      );
      // Real gap caught in a pre-deploy sweep: if `opened` only ever
      // resolved via the timeout fallback (socket never actually opened —
      // cold gateway, dropped network), every sendChunk below silently
      // no-ops forever. Unlike the explicit "error"/"text_only_fallback"
      // event path above, nothing ever told the user why no audio played —
      // the text reply just appeared with silence. Surfaced once, not per
      // chunk, the first time this is detected.
      let notifiedUnavailable = false;
      const session = {
        async sendChunk(chunk) {
          await opened;
          // Guards against the timeout (not a real "open") having resolved
          // this — sendText already no-ops if the socket isn't OPEN, but
          // being explicit here documents why that matters.
          if (ws.readyState === WebSocket.OPEN) {
            socket.sendText(chunk);
          } else if (!notifiedUnavailable) {
            notifiedUnavailable = true;
            addMessage("assistant", "Voice reply unavailable: couldn't connect for audio in time.");
          }
        },
        async finish() {
          await opened;
          if (ws.readyState === WebSocket.OPEN) {
            socket.endTTSUtterance();
            await closed;
          }
          // Real bug hit live: this ran unconditionally, but finish() is
          // async and can complete AFTER a newer turn has already opened
          // its own session and set SPEAKING -- a stale, already-aborted
          // session's finish() forced the orb back to IDLE mid-speech for
          // the CURRENT turn. Guarding it the same way as the ref-nulling
          // right below (which was already correctly guarded) fixes it.
          if (activeSpeakSessionRef.current === session) {
            setVoiceState(VoiceState.IDLE);
            activeSpeakSessionRef.current = null;
          }
          return player.finish();
        },
        // Closes without ever speaking anything — for a turn that failed
        // before any real text existed (no sendChunk was ever called), so
        // there's nothing to finish speaking, just a socket to tidy up.
        // Also the general-purpose "stop this session right now" path used
        // by mode changes, conversation switches, and unmount.
        abort() {
          socket.closeTTS();
          player.close();
          setVoiceState(VoiceState.IDLE);
          if (activeSpeakSessionRef.current === session) activeSpeakSessionRef.current = null;
        },
      };
      activeSpeakSessionRef.current = session;
      return session;
    },
    [setVoiceState, addMessage]
  );

  const sendMessage = useCallback(
    (text) =>
      new Promise((resolve) => {
        const assistantId = addMessage("assistant", "", { streaming: true });
        const isVoiceOutput = mode === Mode.SPEECH_TO_SPEECH || mode === Mode.TEXT_TO_SPEECH;
        let speakSession = null;
        // Real bug hit live: switching conversations mid-stream had nothing
        // to stop the old reply's fetch — text deltas kept arriving and
        // audio kept playing over the newly-opened chat. `reset()` (called
        // on mode change / conversation switch / logout / unmount) aborts
        // this so an abandoned turn's stream is actually torn down.
        const abortController = new AbortController();
        turnAbortControllerRef.current = abortController;

        streamChatMessage(
          {
            message: text,
            mode,
            sessionId: ids.current.sessionId,
            conversationId: ids.current.conversationId,
            threadId: ids.current.threadId,
            token,
            signal: abortController.signal,
          },
          {
            onLanguage: (event) => {
              // Opens the TTS socket with the REAL detected language up
              // front — known from the user's own message, before any
              // answer text exists, so voice starts on the very first
              // chunk instead of waiting for the whole reply to also learn
              // what language it turned out to be in.
              if (isVoiceOutput) speakSession = openSpeakSession(event.response_language);
            },
            onTextDelta: (chunk) => {
              appendToMessage(assistantId, chunk);
              speakSession?.sendChunk(chunk);
            },
            onDone: async (doneEvent) => {
              if (turnAbortControllerRef.current === abortController) turnAbortControllerRef.current = null;
              // A failed turn (no LLM provider answered) is not a real reply
              // — never spoken aloud (it wasn't sent as a text_delta, so it
              // never reached TTS) and never saved to history (the backend
              // already skipped persisting it). It's still shown as a
              // message so the user knows what happened, silently.
              if (doneEvent.error) {
                // doneEvent.text isn't in the documented `done` event shape
                // (chatClient.js only lists message_id/self_check_ok/
                // pending_confirmation) -- falling back here instead of
                // rendering the literal string "undefined" if an error-done
                // event ever arrives without it.
                appendToMessage(assistantId, doneEvent.text || "Sorry, something went wrong reaching the assistant.");
                finishMessage(assistantId);
                if (speakSession) speakSession.abort();
                else setVoiceState(VoiceState.IDLE);
                resolve();
                return;
              }
              setLanguageInfo({
                responseLanguage: doneEvent.response_language,
                languageConfidence: doneEvent.language_confidence,
                isCodeMixed: doneEvent.is_code_mixed,
              });
              if (doneEvent.message_id) setMessageServerId(assistantId, doneEvent.message_id);
              finishMessage(assistantId);
              // Real bug hit live: this used to `await speakSession.finish()`
              // (which waits for the TTS socket to actually CLOSE, bounded
              // but still up to TTS_CLOSE_TIMEOUT_MS) before resolving --
              // the composer stayed disabled the whole time, even though
              // nothing about typing a NEXT message depends on the previous
              // reply's audio having finished closing/saving. Detached, same
              // "don't make the user wait on background work" pattern as the
              // backend's own chat_store.record_turn.
              if (speakSession) {
                speakSession
                  .finish()
                  .then((blob) =>
                    saveReplyAudio({
                      userId,
                      messageServerId: doneEvent.message_id,
                      blob,
                      onSaved: setMessageAudioPath,
                    })
                  )
                  .catch(() => {});
              } else {
                setVoiceState(VoiceState.IDLE);
              }
              resolve();
            },
          }
        ).catch((err) => {
          if (turnAbortControllerRef.current === abortController) turnAbortControllerRef.current = null;
          // A deliberate abort (conversation switch, mode change, unmount —
          // see `reset()`) is not a failure — the turn was intentionally
          // abandoned, so nothing about it should be shown as an error.
          if (err.name === "AbortError") {
            resolve();
            return;
          }
          if (err.status === 401) {
            onUnauthorized?.();
            resolve();
            return;
          }
          appendToMessage(assistantId, "Sorry, something went wrong reaching the assistant.");
          finishMessage(assistantId);
          setVoiceState(VoiceState.IDLE);
          resolve();
        });
      }),
    [
      mode,
      token,
      ids,
      onUnauthorized,
      addMessage,
      appendToMessage,
      finishMessage,
      setLanguageInfo,
      setMessageServerId,
      setMessageAudioPath,
      setVoiceState,
      userId,
      openSpeakSession,
    ]
  );

  const send = useCallback(
    (text) => {
      addMessage("user", text);
      return sendMessage(text);
    },
    [addMessage, sendMessage]
  );

  const startListening = useCallback(async (preCreatedContext) => {
    if (sttSocketRef.current || micRef.current) return; // already listening — never overlap two sessions
    setVoiceState(VoiceState.LISTENING);

    // Local voice gate: mic frames stay in the browser until voice is
    // actually detected. The STT socket isn't even OPENED before that — so
    // if the user taps the mic and says nothing, zero frames leave the
    // machine and zero Sarvam/LLM/TTS usage happens. `preRoll` keeps the
    // last ~0.5s of pre-voice audio so the first word isn't clipped.
    let voiceDetected = false;
    let socketOpen = false;
    let socket = null;
    let concluded = false;
    const preRoll = [];
    // Sarvam sends one final transcript PER SPEECH SEGMENT (a mid-sentence
    // pause splits segments). They accumulate here and are joined into the
    // ONE question shown and sent when 3s of real silence concludes the turn
    // — so pausing to think doesn't cut the question short.
    const transcriptSegments = [];

    const stopIdle = () => {
      stopListening();
      setVoiceState(VoiceState.IDLE);
    };

    const finalize = () => {
      stopListening();
      const text = transcriptSegments.join(" ").trim();
      if (text) {
        // The user's spoken question, exactly as the STT model returned it,
        // goes into the chat — then the LLM's reply follows via sendMessage.
        setVoiceState(VoiceState.PROCESSING);
        addMessage("user", text);
        sendMessage(text);
      } else if (voiceDetected) {
        setVoiceState(VoiceState.IDLE);
        addMessage("assistant", "I didn't catch that — tap the mic and try again.");
      } else {
        // Never detected voice — nothing was sent anywhere (zero API usage),
        // but a LOCAL hint beats silent nothing-happened confusion: this
        // exact case was reported in the field as "no response coming".
        setVoiceState(VoiceState.IDLE);
        addMessage(
          "assistant",
          "I didn't hear anything — check the mic level line in the browser console ([voice] level ...) and that the right microphone is selected, then tap to try again."
        );
      }
    };

    const conclude = () => {
      if (concluded) return;
      concluded = true;
      clearTimeout(silenceTimerRef.current);
      micRef.current?.stop();
      micRef.current = null;
      if (transcriptSegments.length > 0 || !voiceDetected) {
        finalize();
        return;
      }
      // Voice was heard but the last segment's transcript is still in
      // flight — hold the socket open briefly so it isn't dropped.
      graceTimerRef.current = setTimeout(finalize, FINAL_TRANSCRIPT_GRACE_MS);
    };
    concludeRef.current = conclude; // exposes this session's conclude() to toggle()'s manual-stop path

    const resetSilenceTimer = () => {
      clearTimeout(silenceTimerRef.current);
      silenceTimerRef.current = setTimeout(conclude, SILENCE_STOP_MS);
    };

    const openSocket = () => {
      socket = new VoiceSocketClient({
        onOpen: () => {
          setConnectionState(ConnectionState.CONNECTED);
          socketOpen = true;
          for (const buffered of preRoll) socket.sendAudioFrame(buffered);
          preRoll.length = 0;
        },
        onReconnecting: () => setConnectionState(ConnectionState.RECONNECTING),
        onEvent: (event) => {
          if (event.type === "transcript" && event.is_final) {
            const text = (event.text || "").trim();
            if (text) transcriptSegments.push(text);
            // After the silence stop, the arrival of the in-flight final
            // segment is what we were holding the socket open for.
            if (concluded) {
              clearTimeout(graceTimerRef.current);
              finalize();
            }
            return;
          }
          // VAD speech_end is deliberately NOT a stop signal anymore — it
          // fires on brief mid-sentence pauses too. The 3s silence timer
          // (fed by local frame energy) is the only thing that ends a turn.
          if (event.type === "error") {
            // Never fail silently — a bare flip back to "Tap to speak" with
            // no explanation reads as "speech to text is not working" with
            // nothing to act on.
            stopIdle();
            addMessage("assistant", `Speech recognition failed: ${event.reason || "unknown error"}. Please try again.`);
          }
        },
      });
      sttSocketRef.current = socket;
      socket.connectSTT({ codec: "pcm_s16le", sample_rate: 16000, mode: "codemix" });
    };

    let noiseFloor = Infinity; // running minimum RMS = this session's silence level
    let consecutiveVoiced = 0;
    let lastLevelLog = 0;

    const voiceThreshold = () =>
      Math.max(MIN_VOICE_RMS, Math.min(noiseFloor, NOISE_FLOOR_CAP) * NOISE_FLOOR_MULTIPLIER);

    const isVoicedFrame = (rms) => {
      noiseFloor = Math.min(noiseFloor, rms);
      return rms >= voiceThreshold();
    };

    const mic = new MicCapture((frame) => {
      if (concluded) return; // silence stop already fired — nothing more leaves the mic
      const rms = frameRMS(frame);
      const voiced = isVoicedFrame(rms);

      // Always-visible field diagnostics (console.info, NOT console.debug —
      // Chrome hides debug by default, which made earlier gate failures
      // undiagnosable): one line every couple of seconds with the exact
      // numbers the gate is deciding on.
      const now = Date.now();
      if (now - lastLevelLog >= LEVEL_LOG_INTERVAL_MS) {
        lastLevelLog = now;
        console.info(
          `[voice] level rms=${rms.toFixed(4)} threshold=${voiceThreshold().toFixed(4)} ` +
            `voiced=${voiced} detected=${voiceDetected}`
        );
      }

      if (!voiceDetected) {
        preRoll.push(frame);
        if (preRoll.length > PRE_ROLL_MAX_FRAMES) preRoll.shift();
        consecutiveVoiced = voiced ? consecutiveVoiced + 1 : 0;
        if (consecutiveVoiced >= CONSECUTIVE_VOICED_FRAMES_TO_OPEN) {
          voiceDetected = true;
          console.info("[voice] detected — rms:", rms.toFixed(4), "threshold:", voiceThreshold().toFixed(4));
          resetSilenceTimer();
          openSocket(); // only NOW does anything leave the browser
        }
        return;
      }
      if (voiced) resetSilenceTimer();
      if (socketOpen) {
        socket.sendAudioFrame(frame);
      } else {
        preRoll.push(frame); // socket still connecting — keep buffering, flushed on open
      }
    });

    try {
      await mic.start(preCreatedContext);
      micRef.current = mic;
    } catch (err) {
      // Real bug hit live: `preCreatedContext` is created synchronously in
      // `toggle()` before this async call even starts (required for mobile
      // gesture timing — see its own comment). If `mic.start` throws before
      // it gets far enough to store the context on itself (e.g.
      // getUserMedia denied), MicCapture never learned about it, so
      // `stopIdle()` -> `stopListening()` -> `micRef.current?.stop()` is a
      // no-op (`micRef.current` is still null) and the context leaks. After
      // ~6 leaked contexts (Chrome's per-page cap), `new AudioContext()`
      // itself starts throwing and voice stops working entirely until reload.
      preCreatedContext?.close().catch(() => {});
      stopIdle();
      const denied = err?.name === "NotAllowedError";
      addMessage(
        "assistant",
        denied
          ? "Microphone access was denied. Please allow microphone access for this site in your browser settings, then try again."
          : `Couldn't start the microphone (${err?.name || "unknown error"}). Check that a working microphone is connected and not in use by another app.`
      );
      return;
    }

    // The ONE stop rule starts counting from here: 2s of silence — whether
    // the user never speaks (mic stops, nothing was ever sent anywhere) or
    // speaks and then goes quiet (their whole question is finalized).
    resetSilenceTimer();

    // Absolute ceiling: even if ambient noise keeps resetting the silence
    // timer, the session concludes (finalizing whatever was transcribed)
    // instead of listening forever.
    maxListenTimerRef.current = setTimeout(conclude, MAX_LISTEN_MS);
  }, [addMessage, sendMessage, setConnectionState, setVoiceState, stopListening]);

  const toggle = useCallback(() => {
    if (useAppStore.getState().voiceState === VoiceState.IDLE) {
      // AudioContext MUST be created synchronously here, inside the click
      // handler, before any async work. Mobile browsers (iOS Safari, Chrome
      // Android) block AudioContext creation in async callbacks because the
      // user-gesture context is lost the moment the call stack goes async.
      const audioCtx = new AudioContext();
      startListening(audioCtx);
    } else if (concludeRef.current) {
      // Tapping again while listening means "I'm done talking, process what
      // I said" — not "throw it away." Previously this called the raw
      // stopListening() (tears down mic+socket with no finalize), silently
      // discarding whatever had already been transcribed instead of sending it.
      concludeRef.current();
    } else {
      stopListening();
      setVoiceState(VoiceState.IDLE);
    }
  }, [startListening, stopListening, setVoiceState]);

  // Switching modes (e.g. Speech to Text -> Speech to Speech) doesn't itself
  // stop an in-progress voice session — without this, the orb kept showing
  // "Listening…"/"Speaking…" in the newly-selected mode because nothing
  // reset it. Call on every mode change; a no-op when already idle.
  //
  // Real bug hit live: this only ever called `stopListening()` — switching
  // modes (or conversations, see App.jsx's handleSwitchConversation) while
  // the assistant was SPEAKING left that reply's audio playing to
  // completion even though the orb had already flipped to IDLE. It also
  // left an in-flight /chat/stream fetch running, so a conversation switch
  // mid-reply kept appending text/audio for a chat the user had already
  // navigated away from. Now stops all three: mic/STT, active TTS
  // playback, and the in-flight stream fetch.
  const reset = useCallback(() => {
    stopListening();
    activeSpeakSessionRef.current?.abort();
    turnAbortControllerRef.current?.abort();
    turnAbortControllerRef.current = null;
    setVoiceState(VoiceState.IDLE);
  }, [stopListening, setVoiceState]);

  // Real bug hit live: nothing ever tore this session down on logout or
  // component unmount — the mic could stay live, TTS could keep playing
  // indefinitely, and the STT WebSocket could keep reconnecting, all after
  // the UI that owned them was gone.
  useEffect(() => {
    return () => {
      stopListening();
      activeSpeakSessionRef.current?.abort();
      turnAbortControllerRef.current?.abort();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return { toggle, send, reset };
}
