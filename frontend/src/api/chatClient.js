import { API_BASE_URL } from "./config";

/**
 * `POST /chat` on the current backend (Phases 2-6) returns one complete JSON
 * response — it does not stream tokens over HTTP yet, even though the
 * underlying LLMRouter/task_agent were built streaming-first (Phase 1).
 * Exposing that as a real SSE/chunked endpoint is a backend change outside
 * this frontend phase's scope. `revealProgressively` below simulates the
 * streaming *display* the requirement asks for by revealing the already-
 * fetched response incrementally — it is a presentation-layer approximation,
 * not real network streaming. Swap this for a real `EventSource`/fetch-stream
 * reader the moment a streaming endpoint exists; nothing else in the UI
 * needs to change, since components only ever see `appendToMessage` calls.
 */
export async function sendChatMessage({ message, mode, sessionId, conversationId, threadId, token, sttLanguageHint }) {
  const resp = await fetch(`${API_BASE_URL}/chat`, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    },
    body: JSON.stringify({
      session_id: sessionId,
      conversation_id: conversationId,
      thread_id: threadId,
      message,
      mode: mode || "text_to_text",
      stt_language_hint: sttLanguageHint ?? null,
    }),
  });
  if (resp.status === 401) {
    const err = new Error("unauthorized");
    err.status = 401;
    throw err;
  }
  if (!resp.ok) {
    throw new Error(`chat request failed: ${resp.status}`);
  }
  return resp.json();
}

/** The authenticated user's persisted chat history (one ongoing conversation
 * — see agent_core/persistence/chat_store.py) — fetched once on login/mount
 * so a page refresh doesn't lose the visible chat log. */
export async function fetchConversationHistory(token) {
  const resp = await fetch(`${API_BASE_URL}/conversations/current/messages`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (resp.status === 401) {
    const err = new Error("unauthorized");
    err.status = 401;
    throw err;
  }
  if (!resp.ok) {
    throw new Error(`fetching chat history failed: ${resp.status}`);
  }
  return resp.json();
}

/** Reveals `text` into the store a few words at a time. Returns a promise
 * that resolves once the whole text has been appended. */
export function revealProgressively(text, onChunk, { wordsPerTick = 3, tickMs = 40 } = {}) {
  // In a hidden/background tab the browser throttles chained setTimeouts
  // (Chrome: down to ~1/minute), freezing the animation mid-text and — worse
  // — leaving this promise unresolved, so the message stays "streaming" and
  // the voice state machine never returns to idle. Nobody is watching a
  // hidden tab's animation anyway: append everything at once.
  if (document.hidden) {
    onChunk(text + " ");
    return Promise.resolve();
  }
  const words = text.split(" ");
  let i = 0;
  return new Promise((resolve) => {
    function tick() {
      if (i >= words.length) {
        resolve();
        return;
      }
      const slice = words.slice(i, i + wordsPerTick).join(" ") + " ";
      onChunk(slice);
      i += wordsPerTick;
      setTimeout(tick, tickMs);
    }
    tick();
  });
}
