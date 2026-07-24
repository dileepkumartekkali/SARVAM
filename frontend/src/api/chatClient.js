import { API_BASE_URL } from "./config";
import { supabase } from "./supabaseClient";

// A tab left open/backgrounded for hours outlives Supabase's ~1h access
// token — supabase-js's autoRefreshToken is timer-driven and browsers
// throttle timers in background tabs, so the token can go stale before it
// refreshes on its own. Every call below used to treat the very next 401 as
// "really logged out" and sign the user out immediately. Real bug hit live:
// this is the one shared place that instead tries a real refresh first and
// retries once — only a 401 that survives a fresh token is a real logout.
async function fetchWithAuthRetry(url, options, token) {
  const withAuth = (t) => ({ ...options, headers: { ...options.headers, Authorization: `Bearer ${t}` } });
  let resp = await fetch(url, withAuth(token));
  if (resp.status === 401) {
    const { data } = await supabase.auth.refreshSession();
    const refreshedToken = data?.session?.access_token;
    if (refreshedToken) resp = await fetch(url, withAuth(refreshedToken));
  }
  return resp;
}

/**
 * `POST /chat/stream` (agent_core.api.main + task_agent.stream_turn) — real
 * SSE token/sentence streaming. Event sequence: one `{"type":"language",
 * response_language, language_confidence, is_code_mixed}` event first (the
 * detected language is known from the user's own message, before any answer
 * text exists — the frontend needs it up front to open a TTS socket with the
 * right voice from chunk one, not after the whole reply is already known),
 * then `{"type":"text_delta","text":...}` events as sentence-bounded chunks
 * become ready, then one final `{"type":"done", message_id, self_check_ok,
 * pending_confirmation}` event. Calls `onLanguage`/`onTextDelta`/`onDone` as
 * those arrive — there is no single resolved return value, since the whole
 * point is not waiting for the full response before doing anything with it.
 */
export async function streamChatMessage(
  { message, mode, sessionId, conversationId, threadId, token, sttLanguageHint, signal },
  { onLanguage, onTextDelta, onDone }
) {
  const resp = await fetchWithAuthRetry(
    `${API_BASE_URL}/chat/stream`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: sessionId,
        conversation_id: conversationId,
        thread_id: threadId,
        message,
        mode: mode || "text_to_text",
        stt_language_hint: sttLanguageHint ?? null,
      }),
      // Optional AbortSignal -- lets a caller (e.g. switching conversations
      // mid-stream) stop consuming a reply that's no longer wanted, instead
      // of it continuing to append text/audio to a conversation the user
      // has already navigated away from.
      signal,
    },
    token
  );
  if (resp.status === 401) {
    const err = new Error("unauthorized");
    err.status = 401;
    throw err;
  }
  if (!resp.ok || !resp.body) {
    throw new Error(`chat stream request failed: ${resp.status}`);
  }

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let boundary;
    while ((boundary = buffer.indexOf("\n\n")) !== -1) {
      const block = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      for (const line of block.split("\n")) {
        if (!line.startsWith("data:")) continue;
        // A single malformed/partial `data:` line (keepalive payload, an
        // edge-case flush) used to throw here and abort the whole read loop
        // -- the rest of a perfectly good reply was silently dropped and the
        // user just saw a generic "something went wrong" error.
        let event;
        try {
          event = JSON.parse(line.slice(5).trim());
        } catch {
          continue;
        }
        if (event.type === "language") onLanguage?.(event);
        else if (event.type === "text_delta") onTextDelta?.(event.text);
        else if (event.type === "done") onDone?.(event);
      }
    }
  }
}

function _unauthorizedOr(resp, message) {
  if (resp.status === 401) {
    const err = new Error("unauthorized");
    err.status = 401;
    return err;
  }
  if (!resp.ok) {
    const err = new Error(`${message}: ${resp.status}`);
    err.status = resp.status;
    return err;
  }
  return null;
}

/** The authenticated user's conversations (agent_core/persistence/
 * chat_store.py) — ordered most-recently-active first, what populates the
 * conversation switcher/sidebar. */
export async function listConversations(token) {
  const resp = await fetchWithAuthRetry(`${API_BASE_URL}/conversations`, {}, token);
  const err = _unauthorizedOr(resp, "listing conversations failed");
  if (err) throw err;
  return resp.json();
}

/** Starts a new, empty conversation — returns `{id}`. */
export async function createConversation(token) {
  const resp = await fetchWithAuthRetry(`${API_BASE_URL}/conversations`, { method: "POST" }, token);
  const err = _unauthorizedOr(resp, "creating conversation failed");
  if (err) throw err;
  return resp.json();
}

/** One conversation's full message history — fetched on login and whenever
 * the active conversation is switched, so a page refresh (or a switch back
 * to an older chat) doesn't lose what's already been said. */
export async function fetchConversationMessages(token, conversationId) {
  const resp = await fetchWithAuthRetry(`${API_BASE_URL}/conversations/${conversationId}/messages`, {}, token);
  const err = _unauthorizedOr(resp, "fetching conversation messages failed");
  if (err) throw err;
  return resp.json();
}

/** Deletes a conversation and everything in it (cascades server-side —
 * see chat_store.delete_conversation). Does not remove any replay audio
 * already uploaded to Supabase Storage for it. */
export async function deleteConversation(token, conversationId) {
  const resp = await fetchWithAuthRetry(`${API_BASE_URL}/conversations/${conversationId}`, { method: "DELETE" }, token);
  const err = _unauthorizedOr(resp, "deleting conversation failed");
  if (err) throw err;
}
