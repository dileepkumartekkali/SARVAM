import { API_BASE_URL } from "./config";

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
  { message, mode, sessionId, conversationId, threadId, token, sttLanguageHint },
  { onLanguage, onTextDelta, onDone }
) {
  const resp = await fetch(`${API_BASE_URL}/chat/stream`, {
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
        const event = JSON.parse(line.slice(5).trim());
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
  const resp = await fetch(`${API_BASE_URL}/conversations`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  const err = _unauthorizedOr(resp, "listing conversations failed");
  if (err) throw err;
  return resp.json();
}

/** Starts a new, empty conversation — returns `{id}`. */
export async function createConversation(token) {
  const resp = await fetch(`${API_BASE_URL}/conversations`, {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
  });
  const err = _unauthorizedOr(resp, "creating conversation failed");
  if (err) throw err;
  return resp.json();
}

/** One conversation's full message history — fetched on login and whenever
 * the active conversation is switched, so a page refresh (or a switch back
 * to an older chat) doesn't lose what's already been said. */
export async function fetchConversationMessages(token, conversationId) {
  const resp = await fetch(`${API_BASE_URL}/conversations/${conversationId}/messages`, {
    headers: { Authorization: `Bearer ${token}` },
  });
  const err = _unauthorizedOr(resp, "fetching conversation messages failed");
  if (err) throw err;
  return resp.json();
}

/** Deletes a conversation and everything in it (cascades server-side —
 * see chat_store.delete_conversation). Does not remove any replay audio
 * already uploaded to Supabase Storage for it. */
export async function deleteConversation(token, conversationId) {
  const resp = await fetch(`${API_BASE_URL}/conversations/${conversationId}`, {
    method: "DELETE",
    headers: { Authorization: `Bearer ${token}` },
  });
  const err = _unauthorizedOr(resp, "deleting conversation failed");
  if (err) throw err;
}
