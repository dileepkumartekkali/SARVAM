import { useCallback, useEffect, useRef, useState } from "react";
import AppShell from "./components/AppShell";
import ChatView from "./components/ChatView";
import Composer from "./components/Composer";
import LoginScreen from "./components/LoginScreen";
import { createConversation, deleteConversation, fetchConversationMessages, listConversations } from "./api/chatClient";
import { supabase } from "./api/supabaseClient";
import { useVoiceSession } from "./hooks/useVoiceSession";
import { selectIsAuthenticated, useAppStore } from "./store/useAppStore";

function toAppUser(supabaseUser) {
  if (!supabaseUser) return null;
  return {
    id: supabaseUser.id,
    email: supabaseUser.email,
    avatarUrl: supabaseUser.user_metadata?.avatar_url || null,
  };
}

export default function App() {
  const token = useAppStore((s) => s.token);
  const user = useAppStore((s) => s.user);
  const setSession = useAppStore((s) => s.setSession);
  const logout = useAppStore((s) => s.logout);
  const isAuthenticated = useAppStore(selectIsAuthenticated);

  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);
  const messages = useAppStore((s) => s.messages);
  const addMessage = useAppStore((s) => s.addMessage);
  const loadMessages = useAppStore((s) => s.loadMessages);
  const conversations = useAppStore((s) => s.conversations);
  const activeConversationId = useAppStore((s) => s.activeConversationId);
  const setConversations = useAppStore((s) => s.setConversations);
  const setActiveConversationId = useAppStore((s) => s.setActiveConversationId);
  const connectionState = useAppStore((s) => s.connectionState);
  const responseLanguage = useAppStore((s) => s.responseLanguage);
  const languageConfidence = useAppStore((s) => s.languageConfidence);
  const isCodeMixed = useAppStore((s) => s.isCodeMixed);
  const voiceState = useAppStore((s) => s.voiceState);
  const bargeInSignal = useAppStore((s) => s.bargeInSignal);

  const [sending, setSending] = useState(false);
  // Gates the composer: false while the first conversation is being loaded/
  // created. Without this, a message sent in the split-second before that
  // resolves would go out with the placeholder id `ids.current` starts with
  // — a real 404 from the backend, since persistence (when configured)
  // rejects any conversation_id it never actually created.
  const [conversationReady, setConversationReady] = useState(false);

  // Stable per-tab conversation identity — thread_id is the checkpointer key
  // the backend resumes from on reconnect (SessionState.thread_id).
  const ids = useRef({
    sessionId: crypto.randomUUID(),
    conversationId: crypto.randomUUID(),
    threadId: crypto.randomUUID(),
  });
  // Tracks which user's conversations the load-effect below has already run
  // for, so a Supabase token refresh (same user, new token string) doesn't
  // re-trigger it — see that effect's own comment.
  const loadedForUserIdRef = useRef(null);

  function handleLogout() {
    resetVoice(); // stop mic/TTS/in-flight stream before the session that owns them goes away
    supabase.auth.signOut();
    logout();
  }

  const {
    toggle: onVoiceToggle,
    send: sendViaHook,
    reset: resetVoice,
  } = useVoiceSession({ token, ids, onUnauthorized: handleLogout });

  function handleModeChange(newMode) {
    resetVoice(); // stop a stale voice session bleeding "Listening…" into the newly-selected mode
    setMode(newMode);
  }

  // Supabase owns the session's lifecycle (storage, refresh) — this just
  // mirrors its current session into the app store. Fires once on mount with
  // whatever session already exists (including right after the Google OAuth
  // redirect completes), then on every subsequent sign-in/sign-out/refresh.
  useEffect(() => {
    supabase.auth.getSession().then(({ data: { session } }) => {
      if (session) setSession(session.access_token, toAppUser(session.user));
    });
    const {
      data: { subscription },
    } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session?.access_token ?? null, toAppUser(session?.user));
    });
    return () => subscription.unsubscribe();
  }, [setSession]);

  // Points ids.current (and thus every /chat call) at a given conversation —
  // thread_id = conversation_id is deliberate: one LangGraph thread per
  // conversation, not per browser tab, so switching chats can't leak the
  // agent's own short-term reasoning context between them.
  function activateConversation(id) {
    ids.current.conversationId = id;
    ids.current.threadId = id;
    setActiveConversationId(id);
  }

  // On login: load the conversation list, creating the first one if this is
  // a brand-new user, then load its messages — the one place a page refresh
  // would otherwise lose everything, since /chat itself is stateless from
  // the frontend's perspective (see chatClient.js). The composer stays
  // disabled (conversationReady=false) until this settles, one way or
  // another — see the note on conversationReady above.
  //
  // Real bug hit live: this used to be keyed on [token] alone, which also
  // re-fires on Supabase's periodic TOKEN_REFRESHED event (same user, brand
  // new access_token string) -- roughly hourly, or on tab refocus, the user
  // was bounced back to their most-recently-active conversation and any
  // in-progress streaming reply on screen was wiped. Gating on the actual
  // user id (which a token refresh never changes) makes this run once per
  // real login instead of once per token.
  useEffect(() => {
    if (!token) {
      loadedForUserIdRef.current = null;
      return;
    }
    if (loadedForUserIdRef.current === user?.id) return;
    loadedForUserIdRef.current = user?.id;
    setConversationReady(false);
    (async () => {
      try {
        let list = await listConversations(token);
        let isBrandNew = false;
        if (list.length === 0) {
          const created = await createConversation(token);
          list = [{ id: created.id, title: null, updated_at: new Date().toISOString() }];
          isBrandNew = true;
        }
        setConversations(list);
        activateConversation(list[0].id);
        // A conversation we just created has no messages -- fetching them
        // is a pointless extra round-trip that only made the loading spinner
        // last longer for a first-time user, for a result we already know.
        loadMessages(isBrandNew ? [] : await fetchConversationMessages(token, list[0].id));
        setConversationReady(true);
      } catch (err) {
        if (err.status === 401) {
          handleLogout();
          return;
        }
        if (err.status === 503) {
          // POST /conversations only ever 503s when POSTGRES_DSN isn't set
          // on the backend at all — no persistence configured, so the
          // random local id from initial mount is exactly what a stateless
          // /chat expects, same as before multi-chat existed.
          setConversationReady(true);
          return;
        }
        // Persistence IS configured but this failed (network blip, backend
        // hiccup) — sending now would just 404 against a conversation id the
        // backend never created, so surface it and leave the composer
        // disabled rather than let that happen silently.
        addMessage("assistant", "Couldn't start a conversation — check your connection and reload the page.");
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const handleSwitchConversation = useCallback(
    async (id) => {
      if (id === activeConversationId) return;
      // Real bug hit live: switching conversations while a reply was still
      // streaming orphaned it — text deltas kept targeting the old
      // (about-to-be-replaced) message id and, in voice modes, the previous
      // conversation's audio kept playing over the newly-opened chat.
      // `resetVoice()` aborts the in-flight stream fetch and any active TTS
      // playback before the switch.
      resetVoice();
      activateConversation(id);
      try {
        loadMessages(await fetchConversationMessages(token, id));
      } catch {
        loadMessages([]);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [activeConversationId, token, loadMessages, resetVoice]
  );

  const handleNewConversation = useCallback(async () => {
    try {
      const created = await createConversation(token);
      setConversations([{ id: created.id, title: null, updated_at: new Date().toISOString() }, ...conversations]);
      activateConversation(created.id);
      loadMessages([]);
    } catch {
      // Persistence not configured — nothing meaningful to create.
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token, conversations, loadMessages]);

  const handleDeleteConversation = useCallback(
    async (id) => {
      try {
        await deleteConversation(token, id);
      } catch {
        return; // best-effort -- leave the list as-is if the delete itself failed
      }
      const remaining = conversations.filter((c) => c.id !== id);
      setConversations(remaining);
      if (id !== activeConversationId) return; // deleted a chat that wasn't open -- nothing else to do

      if (remaining.length > 0) {
        activateConversation(remaining[0].id);
        try {
          loadMessages(await fetchConversationMessages(token, remaining[0].id));
        } catch {
          loadMessages([]);
        }
        return;
      }
      // Deleted the only conversation -- start a fresh one so there's always
      // something to chat in, same as a brand-new user's first load.
      try {
        const created = await createConversation(token);
        setConversations([{ id: created.id, title: null, updated_at: new Date().toISOString() }]);
        activateConversation(created.id);
        loadMessages([]);
      } catch {
        // Persistence hiccup right after a delete — rare; the composer will
        // still point at the now-deleted id until the user reloads.
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [token, conversations, activeConversationId, setConversations, loadMessages]
  );

  // Sidebar titles/ordering only need to be fresh when it's actually opened,
  // not after every single turn — cheap, and avoids an extra request per
  // message for a purely cosmetic refresh.
  const handleRefreshConversations = useCallback(() => {
    if (!token) return;
    listConversations(token)
      .then(setConversations)
      .catch(() => {});
  }, [token, setConversations]);

  async function handleSend(text) {
    setSending(true);
    try {
      await sendViaHook(text);
    } finally {
      setSending(false);
    }
  }

  if (!isAuthenticated) {
    return <LoginScreen />;
  }

  return (
    <AppShell
      connectionState={connectionState}
      responseLanguage={responseLanguage}
      languageConfidence={languageConfidence}
      isCodeMixed={isCodeMixed}
      user={user}
      onLogout={handleLogout}
      conversations={conversations}
      activeConversationId={activeConversationId}
      onSwitchConversation={handleSwitchConversation}
      onNewConversation={handleNewConversation}
      onDeleteConversation={handleDeleteConversation}
      onOpenConversations={handleRefreshConversations}
    >
      <ChatView messages={messages} loading={!conversationReady} />
      <Composer
        mode={mode}
        onModeChange={handleModeChange}
        onSend={handleSend}
        voiceState={voiceState}
        bargeInSignal={bargeInSignal}
        onVoiceToggle={onVoiceToggle}
        disabled={sending || !conversationReady}
      />
    </AppShell>
  );
}
