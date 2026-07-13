import { useCallback, useEffect, useRef, useState } from "react";
import AppShell from "./components/AppShell";
import ChatView from "./components/ChatView";
import Composer from "./components/Composer";
import LoginScreen from "./components/LoginScreen";
import { createConversation, fetchConversationMessages, listConversations } from "./api/chatClient";
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

  // Stable per-tab conversation identity — thread_id is the checkpointer key
  // the backend resumes from on reconnect (SessionState.thread_id).
  const ids = useRef({
    sessionId: crypto.randomUUID(),
    conversationId: crypto.randomUUID(),
    threadId: crypto.randomUUID(),
  });

  function handleLogout() {
    supabase.auth.signOut();
    logout();
  }

  const { toggle: onVoiceToggle, send: sendViaHook } = useVoiceSession({ token, ids, onUnauthorized: handleLogout });

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
  // the frontend's perspective (see chatClient.js). If persistence isn't
  // configured on the backend (no POSTGRES_DSN), these calls fail and the
  // app falls back to the random local id ids.current already started with
  // — same stateless behavior as before multi-chat existed.
  useEffect(() => {
    if (!token) return;
    (async () => {
      try {
        let list = await listConversations(token);
        if (list.length === 0) {
          const created = await createConversation(token);
          list = [{ id: created.id, title: null, updated_at: new Date().toISOString() }];
        }
        setConversations(list);
        activateConversation(list[0].id);
        const history = await fetchConversationMessages(token, list[0].id);
        loadMessages(history);
      } catch (err) {
        if (err.status === 401) handleLogout();
        // else: persistence not configured, or a transient error — keep the
        // random local conversation id from initial mount and carry on
        // stateless, rather than blocking the app from working at all.
      }
    })();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [token]);

  const handleSwitchConversation = useCallback(
    async (id) => {
      if (id === activeConversationId) return;
      activateConversation(id);
      try {
        loadMessages(await fetchConversationMessages(token, id));
      } catch {
        loadMessages([]);
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [activeConversationId, token, loadMessages]
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
      onOpenConversations={handleRefreshConversations}
    >
      <ChatView messages={messages} />
      <Composer
        mode={mode}
        onModeChange={setMode}
        onSend={handleSend}
        voiceState={voiceState}
        bargeInSignal={bargeInSignal}
        onVoiceToggle={onVoiceToggle}
        disabled={sending}
      />
    </AppShell>
  );
}
