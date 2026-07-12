import { useEffect, useRef, useState } from "react";
import AppShell from "./components/AppShell";
import ChatView from "./components/ChatView";
import Composer from "./components/Composer";
import LoginScreen from "./components/LoginScreen";
import { fetchConversationHistory } from "./api/chatClient";
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
  const setSession = useAppStore((s) => s.setSession);
  const logout = useAppStore((s) => s.logout);
  const isAuthenticated = useAppStore(selectIsAuthenticated);

  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);
  const messages = useAppStore((s) => s.messages);
  const loadMessages = useAppStore((s) => s.loadMessages);
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

  // Rehydrate the chat box once a session exists — the one place a page
  // refresh would otherwise lose everything, since /chat itself is stateless
  // from the frontend's perspective (see chatClient.js).
  useEffect(() => {
    if (!token) return;
    fetchConversationHistory(token)
      .then(loadMessages)
      .catch(() => {}); // best-effort — an empty chat box beats a crash
  }, [token, loadMessages]);

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
      onLogout={handleLogout}
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
