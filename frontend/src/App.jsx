import { useEffect, useRef, useState } from "react";
import AppShell from "./components/AppShell";
import ChatView from "./components/ChatView";
import Composer from "./components/Composer";
import LoginScreen from "./components/LoginScreen";
import { useVoiceSession } from "./hooks/useVoiceSession";
import { selectIsAuthenticated, useAppStore } from "./store/useAppStore";

const TOKEN_STORAGE_KEY = "vaani_token";

export default function App() {
  const token = useAppStore((s) => s.token);
  const setToken = useAppStore((s) => s.setToken);
  const logout = useAppStore((s) => s.logout);
  const isAuthenticated = useAppStore(selectIsAuthenticated);

  const mode = useAppStore((s) => s.mode);
  const setMode = useAppStore((s) => s.setMode);
  const messages = useAppStore((s) => s.messages);
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
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    logout();
  }

  const { toggle: onVoiceToggle, send: sendViaHook } = useVoiceSession({ token, ids, onUnauthorized: handleLogout });

  useEffect(() => {
    const saved = localStorage.getItem(TOKEN_STORAGE_KEY);
    if (saved) setToken(saved);
  }, [setToken]);

  function handleLogin(newToken) {
    localStorage.setItem(TOKEN_STORAGE_KEY, newToken);
    setToken(newToken);
  }

  async function handleSend(text) {
    setSending(true);
    try {
      await sendViaHook(text);
    } finally {
      setSending(false);
    }
  }

  if (!isAuthenticated) {
    return <LoginScreen onLogin={handleLogin} />;
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
