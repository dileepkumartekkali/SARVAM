import { useState } from "react";
import ConnectionBanner from "./ConnectionBanner";
import ConversationDrawer from "./ConversationDrawer";
import LanguageBadge from "./LanguageBadge";
import ProfileMenu from "./ProfileMenu";

export default function AppShell({
  connectionState,
  responseLanguage,
  languageConfidence,
  isCodeMixed,
  user,
  onLogout,
  conversations,
  activeConversationId,
  onSwitchConversation,
  onNewConversation,
  onOpenConversations,
  children,
}) {
  const [drawerOpen, setDrawerOpen] = useState(false);

  function openDrawer() {
    onOpenConversations?.();
    setDrawerOpen(true);
  }

  return (
    <div className="app-shell">
      <header className="app-shell__topbar">
        <div className="app-shell__topbar-left">
          <button type="button" className="app-shell__menu-btn" onClick={openDrawer} aria-label="Conversations">
            <svg viewBox="0 0 24 24" width="20" height="20" aria-hidden="true">
              <path fill="currentColor" d="M4 6h16v2H4V6Zm0 5h16v2H4v-2Zm0 5h16v2H4v-2Z" />
            </svg>
          </button>
          <span className="app-shell__brand">Mvoice</span>
        </div>
        <div className="app-shell__topbar-right">
          <LanguageBadge language={responseLanguage} confidence={languageConfidence} isCodeMixed={isCodeMixed} />
          <ProfileMenu user={user} onLogout={onLogout} />
        </div>
      </header>
      <ConnectionBanner connectionState={connectionState} />
      <main className="app-shell__main">{children}</main>
      <ConversationDrawer
        open={drawerOpen}
        onClose={() => setDrawerOpen(false)}
        conversations={conversations}
        activeConversationId={activeConversationId}
        onSwitchConversation={onSwitchConversation}
        onNewConversation={onNewConversation}
      />
    </div>
  );
}
