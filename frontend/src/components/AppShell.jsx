import ConnectionBanner from "./ConnectionBanner";
import LanguageBadge from "./LanguageBadge";

export default function AppShell({ connectionState, responseLanguage, languageConfidence, isCodeMixed, onLogout, children }) {
  return (
    <div className="app-shell">
      <header className="app-shell__topbar">
        <span className="app-shell__brand">Vaani</span>
        <div className="app-shell__topbar-right">
          <LanguageBadge language={responseLanguage} confidence={languageConfidence} isCodeMixed={isCodeMixed} />
          <button type="button" className="app-shell__logout" onClick={onLogout}>
            Sign out
          </button>
        </div>
      </header>
      <ConnectionBanner connectionState={connectionState} />
      <main className="app-shell__main">{children}</main>
    </div>
  );
}
