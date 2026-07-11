import { useState } from "react";
import { devLogin } from "../api/authClient";

export default function LoginScreen({ onLogin }) {
  const [username, setUsername] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function submit(e) {
    e.preventDefault();
    if (!username.trim()) return;
    setBusy(true);
    setError(null);
    try {
      const token = await devLogin(username.trim());
      onLogin(token);
    } catch (err) {
      setError("Couldn't sign in — check that the backend is running and DEV_AUTH_ENABLED is set.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="login-screen">
      <div className="login-card">
        <div className="login-card__brand">Vaani</div>
        <p className="login-card__tagline">Your multilingual voice assistant</p>
        <form onSubmit={submit}>
          <label className="login-card__label" htmlFor="username">
            Username
          </label>
          <input
            id="username"
            className="login-card__input"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            placeholder="you@example.com"
            autoFocus
          />
          {error && <div className="login-card__error">{error}</div>}
          <button type="submit" className="login-card__submit" disabled={busy || !username.trim()}>
            {busy ? "Signing in…" : "Continue"}
          </button>
        </form>
        <p className="login-card__footnote">
          Dev sign-in only — a real OAuth flow isn't wired up yet (see docs/THREAT_MODEL.md).
        </p>
      </div>
    </div>
  );
}
