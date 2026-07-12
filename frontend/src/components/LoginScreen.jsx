import { useState } from "react";
import { supabase } from "../api/supabaseClient";

export default function LoginScreen() {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  async function signInWithGoogle() {
    setBusy(true);
    setError(null);
    const { error: signInError } = await supabase.auth.signInWithOAuth({
      provider: "google",
      options: { redirectTo: window.location.origin },
    });
    if (signInError) {
      setError("Couldn't start Google sign-in — please try again.");
      setBusy(false);
    }
    // On success the browser navigates away to Google's consent screen, so
    // there's nothing to render here — App.jsx picks up the session via
    // supabase.auth.onAuthStateChange once the redirect back completes.
  }

  return (
    <div className="login-screen">
      <div className="login-card">
        <div className="login-card__brand">Vaani</div>
        <p className="login-card__tagline">Your multilingual voice assistant</p>
        {error && <div className="login-card__error">{error}</div>}
        <button type="button" className="login-card__submit login-card__google" onClick={signInWithGoogle} disabled={busy}>
          <svg viewBox="0 0 24 24" width="18" height="18" aria-hidden="true">
            <path
              fill="currentColor"
              d="M22.56 12.25c0-.78-.07-1.53-.2-2.25H12v4.26h5.92a5.06 5.06 0 0 1-2.2 3.32v2.77h3.57c2.08-1.92 3.28-4.74 3.28-8.1Z"
            />
            <path
              fill="currentColor"
              d="M12 23c2.97 0 5.46-.98 7.28-2.66l-3.57-2.77c-.99.66-2.25 1.06-3.71 1.06-2.86 0-5.29-1.93-6.16-4.53H2.18v2.85A11 11 0 0 0 12 23Z"
            />
            <path
              fill="currentColor"
              d="M5.84 14.09A6.6 6.6 0 0 1 5.5 12c0-.73.12-1.43.34-2.09V7.06H2.18A11 11 0 0 0 1 12c0 1.78.43 3.46 1.18 4.94l3.66-2.85Z"
            />
            <path
              fill="currentColor"
              d="M12 5.38c1.62 0 3.06.56 4.21 1.64l3.15-3.15C17.45 2.09 14.97 1 12 1A11 11 0 0 0 2.18 7.06l3.66 2.85C6.71 7.31 9.14 5.38 12 5.38Z"
            />
          </svg>
          {busy ? "Redirecting…" : "Continue with Google"}
        </button>
      </div>
    </div>
  );
}
