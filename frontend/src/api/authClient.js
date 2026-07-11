import { API_BASE_URL } from "./config";

/**
 * Hits the backend's dev-only login endpoint (see agent_core/api/main.py's
 * `/auth/dev-login`, gated behind DEV_AUTH_ENABLED). There is no real OAuth
 * IdP wired up yet — Phase 6's threat model lists that as accepted risk for
 * v1. This is a working dev convenience so the auth UI is real and clickable
 * end-to-end, not a mock; swap this call for a real OAuth redirect flow when
 * an IdP is integrated, without touching any other component.
 */
export async function devLogin(username) {
  const resp = await fetch(`${API_BASE_URL}/auth/dev-login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username }),
  });
  if (!resp.ok) {
    throw new Error(`login failed: ${resp.status}`);
  }
  const data = await resp.json();
  return data.access_token;
}
