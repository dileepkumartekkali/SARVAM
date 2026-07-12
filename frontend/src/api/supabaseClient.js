import { createClient } from "@supabase/supabase-js";
import { SUPABASE_ANON_KEY, SUPABASE_URL } from "./config";

if (!SUPABASE_URL || !SUPABASE_ANON_KEY) {
  throw new Error(
    "Missing VITE_SUPABASE_URL / VITE_SUPABASE_ANON_KEY — set both in your .env (see .env.example) before running the app."
  );
}

// Single shared client — handles session storage/refresh internally
// (localStorage under its own key), so nothing else in this app needs to
// manage the Supabase access token's lifecycle by hand.
export const supabase = createClient(SUPABASE_URL, SUPABASE_ANON_KEY);
