import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import { supabase } from "../api/supabaseClient";

const SIGNED_URL_TTL_SECONDS = 60;

// Real bug hit live: each MessageBubble only tracked its OWN `playing` state
// -- tapping "Play" on message A then message B played both recordings
// simultaneously, since nothing coordinated across instances. Module-level
// (not component state) so every bubble shares the same "one replay at a
// time" gate regardless of which one is mounted where.
let _activeReplay = null;

/** TEXT_MODE allows markdown (lists/headers/bold) per agent_system_prompt.md
 * — rendered here for assistant messages only. User messages are shown as
 * plain text: it's an echo of what they typed, not an LLM output the
 * TEXT_MODE formatting rules apply to. */
export default function MessageBubble({ role, text, streaming, audioPath }) {
  const isUser = role === "user";
  const [playing, setPlaying] = useState(false);
  const audioRef = useRef(null);
  const mountedRef = useRef(true);

  // Real bug hit live: nothing stopped a replay's audio when the bubble
  // unmounted (switching conversations, message list re-rendering) -- the
  // `Audio` element had no ref and no teardown, so it kept playing to
  // completion over whatever the user navigated to next.
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
      if (audioRef.current === _activeReplay) _activeReplay = null;
      audioRef.current?.pause();
    };
  }, []);

  async function playRecording() {
    if (playing) return;
    // Stop whatever else is currently playing before starting this one --
    // see the module-level `_activeReplay` comment above.
    _activeReplay?.pause();
    setPlaying(true);
    try {
      // Bucket is private (per-user RLS) — a signed URL is required, not the
      // public object URL.
      const { data, error } = await supabase.storage
        .from("tts-audio")
        .createSignedUrl(audioPath, SIGNED_URL_TTL_SECONDS);
      if (error || !data?.signedUrl) return;
      const audio = new Audio(data.signedUrl);
      audioRef.current = audio;
      _activeReplay = audio;
      await new Promise((resolve) => {
        audio.onended = resolve;
        audio.onerror = resolve;
        // Fires when another bubble's playRecording() pauses THIS audio to
        // start its own (the only thing that ever calls .pause() here,
        // since there's no user-facing pause control) -- without this, an
        // interrupted bubble's promise never settles and its "Playing…"
        // state never resets.
        audio.onpause = resolve;
        audio.play().catch(resolve);
      });
    } finally {
      if (audioRef.current === _activeReplay) _activeReplay = null;
      if (mountedRef.current) setPlaying(false);
    }
  }

  return (
    <div className={`message-row message-row--${role}`}>
      <div className={`message-bubble message-bubble--${role}`}>
        {isUser ? (
          <span>{text}</span>
        ) : (
          <div className="message-bubble__markdown">
            <ReactMarkdown>{text || " "}</ReactMarkdown>
          </div>
        )}
        {streaming && <span className="message-bubble__cursor" aria-hidden="true" />}
        {!isUser && audioPath && (
          <button
            type="button"
            className="message-bubble__replay"
            onClick={playRecording}
            disabled={playing}
            aria-label={playing ? "Playing" : "Play voice reply"}
          >
            <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden="true">
              {playing ? (
                <path fill="currentColor" d="M6 5h4v14H6V5Zm8 0h4v14h-4V5Z" />
              ) : (
                <path fill="currentColor" d="M8 5v14l11-7L8 5Z" />
              )}
            </svg>
            {playing ? "Playing…" : "Play"}
          </button>
        )}
      </div>
    </div>
  );
}
