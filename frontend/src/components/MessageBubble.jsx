import { useState } from "react";
import ReactMarkdown from "react-markdown";
import { supabase } from "../api/supabaseClient";

const SIGNED_URL_TTL_SECONDS = 60;

/** TEXT_MODE allows markdown (lists/headers/bold) per agent_system_prompt.md
 * — rendered here for assistant messages only. User messages are shown as
 * plain text: it's an echo of what they typed, not an LLM output the
 * TEXT_MODE formatting rules apply to. */
export default function MessageBubble({ role, text, streaming, audioPath }) {
  const isUser = role === "user";
  const [playing, setPlaying] = useState(false);

  async function playRecording() {
    if (playing) return;
    setPlaying(true);
    try {
      // Bucket is private (per-user RLS) — a signed URL is required, not the
      // public object URL.
      const { data, error } = await supabase.storage
        .from("tts-audio")
        .createSignedUrl(audioPath, SIGNED_URL_TTL_SECONDS);
      if (error || !data?.signedUrl) return;
      const audio = new Audio(data.signedUrl);
      await new Promise((resolve) => {
        audio.onended = resolve;
        audio.onerror = resolve;
        audio.play().catch(resolve);
      });
    } finally {
      setPlaying(false);
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
