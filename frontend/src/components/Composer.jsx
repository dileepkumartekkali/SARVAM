import { useState } from "react";
import { isVoiceInputMode, Mode } from "../store/useAppStore";
import VoiceOrb from "./VoiceOrb";

const MODE_OPTIONS = [
  { value: Mode.TEXT_TO_TEXT, label: "Text" },
  { value: Mode.SPEECH_TO_TEXT, label: "Speech to Text" },
  { value: Mode.TEXT_TO_SPEECH, label: "Text to Speech" },
  { value: Mode.SPEECH_TO_SPEECH, label: "Speech to Speech" },
];

export default function Composer({ mode, onModeChange, onSend, voiceState, bargeInSignal, onVoiceToggle, disabled }) {
  const [draft, setDraft] = useState("");
  const voiceInput = isVoiceInputMode(mode);

  function submit(e) {
    e.preventDefault();
    const text = draft.trim();
    if (!text || disabled) return;
    onSend(text);
    setDraft("");
  }

  return (
    <div className="composer">
      <div className="composer__mode-switch" role="tablist" aria-label="Input/output mode">
        {MODE_OPTIONS.map((opt) => (
          <button
            key={opt.value}
            type="button"
            role="tab"
            aria-selected={mode === opt.value}
            className={`composer__mode-btn${mode === opt.value ? " composer__mode-btn--active" : ""}`}
            onClick={() => onModeChange(opt.value)}
          >
            {opt.label}
          </button>
        ))}
      </div>

      {voiceInput ? (
        <div className="composer__voice">
          <VoiceOrb voiceState={voiceState} bargeInSignal={bargeInSignal} onToggle={onVoiceToggle} />
        </div>
      ) : (
        <form className="composer__form" onSubmit={submit}>
          <textarea
            className="composer__input"
            placeholder="Type a message…"
            value={draft}
            rows={1}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) submit(e);
            }}
            disabled={disabled}
          />
          <button type="submit" className="composer__send" disabled={disabled || !draft.trim()} aria-label="Send">
            <svg viewBox="0 0 24 24" width="20" height="20">
              <path fill="currentColor" d="M3 20v-6l8-2-8-2V4l19 8-19 8Z" />
            </svg>
          </button>
        </form>
      )}
    </div>
  );
}
