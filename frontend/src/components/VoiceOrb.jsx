import { useEffect, useRef, useState } from "react";
import { VoiceState } from "../store/useAppStore";

const LABELS = {
  [VoiceState.IDLE]: "Tap to speak",
  [VoiceState.LISTENING]: "Listening…",
  [VoiceState.PROCESSING]: "Thinking…",
  [VoiceState.SPEAKING]: "Speaking…",
};

/** The mic button + state-machine visual. `bargeInSignal` is a counter (not
 * a boolean) so re-triggering while already flashed still re-fires the
 * animation — the visual cue the task asked for: the user should SEE the
 * interruption land, not just hear the audio cut off. */
export default function VoiceOrb({ voiceState, bargeInSignal, onToggle }) {
  const [flashing, setFlashing] = useState(false);
  const lastSignal = useRef(bargeInSignal);

  useEffect(() => {
    if (bargeInSignal !== lastSignal.current) {
      lastSignal.current = bargeInSignal;
      setFlashing(true);
      const t = setTimeout(() => setFlashing(false), 420);
      return () => clearTimeout(t);
    }
  }, [bargeInSignal]);

  return (
    <div className="voice-orb-wrap">
      <button
        type="button"
        className={`voice-orb voice-orb--${voiceState}${flashing ? " voice-orb--flash" : ""}`}
        onClick={onToggle}
        aria-label={LABELS[voiceState]}
      >
        {voiceState === VoiceState.SPEAKING ? (
          <span className="voice-orb__bars" aria-hidden="true">
            <span />
            <span />
            <span />
            <span />
          </span>
        ) : voiceState === VoiceState.PROCESSING ? (
          <span className="voice-orb__spinner" aria-hidden="true" />
        ) : (
          <svg viewBox="0 0 24 24" width="28" height="28" aria-hidden="true">
            <path
              fill="currentColor"
              d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Zm5-3a5 5 0 0 1-10 0H5a7 7 0 0 0 6 6.93V21h2v-2.07A7 7 0 0 0 19 12h-2Z"
            />
          </svg>
        )}
      </button>
      <span className="voice-orb__label">{LABELS[voiceState]}</span>
    </div>
  );
}
