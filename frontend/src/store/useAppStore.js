/**
 * State management: Zustand, chosen specifically against this app's traffic
 * pattern, not as a generic default.
 *
 * Why not Redux: streaming tokens and voice-state transitions fire at a rate
 * (multiple times per second during TTS playback / LLM streaming) where
 * action-creator/reducer ceremony buys nothing — there's no complex derived
 * state or time-travel debugging need here, just "append this token" and
 * "set this enum." Redux's indirection would slow iteration without solving
 * a problem this app has.
 *
 * Why not Context: Context re-renders every consumer on every Provider value
 * change unless the tree is manually split into many fine-grained providers
 * and memoized — exactly the failure mode a fast-churning voice session
 * triggers (a token stream update would re-render the mic button; a VAD
 * level tick would re-render the chat log). Zustand's selector subscriptions
 * (`useAppStore(s => s.voiceState)`) mean each component only re-renders when
 * ITS slice changes — the chat log doesn't re-render on every audio-level
 * tick, and the mic button doesn't re-render on every streamed token.
 *
 * Why Zustand specifically: no Provider wrapper (one less thing to get
 * wrong with WebSocket-lifecycle effects that need to run outside React's
 * render cycle anyway), plain function calls from non-component code (the
 * WS client updates state directly from `onmessage`, no dispatch plumbing),
 * and selector-based subscription is the exact shape this app's update
 * pattern needs.
 */
import { create } from "zustand";

export const VoiceState = {
  IDLE: "idle",
  LISTENING: "listening",
  PROCESSING: "processing",
  SPEAKING: "speaking",
};

// Mirrors backend agent_core.supervisor.state.Mode exactly — "Speech/Text"
// names which side takes the INPUT, "to Speech/Text" names the OUTPUT.
export const Mode = {
  TEXT_TO_TEXT: "text_to_text",
  SPEECH_TO_TEXT: "speech_to_text",
  TEXT_TO_SPEECH: "text_to_speech",
  SPEECH_TO_SPEECH: "speech_to_speech",
};

export const isVoiceInputMode = (mode) => mode === Mode.SPEECH_TO_TEXT || mode === Mode.SPEECH_TO_SPEECH;
export const isVoiceOutputMode = (mode) => mode === Mode.TEXT_TO_SPEECH || mode === Mode.SPEECH_TO_SPEECH;

export const ConnectionState = {
  CONNECTED: "connected",
  RECONNECTING: "reconnecting",
  DISCONNECTED: "disconnected",
};

let nextMessageId = 1;

export const useAppStore = create((set, get) => ({
  // --- Auth ---
  token: null,
  setToken: (token) => set({ token }),
  logout: () => set({ token: null }),

  // --- Mode ---
  // Voice-first product: default to full voice conversation, not the
  // (now-removed) type-and-read text mode.
  mode: Mode.SPEECH_TO_SPEECH,
  setMode: (mode) => set({ mode }),

  // --- Connection (backend REST / speech gateway WS) ---
  connectionState: ConnectionState.CONNECTED,
  setConnectionState: (connectionState) => set({ connectionState }),

  // --- Chat (text mode) ---
  messages: [], // {id, role: "user"|"assistant", text, streaming}
  addMessage: (role, text, { streaming = false } = {}) => {
    const id = nextMessageId++;
    set((s) => ({ messages: [...s.messages, { id, role, text, streaming }] }));
    return id;
  },
  appendToMessage: (id, delta) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, text: m.text + delta } : m)),
    })),
  finishMessage: (id) =>
    set((s) => ({
      messages: s.messages.map((m) => (m.id === id ? { ...m, streaming: false } : m)),
    })),

  // --- Language indicator ---
  responseLanguage: null,
  languageConfidence: null,
  isCodeMixed: false,
  setLanguageInfo: ({ responseLanguage, languageConfidence, isCodeMixed }) =>
    set({ responseLanguage, languageConfidence, isCodeMixed }),

  // --- Voice state machine ---
  voiceState: VoiceState.IDLE,
  setVoiceState: (voiceState) => set({ voiceState }),

  // Barge-in is a transient visual pulse, not a steady state — a component
  // watches this counter (not a boolean) so re-triggering while already
  // "flashed" still re-fires the animation.
  bargeInSignal: 0,
  triggerBargeIn: () => set((s) => ({ bargeInSignal: s.bargeInSignal + 1, voiceState: VoiceState.LISTENING })),

  reset: () =>
    set({
      messages: [],
      voiceState: VoiceState.IDLE,
      responseLanguage: null,
      languageConfidence: null,
      isCodeMixed: false,
    }),
}));

export const selectIsAuthenticated = (s) => Boolean(s.token);
