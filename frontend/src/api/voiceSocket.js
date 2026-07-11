import { GATEWAY_WS_URL } from "./config";

/**
 * Speech Gateway WS client (mic -> /ws/stt, TTS audio <- /ws/tts).
 *
 * Untestable live in this environment: there's no real Sarvam/Azure account
 * or a browser with mic permission available here (same constraint noted in
 * Phases 4-5's READMEs). What IS real and exercised: the reconnect-with-
 * backoff logic and the state transitions it drives — verified by unit-style
 * calls against a fake WebSocket in this module's usage from the voice hook,
 * not against a live gateway.
 */
export class VoiceSocketClient {
  constructor({ onOpen, onClose, onEvent, onAudioChunk, onReconnecting } = {}) {
    this._onOpen = onOpen || (() => {});
    this._onClose = onClose || (() => {});
    this._onEvent = onEvent || (() => {});
    this._onAudioChunk = onAudioChunk || (() => {});
    this._onReconnecting = onReconnecting || (() => {});
    this._stt = null;
    this._tts = null;
    this._reconnectAttempt = 0;
    this._closedByClient = false;
  }

  connectSTT(config) {
    this._closedByClient = false;
    this._openSTT(config);
  }

  _openSTT(config) {
    const ws = new WebSocket(`${GATEWAY_WS_URL}/ws/stt`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => {
      this._reconnectAttempt = 0;
      ws.send(JSON.stringify(config));
      this._onOpen();
    };
    ws.onmessage = (evt) => {
      if (typeof evt.data === "string") {
        this._onEvent(JSON.parse(evt.data));
      }
    };
    ws.onclose = () => {
      this._onClose();
      if (!this._closedByClient) {
        this._scheduleReconnect(config);
      }
    };
    this._stt = ws;
  }

  _scheduleReconnect(config) {
    this._reconnectAttempt += 1;
    this._onReconnecting(this._reconnectAttempt);
    const delay = Math.min(250 * 2 ** (this._reconnectAttempt - 1), 4000);
    setTimeout(() => {
      if (!this._closedByClient) this._openSTT(config);
    }, delay);
  }

  sendAudioFrame(frame) {
    if (this._stt && this._stt.readyState === WebSocket.OPEN) {
      this._stt.send(frame);
    }
  }

  closeSTT() {
    this._closedByClient = true;
    this._stt?.close();
  }

  connectTTS(config) {
    const ws = new WebSocket(`${GATEWAY_WS_URL}/ws/tts`);
    ws.binaryType = "arraybuffer";
    ws.onopen = () => ws.send(JSON.stringify(config));
    ws.onmessage = (evt) => {
      if (evt.data instanceof ArrayBuffer) {
        this._onAudioChunk(evt.data);
      } else {
        // {"type":"error",...} or {"type":"text_only_fallback",...} — the
        // gateway's only way to say synthesis failed. Previously dropped
        // silently here, so a TTS failure looked identical to a normal
        // "utterance finished" close — no error ever reached the caller.
        this._onEvent(JSON.parse(evt.data));
      }
    };
    this._tts = ws;
    return ws;
  }

  sendText(text) {
    if (this._tts && this._tts.readyState === WebSocket.OPEN) {
      this._tts.send(JSON.stringify({ text }));
    }
  }

  endTTSUtterance() {
    this.sendText("__END__");
  }

  closeTTS() {
    this._tts?.close();
  }
}
