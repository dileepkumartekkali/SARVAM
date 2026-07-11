/**
 * Real microphone capture -> PCM16LE frames at 16kHz, matching the gateway's
 * `/ws/stt` contract exactly (agent_core/speech/audio_validation.py expects
 * ~32ms frames = 512 samples at 16kHz = 1024 bytes).
 *
 * Uses AudioWorkletNode (the worklet module lives at public/mic-worklet.js
 * and only forwards raw float blocks) — replaced the deprecated
 * ScriptProcessorNode, which logged a deprecation warning on every single
 * mic start. Resampling and PCM16 conversion stay here on the main thread.
 *
 * Never trusts the context to capture at 16kHz — some browser/hardware
 * combinations keep their native rate (44.1/48kHz) silently. Frames are
 * always explicitly resampled to TARGET_SAMPLE_RATE; mislabeled-rate frames
 * reach Sarvam as garbage it can't transcribe (hit live once).
 *
 * `autoGainControl: true` matters for the voice gate in useVoiceSession.js:
 * without it, quiet mics produce speech at RMS levels barely above the noise
 * floor and voice detection becomes guesswork (hit live: a real mic whose
 * speech never crossed the old fixed threshold, so the gate cut the mic off
 * mid-sentence).
 */
const TARGET_SAMPLE_RATE = 16000;
const FRAME_SAMPLES = 512; // 32ms at 16kHz — one gateway frame per emitted chunk

export class MicCapture {
  constructor(onFrame) {
    this._onFrame = onFrame;
    this._stream = null;
    this._context = null;
    this._source = null;
    this._worklet = null;
    this._sink = null;
    this._pending = []; // resampled-to-16kHz samples awaiting a full FRAME_SAMPLES chunk
  }

  /**
   * @param {AudioContext} [preCreatedContext] - Pass an AudioContext that was
   * created SYNCHRONOUSLY inside the click handler. Mobile browsers (iOS Safari,
   * Chrome Android) suspend or reject AudioContext creation in async callbacks
   * because the user-gesture context has already been lost by then.
   * Call `new AudioContext()` directly in your onClick handler and pass it here.
   */
  async start(preCreatedContext) {
    this._stream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });
    // Which physical device the browser actually picked — when a mic
    // delivers pure zeros (seen live: rms=0.0000), this label is usually
    // the answer (a virtual/disconnected device was selected).
    this.deviceLabel = this._stream.getAudioTracks()[0]?.label || "unknown device";
    console.info("[voice] using microphone:", this.deviceLabel);
    // Use the pre-created context (gesture-safe on mobile) or create one now
    // as fallback for desktop where timing is not restricted.
    this._context = preCreatedContext || new AudioContext();
    // Mobile browsers auto-suspend AudioContext — must explicitly resume it.
    if (this._context.state === "suspended") {
      await this._context.resume();
    }
    // mic-worklet.js lives in public/ and is served at the site root as /mic-worklet.js.
    // Do NOT use `new URL("./mic-worklet.js", import.meta.url)` here — Vite resolves
    // that to /assets/mic-worklet.js (the bundle output dir), which is a 404 because
    // public/ files are copied to the root, not to assets/.
    await this._context.audioWorklet.addModule("/mic-worklet.js");
    this._source = this._context.createMediaStreamSource(this._stream);
    this._worklet = new AudioWorkletNode(this._context, "mic-capture");
    this._worklet.port.onmessage = (e) => {
      const resampled = resampleTo16kHz(e.data, this._context.sampleRate);
      for (let i = 0; i < resampled.length; i++) this._pending.push(resampled[i]);
      while (this._pending.length >= FRAME_SAMPLES) {
        const frame = this._pending.splice(0, FRAME_SAMPLES);
        this._onFrame(floatTo16BitPCM(frame));
      }
    };
    this._source.connect(this._worklet);
    this._sink = this._context.createGain();
    this._sink.gain.value = 0;
    this._worklet.connect(this._sink);
    this._sink.connect(this._context.destination);
    // Worklet nodes process without being routed to the output — the mic
    // audio is never played back.
  }

  stop() {
    this._worklet?.port.close();
    this._worklet?.disconnect();
    this._sink?.disconnect();
    this._source?.disconnect();
    this._stream?.getTracks().forEach((track) => track.stop());
    this._context?.close();
    this._stream = null;
    this._context = null;
    this._source = null;
    this._worklet = null;
    this._sink = null;
    this._pending = [];
  }
}

function resampleTo16kHz(float32Array, fromRate) {
  if (fromRate === TARGET_SAMPLE_RATE) return float32Array;
  const ratio = fromRate / TARGET_SAMPLE_RATE;
  const outLength = Math.floor(float32Array.length / ratio);
  const result = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcIndex = i * ratio;
    const lo = Math.floor(srcIndex);
    const hi = Math.min(lo + 1, float32Array.length - 1);
    const frac = srcIndex - lo;
    result[i] = float32Array[lo] * (1 - frac) + float32Array[hi] * frac;
  }
  return result;
}

function floatTo16BitPCM(samples) {
  const buffer = new ArrayBuffer(samples.length * 2);
  const view = new DataView(buffer);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }
  return buffer;
}
