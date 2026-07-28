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
   * @param {boolean} [relaxedConstraints] - Confirmed LIVE this did NOT fix
   * the recurring "Intel Smart Sound Technology for Digital Microphones"
   * dead-track case: the negotiated track settings still reported
   * `echoCancellation: true, autoGainControl: true` even with neither
   * requested. That means this specific driver applies its DSP at the
   * OS/driver level, beneath what getUserMedia constraints can reach --
   * kept here since it's still a legitimate (and free) first attempt for
   * OTHER dead-track cases where the browser itself is responsible, but it
   * is not assumed to be sufficient alone; see `deviceId` below for the
   * fallback that actually did something on this driver.
   * @param {string} [deviceId] - Explicit input device to request, bypassing
   * whatever the browser picked as its default. The one lever left once
   * relaxedConstraints is confirmed powerless against driver-level DSP:
   * switch to a DIFFERENT physical device entirely (e.g. a webcam mic
   * instead of the laptop's built-in array), since a different device's
   * driver isn't forcing the same processing.
   */
  async start(preCreatedContext, relaxedConstraints = false, deviceId = null) {
    // `{ideal: true}`, not a bare `true`. A bare boolean is a HARD constraint
    // — reported live: on certain mic/driver stacks (confirmed across two
    // completely different devices/OSes, ruling out one bad mic), Chrome
    // couldn't actually satisfy all three simultaneously and, instead of
    // throwing, silently handed back a track present in name but delivering
    // literal all-zero samples (exact 0.0000 RMS — real ambient noise almost
    // never rounds to exactly zero, which was the tell). `ideal` lets the
    // browser do its best without ever degrading to a dead track -- but on
    // some driver stacks even that isn't enough (see relaxedConstraints).
    const audioConstraints = relaxedConstraints
      ? { channelCount: 1 }
      : {
          channelCount: 1,
          echoCancellation: { ideal: true },
          noiseSuppression: { ideal: true },
          autoGainControl: { ideal: true },
        };
    if (deviceId) audioConstraints.deviceId = { exact: deviceId };
    this._stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints });
    // Which physical device the browser actually picked — when a mic
    // delivers pure zeros (seen live: rms=0.0000), this label is usually
    // the answer (a virtual/disconnected device was selected).
    const track = this._stream.getAudioTracks()[0];
    this.deviceLabel = track?.label || "unknown device";
    this.deviceId = track?.getSettings?.().deviceId || null;
    console.info("[voice] using microphone:", this.deviceLabel, relaxedConstraints ? "(relaxed constraints)" : "");
    // Real gap: nothing ever logged what the browser ACTUALLY negotiated --
    // `{ideal: true}` can silently resolve to false (or true, on a driver
    // that then delivers a dead track anyway) with zero visibility either
    // way. This is the one log line that would have made the very first
    // occurrence of the all-zero-RMS bug immediately diagnosable instead of
    // guessed at.
    console.info("[voice] negotiated track settings:", track?.getSettings?.());
    // MediaStreamTrack.muted is the browser's OWN verdict on whether this
    // track is currently delivering real data -- distinct from the app's
    // own RMS-based dead-track guess, and the one signal that can tell an
    // OS/driver-level mute (this fires) apart from "the track looks live
    // but every sample happens to be zero for some other reason" (this
    // does NOT fire, since the browser itself still thinks data is
    // flowing). Never logged before this -- exactly the missing piece
    // needed to tell those two cases apart from a live report.
    console.info("[voice] track.muted at negotiation:", track?.muted, "readyState:", track?.readyState);
    if (track) {
      track.onmute = () => console.warn("[voice] track muted mid-session (browser/OS signaled no more data):", this.deviceLabel);
      track.onunmute = () => console.info("[voice] track unmuted mid-session:", this.deviceLabel);
    }
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

  /** Same teardown as stop(), but leaves the AudioContext open and returns
   * it — for the dead-track retry in useVoiceSession.js, which needs a NEW
   * MediaStream (via a fresh getUserMedia call with different constraints)
   * but must NOT create a brand new AudioContext this deep into an async
   * frame-processing callback: mobile browsers only allow AudioContext
   * creation synchronously inside the original click gesture (see start()'s
   * own preCreatedContext doc) -- by the time frames have been processing
   * for a second, that gesture window is long gone. */
  stopKeepingContext() {
    this._worklet?.port.close();
    this._worklet?.disconnect();
    this._sink?.disconnect();
    this._source?.disconnect();
    this._stream?.getTracks().forEach((track) => track.stop());
    const context = this._context;
    this._stream = null;
    this._context = null;
    this._source = null;
    this._worklet = null;
    this._sink = null;
    this._pending = [];
    return context;
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
