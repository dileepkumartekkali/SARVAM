/**
 * Plays TTS audio chunks arriving over `/ws/tts`, strictly ONE AT A TIME.
 *
 * `playChunk` used to start each chunk's playback immediately on arrival —
 * fine for the first chunk, but every chunk after it starts on the same
 * shared AudioContext before the previous one finished playing, since the
 * gateway streams chunks faster than they take to speak. The result: two (or
 * more) chunks audibly overlapping, heard as a second voice cutting into the
 * first. Fixed with an explicit queue — `playChunk` enqueues and returns a
 * promise that resolves only once THIS chunk has actually played, and chunks
 * are drained one at a time, each waiting for the previous chunk's
 * `onended`.
 *
 * Real bug hit live, reported as "TTS speaking not clear": Sarvam's own
 * docs (confirmed against the real API, not assumed) say `output_audio_codec`
 * defaults to MP3 when the backend's config omits it — which it always did.
 * Each streamed chunk is a FRAGMENT of a continuous MP3 stream, not a
 * self-contained file; MP3's frame-to-frame bit-reservoir dependencies mean
 * decoding arbitrary fragments in isolation (exactly this queue's
 * chunk-by-chunk model) produced glitchy, unclear audio. The backend
 * (agent_core/speech/sarvam_tts.py) now explicitly requests uncompressed
 * `linear16` PCM at a fixed 24000Hz — confirmed live to work (a real
 * synthesize() call returned real audio bytes at this rate) — NOT the
 * 22050Hz this constant used to guess, which was never actually verified
 * against a real payload. Separately: an earlier key on this account had no
 * bulbul:v3 access at all (confirmed live — every v3-only speaker rejected
 * as incompatible with bulbul:v2); a key with real v3 access is now in use
 * (useVoiceSession.js, sarvam_tts.py). `decodeAudioData` below is expected to
 * always fail on raw PCM (it has no container header to parse) and fall
 * through to the PCM16 path every time — that's the normal path now, not a
 * rare fallback.
 */
const FALLBACK_SAMPLE_RATE = 24000;

export class TTSPlayer {
  constructor() {
    this._context = null;
    this._queue = [];
    this._draining = false;
    // Decoded (post-decode, uniform Float32 PCM) samples for the current
    // utterance — capturing here rather than the raw chunk bytes sidesteps
    // Sarvam-vs-Azure-fallback format differences, since everything is
    // already normalized by the time it lands here.
    this._capturedChannels = [];
    this._capturedSampleRate = null;
  }

  playChunk(arrayBuffer) {
    return new Promise((resolve, reject) => {
      this._queue.push({ arrayBuffer, resolve, reject });
      this._drain();
    });
  }

  async _drain() {
    if (this._draining) return; // already draining — this chunk was just added to the queue it's working through
    this._draining = true;
    while (this._queue.length > 0) {
      const { arrayBuffer, resolve, reject } = this._queue.shift();
      try {
        await this._playOne(arrayBuffer);
        resolve();
      } catch (e) {
        reject(e);
      }
    }
    this._draining = false;
  }

  async _playOne(arrayBuffer) {
    const context = this._ensureContext();
    let audioBuffer;
    try {
      // decodeAudioData detaches/consumes the buffer it's given.
      audioBuffer = await context.decodeAudioData(arrayBuffer.slice(0));
    } catch {
      audioBuffer = this._decodeRawPCM16(context, arrayBuffer);
    }
    if (this._capturedSampleRate == null) this._capturedSampleRate = audioBuffer.sampleRate;
    this._capturedChannels.push(audioBuffer.getChannelData(0).slice());
    return new Promise((resolve) => {
      const source = context.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(context.destination);
      source.onended = resolve;
      source.start();
    });
  }

  _decodeRawPCM16(context, arrayBuffer) {
    const view = new DataView(arrayBuffer);
    const sampleCount = Math.floor(arrayBuffer.byteLength / 2);
    const audioBuffer = context.createBuffer(1, sampleCount, FALLBACK_SAMPLE_RATE);
    const channel = audioBuffer.getChannelData(0);
    for (let i = 0; i < sampleCount; i++) {
      channel[i] = view.getInt16(i * 2, true) / 0x8000;
    }
    return audioBuffer;
  }

  _ensureContext() {
    if (!this._context) this._context = new AudioContext();
    return this._context;
  }

  /** Concatenates everything played since the last `finish()`/`close()` into
   * one WAV blob for replay storage. Returns `null` if nothing played. */
  finish() {
    if (this._capturedChannels.length === 0) return null;
    const totalLength = this._capturedChannels.reduce((sum, c) => sum + c.length, 0);
    const merged = new Float32Array(totalLength);
    let offset = 0;
    for (const chunk of this._capturedChannels) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    const blob = encodeWav(merged, this._capturedSampleRate);
    this._capturedChannels = [];
    this._capturedSampleRate = null;
    return blob;
  }

  close() {
    this._context?.close();
    this._context = null;
    this._queue = [];
    this._draining = false;
    this._capturedChannels = [];
    this._capturedSampleRate = null;
  }
}

/** Minimal 44-byte-header PCM16 mono WAV encoder — no library needed for
 * this one shape (mono, 16-bit, whatever sample rate we decoded at). */
function encodeWav(samples, sampleRate) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const writeString = (offset, str) => {
    for (let i = 0; i < str.length; i++) view.setUint8(offset + i, str.charCodeAt(i));
  };
  writeString(0, "RIFF");
  view.setUint32(4, 36 + samples.length * 2, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // fmt chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, samples.length * 2, true);
  for (let i = 0; i < samples.length; i++) {
    const clamped = Math.max(-1, Math.min(1, samples[i]));
    view.setInt16(44 + i * 2, clamped < 0 ? clamped * 0x8000 : clamped * 0x7fff, true);
  }
  return new Blob([buffer], { type: "audio/wav" });
}
