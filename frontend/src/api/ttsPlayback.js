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
 * Sarvam's streaming TTS defaults to WAV-encoded output, so each chunk is
 * decoded with the standard Web Audio decoder; if a chunk ever arrives as
 * headerless raw PCM instead, this falls back to treating it as PCM16LE mono
 * at Sarvam's documented default TTS sample rate (22050Hz).
 *
 * NOT verified against a real Sarvam TTS audio payload — only the transport
 * framing (base64-over-JSON) was confirmed live this session, never the
 * actual audio bytes. Confirm with one real call before depending on this.
 */
const FALLBACK_SAMPLE_RATE = 22050;

export class TTSPlayer {
  constructor() {
    this._context = null;
    this._queue = [];
    this._draining = false;
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

  close() {
    this._context?.close();
    this._context = null;
    this._queue = [];
    this._draining = false;
  }
}
