/**
 * AudioWorkletProcessor for microphone capture — replaces the deprecated
 * ScriptProcessorNode (which spammed a deprecation warning on every mic
 * start). Runs on the audio rendering thread; accumulates the 128-sample
 * render quanta into ~2048-sample blocks before posting to the main thread,
 * so message traffic stays at ~23/sec instead of ~375/sec.
 *
 * Plain JS served from public/ (not bundled) — AudioWorklet modules load by
 * URL via audioWorklet.addModule('/mic-worklet.js').
 */
const POST_BLOCK_SAMPLES = 2048;

class MicCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._chunks = [];
    this._length = 0;
  }

  process(inputs) {
    const channel = inputs[0] && inputs[0][0];
    if (channel && channel.length > 0) {
      this._chunks.push(new Float32Array(channel)); // copy — the input buffer is reused by the engine
      this._length += channel.length;
      if (this._length >= POST_BLOCK_SAMPLES) {
        const block = new Float32Array(this._length);
        let offset = 0;
        for (const c of this._chunks) {
          block.set(c, offset);
          offset += c.length;
        }
        this.port.postMessage(block, [block.buffer]); // transfer, no copy
        this._chunks = [];
        this._length = 0;
      }
    }
    return true; // keep processing until the node is disconnected
  }
}

registerProcessor("mic-capture", MicCaptureProcessor);
