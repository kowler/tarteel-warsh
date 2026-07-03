// AudioWorklet processor — resamples to 16kHz, sends chunks as transferable ArrayBuffer
// Adapted from Tilawa (github.com/yazinsai/tilawa)
class AudioStreamProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._buffer = [];
    this._bufferSize = 4800; // 300ms at 16kHz
    this._resamplePhase = 0; // fractional position for linear interpolation
    this.port.onmessage = (event) => {
      const msg = event.data || {};
      if (msg.type === "set_config") {
        const chunkMs = Number(msg.audioChunkMs);
        if (Number.isFinite(chunkMs)) {
          const clamped = Math.min(1000, Math.max(100, chunkMs));
          this._bufferSize = Math.max(1, Math.round((16000 * clamped) / 1000));
        }
      }
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input[0]) return true;

    const channelData = input[0];
    const inputSampleRate = sampleRate; // global in AudioWorkletGlobalScope
    const outputSampleRate = 16000;
    const ratio = inputSampleRate / outputSampleRate;

    // Linear interpolation resampling (much better than decimation for speech)
    for (let i = Math.floor(this._resamplePhase); i < channelData.length; i += ratio) {
      const idx = Math.floor(i);
      const frac = i - idx;
      const s1 = channelData[idx] || 0;
      const s2 = channelData[idx + 1] || s1;
      this._buffer.push(s1 + (s2 - s1) * frac);
    }
    // Track fractional remainder for continuity across process() calls
    this._resamplePhase = (this._resamplePhase + channelData.length) % ratio;

    if (this._buffer.length >= this._bufferSize) {
      const chunk = new Float32Array(this._buffer);
      this.port.postMessage(chunk.buffer, [chunk.buffer]); // zero-copy transfer
      this._buffer = [];
    }

    return true;
  }
}

registerProcessor("audio-stream-processor", AudioStreamProcessor);
