// Whisper inference worker — loads Whisper Tiny via transformers.js
// Runs in Web Worker, accumulates 16kHz audio chunks, runs inference every ~2.5s
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.17.0";

// Allow local model files only (no remote downloads)
env.allowRemoteModels = false;
env.localModelPath = "/whisper-model/";

let transcriber = null;
let isProcessing = false;
let audioBuffer = [];
const ACCUMULATE_SAMPLES = 40000; // ~2.5s at 16kHz — Whisper needs at least 1-2s
const MAX_BUFFER_SAMPLES = 16000 * 30; // 30s max — Whisper's max context

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === "init") {
    postMessage({ type: "loading", message: "Loading Whisper model..." });
    try {
      transcriber = await pipeline("automatic-speech-recognition", "whisper-tiny", {
        dtype: "q8",
        progress_callback: (progress) => {
          if (progress.status === "progress") {
            postMessage({
              type: "loading_progress",
              percent: Math.round(progress.progress),
            });
          }
        },
      });
      postMessage({ type: "ready" });
    } catch (err) {
      postMessage({ type: "error", message: err.message });
    }
    return;
  }

  if (msg.type === "audio") {
    if (!transcriber) return;

    // Accumulate audio samples
    const samples = new Float32Array(msg.samples);
    for (let i = 0; i < samples.length; i++) {
      audioBuffer.push(samples[i]);
    }

    // Wait until we have enough audio and we're not already processing
    if (audioBuffer.length < ACCUMULATE_SAMPLES) return;
    if (isProcessing) return; // drop chunk if still processing

    isProcessing = true;

    // Take accumulated audio (cap at MAX_BUFFER_SAMPLES)
    let toProcess;
    if (audioBuffer.length > MAX_BUFFER_SAMPLES) {
      // Keep only the last MAX_BUFFER_SAMPLES
      toProcess = new Float32Array(audioBuffer.slice(-MAX_BUFFER_SAMPLES));
      audioBuffer = audioBuffer.slice(-MAX_BUFFER_SAMPLES);
    } else {
      toProcess = new Float32Array(audioBuffer);
    }

    try {
      // Run Whisper inference with Arabic language hint
      const output = await transcriber(toProcess, {
        language: "ar",
        task: "transcribe",
        chunk_length_s: 30,
        stride_length_s: 5,
        return_timestamps: true,
      });

      // Send back recognized text + timestamps
      postMessage({
        type: "transcription",
        text: output.text || "",
        chunks: output.chunks || [],
      });
    } catch (err) {
      postMessage({ type: "error", message: err.message });
    } finally {
      isProcessing = false;
      // Keep last 0.5s of audio for continuity between inference cycles
      const keepSamples = 8000; // 0.5s at 16kHz
      if (audioBuffer.length > keepSamples) {
        audioBuffer = audioBuffer.slice(-keepSamples);
      }
    }
    return;
  }

  if (msg.type === "reset") {
    audioBuffer = [];
    isProcessing = false;
    return;
  }
};
