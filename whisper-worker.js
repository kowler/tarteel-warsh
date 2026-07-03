// Whisper inference worker — loads tarteel Quran Whisper via transformers.js
// Runs in Web Worker, receives 16kHz audio chunks, returns recognized text
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.17.0";

// Allow local model files
env.allowRemoteModels = false;
env.localModelPath = "/whisper-model/";

let transcriber = null;
let isProcessing = false;

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
    if (!transcriber || isProcessing) return;
    isProcessing = true;

    try {
      const samples = new Float32Array(msg.samples);

      // Run Whisper inference with Arabic language hint
      const output = await transcriber(samples, {
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
    }
    return;
  }

  if (msg.type === "reset") {
    isProcessing = false;
    return;
  }
};
