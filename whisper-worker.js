// Whisper inference worker — loads Whisper Tiny via transformers.js
// WebGPU active → inference ~0.2s. Use rolling 5s window.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1";

env.allowLocalModels = true;
env.allowRemoteModels = false;
env.localModelPath = "/whisper-model/";

self.onerror = (e) => {
  postMessage({ type: "error", message: "Worker uncaught: " + (e?.message || String(e)) });
};

let transcriber = null;
let isProcessing = false;
let audioBuffer = [];
let currentPrompt = "";

const WINDOW_SAMPLES = 16000 * 8;   // 8s window for quality
const TRIGGER_INTERVAL = 16000 * 2; // process every 2s of new audio (overlapping)
let lastProcessedAt = 0; // track how much audio we've processed

self.onmessage = async (e) => {
  const msg = e.data;

  if (msg.type === "init") {
    postMessage({ type: "loading", message: "Loading Whisper model..." });
    try {
      const pipelineOpts = {
        dtype: "fp32",
        progress_callback: (progress) => {
          if (progress.status === "progress") {
            postMessage({
              type: "loading_progress",
              percent: Math.round(progress.progress),
            });
          }
        },
      };
      // Try WebGPU first (10-50x faster), fallback to WASM
      try {
        pipelineOpts.device = "webgpu";
        transcriber = await pipeline("automatic-speech-recognition", "whisper-tiny", pipelineOpts);
        postMessage({ type: "info", message: "WebGPU active" });
      } catch (gpuErr) {
        postMessage({ type: "info", message: "WebGPU unavailable, using WASM" });
        delete pipelineOpts.device;
        transcriber = await pipeline("automatic-speech-recognition", "whisper-tiny", pipelineOpts);
      }
      postMessage({ type: "loading_progress", percent: 100 });
      postMessage({ type: "ready" });
    } catch (err) {
      postMessage({ type: "error", message: err?.message || String(err) || "Unknown error" });
    }
    return;
  }

  if (msg.type === "set_prompt") {
    currentPrompt = msg.text || "";
    return;
  }

  if (msg.type === "audio") {
    if (!transcriber) return;

    // Accumulate audio
    const samples = new Float32Array(msg.samples);
    for (let i = 0; i < samples.length; i++) {
      audioBuffer.push(samples[i]);
    }

    // Wait for at least 8s before first processing
    if (audioBuffer.length < WINDOW_SAMPLES) return;
    // After first processing, trigger every TRIGGER_INTERVAL samples of new audio
    if (isProcessing) return;
    const newAudioSinceLast = audioBuffer.length - lastProcessedAt;
    if (lastProcessedAt > 0 && newAudioSinceLast < TRIGGER_INTERVAL) return;

    isProcessing = true;

    // Take the last 8s (rolling window)
    const startIdx = Math.max(0, audioBuffer.length - WINDOW_SAMPLES);
    const toProcess = new Float32Array(audioBuffer.slice(startIdx));
    lastProcessedAt = audioBuffer.length;

    // Trim buffer: keep last 8s so it doesn't grow forever
    if (audioBuffer.length > WINDOW_SAMPLES * 2) {
      audioBuffer = audioBuffer.slice(-WINDOW_SAMPLES);
      lastProcessedAt = audioBuffer.length;
    }

    try {
      const opts = {
        language: "ar",
        task: "transcribe",
        return_timestamps: true,
      };
      if (currentPrompt) {
        opts.initial_prompt = currentPrompt;
      }
      const output = await transcriber(toProcess, opts);
      postMessage({
        type: "transcription",
        text: output.text || "",
        chunks: output.chunks || [],
      });
    } catch (err) {
      postMessage({ type: "error", message: err?.message || String(err) || "Inference error" });
    } finally {
      isProcessing = false;
      // Trim if buffer grew large during processing
      if (audioBuffer.length > WINDOW_SAMPLES * 3) {
        audioBuffer = audioBuffer.slice(-WINDOW_SAMPLES);
      }
    }
    return;
  }

  if (msg.type === "reset") {
    audioBuffer = [];
    isProcessing = false;
    lastProcessedAt = 0;
    return;
  }
};
