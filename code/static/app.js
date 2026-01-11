import { initVisualizer, updateFrequency } from './visualizer.js';

let socket = null;
let audioContext = null;
let mediaStream = null;
let micWorkletNode = null;
let ttsWorkletNode = null;

// Visualizer Variables
let analyser = null;
let animationFrameId = null;

let isTTSPlaying = false;
let ignoreIncomingTTS = false;

// --- batching setup ---
const BATCH_SAMPLES = 2048;
const HEADER_BYTES  = 8;
const FRAME_BYTES   = BATCH_SAMPLES * 2;
const MESSAGE_BYTES = HEADER_BYTES + FRAME_BYTES;
const bufferPool = [];
let batchBuffer = null;
let batchView = null;
let batchInt16 = null;
let batchOffset = 0;

// Initialize 3D Visualizer immediately
initVisualizer(document.getElementById('canvas-container'));

function initBatch() {
  if (! batchBuffer) {
    batchBuffer = bufferPool.pop() || new ArrayBuffer(MESSAGE_BYTES);
    batchView   = new DataView(batchBuffer);
    batchInt16  = new Int16Array(batchBuffer, HEADER_BYTES);
    batchOffset = 0;
  }
}

function flushBatch() {
  const ts = Date.now() & 0xFFFFFFFF;
  batchView.setUint32(0, ts, false);
  const flags = isTTSPlaying ? 1 : 0;
  batchView.setUint32(4, flags, false);
  if (socket && socket.readyState === WebSocket.OPEN) {
      socket.send(batchBuffer);
  }
  bufferPool.push(batchBuffer);
  batchBuffer = null;
}

function flushRemainder() {
  if (batchOffset > 0) {
    for (let i = batchOffset; i < BATCH_SAMPLES; i++) { batchInt16[i] = 0; }
    flushBatch();
  }
}

// ---------------------------------------------------------
// AUDIO LOGIC
// ---------------------------------------------------------

function createAudioContext() {
  if (! audioContext) {
    audioContext = new AudioContext(); 
    console.log("AudioContext created with sampleRate:", audioContext.sampleRate);
  }
}

function base64ToInt16Array(b64) {
  const raw = atob(b64);
  const buf = new ArrayBuffer(raw.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < raw.length; i++) { view[i] = raw.charCodeAt(i); }
  return new Int16Array(buf);
}

// --- VISUALIZER LOOP ---
function animateVisualizer() {
  if (!isTTSPlaying || !analyser) {
    updateFrequency(0);
    return;
  }

  animationFrameId = requestAnimationFrame(animateVisualizer);

  const dataArray = new Uint8Array(analyser.frequencyBinCount);
  analyser.getByteFrequencyData(dataArray);

  let sum = 0;
  const relevantBins = Math.floor(dataArray.length / 2); 
  for (let i = 0; i < relevantBins; i++) {
    sum += dataArray[i];
  }
  const average = sum / relevantBins;
  updateFrequency(average);
}

async function startRawPcmCapture() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { 
        channelCount: 1, 
        echoCancellation: true, 
        noiseSuppression: true,
        autoGainControl: true
      }
    });
    mediaStream = stream;
    
    createAudioContext();
    if (audioContext.state === 'suspended') {
      await audioContext.resume();
    }

    try {
        await audioContext.audioWorklet.addModule(new URL('./pcmWorkletProcessor.js', import.meta.url));
    } catch (e) {
        await audioContext.audioWorklet.addModule('./pcmWorkletProcessor.js');
    }

    micWorkletNode = new AudioWorkletNode(audioContext, 'pcm-worklet-processor');

    micWorkletNode.port.onmessage = ({ data }) => {
      const incoming = new Int16Array(data);
      let read = 0;
      while (read < incoming.length) {
        initBatch();
        const toCopy = Math.min(incoming.length - read, BATCH_SAMPLES - batchOffset);
        batchInt16.set(incoming.subarray(read, read + toCopy), batchOffset);
        batchOffset += toCopy;
        read += toCopy;
        if (batchOffset === BATCH_SAMPLES) flushBatch();
      }
    };

    const source = audioContext.createMediaStreamSource(stream);
    source.connect(micWorkletNode);
  } catch (err) {
    console.error("Mic/Worklet Error:", err);
  }
}

async function setupTTSPlayback() {
  try {
      try {
          await audioContext.audioWorklet.addModule(new URL('./ttsPlaybackProcessor.js', import.meta.url));
      } catch (e) {
          await audioContext.audioWorklet.addModule('./ttsPlaybackProcessor.js');
      }

      ttsWorkletNode = new AudioWorkletNode(audioContext, 'tts-playback-processor');

      analyser = audioContext.createAnalyser();
      analyser.fftSize = 512; 
      analyser.smoothingTimeConstant = 0.8;

      ttsWorkletNode.port.onmessage = (event) => {
        const { type } = event.data;

        if (type === 'ttsPlaybackStarted') {
          if (! isTTSPlaying && socket && socket.readyState === WebSocket.OPEN) {
            isTTSPlaying = true;
            animateVisualizer();
            socket.send(JSON.stringify({ type: 'tts_start' }));
          }
        } else if (type === 'ttsPlaybackStopped') {
          if (isTTSPlaying && socket && socket.readyState === WebSocket.OPEN) {
            isTTSPlaying = false;
            cancelAnimationFrame(animationFrameId);
            updateFrequency(0);
            socket.send(JSON.stringify({ type: 'tts_stop' }));
          }
        }
      };

      ttsWorkletNode.connect(analyser);
      analyser.connect(audioContext.destination);
  } catch (e) {
      console.error("TTS Setup Error:", e);
  }
}

function cleanupAudio() {
  if (micWorkletNode) { micWorkletNode.disconnect(); micWorkletNode = null; }
  if (ttsWorkletNode) { ttsWorkletNode.disconnect(); ttsWorkletNode = null; }
  if (analyser) { analyser.disconnect(); analyser = null; }
  if (audioContext && audioContext.state !== 'closed') { audioContext.suspend(); }
  if (mediaStream) { mediaStream.getAudioTracks().forEach(track => track.stop()); mediaStream = null; }
}

function handleJSONMessage({ type, content }) {
  if (type === "tts_chunk") {
    if (ignoreIncomingTTS) return;
    const int16Data = base64ToInt16Array(content);
    if (ttsWorkletNode) ttsWorkletNode.port.postMessage(int16Data);
    return;
  }
  if (type === "stop_tts" || type === "tts_interruption") {
    if (ttsWorkletNode) ttsWorkletNode.port.postMessage({ type: "clear" });
    isTTSPlaying = false;
    ignoreIncomingTTS = (type === "stop_tts");
    
    cancelAnimationFrame(animationFrameId);
    updateFrequency(0);

    if (type === "stop_tts" && socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify({ type: 'tts_stop' }));
    }
    return;
  }
}

// --- MAIN START HANDLER ---
let isConnecting = false;

async function startSession() {
  if (isConnecting) return;
  if (socket && (socket.readyState === WebSocket.OPEN || socket.readyState === WebSocket.CONNECTING)) return;

  isConnecting = true;
  
  try {
    createAudioContext();
    await audioContext.resume();
    
    const wsProto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const port = window.location.port;
    const host = window.location.hostname;

    socket = new WebSocket(`${wsProto}//${host}:${port}/ws`);

    socket.onopen = async () => {
      await startRawPcmCapture();
      await setupTTSPlayback();
      socket.send(JSON.stringify({ type: 'set_speed', speed:  30 }));
    };

    socket.onmessage = (evt) => {
      if (typeof evt.data === "string") {
        try {
          const msg = JSON.parse(evt.data);
          handleJSONMessage(msg);
        } catch (e) {
          console.error("Error parsing message:", e);
        }
      }
    };

    socket.onclose = () => {
      isConnecting = false;
      flushRemainder();
      cleanupAudio();
      setTimeout(() => startSession(), 2000);
    };

    socket.onerror = (err) => {
      isConnecting = false;
      console.error(err);
      cleanupAudio();
    };
  } catch (e) {
    isConnecting = false;
    throw e;
  }
}

// Auto-start when the page loads
document.addEventListener('click', async function initOnClick() {
  document.removeEventListener('click', initOnClick);
  await startSession();
}, { once: true });

document.addEventListener('DOMContentLoaded', async () => {
  setTimeout(async () => {
    try {
      await startSession();
    } catch (e) {
      console.log("Waiting for user interaction to start audio...");
    }
  }, 500);
});