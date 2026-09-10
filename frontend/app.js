/**
 * RPS AI BATTLE — 100% Client-Side ONNX Runtime Web Controller
 * Runs deep learning model 'rps_model.onnx' directly inside the browser using WebGL/WASM.
 * Matches original python logic from main.py.
 */

// Model & Game Configuration
const MODEL_PATH = './rps_model.onnx';
const MODEL_TOTAL_BYTES = 13898247; // 13.9 MB
const CLASSES = ['paper', 'rock', 'scissors']; // Index 0: paper, 1: rock, 2: scissors
const EMOJIS = {
  paper: '✋',
  rock: '✊',
  scissors: '✌️'
};

// Global App State
const state = {
  session: null,
  mode: 'battle', // 'battle' or 'live'
  gameState: 'IDLE', // 'IDLE', 'COUNTDOWN', 'CAPTURE', 'RESULT'
  modelLoaded: false,
  isInferring: false,
  soundEnabled: true,
  isMirrored: true,
  showPip: true,
  
  // Game Stats
  playerScore: 0,
  cpuScore: 0,
  tieScore: 0,
  streak: 0,
  round: 1,

  // Last Moves
  lastPlayerMove: null,
  lastCpuMove: null,
  lastOutcome: null,
  
  // Inference Engine Mode
  inferenceEngine: 'client', // 'client' or 'backend'
  backendUrl: 'http://localhost:8080',
  
  // Stream & Video
  stream: null,
  videoEl: null,
  offscreenCanvas: null,
  offscreenCtx: null,

  // Loop & Timer IDs
  liveLoopId: null,
  countdownTimerId: null,
  shuffleIntervalId: null
};

// DOM Elements Cache
const DOM = {};

document.addEventListener('DOMContentLoaded', () => {
  cacheDOMElements();
  setupAudioContext();
  setupEventListeners();
  initOffscreenCanvas();
  startCamera();
  loadOnnxModel();
});

function cacheDOMElements() {
  DOM.webcam = document.getElementById('webcam');
  DOM.overlayCanvas = document.getElementById('overlayCanvas');
  DOM.aiSeeCanvas = document.getElementById('aiSeeCanvas');
  DOM.debugPipCard = document.getElementById('debugPipCard');
  DOM.handTargetBox = document.getElementById('handTargetBox');
  DOM.targetBoxLabel = document.getElementById('targetBoxLabel');
  DOM.countdownOverlay = document.getElementById('countdownOverlay');
  DOM.countdownText = document.getElementById('countdownText');
  DOM.countdownSub = document.getElementById('countdownSub');
  DOM.cameraFallback = document.getElementById('cameraFallback');
  DOM.enableCamBtn = document.getElementById('enableCamBtn');

  // Status & Loaders
  DOM.modelLoaderOverlay = document.getElementById('modelLoaderOverlay');
  DOM.modelProgressBar = document.getElementById('modelProgressBar');
  DOM.modelProgressPct = document.getElementById('modelProgressPct');
  DOM.modelProgressDetail = document.getElementById('modelProgressDetail');
  DOM.modelLoadedMb = document.getElementById('modelLoadedMb');
  DOM.engineStatusPill = document.getElementById('engineStatusPill');
  DOM.engineStatusDot = document.getElementById('engineStatusDot');
  DOM.engineStatusText = document.getElementById('engineStatusText');
  DOM.inferenceFpsBadge = document.getElementById('inferenceFpsBadge');
  DOM.sfxToggleBtn = document.getElementById('sfxToggleBtn');
  DOM.sfxIcon = document.getElementById('sfxIcon');

  // Buttons & Controls
  DOM.modeBattleBtn = document.getElementById('modeBattleBtn');
  DOM.modeLiveBtn = document.getElementById('modeLiveBtn');
  DOM.startRoundBtn = document.getElementById('startRoundBtn');
  DOM.startBtnText = document.getElementById('startBtnText');
  DOM.resetScoreBtn = document.getElementById('resetScoreBtn');
  DOM.toggleCameraBtn = document.getElementById('toggleCameraBtn');
  DOM.camToggleText = document.getElementById('camToggleText');
  DOM.flipCamBtn = document.getElementById('flipCamBtn');
  DOM.togglePipBtn = document.getElementById('togglePipBtn');
  DOM.clearHistoryBtn = document.getElementById('clearHistoryBtn');

  // Scoreboard
  DOM.playerScore = document.getElementById('playerScore');
  DOM.cpuScore = document.getElementById('cpuScore');
  DOM.tieScore = document.getElementById('tieScore');
  DOM.playerWinPct = document.getElementById('playerWinPct');
  DOM.cpuWinPct = document.getElementById('cpuWinPct');
  DOM.streakCount = document.getElementById('streakCount');
  DOM.roundCounter = document.getElementById('roundCounter');

  // Player & CPU Arena Cards
  DOM.playerMovePill = document.getElementById('playerMovePill');
  DOM.playerMoveIcon = document.getElementById('playerMoveIcon');
  DOM.playerMoveText = document.getElementById('playerMoveText');
  DOM.playerMoveConf = document.getElementById('playerMoveConf');

  DOM.cpuStateText = document.getElementById('cpuStateText');
  DOM.cpuHandCard = document.getElementById('cpuHandCard');
  DOM.cpuMoveIcon = document.getElementById('cpuMoveIcon');
  DOM.cpuMoveName = document.getElementById('cpuMoveName');
  DOM.cpuMoveSub = document.getElementById('cpuMoveSub');
  DOM.cpuMindText = document.getElementById('cpuMindText');

  // Center Stage
  DOM.vsEmblem = document.getElementById('vsEmblem');
  DOM.outcomeBanner = document.getElementById('outcomeBanner');
  DOM.outcomeIcon = document.getElementById('outcomeIcon');
  DOM.outcomeTitle = document.getElementById('outcomeTitle');
  DOM.outcomeSub = document.getElementById('outcomeSub');

  // Telemetry Bars
  DOM.probValRock = document.getElementById('probValRock');
  DOM.probFillRock = document.getElementById('probFillRock');
  DOM.probValPaper = document.getElementById('probValPaper');
  DOM.probFillPaper = document.getElementById('probFillPaper');
  DOM.probValScissors = document.getElementById('probValScissors');
  DOM.probFillScissors = document.getElementById('probFillScissors');
  DOM.telemetryLatencyTag = document.getElementById('telemetryLatencyTag');

  // Match History
  DOM.historyChips = document.getElementById('historyChips');

  // Settings Modal Elements
  DOM.settingsToggleBtn = document.getElementById('settingsToggleBtn');
  DOM.settingsModal = document.getElementById('settingsModal');
  DOM.closeSettingsBtn = document.getElementById('closeSettingsBtn');
  DOM.saveSettingsBtn = document.getElementById('saveSettingsBtn');
  DOM.radioCardClient = document.getElementById('radioCardClient');
  DOM.radioCardBackend = document.getElementById('radioCardBackend');
  DOM.backendUrlGroup = document.getElementById('backendUrlGroup');
  DOM.backendUrlInput = document.getElementById('backendUrlInput');
  DOM.testBackendBtn = document.getElementById('testBackendBtn');
  DOM.backendTestStatus = document.getElementById('backendTestStatus');
}

/* =========================================================
   SYNTHESIZED AUDIO (Web Audio API - Zero Assets Required)
   ========================================================= */
let audioCtx = null;

function setupAudioContext() {
  const AudioContext = window.AudioContext || window.webkitAudioContext;
  if (AudioContext) {
    audioCtx = new AudioContext();
  }
}

function ensureAudioReady() {
  if (audioCtx && audioCtx.state === 'suspended') {
    audioCtx.resume();
  }
}

function playSound(type) {
  if (!state.soundEnabled || !audioCtx) return;
  ensureAudioReady();

  try {
    const now = audioCtx.currentTime;

    if (type === 'tick') {
      // Countdown tick
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sine';
      osc.frequency.setValueAtTime(880, now);
      gain.gain.setValueAtTime(0.15, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.12);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.12);

    } else if (type === 'snap') {
      // Shutter snap / capture sound
      const bufferSize = audioCtx.sampleRate * 0.05;
      const buffer = audioCtx.createBuffer(1, bufferSize, audioCtx.sampleRate);
      const data = buffer.getChannelData(0);
      for (let i = 0; i < bufferSize; i++) {
        data[i] = (Math.random() * 2 - 1) * Math.exp(-i / (bufferSize * 0.2));
      }
      const noise = audioCtx.createBufferSource();
      noise.buffer = buffer;
      const filter = audioCtx.createBiquadFilter();
      filter.type = 'highpass';
      filter.frequency.value = 1000;
      const gain = audioCtx.createGain();
      gain.gain.value = 0.2;
      noise.connect(filter);
      filter.connect(gain);
      gain.connect(audioCtx.destination);
      noise.start(now);

    } else if (type === 'win') {
      // Victory Fanfare (Arpeggio: C5, E5, G5, C6)
      const freqs = [523.25, 659.25, 783.99, 1046.5];
      freqs.forEach((freq, idx) => {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'triangle';
        osc.frequency.setValueAtTime(freq, now + idx * 0.1);
        gain.gain.setValueAtTime(0.2, now + idx * 0.1);
        gain.gain.exponentialRampToValueAtTime(0.001, now + idx * 0.1 + 0.35);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now + idx * 0.1);
        osc.stop(now + idx * 0.1 + 0.35);
      });

    } else if (type === 'lose') {
      // Defeat Thud (Descending tone)
      const osc = audioCtx.createOscillator();
      const gain = audioCtx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(280, now);
      osc.frequency.exponentialRampToValueAtTime(110, now + 0.35);
      gain.gain.setValueAtTime(0.2, now);
      gain.gain.exponentialRampToValueAtTime(0.001, now + 0.35);
      osc.connect(gain);
      gain.connect(audioCtx.destination);
      osc.start(now);
      osc.stop(now + 0.35);

    } else if (type === 'tie') {
      // Neutral Dual Tone
      [440, 554.37].forEach(freq => {
        const osc = audioCtx.createOscillator();
        const gain = audioCtx.createGain();
        osc.type = 'sine';
        osc.frequency.setValueAtTime(freq, now);
        gain.gain.setValueAtTime(0.12, now);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.3);
        osc.connect(gain);
        gain.connect(audioCtx.destination);
        osc.start(now);
        osc.stop(now + 0.3);
      });
    }
  } catch (err) {
    console.warn('Audio playback error:', err);
  }
}

/* =========================================================
   ONNX RUNTIME WEB — STREAMING LOADER & INITIALIZATION
   ========================================================= */
async function loadOnnxModel() {
  try {
    DOM.engineStatusText.textContent = 'Streaming Model (13.9 MB)...';

    // Configure WASM worker paths
    if (window.ort && ort.env && ort.env.wasm) {
      ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/';
      ort.env.wasm.numThreads = Math.min(navigator.hardwareConcurrency || 4, 4);
    }

    // Progressive Chunk Streaming
    const response = await fetch(MODEL_PATH);
    if (!response.ok) {
      throw new Error(`HTTP Error ${response.status}: Failed to fetch ${MODEL_PATH}`);
    }

    const contentLength = +(response.headers.get('Content-Length') || MODEL_TOTAL_BYTES);
    const reader = response.body.getReader();
    let receivedBytes = 0;
    const chunks = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      receivedBytes += value.length;

      const pct = Math.min(Math.round((receivedBytes / contentLength) * 100), 100);
      const loadedMb = (receivedBytes / (1024 * 1024)).toFixed(1);
      const totalMb = (contentLength / (1024 * 1024)).toFixed(1);

      DOM.modelProgressBar.style.width = `${pct}%`;
      DOM.modelProgressPct.textContent = `${pct}%`;
      DOM.modelLoadedMb.textContent = `${loadedMb} / ${totalMb} MB`;
    }

    DOM.modelProgressDetail.textContent = 'Compiling WebGL GPU Tensor Engine...';

    // Assemble ArrayBuffer
    const modelBuffer = new Uint8Array(receivedBytes);
    let position = 0;
    for (const chunk of chunks) {
      modelBuffer.set(chunk, position);
      position += chunk.length;
    }

    // Create InferenceSession with WebGL acceleration + WASM fallback
    let session = null;
    let providerUsed = 'WebGL (GPU)';

    try {
      session = await ort.InferenceSession.create(modelBuffer.buffer, {
        executionProviders: ['webgl', 'wasm'],
        graphOptimizationLevel: 'all'
      });
    } catch (gpuErr) {
      console.warn('WebGL initialization fallback to WASM:', gpuErr);
      providerUsed = 'WASM (CPU)';
      session = await ort.InferenceSession.create(modelBuffer.buffer, {
        executionProviders: ['wasm']
      });
    }

    state.session = session;
    state.modelLoaded = true;

    // Warm-up inference with dummy [1, 150, 150, 3] tensor
    const dummyData = new Float32Array(1 * 150 * 150 * 3);
    const dummyTensor = new ort.Tensor('float32', dummyData, [1, 150, 150, 3]);
    await session.run({ [session.inputNames[0]]: dummyTensor });

    // Update UI
    DOM.engineStatusDot.classList.remove('loading');
    DOM.engineStatusDot.classList.add('ready');
    DOM.engineStatusText.textContent = `AI Ready • ${providerUsed}`;
    DOM.inferenceFpsBadge.textContent = providerUsed;

    // Hide Loader Card
    setTimeout(() => {
      DOM.modelLoaderOverlay.classList.add('hidden');
    }, 400);

    DOM.startRoundBtn.disabled = false;

    // If already in live mode, kick off loop
    if (state.mode === 'live') {
      startLiveDetection();
    }

  } catch (err) {
    console.error('Model loading error:', err);
    DOM.engineStatusDot.classList.remove('ready');
    DOM.engineStatusDot.classList.add('loading');
    DOM.engineStatusText.textContent = 'Model Load Failed';
    DOM.modelProgressDetail.textContent = `Error: ${err.message}. Please refresh or check local server.`;
  }
}

/* =========================================================
   CAMERA MANAGEMENT & PREPROCESSING
   ========================================================= */
async function startCamera() {
  try {
    const constraints = {
      video: {
        width: { ideal: 640 },
        height: { ideal: 480 },
        facingMode: 'user'
      },
      audio: false
    };

    const stream = await navigator.mediaDevices.getUserMedia(constraints);
    state.stream = stream;
    DOM.webcam.srcObject = stream;

    DOM.webcam.onloadedmetadata = () => {
      DOM.webcam.play();
      DOM.cameraFallback.classList.add('hidden');
      DOM.camToggleText.textContent = '📷 Camera Active';
    };
  } catch (err) {
    console.warn('Webcam access error:', err);
    DOM.cameraFallback.classList.remove('hidden');
    DOM.camToggleText.textContent = '📷 Camera Off';
  }
}

function stopCamera() {
  if (state.stream) {
    state.stream.getTracks().forEach(track => track.stop());
    state.stream = null;
    DOM.webcam.srcObject = null;
    DOM.cameraFallback.classList.remove('hidden');
    DOM.camToggleText.textContent = '📷 Camera Off';
  }
}

function initOffscreenCanvas() {
  state.offscreenCanvas = document.createElement('canvas');
  state.offscreenCanvas.width = 150;
  state.offscreenCanvas.height = 150;
  state.offscreenCtx = state.offscreenCanvas.getContext('2d', { willReadFrequently: true });
}

/**
 * Extracts the 150x150 hand ROI tensor matching main.py preprocessing:
 * 1. Crop region of interest
 * 2. Resize to 150x150
 * 3. Convert RGBA -> RGB normalized Float32Array [0.0 - 1.0]
 * 4. Reshape to [1, 150, 150, 3]
 */
function extractRoiTensor() {
  const video = DOM.webcam;
  if (!video || video.readyState < 2) return null;

  const vw = video.videoWidth || 640;
  const vh = video.videoHeight || 480;

  // Calculate Region of Interest (matches 60% width square in center-middle of video)
  const cropSize = Math.floor(Math.min(vw * 0.6, vh * 0.65));
  const cropX = Math.floor((vw - cropSize) / 2);
  const cropY = Math.floor((vh - cropSize) / 2);

  // Draw crop into 150x150 offscreen canvas
  state.offscreenCtx.save();
  if (state.isMirrored) {
    // Mirror draw so offscreen canvas matches what the player sees
    state.offscreenCtx.translate(150, 0);
    state.offscreenCtx.scale(-1, 1);
  }
  state.offscreenCtx.drawImage(
    video,
    cropX, cropY, cropSize, cropSize,
    0, 0, 150, 150
  );
  state.offscreenCtx.restore();

  // Draw preview to "What AI Sees" debug canvas
  const aiCtx = DOM.aiSeeCanvas.getContext('2d');
  aiCtx.drawImage(state.offscreenCanvas, 0, 0, 150, 150);

  // Extract pixel data
  const imgData = state.offscreenCtx.getImageData(0, 0, 150, 150).data;
  const float32Data = new Float32Array(1 * 150 * 150 * 3);

  // Convert RGBA -> RGB Float32 normalized 0..1
  for (let i = 0, j = 0; i < imgData.length; i += 4, j += 3) {
    float32Data[j] = imgData[i] / 255.0;       // Red
    float32Data[j + 1] = imgData[i + 1] / 255.0; // Green
    float32Data[j + 2] = imgData[i + 2] / 255.0; // Blue
  }

  return new ort.Tensor('float32', float32Data, [1, 150, 150, 3]);
}

/* =========================================================
   INFERENCE & CLASSIFICATION PIPELINE
   ========================================================= */
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((acc, val) => acc + val, 0);
  return exps.map(x => x / sum);
}

async function runModelInference() {
  if (state.inferenceEngine === 'backend') {
    return await runBackendInference();
  }
  return await runOnnxInference();
}

async function runOnnxInference() {
  if (!state.session || state.isInferring) return null;
  state.isInferring = true;

  try {
    const inputTensor = extractRoiTensor();
    if (!inputTensor) return null;

    const t0 = performance.now();
    const feeds = { [state.session.inputNames[0]]: inputTensor };
    const results = await state.session.run(feeds);
    const latency = Math.round(performance.now() - t0);

    const rawOutput = results[state.session.outputNames[0]].data;
    const probs = softmax(Array.from(rawOutput));

    // Find class with highest probability
    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }

    const predictedClass = CLASSES[maxIdx];
    const confidence = probs[maxIdx];

    // Update Telemetry Display
    updateTelemetry(probs, latency);

    return {
      move: predictedClass,
      confidence: confidence,
      probabilities: {
        paper: probs[0],
        rock: probs[1],
        scissors: probs[2]
      },
      latency: latency
    };
  } catch (err) {
    console.error('ONNX Inference execution error:', err);
    return null;
  } finally {
    state.isInferring = false;
  }
}

async function runBackendInference() {
  if (state.isInferring) return null;
  state.isInferring = true;

  try {
    extractRoiTensor();
    const dataUrl = state.offscreenCanvas.toDataURL('image/jpeg', 0.85);

    const t0 = performance.now();
    const res = await fetch(`${state.backendUrl}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: dataUrl })
    });

    if (!res.ok) throw new Error(`Backend HTTP ${res.status}`);
    const data = await res.json();
    const latency = Math.round(performance.now() - t0);

    const probs = [
      data.probabilities.paper || 0,
      data.probabilities.rock || 0,
      data.probabilities.scissors || 0
    ];
    updateTelemetry(probs, latency);

    return {
      move: data.player_move,
      confidence: data.confidence,
      cpuMove: data.cpu_move,
      winner: data.winner,
      probabilities: data.probabilities,
      latency: latency
    };
  } catch (err) {
    console.warn('Backend inference failed, falling back to in-browser ONNX:', err);
    return await runOnnxInference();
  } finally {
    state.isInferring = false;
  }
}

function updateTelemetry(probs, latency) {
  // Update percentage text & progress bars
  // CLASSES = ['paper', 'rock', 'scissors']
  const paperPct = Math.round(probs[0] * 100);
  const rockPct = Math.round(probs[1] * 100);
  const scissorsPct = Math.round(probs[2] * 100);

  DOM.probValPaper.textContent = `${paperPct}%`;
  DOM.probFillPaper.style.width = `${paperPct}%`;

  DOM.probValRock.textContent = `${rockPct}%`;
  DOM.probFillRock.style.width = `${rockPct}%`;

  DOM.probValScissors.textContent = `${scissorsPct}%`;
  DOM.probFillScissors.style.width = `${scissorsPct}%`;

  DOM.telemetryLatencyTag.textContent = `${latency}ms • 150×150`;
}

/* =========================================================
   GAME ENGINE & STATE MACHINE (Matches main.py logic)
   ========================================================= */

function getWinner(player, computer) {
  const p = player.toLowerCase();
  const c = computer.toLowerCase();
  if (p === c) return 'Tie';
  if (
    (p === 'rock' && c === 'scissors') ||
    (p === 'scissors' && c === 'paper') ||
    (p === 'paper' && c === 'rock')
  ) {
    return 'Player Wins!';
  }
  return 'Computer Wins!';
}

function startRound() {
  if (!state.modelLoaded) return;
  if (state.gameState !== 'IDLE' && state.gameState !== 'RESULT') return;

  state.gameState = 'COUNTDOWN';
  DOM.startRoundBtn.disabled = true;
  DOM.startBtnText.textContent = 'ROUND IN PROGRESS...';

  // Visual cues
  DOM.countdownOverlay.classList.add('active');
  DOM.handTargetBox.className = 'hand-target-box countdown-state';
  DOM.targetBoxLabel.textContent = 'STEADY YOUR HAND';
  DOM.cpuStateText.textContent = 'ANALYZING...';
  DOM.cpuMindText.textContent = 'Generating counter-strategy...';

  // Shuffling CPU Card
  DOM.cpuHandCard.classList.add('shuffling');
  DOM.cpuHandCard.classList.remove('revealed');
  let shuffleStep = 0;
  const shuffleEmojis = ['✊', '✋', '✌️'];
  state.shuffleIntervalId = setInterval(() => {
    shuffleStep = (shuffleStep + 1) % shuffleEmojis.length;
    DOM.cpuMoveIcon.textContent = shuffleEmojis[shuffleStep];
    DOM.cpuMoveName.textContent = 'SHUFFLING...';
  }, 100);

  // 3-2-1 Countdown Sequence
  let count = 3;
  DOM.countdownText.textContent = count;
  DOM.countdownSub.textContent = 'GET READY!';
  playSound('tick');

  state.countdownTimerId = setInterval(() => {
    count--;
    if (count > 0) {
      DOM.countdownText.textContent = count;
      playSound('tick');
    } else {
      clearInterval(state.countdownTimerId);
      triggerCapturePhase();
    }
  }, 1000);
}

async function triggerCapturePhase() {
  state.gameState = 'CAPTURE';
  DOM.countdownText.textContent = 'HOLD IT!';
  DOM.countdownSub.textContent = 'AI CAPTURING HAND...';
  DOM.handTargetBox.className = 'hand-target-box capture-state';
  DOM.targetBoxLabel.textContent = 'HOLD HAND STILL!';

  // Wait 0.5s before snapshot (exactly matching line 92 in main.py)
  setTimeout(async () => {
    playSound('snap');

    // Run deep learning model on current camera crop
    const result = await runModelInference();
    const playerMove = result ? result.move : 'rock'; // Fallback
    const playerConf = result ? Math.round(result.confidence * 100) : 85;

    // CPU Move Selection (Random from classes, like line 115 in main.py)
    const cpuMove = CLASSES[Math.floor(Math.random() * CLASSES.length)];

    // End countdown & shuffle
    clearInterval(state.shuffleIntervalId);
    DOM.cpuHandCard.classList.remove('shuffling');
    DOM.cpuHandCard.classList.add('revealed');
    DOM.countdownOverlay.classList.remove('active');

    // Evaluate Winner
    const winnerMessage = getWinner(playerMove, cpuMove);
    resolveRound(playerMove, playerConf, cpuMove, winnerMessage);
  }, 500);
}

function resolveRound(playerMove, playerConf, cpuMove, winnerMessage) {
  state.gameState = 'RESULT';
  state.lastPlayerMove = playerMove;
  state.lastCpuMove = cpuMove;
  state.lastOutcome = winnerMessage;

  // Display Player Hand
  DOM.playerMoveIcon.textContent = EMOJIS[playerMove];
  DOM.playerMoveText.textContent = capitalize(playerMove);
  DOM.playerMoveConf.textContent = `AI Confidence: ${playerConf}%`;

  // Display CPU Hand
  DOM.cpuMoveIcon.textContent = EMOJIS[cpuMove];
  DOM.cpuMoveName.textContent = capitalize(cpuMove);
  DOM.cpuMoveSub.textContent = 'Locked In!';
  DOM.cpuStateText.textContent = 'MOVE REVEALED';

  // Apply Outcome Styling & Audio
  DOM.outcomeBanner.className = 'round-outcome-banner';
  let chipClass = 'chip-tie';

  if (winnerMessage === 'Player Wins!') {
    DOM.outcomeBanner.classList.add('player-win');
    DOM.outcomeIcon.textContent = '🎉';
    DOM.outcomeTitle.textContent = 'PLAYER WINS!';
    DOM.outcomeSub.textContent = `${capitalize(playerMove)} beats ${capitalize(cpuMove)}`;
    state.playerScore++;
    state.streak++;
    playSound('win');
    chipClass = 'chip-win';
    DOM.cpuMindText.textContent = 'Opponent gesture outmatched CPU!';
  } else if (winnerMessage === 'Computer Wins!') {
    DOM.outcomeBanner.classList.add('cpu-win');
    DOM.outcomeIcon.textContent = '💀';
    DOM.outcomeTitle.textContent = 'CPU WINS!';
    DOM.outcomeSub.textContent = `${capitalize(cpuMove)} beats ${capitalize(playerMove)}`;
    state.cpuScore++;
    state.streak = 0;
    playSound('lose');
    chipClass = 'chip-loss';
    DOM.cpuMindText.textContent = 'Neural prediction successful.';
  } else {
    DOM.outcomeBanner.classList.add('tie');
    DOM.outcomeIcon.textContent = '🤝';
    DOM.outcomeTitle.textContent = 'MATCH DRAW!';
    DOM.outcomeSub.textContent = `Both selected ${capitalize(playerMove)}`;
    state.tieScore++;
    playSound('tie');
    chipClass = 'chip-tie';
    DOM.cpuMindText.textContent = 'Symmetrical move detected.';
  }

  // Update Scoreboard Numbers
  updateScoreboard();

  // Add Match History Chip
  addHistoryChip(chipClass, playerMove, cpuMove, winnerMessage);

  // Reset Button & Target Box for next round
  DOM.handTargetBox.className = 'hand-target-box';
  DOM.targetBoxLabel.textContent = 'PLACE HAND HERE';
  DOM.startRoundBtn.disabled = false;
  DOM.startBtnText.textContent = 'PLAY NEXT ROUND';
  state.round++;
  DOM.roundCounter.textContent = `ROUND ${state.round}`;
}

function updateScoreboard() {
  DOM.playerScore.textContent = state.playerScore;
  DOM.cpuScore.textContent = state.cpuScore;
  DOM.tieScore.textContent = state.tieScore;
  DOM.streakCount.textContent = `${state.streak} STREAK`;

  const total = state.playerScore + state.cpuScore + state.tieScore;
  if (total > 0) {
    const pPct = Math.round((state.playerScore / total) * 100);
    const cPct = Math.round((state.cpuScore / total) * 100);
    DOM.playerWinPct.textContent = `Win Rate: ${pPct}%`;
    DOM.cpuWinPct.textContent = `Win Rate: ${cPct}%`;
  }
}

function addHistoryChip(chipClass, pMove, cMove, outcome) {
  // Clear placeholder if first round
  const empty = DOM.historyChips.querySelector('.history-empty');
  if (empty) empty.remove();

  const chip = document.createElement('div');
  chip.className = `history-chip ${chipClass}`;
  
  let label = 'TIE';
  if (outcome === 'Player Wins!') label = 'WON';
  else if (outcome === 'Computer Wins!') label = 'LOST';

  chip.innerHTML = `
    <span>${EMOJIS[pMove]} vs ${EMOJIS[cMove]}</span>
    <strong>${label}</strong>
    <span style="opacity: 0.6;">(R${state.round})</span>
  `;

  DOM.historyChips.prepend(chip);

  // Keep max 15 chips
  while (DOM.historyChips.children.length > 15) {
    DOM.historyChips.removeChild(DOM.historyChips.lastChild);
  }
}

/* =========================================================
   LIVE PRACTICE MODE (Continuous Streaming Inference)
   ========================================================= */
function setGameMode(mode) {
  if (state.mode === mode) return;
  state.mode = mode;

  if (mode === 'live') {
    DOM.modeLiveBtn.classList.add('active');
    DOM.modeBattleBtn.classList.remove('active');
    DOM.startRoundBtn.disabled = true;
    DOM.startBtnText.textContent = 'CONTINUOUS LIVE DETECTION';
    DOM.targetBoxLabel.textContent = 'LIVE AI RECOGNITION';
    DOM.outcomeTitle.textContent = 'PRACTICE MODE ACTIVE';
    DOM.outcomeSub.textContent = 'Hold any hand gesture in the box';
    DOM.outcomeIcon.textContent = '⚡';
    startLiveDetection();
  } else {
    DOM.modeBattleBtn.classList.add('active');
    DOM.modeLiveBtn.classList.remove('active');
    DOM.startRoundBtn.disabled = !state.modelLoaded;
    DOM.startBtnText.textContent = 'START ROUND';
    DOM.targetBoxLabel.textContent = 'PLACE HAND HERE';
    DOM.outcomeTitle.textContent = 'PRESS START';
    DOM.outcomeSub.textContent = 'Spacebar or click button below';
    DOM.outcomeIcon.textContent = '🏆';
    stopLiveDetection();
    state.gameState = 'IDLE';
  }
}

function startLiveDetection() {
  if (state.liveLoopId) return;

  const loop = async () => {
    if (state.mode !== 'live') return;

    if (state.modelLoaded && state.stream) {
      const result = await runModelInference();
      if (result) {
        DOM.playerMoveIcon.textContent = EMOJIS[result.move];
        DOM.playerMoveText.textContent = capitalize(result.move);
        DOM.playerMoveConf.textContent = `Live Confidence: ${Math.round(result.confidence * 100)}%`;
      }
    }

    state.liveLoopId = requestAnimationFrame(loop);
  };

  state.liveLoopId = requestAnimationFrame(loop);
}

function stopLiveDetection() {
  if (state.liveLoopId) {
    cancelAnimationFrame(state.liveLoopId);
    state.liveLoopId = null;
  }
}

/* =========================================================
   EVENT LISTENERS & KEYBOARD SHORTCUTS
   ========================================================= */
function setupEventListeners() {
  // Mode switch buttons
  DOM.modeBattleBtn.addEventListener('click', () => setGameMode('battle'));
  DOM.modeLiveBtn.addEventListener('click', () => setGameMode('live'));

  // Main Action button
  DOM.startRoundBtn.addEventListener('click', () => {
    ensureAudioReady();
    if (state.mode === 'battle') startRound();
  });

  // Camera permissions placeholder button
  DOM.enableCamBtn.addEventListener('click', () => {
    startCamera();
  });

  // Toggle Camera
  DOM.toggleCameraBtn.addEventListener('click', () => {
    if (state.stream) {
      stopCamera();
    } else {
      startCamera();
    }
  });

  // Mirror view flip
  DOM.flipCamBtn.addEventListener('click', () => {
    state.isMirrored = !state.isMirrored;
    DOM.webcam.classList.toggle('no-mirror', !state.isMirrored);
  });

  // Toggle Debug "What AI Sees" PIP Canvas
  DOM.togglePipBtn.addEventListener('click', () => {
    state.showPip = !state.showPip;
    DOM.debugPipCard.classList.toggle('hidden', !state.showPip);
    DOM.togglePipBtn.classList.toggle('active', state.showPip);
  });

  // SFX Toggle
  DOM.sfxToggleBtn.addEventListener('click', () => {
    state.soundEnabled = !state.soundEnabled;
    DOM.sfxIcon.textContent = state.soundEnabled ? '🔊' : '🔇';
    DOM.sfxToggleBtn.style.opacity = state.soundEnabled ? '1' : '0.6';
  });

  // Reset Scoreboard
  DOM.resetScoreBtn.addEventListener('click', () => {
    state.playerScore = 0;
    state.cpuScore = 0;
    state.tieScore = 0;
    state.streak = 0;
    state.round = 1;
    DOM.roundCounter.textContent = 'ROUND 1';
    DOM.playerWinPct.textContent = 'Win Rate: 0%';
    DOM.cpuWinPct.textContent = 'Win Rate: 0%';
    updateScoreboard();
  });

  // Clear History
  DOM.clearHistoryBtn.addEventListener('click', () => {
    DOM.historyChips.innerHTML = '<div class="history-empty">No rounds played yet. Hit Start Round to begin!</div>';
  });

  // Settings Modal Handlers
  DOM.settingsToggleBtn.addEventListener('click', () => {
    DOM.settingsModal.classList.remove('hidden');
  });

  DOM.closeSettingsBtn.addEventListener('click', () => {
    DOM.settingsModal.classList.add('hidden');
  });

  DOM.settingsModal.addEventListener('click', (e) => {
    if (e.target === DOM.settingsModal) {
      DOM.settingsModal.classList.add('hidden');
    }
  });

  // Inference Mode Radio Toggle
  const radioInputs = DOM.settingsModal.querySelectorAll('input[name="inferenceMode"]');
  radioInputs.forEach(input => {
    input.addEventListener('change', (e) => {
      const isBackend = e.target.value === 'backend';
      DOM.radioCardClient.classList.toggle('active', !isBackend);
      DOM.radioCardBackend.classList.toggle('active', isBackend);
      DOM.backendUrlGroup.classList.toggle('hidden', !isBackend);
    });
  });

  // Test Backend Connection Button
  DOM.testBackendBtn.addEventListener('click', async () => {
    const url = DOM.backendUrlInput.value.trim().replace(/\/+$/, '');
    DOM.backendTestStatus.textContent = 'Testing connection...';
    DOM.backendTestStatus.className = 'backend-test-status';

    try {
      const res = await fetch(`${url}/health`, { method: 'GET' });
      if (res.ok) {
        const data = await res.json();
        DOM.backendTestStatus.textContent = `Connected! Model status: ${data.model_loaded ? 'Loaded' : 'Not Loaded'}`;
        DOM.backendTestStatus.className = 'backend-test-status success';
      } else {
        DOM.backendTestStatus.textContent = `Backend returned HTTP ${res.status}`;
        DOM.backendTestStatus.className = 'backend-test-status error';
      }
    } catch (err) {
      DOM.backendTestStatus.textContent = `Connection failed: ${err.message}`;
      DOM.backendTestStatus.className = 'backend-test-status error';
    }
  });

  // Save Settings Button
  DOM.saveSettingsBtn.addEventListener('click', () => {
    const selectedMode = DOM.settingsModal.querySelector('input[name="inferenceMode"]:checked').value;
    state.inferenceEngine = selectedMode;
    state.backendUrl = DOM.backendUrlInput.value.trim().replace(/\/+$/, '');

    if (state.inferenceEngine === 'backend') {
      DOM.engineStatusDot.className = 'status-indicator ready';
      DOM.engineStatusText.textContent = 'Backend API Active';
      DOM.inferenceFpsBadge.textContent = 'Python FastAPI';
    } else {
      DOM.engineStatusDot.className = 'status-indicator ready';
      DOM.engineStatusText.textContent = 'AI Ready • In-Browser ONNX';
      DOM.inferenceFpsBadge.textContent = 'WebGL (GPU)';
    }

    DOM.settingsModal.classList.add('hidden');
  });

  // Keyboard Shortcuts (Matching main.py keys)
  window.addEventListener('keydown', (e) => {
    // Ignore if typing in input
    if (['INPUT', 'TEXTAREA'].includes(e.target.tagName)) return;

    if (e.code === 'Space' || e.key.toLowerCase() === 's') {
      e.preventDefault();
      if (state.mode === 'battle' && (state.gameState === 'IDLE' || state.gameState === 'RESULT')) {
        startRound();
      }
    } else if (e.key.toLowerCase() === 'r') {
      e.preventDefault();
      if (state.gameState === 'RESULT') {
        startRound();
      }
    } else if (e.key.toLowerCase() === 'm') {
      DOM.sfxToggleBtn.click();
    } else if (e.key.toLowerCase() === 'l') {
      setGameMode(state.mode === 'battle' ? 'live' : 'battle');
    }
  });
}

function capitalize(str) {
  if (!str) return '';
  return str.charAt(0).toUpperCase() + str.slice(1);
}
