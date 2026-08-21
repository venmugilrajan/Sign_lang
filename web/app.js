/**
 * SignLens — MediaPipe Landmark Neural Network Edition
 *
 * Runs classification directly in the browser using the weights
 * of our custom MLP Neural Network trained on the 21 hand landmarks.
 */

// ── Config ──────────────────────────────────────────────────────────────────
const STREAK_NEEDED  = 5;     // frames of same letter to capture
const COOLDOWN_MS    = 1000;  // ms after capture before detecting again
const SPACE_FRAMES   = 15;    // no-hand frames before auto word push
const MIN_CONFIDENCE = 0.65;  // minimum classifier confidence

// ── MediaPipe Hand Connections ───────────────────────────────────────────────
const CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

const NODE_COLORS = {
  0:'#ffffff',
  1:'#ff6b9d',2:'#ff6b9d',3:'#ff6b9d',4:'#ff6b9d',
  5:'#00d4ff',6:'#00d4ff',7:'#00d4ff',8:'#00d4ff',
  9:'#22d67a',10:'#22d67a',11:'#22d67a',12:'#22d67a',
  13:'#ffe066',14:'#ffe066',15:'#ffe066',16:'#ffe066',
  17:'#ff9240',18:'#ff9240',19:'#ff9240',20:'#ff9240',
};

const EDGE_COLORS = [
  '#ff6b9d','#ff6b9d','#ff6b9d',
  '#00d4ff','#00d4ff','#00d4ff','#00d4ff',
  '#22d67a','#22d67a','#22d67a','#22d67a',
  '#ffe066','#ffe066','#ffe066','#ffe066',
  '#ff9240','#ff9240','#ff9240','#ff9240',
  '#7c6aff','#7c6aff','#7c6aff',
];

// ── State ────────────────────────────────────────────────────────────────────
let currentMode   = 'asl';
let paused        = false;
let inCooldown    = false;
let cameraActive  = false;

let currentWord   = '';
let sentenceWords = [];

let streakLetter  = null;
let streakCount   = 0;
let noHandCount   = 0;

// Trained Neural Network model weights loaded from JSON
let aslWeights     = null;
let islWeights     = null;
let currentWeights = null;

// ── DOM ──────────────────────────────────────────────────────────────────────
const video           = document.getElementById('webcam-video');
const displayCanvas   = document.getElementById('display-canvas');
const dctx            = displayCanvas.getContext('2d');
const noHandOverlay   = document.getElementById('no-hand-overlay');
const detectedLetter  = document.getElementById('detected-letter');
const detectedConf    = document.getElementById('detected-conf');
const streakBar       = document.getElementById('streak-bar');
const streakLabel     = document.getElementById('streak-label');
const handBadge       = document.getElementById('hand-badge');
const statusPill      = document.getElementById('status-pill');
const statusText      = document.getElementById('status-text');
const wordDisplay     = document.getElementById('word-display');
const sentenceDisplay = document.getElementById('sentence-display');
const wordChips       = document.getElementById('word-chips');
const modeToggle      = document.getElementById('mode-toggle');
const toast           = document.getElementById('toast');

// ── Utilities ────────────────────────────────────────────────────────────────
function showToast(msg, ms = 2000) {
  toast.textContent = msg;
  toast.classList.add('show');
  setTimeout(() => toast.classList.remove('show'), ms);
}
function setStatus(cls, msg) {
  statusPill.className = `status-pill ${cls}`;
  statusText.textContent = msg;
}
function resetStreak() {
  streakLetter = null; streakCount = 0;
  streakBar.style.width = '0%';
  streakLabel.textContent = `0 / ${STREAK_NEEDED}`;
}

// ── Word / Sentence ───────────────────────────────────────────────────────────
function renderWord() {
  wordDisplay.innerHTML = currentWord
    ? currentWord
    : '<span class="ph">Start signing…</span>';
}
function renderSentence() {
  sentenceDisplay.innerHTML = sentenceWords.length
    ? sentenceWords.join(' ')
    : '<em class="ph">Your sentence will appear here…</em>';
  wordChips.innerHTML = '';
  sentenceWords.forEach(w => {
    const c = document.createElement('span');
    c.className = 'word-chip'; c.textContent = w;
    wordChips.appendChild(c);
  });
}
async function pushWord() {
  let w = currentWord.trim();
  if (w) {
    // Attempt spellcheck via backend API if available
    try {
      const res = await fetch('/spellcheck', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({ word: w })
      });
      if (res.ok) {
        const data = await res.json();
        if (data.corrected) {
          w = data.corrected.toUpperCase();
        }
      }
    } catch (e) {
      // Backend not present or offline, keep raw word
    }

    sentenceWords.push(w);
    renderSentence();
    showToast(`✓ "${w}"`);
  }
  currentWord = '';
  renderWord();
  resetStreak();
  noHandCount = 0;
}

// ── Letter capture ────────────────────────────────────────────────────────────
function captureLetter(letter) {
  if (inCooldown) return;
  const l = letter.toLowerCase();
  if (l === 'space' || l === '_') { pushWord(); return; }
  if (l === 'del' || l === 'delete') {
    currentWord = currentWord.slice(0, -1);
    renderWord();
    resetStreak();
    return;
  }
  if (l === 'nothing') return;

  currentWord += letter.toUpperCase();
  renderWord();

  detectedLetter.classList.remove('flash');
  void detectedLetter.offsetWidth;
  detectedLetter.classList.add('flash');

  resetStreak();
  noHandCount = 0;
  inCooldown = true;
  setTimeout(() => { inCooldown = false; }, COOLDOWN_MS);
}

// ════════════════════════════════════════════════════════════════════════════
// ── Landmark Neural Network Classifier ──────────────────────────────────────
// ════════════════════════════════════════════════════════════════════════════

function dist2D(a, b) {
  return Math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2);
}

/** Normalize landmarks so hand size = 1, centered at wrist */
function normalizeLandmarks(lm) {
  const wrist = lm[0];
  const scale = dist2D(wrist, lm[9]) || 1; // wrist-to-middle-MCP as unit
  const features = [];
  lm.forEach(p => {
    features.push((p.x - wrist.x) / scale);
    features.push((p.y - wrist.y) / scale);
    features.push((p.z - wrist.z) / scale);
  });
  return features;
}

/** Matrix multiplication helper */
function matMul(input, weights, bias) {
  const output = [];
  const numOutputs = bias.length;
  const numInputs = input.length;
  for (let j = 0; j < numOutputs; j++) {
    let sum = bias[j];
    for (let i = 0; i < numInputs; i++) {
      sum += input[i] * weights[i][j];
    }
    output.push(sum);
  }
  return output;
}

/** ReLU activation function */
function relu(arr) {
  return arr.map(x => Math.max(0, x));
}

/** Softmax activation function */
function softmax(arr) {
  const max = Math.max(...arr);
  const exps = arr.map(x => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map(x => x / sum);
}

/** Load MLP weights from JSON files */
async function loadLandmarkModel() {
  try {
    const resAsl = await fetch('asl_landmarks_weights.json');
    if (resAsl.ok) {
      aslWeights = await resAsl.json();
      console.log("🧠 ASL Neural Network loaded!");
    }
    const resIsl = await fetch('isl_landmarks_weights.json');
    if (resIsl.ok) {
      islWeights = await resIsl.json();
      console.log("🧠 ISL Neural Network loaded!");
    }
    currentWeights = currentMode === 'asl' ? aslWeights : islWeights;
  } catch (err) {
    console.error("Failed to load MLP weights:", err);
    showToast("⚠️ Neural network weights not fully loaded.", 4000);
  }
}

/** Classify hand landmarks using dynamic 2 or 3-layer neural network model */
function classify(rawLandmarks) {
  if (!currentWeights) {
    return { letter: 'nothing', confidence: 0.0 };
  }

  // 1. Prepare feature vector (63 floats)
  const x = normalizeLandmarks(rawLandmarks);

  // Layer 0
  let h = relu(matMul(x, currentWeights.w0, currentWeights.b0));

  // Layer 1
  h = relu(matMul(h, currentWeights.w1, currentWeights.b1));

  // Layer 2 (if present)
  if (currentWeights.w2 && currentWeights.b2) {
    if (currentWeights.w3 && currentWeights.b3) {
      h = relu(matMul(h, currentWeights.w2, currentWeights.b2));
      h = matMul(h, currentWeights.w3, currentWeights.b3);
    } else {
      h = matMul(h, currentWeights.w2, currentWeights.b2);
    }
  }

  // Softmax to get probabilities
  const probs = softmax(h);

  // Find max confidence index
  let maxIdx = 0;
  let maxProb = 0;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > maxProb) {
      maxProb = probs[i];
      maxIdx = i;
    }
  }

  return {
    letter: currentWeights.classes[maxIdx],
    confidence: maxProb
  };
}

// ── Process classification result ─────────────────────────────────────────────
function processResult(letter, confidence) {
  const conf = Math.round(confidence * 100);

  detectedLetter.textContent = (letter === 'nothing' || confidence < MIN_CONFIDENCE)
    ? '—'
    : letter.toUpperCase();

  detectedConf.textContent = letter === 'nothing'
    ? 'Unclear gesture'
    : `${conf}% · ${letter.toUpperCase()}`;

  const effective = (confidence >= MIN_CONFIDENCE && letter !== 'nothing') ? letter : null;

  if (paused || inCooldown || !effective) {
    if (!effective) resetStreak();
    return;
  }

  noHandCount = 0;

  if (effective === streakLetter) {
    streakCount++;
  } else {
    streakLetter = effective;
    streakCount  = 1;
  }

  const pct = Math.min(100, Math.round((streakCount / STREAK_NEEDED) * 100));
  streakBar.style.width   = `${pct}%`;
  streakLabel.textContent = `${Math.min(streakCount, STREAK_NEEDED)} / ${STREAK_NEEDED}`;

  if (streakCount >= STREAK_NEEDED) {
    captureLetter(streakLetter);
  }
}

// ── MediaPipe Camera Loop ─────────────────────────────────────────────────────
let camera = null;
let hands  = null;

function onResults(results) {
  dctx.save();
  dctx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
  dctx.drawImage(results.image, 0, 0, displayCanvas.width, displayCanvas.height);

  if (results.multiHandLandmarks && results.multiHandLandmarks.length > 0) {
    noHandOverlay.classList.remove('visible');
    handBadge.className   = 'hand-badge live';
    handBadge.textContent = 'HAND DETECTED';

    const lm = results.multiHandLandmarks[0];
    drawSkeleton(lm);

    const res = classify(lm);
    processResult(res.letter, res.confidence);

  } else {
    noHandOverlay.classList.add('visible');
    handBadge.className   = 'hand-badge';
    handBadge.textContent = 'NO HAND';
    detectedLetter.textContent = '—';
    detectedConf.textContent   = 'Waiting…';
    resetStreak();

    if (currentWord) {
      noHandCount++;
      if (noHandCount >= SPACE_FRAMES) {
        pushWord();
      }
    }
  }
  dctx.restore();
}

function drawSkeleton(lm) {
  const w = displayCanvas.width;
  const h = displayCanvas.height;

  // Draw lines
  CONNECTIONS.forEach(([i, j], idx) => {
    const p1 = lm[i];
    const p2 = lm[j];
    dctx.beginPath();
    dctx.moveTo(p1.x * w, p1.y * h);
    dctx.lineTo(p2.x * w, p2.y * h);
    dctx.strokeStyle = EDGE_COLORS[idx] || '#00d4ff';
    dctx.lineWidth   = 3;
    dctx.lineCap     = 'round';
    dctx.stroke();
  });

  // Draw points
  lm.forEach((p, i) => {
    const x = p.x * w;
    const y = p.y * h;
    dctx.beginPath();
    dctx.arc(x, y, i === 0 ? 7 : 5, 0, 2 * Math.PI);
    dctx.fillStyle = NODE_COLORS[i] || '#ffffff';
    dctx.fill();
    dctx.strokeStyle = '#ffffff';
    dctx.lineWidth   = 1.5;
    dctx.stroke();
  });
}

async function initMediaPipe() {
  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 1,
    modelComplexity: 1,
    minDetectionConfidence: 0.6,
    minTrackingConfidence: 0.6,
  });

  hands.onResults(onResults);

  camera = new Camera(video, {
    onFrame: async () => {
      if (cameraActive) {
        await hands.send({ image: video });
      }
    },
    width: 640,
    height: 480,
  });

  await camera.start();
  cameraActive = true;
  setStatus('ready', 'Connected');
}

// ── Controls Setup ────────────────────────────────────────────────────────────
function setupControls() {
  document.getElementById('btn-backspace').addEventListener('click', () => {
    currentWord = currentWord.slice(0, -1);
    renderWord();
    resetStreak();
  });

  document.getElementById('btn-add-space').addEventListener('click', () => {
    pushWord();
  });

  document.getElementById('btn-clear-word').addEventListener('click', () => {
    currentWord = '';
    renderWord();
    resetStreak();
  });

  document.getElementById('btn-clear-sentence').addEventListener('click', () => {
    sentenceWords = [];
    currentWord = '';
    renderWord();
    renderSentence();
    resetStreak();
  });

  document.getElementById('btn-pause-detection').addEventListener('click', function() {
    paused = !paused;
    this.textContent = paused ? '▶ Resume' : '⏸ Pause';
    this.classList.toggle('active-btn', paused);
  });

  document.getElementById('btn-camera-toggle').addEventListener('click', function() {
    cameraActive = !cameraActive;
    this.textContent = cameraActive ? '📷 Stop Camera' : '📷 Start Camera';
    this.classList.toggle('active-btn', cameraActive);
  });

  document.getElementById('btn-copy').addEventListener('click', () => {
    if (sentenceWords.length) {
      navigator.clipboard.writeText(sentenceWords.join(' '));
      showToast('📋 Copied to clipboard!');
    }
  });

  document.getElementById('btn-speak').addEventListener('click', () => {
    if (sentenceWords.length && 'speechSynthesis' in window) {
      const u = new SpeechSynthesisUtterance(sentenceWords.join(' '));
      speechSynthesis.speak(u);
    }
  });

  // Mode buttons
  const btnAsl = document.getElementById('btn-asl');
  const btnIsl = document.getElementById('btn-isl');

  btnAsl.addEventListener('click', () => {
    currentMode = 'asl';
    currentWeights = aslWeights;
    btnAsl.classList.add('active');
    btnIsl.classList.remove('active');
    resetStreak();
    showToast('Switched to ASL');
  });

  btnIsl.addEventListener('click', () => {
    currentMode = 'isl';
    currentWeights = islWeights;
    btnIsl.classList.add('active');
    btnAsl.classList.remove('active');
    resetStreak();
    showToast('Switched to ISL');
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault();
      pushWord();
    } else if (e.key === 'Backspace') {
      currentWord = currentWord.slice(0, -1);
      renderWord();
    } else if (e.key === '1') {
      btnAsl.click();
    } else if (e.key === '2') {
      btnIsl.click();
    }
  });
}

// ── Startup ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  displayCanvas.width  = 640;
  displayCanvas.height = 480;

  setStatus('loading', 'Loading Model…');
  await loadLandmarkModel();
  await initMediaPipe();
  setupControls();
});
