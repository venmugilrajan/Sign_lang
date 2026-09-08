/**
 * SignLens — MediaPipe Landmark Neural Network Edition
 *
 * Runs classification directly in the browser using the weights
 * of our custom MLP Neural Network trained on the 21 hand landmarks.
 */

// ── Config ──────────────────────────────────────────────────────────────────
let   STREAK_NEEDED  = 5;     // frames of same letter to capture (settings sheet)
const COOLDOWN_MS    = 1000;  // ms after capture before detecting again
const SPACE_FRAMES   = 15;    // no-hand frames before auto word push
let   MIN_CONFIDENCE = 0.65;  // minimum classifier confidence (settings sheet)

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
  1:'#f2856a',2:'#f2856a',3:'#f2856a',4:'#f2856a',
  5:'#f0a53f',6:'#f0a53f',7:'#f0a53f',8:'#f0a53f',
  9:'#6ec9a0',10:'#6ec9a0',11:'#6ec9a0',12:'#6ec9a0',
  13:'#5aa9e6',14:'#5aa9e6',15:'#5aa9e6',16:'#5aa9e6',
  17:'#b98ae0',18:'#b98ae0',19:'#b98ae0',20:'#b98ae0',
};

const EDGE_COLORS = [
  '#f2856a','#f2856a','#f2856a','#f2856a',   // thumb   0-3
  '#f0a53f','#f0a53f','#f0a53f','#f0a53f',   // index   4-7
  '#6ec9a0','#6ec9a0','#6ec9a0','#6ec9a0',   // middle  8-11
  '#5aa9e6','#5aa9e6','#5aa9e6','#5aa9e6',   // ring    12-15
  '#b98ae0','#b98ae0','#b98ae0','#b98ae0',   // pinky   16-19
  '#6a6a76','#6a6a76','#6a6a76',             // palm    20-22
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
function setStreakVisual(pct) {
  streakBar.style.width = `${pct}%`;
  // .dial-ring reads --streak as a registered <number> and interpolates it.
  document.documentElement.style.setProperty('--streak', pct);
}
function resetStreak() {
  streakLetter = null; streakCount = 0;
  setStreakVisual(0);
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

/** 15 finger flexion triplets (vertex is the middle index) */
const ANGLE_TRIPLETS = [
  [0,1,2],   [1,2,3],    [2,3,4],
  [0,5,6],   [5,6,7],    [6,7,8],
  [0,9,10],  [9,10,11],  [10,11,12],
  [0,13,14], [13,14,15], [14,15,16],
  [0,17,18], [17,18,19], [18,19,20],
];

/** Angle at vertex b formed by a-b-c, in radians normalized to [0,1] */
function calculateAngle3D(a, b, c) {
  const bax = a[0]-b[0], bay = a[1]-b[1], baz = a[2]-b[2];
  const bcx = c[0]-b[0], bcy = c[1]-b[1], bcz = c[2]-b[2];
  const dot = bax*bcx + bay*bcy + baz*bcz;
  const nba = Math.sqrt(bax*bax + bay*bay + baz*baz);
  const nbc = Math.sqrt(bcx*bcx + bcy*bcy + bcz*bcz);
  const cos = Math.min(1, Math.max(-1, dot / (nba*nbc + 1e-6)));
  return Math.acos(cos) / Math.PI;
}

/**
 * Builds the 78-D feature vector the MLP was trained on:
 * 63 aspect-corrected, wrist-centred, palm-scaled, rotation-aligned coords
 * + 15 finger flexion angles.
 *
 * Mirrors train_landmark_classifier.extract_features_from_image() exactly,
 * including its rotation convention. Do not "fix" the rotation here alone --
 * the shipped weights were trained with it (see notes).
 */
function normalizeLandmarks(lm, w, h) {
  w = w || video.videoWidth  || 640;
  h = h || video.videoHeight || 480;

  // Aspect-ratio corrected pixel space: (x*w, y*h, z*w)
  const pts = lm.map(p => [p.x * w, p.y * h, p.z * w]);

  // Centre on the wrist
  const wrist = pts[0];
  const cen = pts.map(p => [p[0]-wrist[0], p[1]-wrist[1], p[2]-wrist[2]]);

  // Palm scale = 2D wrist -> middle MCP distance
  let scale = Math.sqrt(cen[9][0]*cen[9][0] + cen[9][1]*cen[9][1]);
  if (scale < 1e-4) scale = 1;

  // 2D rotation keyed on landmark 9 against the negative Y axis
  const angle = Math.atan2(cen[9][0]/scale, -cen[9][1]/scale);
  const ca = Math.cos(angle), sa = Math.sin(angle);

  const features = [];
  cen.forEach(p => {
    const x = p[0]/scale, y = p[1]/scale;
    features.push(x*ca - y*sa);
    features.push(x*sa + y*ca);
    features.push(p[2]/scale);          // z is scaled but not rotated
  });

  // Angles come from the raw pixel points, not the normalized ones
  ANGLE_TRIPLETS.forEach(([i, j, k]) =>
    features.push(calculateAngle3D(pts[i], pts[j], pts[k]))
  );

  return features;                       // 63 + 15 = 78
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
/**
 * Assembles the vector the active model expects.
 *  - single-hand (78-D): the first detected hand
 *  - dual-hand (156-D): slot 1 = leftmost hand on screen, slot 2 = rightmost
 *    (zeros when only one hand is up)
 *
 * Sorting is on the RAW wrist x of the unflipped frame, identical to
 * run_native.extract_landmarks(). If these two ever disagree, every
 * two-handed sign silently decodes to the wrong letter.
 */
function buildFeatureVector(multiHandLandmarks, isDualMode, w, h) {
  if (!isDualMode) {
    return normalizeLandmarks(multiHandLandmarks[0], w, h);
  }

  const hands = multiHandLandmarks
    .map(lm => ({ x: lm[0].x, feat: normalizeLandmarks(lm, w, h) }))
    .sort((a, b) => a.x - b.x);

  const slot1 = hands[0].feat;
  const slot2 = hands.length > 1 ? hands[1].feat : new Array(78).fill(0);
  return slot1.concat(slot2);
}

function classify(multiHandLandmarks) {
  if (!currentWeights || !multiHandLandmarks || multiHandLandmarks.length === 0) {
    return { letter: 'nothing', confidence: 0.0 };
  }

  // 1. Prepare feature vector (78 single-hand, or 156 dual-hand)
  const isDualMode = currentWeights.w0.length === 156;
  const x = buildFeatureVector(multiHandLandmarks, isDualMode);

  // matMul walks input.length, so a short vector would silently use only the
  // first N rows of w0 and return a confident wrong answer. Fail loudly instead.
  if (x.length !== currentWeights.w0.length) {
    console.error(`Feature/weight mismatch: built ${x.length}-D, model expects ${currentWeights.w0.length}-D`);
    return { letter: 'nothing', confidence: 0.0 };
  }

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
    ? ''
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
  setStreakVisual(pct);
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

    const hands = results.multiHandLandmarks;
    handBadge.textContent = hands.length > 1 ? '2 HANDS DETECTED' : 'HAND DETECTED';
    hands.forEach(drawSkeleton);

    const res = classify(hands);
    processResult(res.letter, res.confidence);

  } else {
    noHandOverlay.classList.add('visible');
    handBadge.className   = 'hand-badge';
    handBadge.textContent = 'NO HAND';
    detectedLetter.textContent = '';
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
    dctx.strokeStyle = EDGE_COLORS[idx] || '#6a6a76';   // arrays are aligned; neutral if ever not
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

/**
 * Renders a styled, in-theme camera failure inside the camera panel.
 * Replaces the native alert() MediaPipe would otherwise raise.
 */
function showCameraError(err) {
  const name = (err && err.name) ? err.name : 'Error';
  const HINTS = {
    NotReadableError:     'Another application is already using the camera. Close it (video call, another browser tab, your editor\u2019s preview pane) and reload.',
    NotAllowedError:      'Camera permission was denied. Allow camera access for this site, then reload.',
    NotFoundError:        'No camera device was found on this machine.',
    NotSupportedError:    'This browser cannot access a camera on this page. A secure origin (https:// or localhost) is required.',
    OverconstrainedError: 'No camera matches the requested 640\u00d7480 video mode.',
    AbortError:           'The camera was released before it could start. Reload to retry.',
  };
  const msg = HINTS[name] || (err && err.message) || 'The camera could not be started.';

  noHandOverlay.innerHTML =
    '<div class="cam-error-icon" aria-hidden="true">\u26A0</div>' +
    '<div class="cam-error-title">Camera unavailable</div>' +
    '<div class="cam-error-msg"></div>' +
    '<div class="cam-error-code"></div>';
  // textContent, not innerHTML -- the error string is not ours to trust as markup
  noHandOverlay.querySelector('.cam-error-msg').textContent = msg;
  noHandOverlay.querySelector('.cam-error-code').textContent = name;
  noHandOverlay.classList.add('error');
  noHandOverlay.classList.remove('hidden');
  noHandOverlay.setAttribute('role', 'alert');

  setStatus('err', 'Camera unavailable');
  detectedLetter.textContent = '';
  detectedConf.textContent = 'No camera';
  console.error('[SignLens] camera init failed:', err);
}

async function initMediaPipe() {
  // Probe getUserMedia ourselves first. MediaPipe's camera_utils.js calls a
  // native alert() when acquisition fails, which cannot be styled or caught --
  // so fail here, before it ever runs.
  try {
    const probe = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 }
    });
    probe.getTracks().forEach(t => t.stop());   // release for MediaPipe to reopen
  } catch (err) {
    showCameraError(err);
    return false;
  }

  hands = new Hands({
    locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
  });

  hands.setOptions({
    maxNumHands: 2,
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

  try {
    await camera.start();
  } catch (err) {
    showCameraError(err);
    return false;
  }
  cameraActive = true;
  setStatus('ready', 'Connected');
  return true;
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
    modeToggle.classList.remove('isl-mode');   // drives .mode-slider transform
    resetStreak();
    showToast('Switched to ASL');
  });

  btnIsl.addEventListener('click', () => {
    currentMode = 'isl';
    currentWeights = islWeights;
    btnIsl.classList.add('active');
    btnAsl.classList.remove('active');
    modeToggle.classList.add('isl-mode');      // drives .mode-slider transform
    resetStreak();
    showToast('Switched to ISL');
  });

  // Settings sheet. Both thresholds are read live by processResult(),
  // so no restart is needed after a change.
  const setConf   = document.getElementById('set-confidence');
  const outConf   = document.getElementById('out-confidence');
  const setStreak = document.getElementById('set-streak');
  const outStreak = document.getElementById('out-streak');

  setConf.value   = Math.round(MIN_CONFIDENCE * 100);
  outConf.value   = `${setConf.value}%`;
  setStreak.value = STREAK_NEEDED;
  outStreak.value = STREAK_NEEDED;

  setConf.addEventListener('input', () => {
    MIN_CONFIDENCE = Number(setConf.value) / 100;
    outConf.value  = `${setConf.value}%`;
  });

  setStreak.addEventListener('input', () => {
    STREAK_NEEDED   = Number(setStreak.value);
    outStreak.value = STREAK_NEEDED;
    resetStreak();                 // rescale the dial against the new target
  });

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    // Ignore shortcuts while a control has focus, or a sheet is open.
    if (e.target.closest('input, [popover]')) return;

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

  // Wire controls FIRST: a camera failure must not leave every button dead.
  setupControls();

  setStatus('loading', 'Loading Model…');
  try {
    await loadLandmarkModel();
    await initMediaPipe();
  } catch (err) {
    showCameraError(err);
  }
});
