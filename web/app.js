/**
 * SignLens — MediaPipe Landmark Neural Network Edition
 *
 * Runs classification directly in the browser using the weights
 * of our custom MLP Neural Network trained on the 21 hand landmarks.
 */

// ── Config ──────────────────────────────────────────────────────────────────
let   STREAK_NEEDED   = 5;     // frames of same letter to capture (settings sheet, matches native)
const COOLDOWN_MS     = 350;   // ms after capture before detecting again (fluid sign transitions)
let   SPACE_FRAMES    = 90;    // no-hand frames before auto word push (~3.0s at 30fps, plenty of transition time)
let   MIN_CONFIDENCE  = 0.50;  // minimum classifier confidence (settings sheet, matches native 0.40-0.50)
const MIN_HAND_SPAN   = 35;    // minimum pixel distance between wrist and middle MCP to reject ghost hands

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
let currentMode       = 'asl';
let paused            = false;
let inCooldown        = false;
let cameraActive      = false;

let currentWord       = '';
let letterConfidences = [];
let sentenceWords     = [];

let streakLetter  = null;
let streakCount   = 0;
let noHandCount   = 0;

// State tracking to prevent runaway letter repeat without releasing hand
let lastCapturedLetter = '';
let releasedSinceLastCapture = true;
let releaseFrames = 0;
const RELEASE_FRAMES_NEEDED = 3;

// State tracking to prevent runaway word repeat without clear user intent
let lastCommittedWord = '';

// Trained Neural Network model weights (kept for OFFLINE fallback only)
let aslWeights     = null;
let islWeights     = null;
let currentWeights = null;

// ── Server-side inference state ──────────────────────────────────────────────
// When Flask is running, all classification is done server-side (predict_proba
// on the sklearn model) for feature parity with run_native.py.
// If the server is unreachable, we fall back to the JS MLP.
let serverAvailable   = false;   // set true after first successful /health probe
let serverCheckPending = false;

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
        body: JSON.stringify({ word: w, confidences: letterConfidences })
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

    // Gating: Don't repeat the exact same word back-to-back if triggered by background jitter
    const isDuplicateAutoPush = (sentenceWords.length > 0 && sentenceWords[sentenceWords.length - 1] === w && lastCommittedWord === w);
    if (!isDuplicateAutoPush) {
      sentenceWords.push(w);
      lastCommittedWord = w;
      renderSentence();
      showToast(`✓ "${w}"`);
    }
  }
  currentWord = '';
  letterConfidences = [];
  renderWord();
  resetStreak();
  noHandCount = 0;
  lastCapturedLetter = '';
  releasedSinceLastCapture = true;
  releaseFrames = 0;
}

// ── Letter capture ────────────────────────────────────────────────────────────
function captureLetter(letter, confidence = 0.85) {
  if (inCooldown) return;
  const l = letter.toLowerCase();
  if (l === 'space' || l === '_') { pushWord(); return; }
  if (l === 'del' || l === 'delete') {
    currentWord = currentWord.slice(0, -1);
    letterConfidences.pop();
    renderWord();
    resetStreak();
    lastCapturedLetter = '';
    releasedSinceLastCapture = true;
    return;
  }
  if (l === 'nothing') return;

  currentWord += letter.toUpperCase();
  letterConfidences.push(confidence);
  renderWord();

  lastCapturedLetter = letter.toUpperCase();
  releasedSinceLastCapture = false;
  releaseFrames = 0;

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
function normalizeLandmarks(lm, w, h, handedness = "Right", applyHandednessMirror = false) {
  w = w || video.videoWidth  || 640;
  h = h || video.videoHeight || 480;

  // Aspect-ratio corrected pixel space: (x*w, y*h, z*w)
  const pts = lm.map(p => [p.x * w, p.y * h, p.z * w]);

  // Mirror Left hand to match Right hand dataset distribution (ASL single hand)
  if (applyHandednessMirror && handedness === "Left") {
    const wristX = pts[0][0];
    pts.forEach(p => {
      p[0] = wristX - (p[0] - wristX);
    });
  }

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
  // ISL uses the mathematically correct rotation matrix [[cos, -sin], [sin, cos]]
  // that aligns landmark 9 to vertical 12 o'clock, cancelling any hand tilt.
  const isIslMode = (typeof currentMode !== 'undefined' && currentMode === 'isl');
  cen.forEach(p => {
    const x = p[0]/scale, y = p[1]/scale;
    if (isIslMode) {
      features.push(x*ca + y*sa);
      features.push(-x*sa + y*ca);
    } else {
      // ASL legacy rotation
      features.push(x*ca - y*sa);
      features.push(x*sa + y*ca);
    }
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

/**
 * Probes /health to find out whether the Flask backend is up.
 * Called once at startup and retried whenever a classify() call fails.
 */
async function probeServer() {
  if (serverCheckPending) return;
  serverCheckPending = true;
  try {
    const r = await fetch('/health', { signal: AbortSignal.timeout(1500) });
    if (r.ok) {
      if (!serverAvailable) {
        serverAvailable = true;
        console.log('✅ Flask backend detected — using server-side inference');
        setStatus('ready', 'Connected');
      }
    } else {
      serverAvailable = false;
    }
  } catch {
    serverAvailable = false;
  } finally {
    serverCheckPending = false;
    // If server is currently offline, automatically retry in 2.5s
    if (!serverAvailable && cameraActive) {
      setTimeout(probeServer, 2500);
    }
  }
}

/** Load MLP weights from JSON files (used for OFFLINE fallback only) */
async function loadLandmarkModel() {
  try {
    const resAsl = await fetch('asl_landmarks_weights.json');
    if (resAsl.ok) {
      aslWeights = await resAsl.json();
      console.log('🧠 ASL Neural Network loaded (offline fallback)');
    }
    const resIsl = await fetch('isl_landmarks_weights.json');
    if (resIsl.ok) {
      islWeights = await resIsl.json();
      console.log('🧠 ISL Neural Network loaded (offline fallback)');
    }
    currentWeights = currentMode === 'asl' ? aslWeights : islWeights;
  } catch (err) {
    console.error('Failed to load MLP weights:', err);
  }
}

/**
 * Assembles the 78-D (single-hand) or 156-D (dual-hand) feature vector.
 *
 * Sorting is on the RAW wrist x of the unflipped frame, identical to
 * run_native.extract_landmarks(). If these two ever disagree, every
 * two-handed sign silently decodes to the wrong letter.
 */
function buildFeatureVector(multiHandLandmarks, isDualMode, multiHandedness, w, h) {
  if (!isDualMode) {
    const handedness = (multiHandedness && multiHandedness[0] && multiHandedness[0].label)
      ? multiHandedness[0].label
      : 'Right';
    // In ASL single hand mode, mirror Left hands to match Right hand dataset
    return normalizeLandmarks(multiHandLandmarks[0], w, h, handedness, true);
  }

  // Dual mode (ISL):
  // Single-hand ISL model is trained on BOTH Left & Right hands natively,
  // and two-handed classes strictly require both hands.
  // No fragile handedness guessing/mirroring is needed.
  const hands = multiHandLandmarks
    .map(lm => ({ x: lm[0].x, feat: normalizeLandmarks(lm, w, h, 'Right', false) }))
    .sort((a, b) => a.x - b.x);

  const slot1 = hands[0].feat;
  const slot2 = hands.length > 1 ? hands[1].feat : new Array(78).fill(0);
  return slot1.concat(slot2);
}

// Helper to check if hand landmarks represent a valid, non-ghost hand
function isValidHand(lm) {
  if (!lm || lm.length < 21) return false;
  const w = displayCanvas.width  || 640;
  const h = displayCanvas.height || 480;

  // Wrist (0) to Middle MCP (9) distance in pixels
  const dx = (lm[9].x - lm[0].x) * w;
  const dy = (lm[9].y - lm[0].y) * h;
  const span = Math.sqrt(dx * dx + dy * dy);

  // Reject tiny noise clusters / background artifacts
  if (span < MIN_HAND_SPAN) return false;

  // Check landmark coordinates are reasonably within frame [-0.2, 1.2]
  for (let i = 0; i < lm.length; i++) {
    if (isNaN(lm[i].x) || isNaN(lm[i].y)) return false;
    if (lm[i].x < -0.2 || lm[i].x > 1.2 || lm[i].y < -0.2 || lm[i].y > 1.2) return false;
  }
  return true;
}

/**
 * JS MLP fallback — used ONLY when Flask server is unreachable.
 * Runs the MLP forward pass in-browser using the cached weight JSON.
 */
function classifyJS(featureVector) {
  if (!currentWeights) return { letter: 'nothing', confidence: 0.0 };

  const x = featureVector;
  if (x.length !== currentWeights.w0.length) {
    return { letter: 'nothing', confidence: 0.0 };
  }

  let h = relu(matMul(x, currentWeights.w0, currentWeights.b0));
  h = relu(matMul(h, currentWeights.w1, currentWeights.b1));
  if (currentWeights.w2 && currentWeights.b2) {
    if (currentWeights.w3 && currentWeights.b3) {
      h = relu(matMul(h, currentWeights.w2, currentWeights.b2));
      h = matMul(h, currentWeights.w3, currentWeights.b3);
    } else {
      h = matMul(h, currentWeights.w2, currentWeights.b2);
    }
  }

  let probs = softmax(h);

  // ── ISL Hand-Count Constrained Gating ─────────────────────────────────
  if (currentMode === 'isl' && x.length === 156) {
    const isOneHand = x.slice(78).every(v => Math.abs(v) < 1e-4);
    const islSingleHand = new Set(['1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'I', 'J', 'L', 'O', 'Q', 'U', 'V']);
    for (let i = 0; i < probs.length; i++) {
      const cls = currentWeights.classes[i];
      if (isOneHand ? !islSingleHand.has(cls) : islSingleHand.has(cls)) {
        probs[i] = 0;
      }
    }
    const pSum = probs.reduce((a, b) => a + b, 0);
    if (pSum > 0) {
      probs = probs.map(p => p / pSum);
    }
  }

  let maxIdx = 0, maxProb = 0, secondProb = 0;
  for (let i = 0; i < probs.length; i++) {
    if (probs[i] > maxProb) { secondProb = maxProb; maxProb = probs[i]; maxIdx = i; }
    else if (probs[i] > secondProb) { secondProb = probs[i]; }
  }
  if ((maxProb - secondProb) < 0.08 && maxProb < 0.85) {
    return { letter: 'nothing', confidence: maxProb };
  }
  return { letter: currentWeights.classes[maxIdx], confidence: maxProb };
}

/**
 * PRIMARY classify() — async.
 *
 * Strategy:
 *   1. Build the feature vector in JS (normalization is correct and fast).
 *   2. POST it to Flask /predict_landmarks — uses sklearn's calibrated
 *      predict_proba(), identical to run_native.py. This is why signs that
 *      work in native Python also work in the browser.
 *   3. If the server is unavailable, fall back to the JS MLP.
 */
async function classify(multiHandLandmarks, multiHandedness) {
  if (!multiHandLandmarks || multiHandLandmarks.length === 0) {
    return { letter: 'nothing', confidence: 0.0 };
  }

  // Determine if active model is dual-hand (156-D) or single-hand (78-D)
  const isDualMode = currentWeights ? currentWeights.w0.length === 156 : false;

  // Build feature vector with true display/camera aspect ratio
  const cw = (displayCanvas && displayCanvas.width) || (video && video.videoWidth) || 640;
  const ch = (displayCanvas && displayCanvas.height) || (video && video.videoHeight) || 480;
  const featureVector = buildFeatureVector(multiHandLandmarks, isDualMode, multiHandedness, cw, ch);

  // ── PATH A: Server-side inference (preferred) ────────────────────────────
  if (serverAvailable) {
    try {
      const response = await fetch('/predict_landmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: currentMode, landmarks: featureVector }),
        signal: AbortSignal.timeout(1500),   // 1500ms resilient timeout per frame
      });

      if (response.ok) {
        const data = await response.json();
        return {
          letter:     data.letter     || 'nothing',
          confidence: data.confidence || 0.0,
          top3:       data.top3       || [],
        };
      }

      // Non-200 response — server up but something is wrong
      console.warn('[classify] Server error:', response.status);
      return { letter: 'nothing', confidence: 0.0 };

    } catch (err) {
      // Timeout or network error → mark server down and re-probe
      console.warn('[classify] Server unreachable, switching to JS fallback:', err.name);
      serverAvailable = false;
      showToast('⚠️ Server offline — using JS fallback', 3000);
      setStatus('loading', 'JS fallback');
      probeServer();   // async retry in background
    }
  }

  // ── PATH B: JS MLP fallback ──────────────────────────────────────────────
  if (!currentWeights) return { letter: 'nothing', confidence: 0.0 };
  return classifyJS(featureVector);
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

  // Track release for repeating the same letter
  if (!effective || effective !== lastCapturedLetter) {
    releaseFrames++;
    if (releaseFrames >= RELEASE_FRAMES_NEEDED) {
      releasedSinceLastCapture = true;
    }
  } else {
    releaseFrames = 0;
  }

  if (paused || inCooldown || !effective) {
    if (!effective) resetStreak();
    return;
  }

  // Prevent spamming the exact same letter repeatedly without releasing hand
  if (effective.toUpperCase() === lastCapturedLetter && !releasedSinceLastCapture) {
    resetStreak();
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
    captureLetter(streakLetter, confidence);
  }
}

// (handLandmarker and animFrameId declared near initMediaPipe below)

// Prevent concurrent classify calls from piling up while a fetch is in flight
let classifyInFlight = false;

async function onResults(results) {
  dctx.save();
  dctx.clearRect(0, 0, displayCanvas.width, displayCanvas.height);
  dctx.drawImage(results.image, 0, 0, displayCanvas.width, displayCanvas.height);

  // Filter for genuine hands with valid size and geometry
  const validHands = (results.multiHandLandmarks || []).filter(isValidHand);

  if (validHands.length > 0) {
    noHandOverlay.classList.remove('visible');
    handBadge.className   = 'hand-badge live';
    if (currentMode === 'isl') {
      handBadge.textContent = validHands.length > 1 ? '2 HANDS · ISL (A-Z)' : '1 HAND · ISL (1-9, C, I, L, O, U, V)';
    } else {
      handBadge.textContent = validHands.length > 1 ? '2 HANDS DETECTED' : 'HAND DETECTED';
    }
    validHands.forEach(drawSkeleton);

    // Skip this frame if the previous classify() hasn't returned yet
    // (avoids a queue of stale requests building up on slow networks)
    if (!classifyInFlight) {
      classifyInFlight = true;
      try {
        const res = await classify(validHands, results.multiHandedness);
        processResult(res.letter, res.confidence);
      } finally {
        classifyInFlight = false;
      }
    }

  } else {
    noHandOverlay.classList.add('visible');
    handBadge.className   = 'hand-badge';
    handBadge.textContent = 'NO HAND';
    detectedLetter.textContent = '';
    detectedConf.textContent   = 'Waiting…';
    resetStreak();

    // Hand was dropped/removed -> immediately grant release permission for next letter
    releaseFrames++;
    if (releaseFrames >= RELEASE_FRAMES_NEEDED) {
      releasedSinceLastCapture = true;
    }

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

// ── MediaPipe Tasks Vision Camera Loop ───────────────────────────────────────
let handLandmarker = null;
let animFrameId    = null;

/**
 * Converts new Tasks Vision HandLandmarker results into the same shape
 * the rest of app.js expects:
 *   multiHandLandmarks: Array of [{x,y,z}, ...21]  (same as old API)
 *   multiHandedness:    Array of {label: 'Right'|'Left'}  (same field name)
 */
function adaptResults(result) {
  const rawLandmarks = result.landmarks || [];
  const rawHandedness = result.handedness || [];
  
  const multiHandLandmarks = [];
  const multiHandedness = [];

  for (let i = 0; i < rawLandmarks.length; i++) {
    const lms = rawLandmarks[i];
    if (!lms || lms.length < 21) continue;

    // Anatomical validation:
    // Rejects facial false positives while fully supporting profile/curled hand signs like 'O', 'C', 'E', 'S'
    const palmLen = Math.hypot(lms[9].x - lms[0].x, lms[9].y - lms[0].y);
    const totalSpan = Math.max(...lms.map(p => Math.hypot(p.x - lms[0].x, p.y - lms[0].y)));
    if (palmLen < 0.04 || totalSpan < 0.08) {
      continue;
    }

    multiHandLandmarks.push(lms);
    const h = rawHandedness[i];
    multiHandedness.push({
      label: (h && h[0] && h[0].categoryName) ? h[0].categoryName : 'Right'
    });
  }

  return { multiHandLandmarks, multiHandedness };
}

async function initMediaPipe() {
  // Probe getUserMedia first so we can show a styled error instead of a
  // native alert() on failure.
  try {
    const probe = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480 }
    });
    probe.getTracks().forEach(t => t.stop());
  } catch (err) {
    showCameraError(err);
    return false;
  }

  // Dynamically import the Tasks Vision ES-module bundle.
  // More reliable than a <script> UMD tag: exports land directly in scope,
  // no window.* namespace hunting required.
  const { FilesetResolver, HandLandmarker } = await import(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs'
  );

  // Resolve WASM files from the same CDN version
  const vision = await FilesetResolver.forVisionTasks(
    'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm'
  );

  handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      // Same hand_landmarker model as Python's hand_landmarker.task — identical coordinate space
      modelAssetPath: 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task',
      delegate: 'GPU',
    },
    runningMode:   'VIDEO',
    numHands:      2,
    minHandDetectionConfidence: 0.40,
    minHandPresenceConfidence:  0.40,
    minTrackingConfidence:      0.40,
  });

  // Start camera stream
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' }
    });
  } catch (err) {
    showCameraError(err);
    return false;
  }

  video.srcObject = stream;
  await new Promise(resolve => { video.onloadedmetadata = resolve; });
  video.play();

  cameraActive = true;
  setStatus('ready', 'Connected');

  // Detection loop using requestAnimationFrame for smooth 30fps
  let lastVideoTime = -1;
  function detect() {
    if (!cameraActive) { animFrameId = null; return; }
    animFrameId = requestAnimationFrame(detect);

    if (video.currentTime === lastVideoTime) return;  // no new frame yet
    lastVideoTime = video.currentTime;

    displayCanvas.width  = video.videoWidth  || 640;
    displayCanvas.height = video.videoHeight || 480;

    // detectForVideo uses the video timestamp in ms for temporal smoothing
    const result = handLandmarker.detectForVideo(video, performance.now());
    const adapted = adaptResults(result);

    // Synthesise a results object matching what onResults() already expects
    onResults({
      image:              video,
      multiHandLandmarks: adapted.multiHandLandmarks,
      multiHandedness:    adapted.multiHandedness,
    });
  }
  detect();
  return true;
}

// ── Controls Setup ────────────────────────────────────────────────────────────
function setupControls() {
  document.getElementById('btn-backspace').addEventListener('click', () => {
    currentWord = currentWord.slice(0, -1);
    letterConfidences.pop();
    renderWord();
    resetStreak();
  });

  document.getElementById('btn-add-space').addEventListener('click', () => {
    pushWord();
  });

  document.getElementById('btn-clear-word').addEventListener('click', () => {
    currentWord = '';
    letterConfidences = [];
    renderWord();
    resetStreak();
  });

  document.getElementById('btn-clear-sentence').addEventListener('click', () => {
    sentenceWords = [];
    currentWord = '';
    letterConfidences = [];
    renderWord();
    renderSentence();
    resetStreak();
  });

  document.getElementById('btn-pause-detection').addEventListener('click', function() {
    paused = !paused;
    this.textContent = paused ? 'Resume' : 'Pause';
    this.classList.toggle('active-btn', paused);
  });

  document.getElementById('btn-camera-toggle').addEventListener('click', function() {
    cameraActive = !cameraActive;
    this.textContent = cameraActive ? 'Stop Camera' : 'Start Camera';
    this.classList.toggle('active-btn', !cameraActive);
    if (!cameraActive && animFrameId) {
      cancelAnimationFrame(animFrameId);
      animFrameId = null;
    } else if (cameraActive && !animFrameId && handLandmarker) {
      // restart loop
      let lastVideoTime = -1;
      function detect() {
        if (!cameraActive) { animFrameId = null; return; }
        animFrameId = requestAnimationFrame(detect);
        if (video.currentTime === lastVideoTime) return;
        lastVideoTime = video.currentTime;
        displayCanvas.width  = video.videoWidth  || 640;
        displayCanvas.height = video.videoHeight || 480;
        const result = handLandmarker.detectForVideo(video, performance.now());
        const adapted = adaptResults(result);
        onResults({ image: video, multiHandLandmarks: adapted.multiHandLandmarks, multiHandedness: adapted.multiHandedness });
      }
      detect();
    }
  });

  document.getElementById('btn-copy').addEventListener('click', () => {
    if (sentenceWords.length) {
      navigator.clipboard.writeText(sentenceWords.join(' '));
      showToast('📋 Copied to clipboard!');
    }
  });

  // Global reference to prevent Chrome garbage-collection bug on SpeechSynthesisUtterance
  window._activeUtterance = null;

  function fallbackServerSpeak(text, btn) {
    fetch('/speak', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    }).then(res => res.json())
      .then(data => {
        console.log('[SignLens] Server TTS spoke:', data);
      })
      .catch(err => console.error('[SignLens] Server TTS failed:', err))
      .finally(() => {
        if (btn) {
          btn.classList.remove('active');
          btn.textContent = 'Speak';
        }
      });
  }

  function speakSentence() {
    const text = (sentenceWords.length
      ? sentenceWords.join(' ')
      : (sentenceDisplay.innerText || '').replace(/Your sentence will appear here[…\.]*/i, '')
    ).trim();

    if (!text) {
      showToast('⚠️ No sentence to speak yet');
      return;
    }

    const btnSpeak = document.getElementById('btn-speak');
    if (btnSpeak) {
      btnSpeak.classList.add('active');
      btnSpeak.textContent = 'Speaking…';
    }
    showToast(`🔊 Speaking: "${text}"`);

    if ('speechSynthesis' in window) {
      try {
        window.speechSynthesis.cancel(); // Reset stuck Chromium speech queue

        const u = new SpeechSynthesisUtterance(text);
        window._activeUtterance = u; // Keep reference to prevent V8 GC drop

        u.lang = 'en-US';
        u.rate = 0.95;
        u.pitch = 1.0;
        u.volume = 1.0;

        const voices = window.speechSynthesis.getVoices();
        if (voices && voices.length > 0) {
          const enVoice = voices.find(v => v.lang.startsWith('en') && !v.localService) ||
                          voices.find(v => v.lang.startsWith('en')) ||
                          voices[0];
          if (enVoice) u.voice = enVoice;
        }

        u.onend = () => {
          window._activeUtterance = null;
          if (btnSpeak) {
            btnSpeak.classList.remove('active');
            btnSpeak.textContent = 'Speak';
          }
        };

        u.onerror = (e) => {
          console.warn('[SignLens] Browser speech error, falling back to server TTS:', e);
          window._activeUtterance = null;
          fallbackServerSpeak(text, btnSpeak);
        };

        window.speechSynthesis.speak(u);

        if (window.speechSynthesis.paused) {
          window.speechSynthesis.resume();
        }

        // Safety timeout to reset button state after utterance
        setTimeout(() => {
          if (btnSpeak && btnSpeak.textContent === 'Speaking…') {
            btnSpeak.classList.remove('active');
            btnSpeak.textContent = 'Speak';
          }
        }, Math.max(3000, text.split(' ').length * 1200));

      } catch (err) {
        console.warn('[SignLens] SpeechSynthesis exception:', err);
        fallbackServerSpeak(text, btnSpeak);
      }
    } else {
      fallbackServerSpeak(text, btnSpeak);
    }
  }

  document.getElementById('btn-speak').addEventListener('click', speakSentence);

  // Mode buttons
  const btnAsl = document.getElementById('btn-asl');
  const btnIsl = document.getElementById('btn-isl');
  const btnVisually = document.getElementById('btn-visually');
  const liveWorkspace = document.getElementById('live-workspace');
  const visuallyWorkspace = document.getElementById('visually-workspace');

  function isInVisuallyMode() {
    return visuallyWorkspace && !visuallyWorkspace.classList.contains('hidden');
  }

  btnAsl.addEventListener('click', () => {
    currentMode = 'asl';
    currentWeights = aslWeights;
    btnAsl.classList.add('active');
    btnIsl.classList.remove('active');
    if (btnVisually) btnVisually.classList.remove('active');
    modeToggle.classList.remove('isl-mode', 'visually-mode');
    if (liveWorkspace) liveWorkspace.classList.remove('hidden');
    if (visuallyWorkspace) visuallyWorkspace.classList.add('hidden');
    resetStreak();
    showToast('Switched to ASL (Live Camera)');
  });

  btnIsl.addEventListener('click', () => {
    currentMode = 'isl';
    currentWeights = islWeights;
    btnIsl.classList.add('active');
    btnAsl.classList.remove('active');
    if (btnVisually) btnVisually.classList.remove('active');
    modeToggle.classList.remove('visually-mode');
    modeToggle.classList.add('isl-mode');
    if (liveWorkspace) liveWorkspace.classList.remove('hidden');
    if (visuallyWorkspace) visuallyWorkspace.classList.add('hidden');
    resetStreak();
    showToast('Switched to ISL (Live Camera)');
  });

  if (btnVisually) {
    btnVisually.addEventListener('click', () => {
      btnVisually.classList.add('active');
      btnAsl.classList.remove('active');
      btnIsl.classList.remove('active');
      modeToggle.classList.remove('isl-mode');
      modeToggle.classList.add('visually-mode');
      if (liveWorkspace) liveWorkspace.classList.add('hidden');
      if (visuallyWorkspace) visuallyWorkspace.classList.remove('hidden');
      activatePracticeStudio();
      showToast('Practice Studio Active');
    });
  }

  // Settings sheet. Both thresholds are read live by processResult(),
  // so no restart is needed after a change.
  const setConf        = document.getElementById('set-confidence');
  const outConf        = document.getElementById('out-confidence');
  const setStreak      = document.getElementById('set-streak');
  const outStreak      = document.getElementById('out-streak');
  const setTimeoutInput = document.getElementById('set-timeout');
  const outTimeout     = document.getElementById('out-timeout');

  setConf.value   = Math.round(MIN_CONFIDENCE * 100);
  outConf.value   = `${setConf.value}%`;
  setStreak.value = STREAK_NEEDED;
  outStreak.value = STREAK_NEEDED;
  if (setTimeoutInput && outTimeout) {
    setTimeoutInput.value = (SPACE_FRAMES / 30).toFixed(1);
    outTimeout.value      = `${setTimeoutInput.value}s`;
  }

  setConf.addEventListener('input', () => {
    MIN_CONFIDENCE = Number(setConf.value) / 100;
    outConf.value  = `${setConf.value}%`;
  });

  setStreak.addEventListener('input', () => {
    STREAK_NEEDED   = Number(setStreak.value);
    outStreak.value = STREAK_NEEDED;
    resetStreak();                 // rescale the dial against the new target
  });

  if (setTimeoutInput && outTimeout) {
    setTimeoutInput.addEventListener('input', () => {
      const sec = Number(setTimeoutInput.value);
      SPACE_FRAMES = Math.round(sec * 30);
      outTimeout.value = `${sec.toFixed(1)}s`;
    });
  }

  // Theme Toggle (Light / Dark Mode)
  const btnThemeToggle = document.getElementById('btn-theme-toggle');
  const iconThemeToggle = document.getElementById('theme-toggle-icon');

  function applyTheme(theme) {
    const isLight = theme === 'light';
    document.documentElement.setAttribute('data-theme', isLight ? 'light' : 'dark');
    localStorage.setItem('signlens_theme', isLight ? 'light' : 'dark');
    if (iconThemeToggle) {
      iconThemeToggle.textContent = isLight ? '🌙' : '☀️';
    }
    if (btnThemeToggle) {
      btnThemeToggle.setAttribute('title', isLight ? 'Switch to Dark Mode' : 'Switch to Light Mode');
      btnThemeToggle.setAttribute('aria-label', isLight ? 'Switch to Dark Mode' : 'Switch to Light Mode');
    }
    // Re-render avatar if in practice mode so canvas background updates immediately
    if (typeof drawAvatar === 'function') {
      try { drawAvatar(); } catch (_) {}
    }
  }

  // Load saved preference or fallback to dark
  const savedTheme = localStorage.getItem('signlens_theme') || 'dark';
  applyTheme(savedTheme);

  if (btnThemeToggle) {
    btnThemeToggle.addEventListener('click', () => {
      const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
      const nextTheme = currentTheme === 'light' ? 'dark' : 'light';
      applyTheme(nextTheme);
      showToast(nextTheme === 'light' ? '☀️ Switched to Light Mode' : '🌙 Switched to Dark Mode');
    });
  }

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    // Ignore shortcuts while a control has focus, or a sheet is open.
    if (e.target.closest('input, textarea, [popover]')) return;

    if (e.key === ' ' || e.code === 'Space') {
      e.preventDefault();
      pushWord();
    } else if (e.key === 'Backspace') {
      currentWord = currentWord.slice(0, -1);
      letterConfidences.pop();
      renderWord();
    } else if (e.key === '1') {
      btnAsl.click();
    } else if (e.key === '2') {
      btnIsl.click();
    } else if (e.key === '3') {
      if (btnVisually) btnVisually.click();
    }
  });
}

// ── Sign Reference Guide Modal ───────────────────────────────────────────────
// ── Sign Reference Guide Modal ───────────────────────────────────────────────
function initSignGuide() {
  const panelGuide = document.getElementById('panel-guide');
  const btnOpenGuide = document.getElementById('btn-open-guide');
  const btnCloseGuide = document.getElementById('btn-close-guide');
  const tabIsl = document.getElementById('tab-guide-isl');
  const tabAsl = document.getElementById('tab-guide-asl');
  const contentIsl = document.getElementById('guide-content-isl');
  const contentAsl = document.getElementById('guide-content-asl');
  const singleGrid = document.getElementById('isl-single-cards');
  const dualGrid = document.getElementById('isl-dual-cards');

  const islDescFallback = {
    '1': 'Index finger extended upwards',
    '2': 'Index and middle fingers extended in V-shape',
    '3': 'Thumb, index, and middle fingers extended',
    '4': 'Four fingers extended upwards',
    '5': 'Open hand with all five fingers spread',
    '6': 'Thumb and little finger extended',
    '7': 'Thumb, index, and little finger extended',
    '8': 'Middle, ring, and little fingers extended',
    '9': 'Thumb touching index finger in circle',
    'C': 'Curved hand forming C-shape',
    'I': 'Little finger extended upwards',
    'L': 'Thumb and index finger forming L-shape',
    'O': 'All fingers curved touching thumb in O-shape',
    'U': 'Index and middle fingers held together vertically',
    'V': 'Index and middle fingers in open V-shape',
    'A': 'Dominant index finger touching non-dominant thumb tip',
    'B': 'Both hands forming double circles / glasses pose',
    'D': 'Dominant index pointing to non-dominant index forming D',
    'E': 'Dominant index touching non-dominant index tip',
    'F': 'Dominant index and middle crossing non-dominant index and middle',
    'G': 'Both fists held together vertically with knuckles facing forward',
    'H': 'Dominant open palm sweeping flat across non-dominant palm',
    'J': 'Index finger drawing J stroke across non-dominant palm',
    'K': 'Dominant index finger hooking over non-dominant index finger',
    'M': 'Dominant three fingers laid across non-dominant palm',
    'N': 'Dominant two fingers laid across non-dominant palm',
    'P': 'Dominant index finger forming circle with thumb touching non-dominant tip',
    'Q': 'Dominant hand forming circle loop with non-dominant index through',
    'R': 'Dominant index finger hooked on non-dominant palm',
    'S': 'Dominant pinky hooking non-dominant pinky / fist lock',
    'T': 'Dominant index finger placed perpendicularly on non-dominant index side',
    'W': 'Interlocking open fingers of both hands upright',
    'X': 'Crossing both index fingers in an X',
    'Y': 'Dominant index pointing between non-dominant thumb and index',
    'Z': 'Dominant open palm held upright against non-dominant palm'
  };

  function createGuideCard(letter, isTwoHanded, desc) {
    const el = document.createElement('div');
    el.className = 'guide-card';
    el.title = `${letter}: ${desc}`;
    el.innerHTML = `
      <div class="guide-card-img-wrap">
        <img src="isl_signs/${letter}.png" alt="Sign ${letter}" loading="lazy" />
      </div>
      <div class="guide-card-body">
        <span class="guide-card-char">${letter}</span>
        <span class="guide-card-hands ${isTwoHanded ? 'pill-blue' : 'pill-amber'} pill-badge">${isTwoHanded ? '2 Hands' : '1 Hand'}</span>
      </div>
      <p class="guide-card-desc" style="padding: 6px 10px 10px; font-size: 0.72rem; color: var(--text-dim); line-height: 1.35; margin: 0;">${desc}</p>
    `;
    return el;
  }

  function renderGuideCards() {
    if (!singleGrid || !dualGrid) return;
    singleGrid.innerHTML = '';
    dualGrid.innerHTML = '';

    const isl = (practiceTemplates && practiceTemplates.isl) ? practiceTemplates.isl : {};
    const singleKeys = ['1', '2', '3', '4', '5', '6', '7', '8', '9', 'C', 'I', 'L', 'O', 'U', 'V'];
    const dualKeys = ['A', 'B', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'W', 'X', 'Y', 'Z'];

    singleKeys.forEach(k => {
      const desc = (isl[k] && isl[k].desc) ? isl[k].desc : (islDescFallback[k] || 'Single hand sign');
      singleGrid.appendChild(createGuideCard(k, false, desc));
    });

    dualKeys.forEach(k => {
      const desc = (isl[k] && isl[k].desc) ? isl[k].desc : (islDescFallback[k] || 'Dual hand sign');
      dualGrid.appendChild(createGuideCard(k, true, desc));
    });
  }

  const aslDescFallback = {
    'A': 'Fist with thumb on side', 'B': 'All fingers together pointing up',
    'C': 'Curved hand in C-shape', 'D': 'Index up, fingers curve to thumb',
    'E': 'Fingers bent, thumb tucked', 'F': 'Index-thumb circle, three fingers up',
    'G': 'Index and thumb point sideways', 'H': 'Two fingers point sideways',
    'I': 'Pinky finger up', 'J': 'Pinky draws J in air',
    'K': 'Index up, middle angled, thumb out', 'L': 'L-shape — thumb and index',
    'M': 'Three fingers over thumb', 'N': 'Two fingers over thumb',
    'O': 'All fingers curve to form O', 'P': 'K shape pointing down',
    'Q': 'G shape pointing down', 'R': 'Two fingers crossed',
    'S': 'Fist with thumb over fingers', 'T': 'Thumb between index and middle',
    'U': 'Two fingers together pointing up', 'V': 'Two fingers spread in V',
    'W': 'Three fingers spread out', 'X': 'Index finger hooks',
    'Y': 'Thumb and pinky out', 'Z': 'Index draws Z in air'
  };

  function renderAslCards() {
    const aslGrid = document.getElementById('asl-cards');
    if (!aslGrid || aslGrid.children.length > 0) return; // render once
    aslGrid.innerHTML = '';
    const aslLetters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
    aslLetters.forEach(letter => {
      const desc = aslDescFallback[letter] || 'ASL hand sign';
      const el = document.createElement('div');
      el.className = 'guide-card';
      el.title = `${letter}: ${desc}`;
      el.innerHTML = `
        <div class="guide-card-img-wrap">
          <img src="asl_signs/${letter}.png" alt="ASL Sign ${letter}" loading="lazy" />
        </div>
        <div class="guide-card-body">
          <span class="guide-card-char">${letter}</span>
          <span class="guide-card-hands pill-amber pill-badge">1 Hand</span>
        </div>
        <p class="guide-card-desc" style="padding: 6px 10px 10px; font-size: 0.72rem; color: var(--text-dim); line-height: 1.35; margin: 0;">${desc}</p>
      `;
      aslGrid.appendChild(el);
    });
  }

  function openGuide() {
    renderGuideCards();
    renderAslCards();
    if (panelGuide) {
      try {
        if (panelGuide.showPopover) {
          panelGuide.showPopover();
        } else {
          panelGuide.style.display = 'block';
          panelGuide.classList.add('open');
        }
      } catch (err) {
        panelGuide.style.display = 'block';
        panelGuide.classList.add('open');
      }
    }
  }

  function closeGuide() {
    if (panelGuide) {
      try {
        if (panelGuide.hidePopover) {
          panelGuide.hidePopover();
        } else {
          panelGuide.style.display = 'none';
          panelGuide.classList.remove('open');
        }
      } catch (err) {
        panelGuide.style.display = 'none';
        panelGuide.classList.remove('open');
      }
    }
  }

  if (btnOpenGuide) {
    btnOpenGuide.addEventListener('click', (e) => {
      e.preventDefault();
      openGuide();
    });
  }

  if (btnCloseGuide) {
    btnCloseGuide.addEventListener('click', (e) => {
      e.preventDefault();
      closeGuide();
    });
  }

  if (tabIsl && tabAsl && contentIsl && contentAsl) {
    tabIsl.addEventListener('click', () => {
      tabIsl.classList.add('active');
      tabAsl.classList.remove('active');
      tabIsl.setAttribute('aria-selected', 'true');
      tabAsl.setAttribute('aria-selected', 'false');
      contentIsl.classList.remove('hidden');
      contentAsl.classList.add('hidden');
    });

    tabAsl.addEventListener('click', () => {
      tabAsl.classList.add('active');
      tabIsl.classList.remove('active');
      tabAsl.setAttribute('aria-selected', 'true');
      tabIsl.setAttribute('aria-selected', 'false');
      contentAsl.classList.remove('hidden');
      contentIsl.classList.add('hidden');
    });
  }

  window._renderGuideCards = renderGuideCards;
  window._openSignGuide = openGuide;
  renderGuideCards();
}

// ── Startup ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  displayCanvas.width  = 640;
  displayCanvas.height = 480;

  // Wire controls FIRST: a camera failure must not leave every button dead.
  setupControls();
  initPracticeStudio();
  initSignGuide();

  setStatus('loading', 'Loading…');
  try {
    // 1. Probe for Flask backend (async, result used by classify())
    probeServer();

    // 2. Always load JS MLP weights for offline fallback
    await loadLandmarkModel();

    // 3. Start camera
    await initMediaPipe();
  } catch (err) {
    showCameraError(err);
  }
});

// ══════════════════════════════════════════════════════════════════════════════
// VISUALLY (PRACTICE STUDIO) IMPLEMENTATION
// ══════════════════════════════════════════════════════════════════════════════

let practiceTemplates = { asl: {}, isl: {} };
let practiceDialect = 'asl'; // 'asl' | 'isl'
let practiceSpeed = 1.0; // 0.5 | 1.0 | 1.5

// Sequence and animation state
let practiceSequence = []; // e.g. ['H', 'I', ' ', 'B', 'A', 'B', 'Y']
let practiceCurrentIndex = 0;
let practiceIsPlaying = false;
let practiceStepStartTime = 0;
let practiceAnimFrameId = null;

// Poses for interpolation: arrays of 21 landmarks [x, y, z]
let practiceFromPose = null;
let practiceToPose = null;
let practiceCurrentPose = null;

// Default neutral resting pose (21 landmarks)
const NEUTRAL_REST_POSE = Array(21).fill(0).map((_, i) => {
  if (i === 0) return [0, 0, 0];
  const fingerIdx = Math.floor((i - 1) / 4);
  const jointIdx = ((i - 1) % 4) + 1;
  const xOffset = (fingerIdx - 2) * 0.18;
  const yOffset = -0.35 * jointIdx;
  return [xOffset, yOffset, 0];
});

function clone21(pts) {
  if (!pts || !Array.isArray(pts)) return null;
  return pts.map(pt => [pt[0], pt[1], pt[2] || 0]);
}

function normalizePose(raw) {
  if (!raw) {
    return {
      twoHanded: false,
      left: null,
      right: clone21(NEUTRAL_REST_POSE),
      wristL: [0.0, 0.0],
      wristR: [0.0, 0.0],
      desc: ''
    };
  }
  if (Array.isArray(raw)) {
    return {
      twoHanded: false,
      left: null,
      right: clone21(raw),
      wristL: [0.0, 0.0],
      wristR: [0.0, 0.0],
      desc: ''
    };
  }
  return {
    twoHanded: !!raw.twoHanded,
    left: raw.left ? clone21(raw.left) : null,
    right: raw.right ? clone21(raw.right) : clone21(NEUTRAL_REST_POSE),
    wristL: raw.wristL ? [raw.wristL[0], raw.wristL[1]] : [0.0, 0.0],
    wristR: raw.wristR ? [raw.wristR[0], raw.wristR[1]] : [0.0, 0.0],
    desc: raw.desc || ''
  };
}

function clonePose(pose) {
  const norm = normalizePose(pose);
  return {
    twoHanded: norm.twoHanded,
    left: clone21(norm.left),
    right: clone21(norm.right),
    wristL: [...norm.wristL],
    wristR: [...norm.wristR],
    desc: norm.desc
  };
}

function initPracticeStudio() {
  const canvas = document.getElementById('avatar-canvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  // 1. Fetch templates
  fetch('/sign_templates.json?v=' + Date.now(), { cache: 'no-store' })
    .then(res => {
      if (!res.ok) throw new Error('HTTP ' + res.status);
      return res.json();
    })
    .then(data => {
      practiceTemplates = data;
      console.log('[PracticeStudio] Loaded sign templates:', Object.keys(data.asl || {}).length, 'ASL,', Object.keys(data.isl || {}).length, 'ISL');
      // Set initial pose
      practiceCurrentPose = clonePose(getTemplatePose('REST'));
      drawAvatar();
      if (window._renderGuideCards) window._renderGuideCards();
    })
    .catch(err => {
      console.warn('[PracticeStudio] Failed to load sign_templates.json:', err);
    });

  // 2. DOM elements
  const txtInput = document.getElementById('practice-input');
  const charCount = document.getElementById('practice-char-count');
  const btnMic = document.getElementById('btn-mic');
  const micLabel = document.getElementById('mic-label');
  const btnSpeakInput = document.getElementById('btn-speak-input');
  const btnClearPractice = document.getElementById('btn-clear-practice');
  const btnStartSign = document.getElementById('btn-start-sign');
  const sequenceChipsContainer = document.getElementById('sequence-chips');
  const seqCounter = document.getElementById('seq-counter');

  const btnPracAsl = document.getElementById('btn-practice-asl');
  const btnPracIsl = document.getElementById('btn-practice-isl');

  const hudChar = document.getElementById('hud-letter-char');
  const hudSub = document.getElementById('hud-letter-sub');
  const hudCue = document.getElementById('hud-letter-cue');
  const hudStatus = document.getElementById('avatar-hud-status');
  const avatarStatusText = document.getElementById('avatar-status-text');

  const btnPrev = document.getElementById('btn-avatar-prev');
  const btnPlayPause = document.getElementById('btn-avatar-playpause');
  const btnNext = document.getElementById('btn-avatar-next');
  const btnReplay = document.getElementById('btn-avatar-replay');
  const scrubber = document.getElementById('avatar-scrubber');

  // Resize canvas handler
  function resizeCanvas() {
    if (!canvas.parentElement) return;
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.floor(rect.width);
    const h = Math.floor(rect.height);
    if (w > 0 && h > 0 && (canvas.width !== w * dpr || canvas.height !== h * dpr)) {
      canvas.width = w * dpr;
      canvas.height = h * dpr;
      drawAvatar();
    }
  }
  window.addEventListener('resize', resizeCanvas);
  setTimeout(resizeCanvas, 100);

  // Get template for a character
  function getTemplatePose(char) {
    const dict = practiceTemplates[practiceDialect] || practiceTemplates.asl || {};
    const upper = String(char).toUpperCase();
    if (dict[upper]) return normalizePose(dict[upper]);
    if (dict['REST']) return normalizePose(dict['REST']);
    return normalizePose(null);
  }

  const ASL_SAMPLES = ['HI', 'HELLO', 'THANK YOU', 'HOW ARE YOU', 'BABY'];
  const ISL_SAMPLES  = ['NAMASTE', 'A B C', '1 2 3', 'ONAM', 'INDIA'];

  function updateSampleChips(samples) {
    const chipsContainer = document.querySelector('.quick-samples');
    if (!chipsContainer) return;
    const chips = chipsContainer.querySelectorAll('.sample-chip');
    chips.forEach((chip, idx) => {
      if (samples[idx] !== undefined) {
        chip.dataset.text = samples[idx];
        chip.textContent  = samples[idx];
        chip.style.display = '';
      } else {
        chip.style.display = 'none';
      }
    });
  }

  // Dialect switch inside Practice
  if (btnPracAsl && btnPracIsl) {
    btnPracAsl.addEventListener('click', () => {
      practiceDialect = 'asl';
      btnPracAsl.classList.add('active');
      btnPracIsl.classList.remove('active');
      updateSampleChips(ASL_SAMPLES);
      rebuildSequence(false);
      showToast('Practice dialect: ASL');
    });
    btnPracIsl.addEventListener('click', () => {
      practiceDialect = 'isl';
      btnPracIsl.classList.add('active');
      btnPracAsl.classList.remove('active');
      updateSampleChips(ISL_SAMPLES);
      rebuildSequence(false);
      showToast('Practice dialect: ISL');
    });
  }

  // Speed controls
  document.querySelectorAll('.speed-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      document.querySelectorAll('.speed-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');
      practiceSpeed = parseFloat(btn.dataset.speed) || 1.0;
    });
  });

  // Quick sample chips
  document.querySelectorAll('.sample-chip').forEach(chip => {
    chip.addEventListener('click', () => {
      const text = chip.dataset.text || '';
      if (txtInput) {
        txtInput.value = text;
        updateCharCount();
      }
      rebuildSequence(true);
    });
  });

  function updateCharCount() {
    if (!txtInput || !charCount) return;
    charCount.textContent = `${txtInput.value.length}/200`;
  }

  if (txtInput) {
    txtInput.addEventListener('input', () => {
      updateCharCount();
      rebuildSequence(false);
    });
    txtInput.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        rebuildSequence(true);
      }
    });
  }

  if (btnClearPractice) {
    btnClearPractice.addEventListener('click', () => {
      if (txtInput) txtInput.value = '';
      updateCharCount();
      practiceSequence = [];
      practiceCurrentIndex = 0;
      pausePlayback();
      renderSequenceChips();
      updateHUD();
      practiceCurrentPose = clonePose(getTemplatePose('REST'));
      drawAvatar();
    });
  }

  // Voice Speech-to-Text
  if (btnMic) {
    let recognition = null;
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

    if (SpeechRecognition) {
      recognition = new SpeechRecognition();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = 'en-US';

      recognition.onstart = () => {
        btnMic.classList.add('listening');
        if (micLabel) micLabel.textContent = 'Listening…';
        showToast('Speak now — listening…');
      };

      recognition.onresult = (event) => {
        const transcript = event.results[0][0].transcript;
        if (txtInput) {
          txtInput.value = transcript.toUpperCase();
          updateCharCount();
          rebuildSequence(true);
        }
      };

      recognition.onerror = (event) => {
        console.warn('[PracticeStudio] Speech recognition error:', event.error);
        btnMic.classList.remove('listening');
        if (micLabel) micLabel.textContent = 'Speak';
        showToast('Voice error: ' + event.error);
      };

      recognition.onend = () => {
        btnMic.classList.remove('listening');
        if (micLabel) micLabel.textContent = 'Speak';
      };

      btnMic.addEventListener('click', () => {
        try {
          if (btnMic.classList.contains('listening')) {
            recognition.stop();
          } else {
            recognition.start();
          }
        } catch (err) {
          console.warn('[PracticeStudio] Speech start exception:', err);
        }
      });
    } else {
      btnMic.addEventListener('click', () => {
        showToast('Speech recognition not supported in this browser. Please type.');
      });
    }
  }

  // TTS speak aloud
  if (btnSpeakInput) {
    btnSpeakInput.addEventListener('click', () => {
      const text = (txtInput ? txtInput.value.trim() : '');
      if (!text) {
        showToast('Type some words to speak aloud.');
        return;
      }
      if ('speechSynthesis' in window) {
        window.speechSynthesis.cancel();
        const utter = new SpeechSynthesisUtterance(text);
        utter.rate = 0.95;
        window.speechSynthesis.speak(utter);
      }
    });
  }

  // Start Demonstration Button
  if (btnStartSign) {
    btnStartSign.addEventListener('click', () => {
      rebuildSequence(true);
    });
  }

  // Parse text into sequence of signs
  function rebuildSequence(autoPlay = false) {
    const raw = (txtInput ? txtInput.value : '').toUpperCase();
    const cleanChars = [];
    for (let i = 0; i < raw.length; i++) {
      const ch = raw[i];
      if ((ch >= 'A' && ch <= 'Z') || (ch >= '0' && ch <= '9')) {
        cleanChars.push(ch);
      } else if (ch === ' ' && cleanChars.length > 0 && cleanChars[cleanChars.length - 1] !== ' ') {
        cleanChars.push(' ');
      }
    }
    // Remove trailing space
    if (cleanChars.length > 0 && cleanChars[cleanChars.length - 1] === ' ') {
      cleanChars.pop();
    }

    practiceSequence = cleanChars;
    practiceCurrentIndex = 0;
    renderSequenceChips();
    updateHUD();

    if (practiceSequence.length === 0) {
      practiceCurrentPose = clonePose(getTemplatePose('REST'));
      drawAvatar();
      return;
    }

    if (autoPlay) {
      startPlayback();
    } else {
      jumpToStep(0);
    }
  }

  function renderSequenceChips() {
    if (!sequenceChipsContainer) return;
    sequenceChipsContainer.innerHTML = '';

    if (practiceSequence.length === 0) {
      sequenceChipsContainer.innerHTML = '<span class="ph">Enter text above to preview gesture timeline…</span>';
      if (seqCounter) seqCounter.textContent = '0 / 0';
      if (scrubber) scrubber.value = 0;
      return;
    }

    if (seqCounter) {
      seqCounter.textContent = `${practiceCurrentIndex + 1} / ${practiceSequence.length}`;
    }

    practiceSequence.forEach((char, idx) => {
      const chip = document.createElement('button');
      chip.className = 'sequence-chip' + (char === ' ' ? ' is-space' : '') + (idx === practiceCurrentIndex ? ' active' : '');
      chip.textContent = (char === ' ' ? '␣' : char);
      chip.title = (char === ' ' ? 'Space (Rest pause)' : `Letter ${char}`);
      chip.addEventListener('click', () => {
        jumpToStep(idx);
      });
      sequenceChipsContainer.appendChild(chip);
    });

    if (scrubber) {
      const pct = (practiceSequence.length > 1) ? (practiceCurrentIndex / (practiceSequence.length - 1)) * 100 : 0;
      scrubber.value = pct;
    }
  }

  function updateHUD() {
    if (practiceSequence.length === 0) {
      if (hudChar) hudChar.textContent = '—';
      if (hudSub) hudSub.textContent = 'Waiting for input';
      if (hudCue) { hudCue.style.display = 'none'; hudCue.textContent = ''; }
      if (avatarStatusText) avatarStatusText.textContent = 'Ready';
      if (hudStatus) hudStatus.classList.remove('playing');
      return;
    }

    const currChar = practiceSequence[practiceCurrentIndex];
    if (hudChar) {
      hudChar.textContent = (currChar === ' ' ? '␣' : currChar);
    }

    const currPose = getTemplatePose(currChar === ' ' ? 'REST' : currChar);
    const is2H = currPose.twoHanded;
    const desc = currPose.desc;

    if (hudSub) {
      if (currChar === ' ') {
        hudSub.textContent = `Word Boundary Pause (${practiceCurrentIndex + 1}/${practiceSequence.length})`;
      } else {
        const handTag = (practiceDialect === 'isl') ? (is2H ? ' • Dual Hand (Two-Handed)' : ' • Single Hand') : ' • Single Hand';
        hudSub.textContent = `${practiceDialect.toUpperCase()} Sign • Step ${practiceCurrentIndex + 1} of ${practiceSequence.length}${handTag}`;
      }
    }

    if (hudCue) {
      if (desc && currChar !== ' ') {
        hudCue.style.display = 'inline-flex';
        hudCue.textContent = `${is2H ? '👐 Two-Handed: ' : '✋ '}${desc}`;
      } else {
        hudCue.style.display = 'none';
        hudCue.textContent = '';
      }
    }

    if (avatarStatusText) {
      avatarStatusText.textContent = practiceIsPlaying ? 'Demonstrating…' : 'Paused';
    }
    if (hudStatus) {
      if (practiceIsPlaying) hudStatus.classList.add('playing');
      else hudStatus.classList.remove('playing');
    }

    // Real human photo reference card preview (both ASL & ISL)
    const photoRef = document.getElementById('avatar-photo-ref');
    const photoImg = document.getElementById('avatar-photo-img');
    const photoHeader = document.getElementById('avatar-photo-header');
    const photoType = document.getElementById('avatar-photo-type');
    if (photoRef && photoImg) {
      if (currChar && currChar !== ' ') {
        const isISL = practiceDialect === 'isl';
        photoImg.src = isISL ? `isl_signs/${currChar}.png` : `asl_signs/${currChar}.png`;
        if (photoHeader) photoHeader.textContent = `${practiceDialect.toUpperCase()}: ${currChar}`;
        if (photoType) photoType.textContent = isISL ? (is2H ? '2-Hands' : '1-Hand') : '1-Hand';
        photoRef.style.display = 'flex';
      } else {
        photoRef.style.display = 'none';
      }
    }
  }

  function jumpToStep(idx) {
    if (practiceSequence.length === 0) return;
    practiceCurrentIndex = Math.max(0, Math.min(idx, practiceSequence.length - 1));
    const char = practiceSequence[practiceCurrentIndex];
    practiceFromPose = practiceCurrentPose ? clonePose(practiceCurrentPose) : clonePose(getTemplatePose('REST'));
    practiceToPose = clonePose(getTemplatePose(char === ' ' ? 'REST' : char));
    practiceStepStartTime = performance.now();
    renderSequenceChips();
    updateHUD();

    // Scroll active chip into view
    const activeChip = sequenceChipsContainer ? sequenceChipsContainer.children[practiceCurrentIndex] : null;
    if (activeChip && activeChip.scrollIntoView) {
      activeChip.scrollIntoView({ behavior: 'smooth', block: 'nearest', inline: 'center' });
    }
  }

  function startPlayback() {
    if (practiceSequence.length === 0) return;
    practiceIsPlaying = true;
    if (btnPlayPause) btnPlayPause.textContent = '⏸';
    jumpToStep(practiceCurrentIndex);
    updateHUD();
  }

  function pausePlayback() {
    practiceIsPlaying = false;
    if (btnPlayPause) btnPlayPause.textContent = '▶';
    updateHUD();
  }

  // Playback control buttons
  if (btnPlayPause) {
    btnPlayPause.addEventListener('click', () => {
      if (practiceIsPlaying) {
        pausePlayback();
      } else {
        if (practiceCurrentIndex >= practiceSequence.length - 1) {
          practiceCurrentIndex = 0;
        }
        startPlayback();
      }
    });
  }

  if (btnPrev) {
    btnPrev.addEventListener('click', () => {
      pausePlayback();
      jumpToStep(practiceCurrentIndex - 1);
    });
  }

  if (btnNext) {
    btnNext.addEventListener('click', () => {
      pausePlayback();
      jumpToStep(practiceCurrentIndex + 1);
    });
  }

  if (btnReplay) {
    btnReplay.addEventListener('click', () => {
      practiceCurrentIndex = 0;
      startPlayback();
    });
  }

  if (scrubber) {
    scrubber.addEventListener('input', () => {
      if (practiceSequence.length === 0) return;
      pausePlayback();
      const val = parseFloat(scrubber.value);
      const targetIdx = Math.round((val / 100) * (practiceSequence.length - 1));
      jumpToStep(targetIdx);
    });
  }

  // Interpolation & Render Loop
  function lerp(a, b, t) {
    return a + (b - a) * t;
  }

  function easeInOutCubic(x) {
    return x < 0.5 ? 4 * x * x * x : 1 - Math.pow(-2 * x + 2, 3) / 2;
  }

  function avatarAnimationLoop(now) {
    practiceAnimFrameId = requestAnimationFrame(avatarAnimationLoop);

    // Calculate step timing based on speed
    const transitionMs = 450 / practiceSpeed;
    const holdMs = 700 / practiceSpeed;
    const totalStepMs = transitionMs + holdMs;

    if (practiceToPose && practiceFromPose) {
      const elapsed = now - practiceStepStartTime;
      const progress = Math.min(1.0, elapsed / transitionMs);
      const t = easeInOutCubic(progress);

      if (!practiceCurrentPose) {
        practiceCurrentPose = clonePose(practiceFromPose);
      }

      // Interpolate right hand
      const fromR = practiceFromPose.right || NEUTRAL_REST_POSE;
      const toR = practiceToPose.right || NEUTRAL_REST_POSE;
      if (!practiceCurrentPose.right) practiceCurrentPose.right = clone21(fromR);
      for (let i = 0; i < 21; i++) {
        practiceCurrentPose.right[i][0] = lerp(fromR[i][0], toR[i][0], t);
        practiceCurrentPose.right[i][1] = lerp(fromR[i][1], toR[i][1], t);
        practiceCurrentPose.right[i][2] = lerp(fromR[i][2], toR[i][2], t);
      }

      // Interpolate left hand
      const fromL = practiceFromPose.left;
      const toL = practiceToPose.left;
      if (fromL || toL) {
        const effFromL = fromL || NEUTRAL_REST_POSE;
        const effToL = toL || NEUTRAL_REST_POSE;
        if (!practiceCurrentPose.left) practiceCurrentPose.left = clone21(effFromL);
        for (let i = 0; i < 21; i++) {
          practiceCurrentPose.left[i][0] = lerp(effFromL[i][0], effToL[i][0], t);
          practiceCurrentPose.left[i][1] = lerp(effFromL[i][1], effToL[i][1], t);
          practiceCurrentPose.left[i][2] = lerp(effFromL[i][2], effToL[i][2], t);
        }
        practiceCurrentPose.twoHanded = (t > 0.5) ? practiceToPose.twoHanded : practiceFromPose.twoHanded;
      } else {
        practiceCurrentPose.left = null;
        practiceCurrentPose.twoHanded = false;
      }

      // Interpolate wrist offsets
      practiceCurrentPose.wristL = [
        lerp(practiceFromPose.wristL ? practiceFromPose.wristL[0] : 0, practiceToPose.wristL ? practiceToPose.wristL[0] : 0, t),
        lerp(practiceFromPose.wristL ? practiceFromPose.wristL[1] : 0, practiceToPose.wristL ? practiceToPose.wristL[1] : 0, t)
      ];
      practiceCurrentPose.wristR = [
        lerp(practiceFromPose.wristR ? practiceFromPose.wristR[0] : 0, practiceToPose.wristR ? practiceToPose.wristR[0] : 0, t),
        lerp(practiceFromPose.wristR ? practiceFromPose.wristR[1] : 0, practiceToPose.wristR ? practiceToPose.wristR[1] : 0, t)
      ];
      practiceCurrentPose.desc = practiceToPose.desc;

      if (practiceIsPlaying && elapsed >= totalStepMs) {
        if (practiceCurrentIndex < practiceSequence.length - 1) {
          jumpToStep(practiceCurrentIndex + 1);
        } else {
          // Completed full sequence
          pausePlayback();
          if (avatarStatusText) avatarStatusText.textContent = 'Complete';
        }
      }
    }

    drawAvatar();
  }

  // Draw 21-landmark hand skeleton with 5 distinct finger colors, palm metacarpals, and pearl beads
  function drawHandSkeleton(ctx, pose21, wristX, wristY, dpr, scale, isLeft) {
    if (!pose21 || pose21.length !== 21) return;

    // Project landmarks to screen coordinates
    const pts = pose21.map(pt => {
      const lx = pt[0];
      const ly = pt[1];
      const lz = pt[2] || 0;
      const depth = 1.0 / (1.0 - lz * 0.18);
      return [
        wristX + lx * scale * depth,
        wristY + ly * scale * depth,
        lz
      ];
    });

    // Palm Metacarpal Structure (cool titanium slate connecting wrist to the 4 knuckles)
    const palmCol = 'rgba(145, 160, 175, 0.85)';
    ctx.strokeStyle = palmCol;
    ctx.lineWidth = 3.5 * dpr;

    // Metacarpal rays: wrist to MCPs (5, 9, 13, 17)
    [5, 9, 13, 17].forEach(mcp => {
      ctx.beginPath();
      ctx.moveTo(wristX, wristY);
      ctx.lineTo(pts[mcp][0], pts[mcp][1]);
      ctx.stroke();
    });

    // Knuckle transverse arch: connecting MCP 5 -> 9 -> 13 -> 17
    ctx.lineWidth = 4.5 * dpr;
    ctx.beginPath();
    ctx.moveTo(pts[5][0], pts[5][1]);
    ctx.lineTo(pts[9][0], pts[9][1]);
    ctx.lineTo(pts[13][0], pts[13][1]);
    ctx.lineTo(pts[17][0], pts[17][1]);
    ctx.stroke();

    // Finger Segments with exact reference colors:
    // Thumb: Coral / Red-Pink (wrist 0 -> 1 -> 2 -> 3 -> 4)
    // Index: Deep Electric Blue (5 -> 6 -> 7 -> 8)
    // Middle: Bright Emerald Green (9 -> 10 -> 11 -> 12)
    // Ring: Vibrant Cyan (13 -> 14 -> 15 -> 16)
    // Pinky: Warm Orange (17 -> 18 -> 19 -> 20)
    const fingerGroups = [
      { indices: [0, 1, 2, 3, 4],    color: '#ff4d4d', width: 6.0 * dpr }, // Thumb
      { indices: [5, 6, 7, 8],        color: '#2979ff', width: 6.0 * dpr }, // Index
      { indices: [9, 10, 11, 12],     color: '#00e676', width: 6.0 * dpr }, // Middle
      { indices: [13, 14, 15, 16],    color: '#00e5ff', width: 6.0 * dpr }, // Ring
      { indices: [17, 18, 19, 20],    color: '#ff9800', width: 6.0 * dpr }, // Pinky
    ];

    fingerGroups.forEach(group => {
      ctx.strokeStyle = group.color;
      ctx.lineWidth = group.width;
      for (let k = 0; k < group.indices.length - 1; k++) {
        const i1 = group.indices[k];
        const i2 = group.indices[k + 1];
        ctx.beginPath();
        ctx.moveTo(pts[i1][0], pts[i1][1]);
        ctx.lineTo(pts[i2][0], pts[i2][1]);
        ctx.stroke();
      }
    });

    // Joint caps: Clean pearl white dots
    ctx.fillStyle = '#ffffff';
    pts.forEach((pt, idx) => {
      if (idx === 0) return; // Skip wrist, drawn specially below
      const radius = (idx % 4 === 0) ? 2.5 * dpr : 2.0 * dpr;
      ctx.beginPath();
      ctx.arc(pt[0], pt[1], radius, 0, Math.PI * 2);
      ctx.fill();
    });

    // Wrist Joint: Clean pearl white bead
    ctx.fillStyle = '#ffffff';
    ctx.beginPath();
    ctx.arc(wristX, wristY, 5 * dpr, 0, Math.PI * 2);
    ctx.fill();
  }

  // Draw Avatar Canvas
  function drawAvatar() {
    if (!canvas || !ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const w = canvas.width;
    const h = canvas.height;

    ctx.save();
    ctx.clearRect(0, 0, w, h);

    // 1. Studio background (theme-aware)
    const isLight = document.documentElement.getAttribute('data-theme') === 'light';
    ctx.fillStyle = isLight ? '#eaedf4' : '#161619';
    ctx.fillRect(0, 0, w, h);

    // 2. Stylized Red Silhouette Avatar (Head, Face, Torso, Arms - matching RyloTranslate)
    const red = '#d82222';
    ctx.strokeStyle = red;
    ctx.fillStyle = red;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.shadowBlur = 0;

    // Shoulder & torso anchor points
    const shoulderY = h * 0.48;
    const leftShoulderX = w * 0.33;
    const rightShoulderX = w * 0.71;

    // Head dimensions
    const headCx = w * 0.51;
    const headCy = h * 0.25;
    const headRx = w * 0.082;
    const headRy = h * 0.125;

    // Head contour
    ctx.lineWidth = 5 * dpr;
    ctx.beginPath();
    ctx.ellipse(headCx, headCy, headRx, headRy, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Eyebrows (bold expressive arches)
    ctx.lineWidth = 7 * dpr;
    ctx.beginPath();
    ctx.arc(headCx - headRx * 0.44, headCy - headRy * 0.35, headRx * 0.34, Math.PI * 1.1, Math.PI * 1.9, false);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(headCx + headRx * 0.44, headCy - headRy * 0.35, headRx * 0.34, Math.PI * 1.1, Math.PI * 1.9, false);
    ctx.stroke();

    // Eyes (almond shaped contours)
    ctx.lineWidth = 3.5 * dpr;
    ctx.beginPath();
    ctx.ellipse(headCx - headRx * 0.42, headCy - headRy * 0.12, headRx * 0.24, headRy * 0.12, 0, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.ellipse(headCx + headRx * 0.42, headCy - headRy * 0.12, headRx * 0.24, headRy * 0.12, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Mouth (expressive open oval)
    ctx.lineWidth = 4.5 * dpr;
    ctx.beginPath();
    ctx.ellipse(headCx, headCy + headRy * 0.38, headRx * 0.32, headRy * 0.16, 0, 0, Math.PI * 2);
    ctx.stroke();

    // Torso Frame:
    ctx.lineWidth = 5 * dpr;

    // Clavicle / shoulder horizontal bar
    ctx.beginPath();
    ctx.moveTo(leftShoulderX, shoulderY);
    ctx.lineTo(rightShoulderX, shoulderY);
    ctx.stroke();

    // Torso body lines (tapering downwards)
    ctx.beginPath();
    ctx.moveTo(leftShoulderX, shoulderY);
    ctx.lineTo(w * 0.39, h * 0.98);
    ctx.moveTo(rightShoulderX, shoulderY);
    ctx.lineTo(w * 0.65, h * 0.98);
    ctx.stroke();

    // Determine dual-hand vs single-hand posture
    const is2H = practiceCurrentPose && practiceCurrentPose.twoHanded && practiceCurrentPose.left;
    const handScale = Math.min(w * 0.20, h * 0.20);

    if (is2H) {
      // TWO-HANDED MODE: Both arms raised into signing space
      const offLx = (practiceCurrentPose.wristL ? practiceCurrentPose.wristL[0] : 0) * handScale;
      const offLy = (practiceCurrentPose.wristL ? practiceCurrentPose.wristL[1] : 0) * handScale;
      const offRx = (practiceCurrentPose.wristR ? practiceCurrentPose.wristR[0] : 0) * handScale;
      const offRy = (practiceCurrentPose.wristR ? practiceCurrentPose.wristR[1] : 0) * handScale;

      const wristL_X = w * 0.37 + offLx;
      const wristL_Y = h * 0.65 + offLy;
      const wristR_X = w * 0.63 + offRx;
      const wristR_Y = h * 0.65 + offRy;

      const elbowL_X = leftShoulderX - w * 0.07;
      const elbowL_Y = (shoulderY + wristL_Y) / 2 + h * 0.08;
      const elbowR_X = rightShoulderX + w * 0.07;
      const elbowR_Y = (shoulderY + wristR_Y) / 2 + h * 0.08;

      // Left Arm (viewer-left): Shoulder -> Elbow -> Wrist
      ctx.beginPath();
      ctx.moveTo(leftShoulderX, shoulderY);
      ctx.lineTo(elbowL_X, elbowL_Y);
      ctx.lineTo(wristL_X, wristL_Y);
      ctx.stroke();

      // Right Arm (viewer-right): Shoulder -> Elbow -> Wrist
      ctx.beginPath();
      ctx.moveTo(rightShoulderX, shoulderY);
      ctx.lineTo(elbowR_X, elbowR_Y);
      ctx.lineTo(wristR_X, wristR_Y);
      ctx.stroke();

      // Render both hand skeletons
      drawHandSkeleton(ctx, practiceCurrentPose.left, wristL_X, wristL_Y, dpr, handScale, true);
      drawHandSkeleton(ctx, practiceCurrentPose.right, wristR_X, wristR_Y, dpr, handScale, false);
    } else {
      // SINGLE-HANDED MODE: Dominant signing arm active, other arm resting naturally
      const wristX = w * 0.34;
      const wristY = h * 0.65;
      const elbowL_X = w * 0.24;
      const elbowL_Y = h * 0.84;

      // Signing Arm (viewer-left): Shoulder -> Elbow -> Wrist
      ctx.beginPath();
      ctx.moveTo(leftShoulderX, shoulderY);
      ctx.lineTo(elbowL_X, elbowL_Y);
      ctx.lineTo(wristX, wristY);
      ctx.stroke();

      // Resting Arm (viewer-right): Shoulder -> Elbow -> Forearm down
      const elbowR_X = w * 0.76;
      const elbowR_Y = h * 0.82;
      ctx.beginPath();
      ctx.moveTo(rightShoulderX, shoulderY);
      ctx.lineTo(elbowR_X, elbowR_Y);
      ctx.lineTo(w * 0.68, h * 0.98);
      ctx.stroke();

      // Render single hand skeleton
      const singlePose = practiceCurrentPose ? (practiceCurrentPose.right || practiceCurrentPose.left) : null;
      if (singlePose) {
        drawHandSkeleton(ctx, singlePose, wristX, wristY, dpr, handScale, false);
      }
    }

    ctx.restore();
  }

  // Start animation loop
  requestAnimationFrame(avatarAnimationLoop);
}

function activatePracticeStudio() {
  const canvas = document.getElementById('avatar-canvas');
  if (canvas && canvas.parentElement) {
    const rect = canvas.parentElement.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    if (rect.width > 0 && rect.height > 0) {
      canvas.width = Math.floor(rect.width) * dpr;
      canvas.height = Math.floor(rect.height) * dpr;
    }
  }
}
