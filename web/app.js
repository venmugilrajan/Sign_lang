/**
 * SignLens — MediaPipe Landmark Neural Network Edition
 *
 * Runs classification directly in the browser using the weights
 * of our custom MLP Neural Network trained on the 21 hand landmarks.
 */

// ── Config ──────────────────────────────────────────────────────────────────
let   STREAK_NEEDED   = 5;     // frames of same letter to capture (settings sheet, matches native)
const COOLDOWN_MS     = 350;   // ms after capture before detecting again (fluid sign transitions)
let   SPACE_FRAMES    = 60;    // no-hand frames before auto word push (~2.0s at 30fps)
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
        setStatus('ready', 'Connected (server)');
      }
    } else {
      serverAvailable = false;
    }
  } catch {
    serverAvailable = false;
  } finally {
    serverCheckPending = false;
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

  const hands = multiHandLandmarks
    .map((lm, idx) => {
      const handedness = (multiHandedness && multiHandedness[idx] && multiHandedness[idx].label)
        ? multiHandedness[idx].label
        : 'Right';
      return { x: lm[0].x, feat: normalizeLandmarks(lm, w, h, handedness, false) };
    })
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

  const probs = softmax(h);
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

  // Build feature vector (always in JS — fast, correct, no round-trip cost)
  const featureVector = buildFeatureVector(multiHandLandmarks, isDualMode, multiHandedness);

  // ── PATH A: Server-side inference (preferred) ────────────────────────────
  if (serverAvailable) {
    try {
      const response = await fetch('/predict_landmarks', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ mode: currentMode, landmarks: featureVector }),
        signal: AbortSignal.timeout(300),   // 300ms hard timeout per frame
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
    handBadge.textContent = validHands.length > 1 ? '2 HANDS DETECTED' : 'HAND DETECTED';
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

    // Geometric validation to reject facial false positives (e.g. nose / glasses bridge)
    const palmLen = Math.hypot(lms[9].x - lms[0].x, lms[9].y - lms[0].y);
    let minX = 1, maxX = 0, minY = 1, maxY = 0;
    for (let j = 0; j < lms.length; j++) {
      const p = lms[j];
      if (p.x < minX) minX = p.x;
      if (p.x > maxX) maxX = p.x;
      if (p.y < minY) minY = p.y;
      if (p.y > maxY) maxY = p.y;
    }
    const bboxDiag = Math.hypot(maxX - minX, maxY - minY);
    if (palmLen < 0.04 || bboxDiag < 0.08) {
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
    minHandDetectionConfidence: 0.50,
    minHandPresenceConfidence:  0.50,
    minTrackingConfidence:      0.50,
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

  // Keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    // Ignore shortcuts while a control has focus, or a sheet is open.
    if (e.target.closest('input, [popover]')) return;

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
    }
  });
}

// ── Startup ──────────────────────────────────────────────────────────────────
window.addEventListener('DOMContentLoaded', async () => {
  displayCanvas.width  = 640;
  displayCanvas.height = 480;

  // Wire controls FIRST: a camera failure must not leave every button dead.
  setupControls();

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
