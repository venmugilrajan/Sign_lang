// Run: node --import ./web/avatar/tests/register.mjs --test web/avatar/tests/retarget.test.mjs
import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { Vector3, Quaternion } from 'three';
import { AvatarRig } from '../rig.js';
import { landmarkToWorld, angleBetween, twistAngle, basisQuaternion } from '../retargetUtils.js';
import { FINGER_NAMES, LANDMARK_CHAINS, BONE_NAMES } from '../boneMap.js';
import { buildTestRig } from './testRig.mjs';

const templates = JSON.parse(readFileSync(new URL('../../sign_templates.json', import.meta.url), 'utf8'));
const DEG = 180 / Math.PI;
const FINGER_TOL_DEG = 0.5; // segment direction / bend angle tolerance
const MCP_TOL_DEG = 1.0;

/** Build a rig with a relaxed REST hand like the app does. */
function makeRig(opts = {}) {
  const { root, nodes } = buildTestRig(opts);
  const restHand = templates.asl.REST.right || templates.asl.REST;
  const rig = new AvatarRig(root, { restHand, silent: !!opts.silent });
  return { rig, root, nodes };
}

/** Same normalisation the app applies to a template entry. */
function poseOf(dialect, key) {
  const raw = templates[dialect][key];
  if (Array.isArray(raw)) return { twoHanded: false, left: null, right: raw, wristL: [0, 0], wristR: [0, 0] };
  return raw;
}

function snap(rig, pose) {
  rig.update(pose, 10); // dt large => alpha = 1 => exact targets
  rig.root.updateMatrixWorld(true);
}

const wp = (rig, name) => rig.bones.get(name).getWorldPosition(new Vector3());
const lmw = (lm, i) => landmarkToWorld(new Vector3(), lm[i]);

/** Max |rig - landmark| angle errors for one hand. */
function handErrors(rig, side, lm) {
  const names = BONE_NAMES[side];
  const handPos = wp(rig, names.hand);
  const lmAlong = lmw(lm, 9).sub(lmw(lm, 0));
  const rigAlong = wp(rig, names.fingers.middle[0]).sub(handPos);
  let dirErr = 0, bendErr = 0, mcpErr = 0;
  for (const f of FINGER_NAMES) {
    const c = LANDMARK_CHAINS[f];
    const rigDirs = [];
    const lmDirs = [];
    for (let k = 0; k < 3; k++) {
      rigDirs.push(wp(rig, names.fingers[f][k + 1]).sub(wp(rig, names.fingers[f][k])));
      lmDirs.push(lmw(lm, c[k + 1]).sub(lmw(lm, c[k])));
      dirErr = Math.max(dirErr, angleBetween(rigDirs[k], lmDirs[k]) * DEG);
    }
    for (let k = 0; k < 2; k++) {
      bendErr = Math.max(bendErr, Math.abs(angleBetween(rigDirs[k], rigDirs[k + 1]) - angleBetween(lmDirs[k], lmDirs[k + 1])) * DEG);
    }
    mcpErr = Math.max(mcpErr, Math.abs(angleBetween(rigDirs[0], rigAlong) - angleBetween(lmDirs[0], lmAlong)) * DEG);
  }
  return { dirErr, bendErr, mcpErr };
}

/** Palm-frame normal (along x across) of the rig hand in world space. */
function rigNormal(rig, side) {
  const n = BONE_NAMES[side];
  const along = wp(rig, n.fingers.middle[0]).sub(wp(rig, n.hand));
  const across = wp(rig, n.fingers.index[0]).sub(wp(rig, n.fingers.pinky[0]));
  return along.cross(across).normalize();
}

function report(label, side, e) {
  console.log(`  ${label.padEnd(18)} ${side.padEnd(5)} segment dir max ${e.dirErr.toFixed(3)} deg | bend max ${e.bendErr.toFixed(3)} deg | MCP max ${e.mcpErr.toFixed(3)} deg`);
}

function assertHandMatches(rig, side, lm, label) {
  const e = handErrors(rig, side, lm);
  report(label, side, e);
  assert.ok(e.dirErr < FINGER_TOL_DEG, `${label} ${side}: finger direction error ${e.dirErr.toFixed(3)} deg`);
  assert.ok(e.bendErr < FINGER_TOL_DEG, `${label} ${side}: bend-angle error ${e.bendErr.toFixed(3)} deg`);
  assert.ok(e.mcpErr < MCP_TOL_DEG, `${label} ${side}: MCP angle error ${e.mcpErr.toFixed(3)} deg`);
}

/** Arms must stay outside the torso volume and keep their bone lengths. */
function assertArm(rig, side, label) {
  const s = rig[side];
  const sh = s.upper.getWorldPosition(new Vector3());
  const el = s.fore.getWorldPosition(new Vector3());
  const wr = s.hand.getWorldPosition(new Vector3());
  assert.ok(Math.abs(sh.distanceTo(el) - s.upperLen) < 1e-4, `${label} ${side}: upper arm length changed`);
  assert.ok(Math.abs(el.distanceTo(wr) - s.foreLen) < 1e-4, `${label} ${side}: forearm length changed`);
  const c = rig.layout.centerX;
  const inTorso = (p) => Math.abs(p.x - c) < 0.15 && p.z < 0.08 && p.y < sh.y + 0.02 && p.y > rig.layout.hipsY;
  assert.ok(!inTorso(el), `${label} ${side}: elbow inside torso (${el.toArray().map((v) => v.toFixed(2))})`);
  assert.ok(!inTorso(wr), `${label} ${side}: wrist inside torso (${wr.toArray().map((v) => v.toFixed(2))})`);
  // Side: right arm stays on viewer-left (-x), left arm on viewer-right (+x). No crossing the midline.
  const sign = side === 'right' ? -1 : 1;
  assert.ok((el.x - c) * sign > 0, `${label} ${side}: elbow crossed the midline`);
  assert.ok((wr.x - c) * sign > 0.02, `${label} ${side}: wrist crossed the midline (${(wr.x - c).toFixed(3)})`);
  // Wrist roll about the forearm axis (swing-twist), relative to rest. Pronation/supination can reach ~150 deg in complex signs.
  const delta = s.hand.quaternion.clone().multiply(s.restHandLocalQ.clone().invert());
  const axis = s.hand.position.clone().normalize();
  const twist = twistAngle(delta, axis) * DEG;
  console.log(`  ${label.padEnd(18)} ${side.padEnd(5)} wrist roll ${twist.toFixed(1)} deg | elbow ${el.toArray().map((v) => v.toFixed(2))} wrist ${wr.toArray().map((v) => v.toFixed(2))}`);
  assert.ok(twist < 160, `${label} ${side}: wrist twist ${twist.toFixed(1)} deg too large (unnatural roll)`);
}

/**
 * Rotation-invariant chirality of a hand: z of the thumb base (landmarks 1-2) in the palm frame
 * (X = Y x Z, Y = wrist->middle MCP, Z = along x across). The thumb base sits on the palm side of the
 * frame, and Z points to the back of a right hand but to the palm of a left hand, so the sign differs per hand.
 */
function landmarkThumbZ(lm) {
  const w = lm.map((p) => landmarkToWorld(new Vector3(), p));
  const q = new Quaternion();
  basisQuaternion(w[9].clone().sub(w[0]), w[5].clone().sub(w[17]), q);
  const inv = q.invert();
  return [1, 2].map((i) => w[i].clone().sub(w[0]).applyQuaternion(inv).z);
}

/** The landmark hand assigned to this avatar side must have that side's chirality (right: z < 0, left: z > 0). */
function assertChirality(lm, side, label) {
  const z = landmarkThumbZ(lm);
  const ok = side === 'right' ? z.every((v) => v < 0) : z.every((v) => v > 0);
  console.log(`  ${label.padEnd(18)} ${side.padEnd(5)} thumb palm-frame z ${z.map((v) => v.toFixed(2))} -> ${ok ? 'correct chirality' : 'MIRRORED'}`);
  assert.ok(ok, `${label}: ${side} hand is fed a mirrored hand shape (thumb z ${z.map((v) => v.toFixed(2))})`);
}

// ------------------------------------------------------------------------------------------------

for (const letter of ['A', 'B', 'L', 'Y']) {
  test(`ASL ${letter}: finger bends match landmarks, right hand not mirrored, arm sane`, () => {
    const { rig } = makeRig();
    const pose = poseOf('asl', letter);
    snap(rig, pose);
    assertHandMatches(rig, 'right', pose.right, `ASL ${letter}`);
    assertChirality(pose.right, 'right', `ASL ${letter}`);
    assertArm(rig, 'right', `ASL ${letter}`);
    assertArm(rig, 'left', `ASL ${letter} (idle arm)`);
    // Signing hand sits on viewer-left (avatar's right), thumb toward the body midline for A
    const wr = wp(rig, BONE_NAMES.right.hand);
    assert.ok(wr.x < rig.layout.centerX, 'signing hand must be on the avatar\'s right (viewer-left)');
  });
}

test('ASL A: thumb is on the viewer-right side of the pinky (not mirrored)', () => {
  const { rig } = makeRig();
  snap(rig, poseOf('asl', 'A'));
  const thumb = wp(rig, BONE_NAMES.right.fingers.thumb[3]);
  const pinky = wp(rig, BONE_NAMES.right.fingers.pinky[0]);
  assert.ok(thumb.x > pinky.x, `thumb.x ${thumb.x.toFixed(3)} should be > pinky.x ${pinky.x.toFixed(3)}`);
});

test('ISL two-handed A: both hands match landmarks, correct sides, no crossing', () => {
  const { rig } = makeRig();
  const pose = poseOf('isl', 'A');
  assert.equal(pose.twoHanded, true);
  snap(rig, pose);
  // template `left` = viewer-left hand = avatar RIGHT; template `right` = avatar LEFT
  assertHandMatches(rig, 'right', pose.left, 'ISL A (2H)');
  assertHandMatches(rig, 'left', pose.right, 'ISL A (2H)');
  assertChirality(pose.left, 'right', 'ISL A (2H)');
  assertChirality(pose.right, 'left', 'ISL A (2H)');
  assertArm(rig, 'right', 'ISL A (2H)');
  assertArm(rig, 'left', 'ISL A (2H)');
  const r = wp(rig, BONE_NAMES.right.hand);
  const l = wp(rig, BONE_NAMES.left.hand);
  assert.ok(r.x < l.x, 'avatar right wrist must be on viewer-left of the left wrist');
});

test('every two-handed ISL sign keeps arms outside the torso and wrists on their own side', () => {
  const { rig } = makeRig();
  let consistent = 0;
  let total = 0;
  for (const key of Object.keys(templates.isl).filter((k) => templates.isl[k].twoHanded)) {
    const pose = poseOf('isl', key);
    snap(rig, pose);
    assertArm(rig, 'right', `ISL ${key}`);
    assertArm(rig, 'left', `ISL ${key}`);
    // Dataset-wide chirality audit (informational for signs with ambiguous thumb geometry).
    for (const [lm, side] of [[pose.left, 'right'], [pose.right, 'left']]) {
      const z = landmarkThumbZ(lm);
      total++;
      if (side === 'right' ? z.every((v) => v < 0) : z.every((v) => v > 0)) consistent++;
    }
  }
  console.log(`  two-handed ISL chirality audit: ${consistent}/${total} hands unambiguously match their assigned side`);
  assert.ok(consistent / total > 0.7, 'most two-handed hands must match their assigned avatar side');
});

test('sequence H-E-L-L-O: smooth playback converges to each letter exactly', () => {
  const { rig } = makeRig();
  let maxStep = 0;
  for (const ch of 'HELLO') {
    const pose = poseOf('asl', ch);
    for (let f = 0; f < 90; f++) { // 1.5 s at 60 fps
      const before = rig.right.fingers[4].quaternion.clone();
      rig.update(pose, 1 / 60);
      rig.root.updateMatrixWorld(true);
      maxStep = Math.max(maxStep, before.angleTo(rig.right.fingers[4].quaternion) * DEG);
    }
    assertHandMatches(rig, 'right', pose.right, `ASL ${ch}`);
    assertArm(rig, 'right', `ASL ${ch}`);
  }
  console.log(`  HELLO: max per-frame finger bone rotation ${maxStep.toFixed(2)} deg (no pops)`);
  assert.ok(maxStep < 60, 'bone rotation popped between frames');
});

test('face: neutral fallback, sign-driven blendshapes, blink', () => {
  const { rig, nodes } = makeRig();
  const face = nodes.get('Head').obj.children.find((c) => c.name === 'Wolf3D_Head');
  rig.update(poseOf('asl', 'A'), 10);
  assert.equal(face.morphTargetInfluences[0], 0, 'jaw must be neutral without face data');
  rig.update({ ...poseOf('asl', 'A'), face: { jawOpen: 0.7, browInnerUp: 0.4 } }, 10);
  assert.ok(Math.abs(face.morphTargetInfluences[0] - 0.7) < 1e-6, 'jawOpen driven from sign data');
  assert.ok(Math.abs(face.morphTargetInfluences[3] - 0.4) < 1e-6, 'browInnerUp driven from sign data');
  let blinked = false;
  for (let i = 0; i < 60 * 8 && !blinked; i++) {
    rig.update(poseOf('asl', 'A'), 1 / 60);
    if (face.morphTargetInfluences[1] > 0.5) blinked = true;
  }
  assert.ok(blinked, 'idle blink should occur within 8 s');
});

test('bone-name warnings: missing bones are listed, mixamo prefix resolves', () => {
  const warns = [];
  const orig = console.warn;
  console.warn = (m) => warns.push(String(m));
  try {
    const omit = ['LeftHandPinky3', 'LeftHandPinky4', 'RightHandThumb4'];
    const { root } = buildTestRig({ omit });
    new AvatarRig(root, {});
  } finally {
    console.warn = orig;
  }
  assert.equal(warns.length, 1);
  assert.match(warns[0], /LeftHandPinky3/);
  assert.doesNotMatch(warns[0], /RightHandThumb4/, 'end bones are optional, only an info notice');

  const { root } = buildTestRig({ prefix: 'mixamorig' });
  const rig = new AvatarRig(root, { silent: true });
  assert.equal(rig.missing.length, 0, `mixamorig prefix should resolve; missing: ${rig.missing}`);
});

test('basis frame is chirality-consistent between landmarks and rig', () => {
  const q = new Quaternion();
  assert.ok(basisQuaternion(new Vector3(0, 1, 0), new Vector3(1, 0, 0), q));
  assert.ok(!basisQuaternion(new Vector3(0, 0, 0), new Vector3(1, 0, 0), q));
});
