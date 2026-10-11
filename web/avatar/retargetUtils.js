/**
 * Pure retargeting math (no DOM, no rendering) so it can be unit-tested in Node.
 *
 * Coordinate conventions
 * ----------------------
 * Landmark space (sign_templates.json / MediaPipe image space):
 *   x -> viewer right, y -> DOWN, z -> away from the viewer (negative = closer).
 *   Templates show the hand as seen by the viewer with the palm facing the viewer.
 * World space (Three.js): x -> viewer right, y -> UP, z -> toward the viewer.
 *   The avatar faces +z, so the avatar's RIGHT side is world -x (viewer-left).
 * Landmark -> world is therefore (x, -y, -z). There is deliberately NO x flip:
 * flipping x would mirror the hand into the wrong chirality.
 */
import { Vector3, Quaternion, Matrix4 } from 'three';
import { FINGER_NAMES, LANDMARK_CHAINS } from './boneMap.js';

/** Approximate metres per landmark unit (wrist -> middle tip is ~1.2 units ~ 0.19 m). */
export const HAND_UNIT_METERS = 0.16;

/** Number of finger bones driven per hand (5 fingers x 3 phalanges). */
export const FINGER_BONE_COUNT = 15;

const _m = new Matrix4();
const _x = new Vector3();
const _y = new Vector3();
const _z = new Vector3();
const _qa = new Quaternion();
/** @type {Vector3[]} */
const _lmw = Array.from({ length: 21 }, () => new Vector3());

/**
 * Landmark -> world vector (no mirroring).
 * @param {Vector3} out
 * @param {ArrayLike<number>} lm [x, y, z]
 */
export function landmarkToWorld(out, lm) {
  return out.set(lm[0], -lm[1], -(lm[2] || 0));
}

/**
 * Build a right-handed orthonormal hand frame from two palm vectors.
 * Y = wrist -> middle knuckle, Z = along x across (palm/back normal), X = Y x Z.
 * The same formula is used on the rig and on the landmarks, so the frame
 * always refers to the same anatomical hand axes (chirality-safe).
 * @param {Vector3} along   wrist -> middle MCP
 * @param {Vector3} across  pinky MCP -> index MCP
 * @param {Quaternion} out
 * @returns {boolean} false when the vectors are degenerate
 */
export function basisQuaternion(along, across, out) {
  _y.copy(along);
  if (_y.lengthSq() < 1e-12) return false;
  _y.normalize();
  _z.crossVectors(_y, across);
  if (_z.lengthSq() < 1e-12) return false;
  _z.normalize();
  _x.crossVectors(_y, _z);
  _m.makeBasis(_x, _y, _z);
  out.setFromRotationMatrix(_m);
  return true;
}

/**
 * Express all 21 landmarks in world axes, in-place into a scratch array.
 * @param {ArrayLike<number>[]} lm
 * @returns {Vector3[]} shared scratch array (valid until the next call)
 */
export function landmarksToWorldScratch(lm) {
  for (let i = 0; i < 21; i++) landmarkToWorld(_lmw[i], lm[i]);
  return _lmw;
}

/**
 * Convert a landmark set into the WRIST-LOCAL frame and extract the 15 finger
 * bone directions in that frame.
 *
 * @param {ArrayLike<number>[]} lm 21 landmarks
 * @param {Quaternion} outBasis  hand frame (world orientation of the palm frame)
 * @param {Vector3[]} outDirs    15 preallocated vectors, order = FINGER_NAMES x 3 bones
 * @returns {boolean}
 */
export function solveHandLandmarks(lm, outBasis, outDirs) {
  if (!lm || lm.length < 21) return false;
  const w = landmarksToWorldScratch(lm);
  _x.subVectors(w[9], w[0]); // along
  _z.subVectors(w[5], w[17]); // across
  // basisQuaternion clobbers _x/_y/_z, so copy through local temporaries first.
  const along = _tmpA.copy(_x);
  const across = _tmpB.copy(_z);
  if (!basisQuaternion(along, across, outBasis)) return false;
  _qa.copy(outBasis).invert();
  let k = 0;
  for (const f of FINGER_NAMES) {
    const c = LANDMARK_CHAINS[f];
    for (let b = 0; b < 3; b++) {
      outDirs[k].subVectors(w[c[b + 1]], w[c[b]]).applyQuaternion(_qa);
      if (outDirs[k].lengthSq() < 1e-12) outDirs[k].set(0, 1, 0);
      outDirs[k].normalize();
      k++;
    }
  }
  return true;
}
const _tmpA = new Vector3();
const _tmpB = new Vector3();

/**
 * Mirror a landmark set across x = 0 (right-hand shape -> left-hand shape).
 * @param {ArrayLike<number>[]} src
 * @param {number[][]} dst preallocated 21 x 3
 */
export function mirrorLandmarksX(src, dst) {
  for (let i = 0; i < 21; i++) {
    dst[i][0] = -src[i][0];
    dst[i][1] = src[i][1];
    dst[i][2] = src[i][2] || 0;
  }
  return dst;
}

const _dir = new Vector3();
const _perp = new Vector3();
const _alt = new Vector3();

/**
 * Two-bone IK (shoulder -> elbow -> wrist) solved analytically.
 * If the target is out of reach the wrist is clamped onto the reachable sphere.
 *
 * @param {Vector3} shoulder
 * @param {Vector3} target
 * @param {number} upperLen
 * @param {number} foreLen
 * @param {Vector3} pole  direction the elbow should bend toward (need not be normalized)
 * @param {Vector3} outElbow
 * @param {Vector3} outWrist
 */
export function solveTwoBoneIK(shoulder, target, upperLen, foreLen, pole, outElbow, outWrist) {
  _dir.subVectors(target, shoulder);
  let dist = _dir.length();
  if (dist < 1e-9) _dir.set(0, -1, 0);
  else _dir.divideScalar(dist);
  const minD = Math.abs(upperLen - foreLen) + 1e-4;
  const maxD = upperLen + foreLen - 1e-4;
  dist = Math.min(maxD, Math.max(minD, dist));

  // Component of the pole vector orthogonal to the shoulder->target axis.
  _perp.copy(pole).addScaledVector(_dir, -pole.dot(_dir));
  if (_perp.lengthSq() < 1e-10) {
    _alt.set(0, 0, 1);
    _perp.copy(_alt).addScaledVector(_dir, -_alt.dot(_dir));
  }
  _perp.normalize();

  const cosA = (upperLen * upperLen + dist * dist - foreLen * foreLen) / (2 * upperLen * dist);
  const along = upperLen * cosA;
  const h = upperLen * Math.sqrt(Math.max(0, 1 - cosA * cosA));
  outElbow.copy(shoulder).addScaledVector(_dir, along).addScaledVector(_perp, h);
  outWrist.copy(shoulder).addScaledVector(_dir, dist);
}

/**
 * Smoothing factor for frame-rate independent exponential smoothing.
 * @param {number} dt seconds
 * @param {number} rate larger = snappier (~18 gives ~60 ms settle)
 */
export function smoothAlpha(dt, rate) {
  return 1 - Math.exp(-Math.max(0, dt) * rate);
}

const _tw = new Quaternion();
const _ax = new Vector3();

/**
 * Twist angle (radians, absolute) of a quaternion about a given axis, via
 * swing-twist decomposition. Used by tests to verify arms do not roll.
 * @param {Quaternion} q
 * @param {Vector3} axis unit axis in the same frame as q
 */
export function twistAngle(q, axis) {
  _ax.set(q.x, q.y, q.z);
  const proj = _ax.dot(axis);
  _tw.set(axis.x * proj, axis.y * proj, axis.z * proj, q.w);
  const len = Math.sqrt(_tw.x * _tw.x + _tw.y * _tw.y + _tw.z * _tw.z + _tw.w * _tw.w);
  if (len < 1e-9) return Math.PI;
  const w = Math.min(1, Math.abs(_tw.w / len));
  return 2 * Math.acos(w);
}

/**
 * Angle in radians between two vectors.
 * @param {Vector3} a
 * @param {Vector3} b
 */
export function angleBetween(a, b) {
  const d = a.dot(b) / Math.sqrt(a.lengthSq() * b.lengthSq());
  return Math.acos(Math.min(1, Math.max(-1, d)));
}
