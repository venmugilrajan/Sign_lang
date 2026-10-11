/**
 * AvatarRig: drives a humanoid skeleton from sign data (MediaPipe-style landmarks).
 *
 *  - Fingers: landmark vectors are converted into the wrist-local frame, then
 *    each phalanx gets `swing(restDir -> desiredDir) * restLocalQuat`
 *    (Quaternion.setFromUnitVectors), applied parent -> child.
 *  - Arms: analytic two-bone IK (shoulder -> elbow -> wrist). Arm bones use a
 *    swing-only rotation from their rest orientation, so they never accumulate roll.
 *  - Hand orientation comes from the landmark palm frame.
 *  - Everything is smoothed with exponential slerp. No per-frame allocations.
 *
 * Handedness: avatar RIGHT = world -x (viewer-left). Template data:
 *   single-handed `pose.right`            -> avatar right hand (matches the old 2D layout)
 *   two-handed   `pose.left`  / `wristL`  -> avatar right hand (viewer-left, thumb toward centre)
 *   two-handed   `pose.right` / `wristR`  -> avatar left hand
 * The hand shapes keep their chirality (no x flip), so no hand is ever mirrored.
 */
import { Vector3, Quaternion } from 'three';
import { BONE_NAMES, FINGER_NAMES, resolveBones, warnMissingBones } from './boneMap.js?v=29';
import {
  HAND_UNIT_METERS,
  FINGER_BONE_COUNT,
  solveHandLandmarks,
  solveTwoBoneIK,
  basisQuaternion,
  mirrorLandmarksX,
  smoothAlpha,
} from './retargetUtils.js';

/** @typedef {import('./boneMap.js').BoneNameConfig} BoneNameConfig */
/** @typedef {ArrayLike<number>[]} Landmarks21 */
/**
 * @typedef {Object} SignPose
 * @property {boolean} [twoHanded]
 * @property {Landmarks21|null} [left]
 * @property {Landmarks21|null} [right]
 * @property {number[]} [wristL]
 * @property {number[]} [wristR]
 * @property {Record<string, number>} [face] optional ARKit-style blendshape weights 0..1
 */

/** Blendshape names the face driver understands (ARKit / Ready Player Me). */
export const FACE_KEYS = [
  'browInnerUp', 'browDownLeft', 'browDownRight', 'browOuterUpLeft', 'browOuterUpRight',
  'eyeBlinkLeft', 'eyeBlinkRight', 'eyeWideLeft', 'eyeWideRight', 'eyeSquintLeft', 'eyeSquintRight',
  'jawOpen', 'mouthSmileLeft', 'mouthSmileRight', 'mouthFunnel', 'mouthPucker',
  'mouthFrownLeft', 'mouthFrownRight',
];

/** Signing-space layout, in units of arm length L (shoulder -> wrist at full reach). */
const SPACE = {
  single: { inward: 0.1, down: 0.25, forward: 0.62 },
  two: { halfGap: 0.23, down: 0.28, forward: 0.58 },
  rest: { outward: 0.05, down: 0.93, forward: 0.02 },
  /** Nominal arm length (m) that HAND_UNIT_METERS was calibrated for. */
  nominalArm: 0.6,
};

/**
 * @typedef {Object} SideRig
 * @property {'left'|'right'} side
 * @property {number} outward world x sign of "outward" for this arm (right: -1, left: +1)
 * @property {any} upper
 * @property {any} fore
 * @property {any} hand
 * @property {(any|null)[]} fingers 15 bones (3 phalanges x 5 fingers)
 * @property {Vector3[]} childDir 15 rest child directions in each bone's local frame
 * @property {Quaternion[]} restQ 15 rest local quaternions
 * @property {Quaternion[]} targetQ 15 target local quaternions
 * @property {Quaternion} restUpperWorldQ
 * @property {Quaternion} restForeWorldQ
 * @property {Quaternion} restHandWorldQ
 * @property {Vector3} restUpperDir
 * @property {Vector3} restForeDir
 * @property {Quaternion} restUpperLocalQ
 * @property {Quaternion} restForeLocalQ
 * @property {Quaternion} restHandLocalQ
 * @property {Quaternion} restBasis palm frame of the rig at rest
 * @property {Quaternion} rrel inv(restHandWorldQ) * restBasis
 * @property {number} upperLen
 * @property {number} foreLen
 * @property {Quaternion} upperT
 * @property {Quaternion} foreT
 * @property {Quaternion} handT
 * @property {Vector3} wristTarget
 * @property {Vector3} shoulderPos
 * @property {Vector3} elbowPos
 * @property {Vector3} wristPos
 * @property {Vector3} pole
 */

// ---- module scratch (reused every frame) ----
const _q1 = new Quaternion();
const _q2 = new Quaternion();
const _q3 = new Quaternion();
const _qAcc = new Quaternion();
const _qBasis = new Quaternion();
const _qInv = new Quaternion();
const _v1 = new Vector3();
const _v2 = new Vector3();
const _v3 = new Vector3();
const _dirs = Array.from({ length: FINGER_BONE_COUNT }, () => new Vector3());
const _mirror = Array.from({ length: 21 }, () => [0, 0, 0]);
const X_AXIS = new Vector3(1, 0, 0);

/** Rotate `rest` direction onto `to`, preserving the bone's rest roll. */
function swingFromRest(out, restDir, to, restQ) {
  out.setFromUnitVectors(restDir, to).multiply(restQ);
  return out;
}

export class AvatarRig {
  /**
   * @param {any} root loaded GLTF scene (or any Object3D with named bones)
   * @param {{ config?: BoneNameConfig, restHand?: Landmarks21|null, silent?: boolean }} [opts]
   *   restHand: a relaxed RIGHT-hand landmark set used for the idle hand shape.
   */
  constructor(root, opts = {}) {
    this.root = root;
    this.config = opts.config || BONE_NAMES;
    root.updateMatrixWorld(true);

    const res = resolveBones(root, this.config);
    this.missing = res.missing;
    this.missingOptional = res.missingOptional;
    this.ok = opts.silent ? res.missing.length === 0 : warnMissingBones(res, this.config);
    this.bones = res.bones;

    /** @type {Landmarks21|null} */
    this.restHandRight = opts.restHand || null;
    /** @type {number[][]|null} */
    this.restHandLeft = null;
    if (this.restHandRight) this.restHandLeft = mirrorLandmarksX(this.restHandRight, _mirror.map((r) => r.slice()));

    this.right = this._buildSide('right', -1);
    this.left = this._buildSide('left', 1);

    // Torso / idle
    this.chest = this.bones.get(this.config.chest) || this.bones.get(this.config.spine) || null;
    this.chestRestQ = this.chest ? this.chest.quaternion.clone() : new Quaternion();

    // Face morph targets
    /** @type {Map<string, {mesh: any, index: number}[]>} */
    this.morphs = new Map();
    this._collectMorphs(root);
    /** @type {Record<string, number>} */
    this.faceState = {};
    for (const k of FACE_KEYS) this.faceState[k] = 0;

    // Blink state
    this._time = 0;
    this._nextBlink = 2.2;
    this._blinkT = -1;
    this.chestAxis = X_AXIS;

    /** Waist-up framing helper data. */
    this.layout = this._computeLayout();
  }

  /** Height/width measurements for camera framing. */
  _computeLayout() {
    const head = this.bones.get(this.config.head);
    const hips = this.bones.get(this.config.hips);
    const out = { headY: 1.7, hipsY: 1.0, shoulderSpan: 0.4, armLength: SPACE.nominalArm, centerX: 0, centerZ: 0 };
    if (head) out.headY = head.getWorldPosition(_v1).y;
    if (hips) out.hipsY = hips.getWorldPosition(_v1).y;
    if (this.right.upper && this.left.upper) {
      this.right.upper.getWorldPosition(_v1);
      this.left.upper.getWorldPosition(_v2);
      out.shoulderSpan = _v1.distanceTo(_v2);
      out.centerX = (_v1.x + _v2.x) / 2;
      out.centerZ = (_v1.z + _v2.z) / 2;
    }
    out.armLength = (this.right.upperLen + this.right.foreLen + this.left.upperLen + this.left.foreLen) / 2 || SPACE.nominalArm;
    return out;
  }

  /**
   * @param {'left'|'right'} side
   * @param {number} outward
   * @returns {SideRig}
   */
  _buildSide(side, outward) {
    const names = this.config[side];
    const get = (n) => this.bones.get(n) || null;
    const upper = get(names.upperArm);
    const fore = get(names.foreArm);
    const hand = get(names.hand);

    /** @type {SideRig} */
    const s = {
      side, outward, upper, fore, hand,
      fingers: new Array(FINGER_BONE_COUNT).fill(null),
      childDir: Array.from({ length: FINGER_BONE_COUNT }, () => new Vector3(0, 1, 0)),
      restQ: Array.from({ length: FINGER_BONE_COUNT }, () => new Quaternion()),
      targetQ: Array.from({ length: FINGER_BONE_COUNT }, () => new Quaternion()),
      restUpperWorldQ: new Quaternion(), restForeWorldQ: new Quaternion(), restHandWorldQ: new Quaternion(),
      restUpperDir: new Vector3(outward, 0, 0), restForeDir: new Vector3(outward, 0, 0),
      restUpperLocalQ: new Quaternion(), restForeLocalQ: new Quaternion(), restHandLocalQ: new Quaternion(),
      restBasis: new Quaternion(), rrel: new Quaternion(),
      upperLen: 0.3, foreLen: 0.27,
      upperT: new Quaternion(), foreT: new Quaternion(), handT: new Quaternion(),
      wristTarget: new Vector3(), shoulderPos: new Vector3(), elbowPos: new Vector3(), wristPos: new Vector3(),
      pole: new Vector3(),
    };
    if (!upper || !fore || !hand) return s;

    upper.getWorldQuaternion(s.restUpperWorldQ);
    fore.getWorldQuaternion(s.restForeWorldQ);
    hand.getWorldQuaternion(s.restHandWorldQ);
    s.restUpperLocalQ.copy(upper.quaternion);
    s.restForeLocalQ.copy(fore.quaternion);
    s.restHandLocalQ.copy(hand.quaternion);
    const pU = upper.getWorldPosition(new Vector3());
    const pF = fore.getWorldPosition(new Vector3());
    const pH = hand.getWorldPosition(new Vector3());
    s.restUpperDir.subVectors(pF, pU);
    s.upperLen = s.restUpperDir.length();
    s.restUpperDir.normalize();
    s.restForeDir.subVectors(pH, pF);
    s.foreLen = s.restForeDir.length();
    s.restForeDir.normalize();

    // Fingers: rest local quaternion + direction toward the child joint, in the bone's own frame.
    FINGER_NAMES.forEach((f, fi) => {
      const chain = names.fingers[f].map(get);
      for (let k = 0; k < 3; k++) {
        const i = fi * 3 + k;
        const bone = chain[k];
        s.fingers[i] = bone;
        if (!bone) continue;
        s.restQ[i].copy(bone.quaternion);
        const child = chain[k + 1];
        if (child && child.position.lengthSq() > 1e-12) {
          s.childDir[i].copy(child.position).normalize();
        } else if (k > 0) {
          s.childDir[i].copy(s.childDir[i - 1]); // missing end bone: reuse previous phalanx axis
        }
      }
    });

    // Rig palm frame (same formula as the landmark frame).
    const idx1 = get(names.fingers.index[0]);
    const mid1 = get(names.fingers.middle[0]);
    const pky1 = get(names.fingers.pinky[0]);
    if (idx1 && mid1 && pky1) {
      _v1.subVectors(mid1.getWorldPosition(_v3), pH);
      _v2.subVectors(idx1.getWorldPosition(_v3), pky1.getWorldPosition(new Vector3()));
      basisQuaternion(_v1, _v2, s.restBasis);
      s.rrel.copy(s.restHandWorldQ).invert().multiply(s.restBasis);
    }
    return s;
  }

  /** @param {any} root */
  _collectMorphs(root) {
    root.traverse((o) => {
      if (!o.morphTargetDictionary || !o.morphTargetInfluences) return;
      for (const key of FACE_KEYS) {
        const index = o.morphTargetDictionary[key];
        if (index === undefined) continue;
        let list = this.morphs.get(key);
        if (!list) this.morphs.set(key, (list = []));
        list.push({ mesh: o, index });
      }
    });
  }

  /**
   * Compute target quaternions for one side.
   * @param {SideRig} s
   * @param {Landmarks21|null|undefined} lm  hand landmarks, or null for a relaxed idle arm
   * @param {boolean} signing  true: raised signing arm with world-oriented hand
   * @param {number[]|undefined} wristOffset [x, y] landmark-unit offset (y down)
   * @param {boolean} twoHanded
   */
  _solveSide(s, lm, signing, wristOffset, twoHanded) {
    if (!s.upper || !s.fore || !s.hand) return;
    const L = this.layout.armLength;
    const scale = L / SPACE.nominalArm;
    s.upper.getWorldPosition(s.shoulderPos);
    const sp = s.shoulderPos;

    // ---- wrist target + elbow pole ----
    if (signing) {
      if (twoHanded) {
        const cx = this.layout.centerX;
        // right arm sits at viewer-left (-x), left arm at viewer-right (+x)
        s.wristTarget.set(
          cx + s.outward * SPACE.two.halfGap * L + (wristOffset ? wristOffset[0] : 0) * HAND_UNIT_METERS * scale,
          sp.y - SPACE.two.down * L - (wristOffset ? wristOffset[1] : 0) * HAND_UNIT_METERS * scale,
          sp.z + SPACE.two.forward * L
        );
      } else {
        s.wristTarget.set(sp.x - s.outward * SPACE.single.inward * L, sp.y - SPACE.single.down * L, sp.z + SPACE.single.forward * L);
      }
      s.pole.set(s.outward * 0.5, -1, -0.3); // elbow drops down and slightly out
    } else {
      s.wristTarget.set(sp.x + s.outward * SPACE.rest.outward * L, sp.y - SPACE.rest.down * L, sp.z + SPACE.rest.forward * L);
      s.pole.set(s.outward * 0.4, 0, -1); // relaxed: elbow slightly back, forearm forward
    }
    solveTwoBoneIK(sp, s.wristTarget, s.upperLen, s.foreLen, s.pole, s.elbowPos, s.wristPos);

    // ---- arm bones: swing-only from rest (no roll accumulation) ----
    _v1.subVectors(s.elbowPos, sp).normalize();
    swingFromRest(_q1, s.restUpperDir, _v1, _q2.copy(s.restUpperWorldQ)); // _q1 = upper world
    _v2.subVectors(s.wristPos, s.elbowPos).normalize();
    swingFromRest(_q3, s.restForeDir, _v2, _qInv.copy(s.restForeWorldQ)); // _q3 = fore world

    // parent (clavicle) world orientation currently in the scene
    if (s.upper.parent) s.upper.parent.getWorldQuaternion(_qAcc);
    else _qAcc.identity();
    s.upperT.copy(_qAcc).invert().multiply(_q1);
    s.foreT.copy(_q1).invert().multiply(_q3);

    // ---- hand + fingers ----
    let shape = lm;
    if (!shape) {
      shape = s.side === 'right' ? this.restHandRight : this.restHandLeft;
    }
    const haveShape = !!shape && solveHandLandmarks(shape, _qBasis, _dirs);

    if (haveShape && signing && lm) {
      // world hand orientation = Btarget * inv(Brest) * restHandWorld
      _q2.copy(s.restBasis).invert();
      _qAcc.copy(_qBasis).multiply(_q2).multiply(s.restHandWorldQ);
    } else {
      // follow the forearm, keeping the rest wrist offset
      _qAcc.copy(_q3).multiply(_qInv.copy(s.restForeWorldQ).invert()).multiply(s.restHandWorldQ);
    }
    s.handT.copy(_q3).invert().multiply(_qAcc);

    if (!haveShape) return;
    for (let fi = 0; fi < 5; fi++) {
      _qAcc.identity(); // accumulates local rotations of already-solved phalanges
      for (let k = 0; k < 3; k++) {
        const i = fi * 3 + k;
        // landmark direction (wrist-local) -> hand-bone frame -> parent-bone frame
        _v1.copy(_dirs[i]).applyQuaternion(s.rrel);
        _qInv.copy(_qAcc).invert();
        _v1.applyQuaternion(_qInv);
        _v2.copy(s.childDir[i]).applyQuaternion(s.restQ[i]); // rest dir in parent frame
        s.targetQ[i].setFromUnitVectors(_v2, _v1).multiply(s.restQ[i]);
        _qAcc.multiply(s.targetQ[i]);
      }
    }
  }

  /**
   * Compute targets from a pose, then ease the skeleton toward them.
   * @param {SignPose|null|undefined} pose
   * @param {number} dt seconds since last call (use a large value to snap)
   */
  update(pose, dt) {
    // Idle first, so arm targets are computed against the chest's current orientation.
    this._idle(dt);
    const two = !!(pose && pose.twoHanded && pose.left && pose.right);
    // avatar right hand <- pose.left (two-handed) or pose.right (single); avatar left hand <- pose.right (two-handed)
    const rightLm = pose ? (two ? pose.left : pose.right || pose.left) : null;
    const leftLm = two && pose ? pose.right : null;
    this._solveSide(this.right, rightLm || null, !!rightLm, pose ? pose.wristL : undefined, two);
    this._solveSide(this.left, leftLm || null, !!leftLm, pose ? pose.wristR : undefined, two);

    const a = smoothAlpha(dt, 22);
    this._ease(this.right, a);
    this._ease(this.left, a);
    this._face(pose && pose.face, dt);
  }

  /** @param {SideRig} s @param {number} a */
  _ease(s, a) {
    if (!s.upper || !s.fore || !s.hand) return;
    s.upper.quaternion.slerp(s.upperT, a);
    s.fore.quaternion.slerp(s.foreT, a);
    s.hand.quaternion.slerp(s.handT, a);
    // parent -> child order
    for (let i = 0; i < FINGER_BONE_COUNT; i++) {
      const b = s.fingers[i];
      if (b) b.quaternion.slerp(s.targetQ[i], a);
    }
  }

  /**
   * Blendshape driver: brows / eyes / jaw / mouth from sign data, neutral fallback.
   * @param {Record<string, number>|undefined|null} face
   * @param {number} dt
   */
  _face(face, dt) {
    const a = smoothAlpha(dt, 14);
    for (const key of FACE_KEYS) {
      const list = this.morphs.get(key);
      if (!list) continue;
      let target = face && typeof face[key] === 'number' ? Math.min(1, Math.max(0, face[key])) : 0;
      if (this._blinkT >= 0 && (key === 'eyeBlinkLeft' || key === 'eyeBlinkRight')) {
        target = Math.max(target, Math.sin(Math.min(1, this._blinkT / 0.16) * Math.PI));
      }
      const cur = this.faceState[key];
      const next = key === 'eyeBlinkLeft' || key === 'eyeBlinkRight' ? target : cur + (target - cur) * a;
      this.faceState[key] = next;
      for (let i = 0; i < list.length; i++) list[i].mesh.morphTargetInfluences[list[i].index] = next;
    }
  }

  /** Breathing + blink timer. @param {number} dt */
  _idle(dt) {
    if (!isFinite(dt)) dt = 0;
    this._time += dt;
    if (this._blinkT >= 0) {
      this._blinkT += dt;
      if (this._blinkT > 0.16) this._blinkT = -1;
    } else if (this._time >= this._nextBlink) {
      this._blinkT = 0;
      this._nextBlink = this._time + 2.4 + Math.random() * 3;
    }
    if (this.chest) {
      _q1.setFromAxisAngle(this.chestAxis, Math.sin(this._time * 1.7) * 0.012);
      this.chest.quaternion.copy(this.chestRestQ).multiply(_q1);
    }
  }
}
