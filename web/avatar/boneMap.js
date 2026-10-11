/**
 * Bone-name configuration for the 3D sign avatar.
 *
 * This is the ONE place to edit if your GLB uses different bone names.
 * Defaults match Ready Player Me (e.g. "RightHandIndex2") and Mixamo
 * ("mixamorig:RightHandIndex2" / "mixamorigRightHandIndex2"). Prefixes are
 * tried in order and name matching is case-insensitive.
 */

/** @typedef {'thumb'|'index'|'middle'|'ring'|'pinky'} FingerName */
/** @typedef {'left'|'right'} Side */

/**
 * @typedef {Object} SideBoneNames
 * @property {string} shoulder
 * @property {string} upperArm
 * @property {string} foreArm
 * @property {string} hand
 * @property {Record<FingerName, string[]>} fingers  4 names per finger: 3 phalanges + optional end bone.
 */

/**
 * @typedef {Object} BoneNameConfig
 * @property {string[]} prefixes  Name prefixes to try (first match wins).
 * @property {string} hips
 * @property {string} spine
 * @property {string} chest
 * @property {string} neck
 * @property {string} head
 * @property {SideBoneNames} left
 * @property {SideBoneNames} right
 */

/** @type {FingerName[]} */
export const FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky'];

/**
 * MediaPipe hand landmark indices per finger (0 is the wrist).
 * Bone k of a finger points from chain[k] to chain[k + 1].
 * @type {Record<FingerName, number[]>}
 */
export const LANDMARK_CHAINS = {
  thumb: [1, 2, 3, 4],
  index: [5, 6, 7, 8],
  middle: [9, 10, 11, 12],
  ring: [13, 14, 15, 16],
  pinky: [17, 18, 19, 20],
};

/** @param {'Left'|'Right'} S @returns {SideBoneNames} */
function sideNames(S) {
  /** @param {string} f */
  const chain = (f) => [1, 2, 3, 4].map((i) => `${S}Hand${f}${i}`);
  return {
    shoulder: `${S}Shoulder`,
    upperArm: `${S}Arm`,
    foreArm: `${S}ForeArm`,
    hand: `${S}Hand`,
    fingers: {
      thumb: chain('Thumb'),
      index: chain('Index'),
      middle: chain('Middle'),
      ring: chain('Ring'),
      pinky: chain('Pinky'),
    },
  };
}

/** Default config: Ready Player Me / Mixamo naming. @type {BoneNameConfig} */
export const BONE_NAMES = {
  prefixes: ['', 'mixamorig', 'mixamorig:', 'mixamorig_'],
  hips: 'Hips',
  spine: 'Spine',
  chest: 'Spine2',
  neck: 'Neck',
  head: 'Head',
  left: sideNames('Left'),
  right: sideNames('Right'),
};

/**
 * Names of the optional end bone of every finger. A missing end bone is only
 * a notice, not a warning, because the last phalanx can borrow its direction.
 * @param {string} name
 */
function isOptionalEndBone(name) {
  return /(Thumb|Index|Middle|Ring|Pinky)4$/.test(name);
}

/**
 * Resolve expected bone names to actual bones in a loaded scene graph.
 * @param {{ traverse: (cb: (o: any) => void) => void }} root
 * @param {BoneNameConfig} [config]
 * @returns {{ bones: Map<string, any>, missing: string[], missingOptional: string[] }}
 */
export function resolveBones(root, config = BONE_NAMES) {
  /** @type {Map<string, any>} */
  const byLowerName = new Map();
  // 1. First index bones from SkinnedMesh skeletons directly so they deform the mesh
  root.traverse((o) => {
    if (o && o.isSkinnedMesh && o.skeleton && o.skeleton.bones) {
      o.skeleton.bones.forEach((b) => {
        if (b && b.name) {
          // If this is a Mixamo leaf dummy bone whose parent has the exact same name, ignore it
          if (b.parent && b.parent.name && b.parent.name.toLowerCase() === b.name.toLowerCase()) {
            return;
          }
          const key = String(b.name).toLowerCase();
          const existing = byLowerName.get(key);
          if (!existing || (b.children && b.children.length > 0 && (!existing.children || existing.children.length === 0))) {
            byLowerName.set(key, b);
          }
        }
      });
    }
  });

  // 2. Index any remaining bones in the general scene graph
  root.traverse((o) => {
    if (o && o.name) {
      // Ignore dummy child bones that share the same name as their parent
      if (o.parent && o.parent.name && o.parent.name.toLowerCase() === o.name.toLowerCase()) {
        return;
      }
      const key = String(o.name).toLowerCase();
      const existing = byLowerName.get(key);
      if (!existing || (o.children && o.children.length > 0 && (!existing.children || existing.children.length === 0))) {
        byLowerName.set(key, o);
      }
    }
  });

  /** @param {string} expected */
  const find = (expected) => {
    for (const p of config.prefixes) {
      const hit = byLowerName.get((p + expected).toLowerCase());
      if (hit) return hit;
    }
    return null;
  };

  /** @type {Map<string, any>} */
  const bones = new Map();
  /** @type {string[]} */
  const missing = [];
  /** @type {string[]} */
  const missingOptional = [];

  /** @param {string} expected */
  const take = (expected) => {
    const b = find(expected);
    if (b) bones.set(expected, b);
    else (isOptionalEndBone(expected) ? missingOptional : missing).push(expected);
  };

  [config.hips, config.spine, config.chest, config.neck, config.head].forEach(take);
  for (const side of /** @type {Side[]} */ (['left', 'right'])) {
    const s = config[side];
    [s.shoulder, s.upperArm, s.foreArm, s.hand].forEach(take);
    for (const f of FINGER_NAMES) s.fingers[f].forEach(take);
  }
  return { bones, missing, missingOptional };
}

/**
 * Log a console warning listing expected bones that are not in the GLB.
 * @param {{ missing: string[], missingOptional: string[] }} res
 * @param {BoneNameConfig} [config]
 * @returns {boolean} true when nothing required is missing
 */
export function warnMissingBones(res, config = BONE_NAMES) {
  if (res.missing.length) {
    console.warn(
      `[Avatar3D] ${res.missing.length} expected bone(s) missing from the GLB: ${res.missing.join(', ')}.\n` +
        `Edit web/avatar/boneMap.js (BONE_NAMES, prefixes: ${JSON.stringify(config.prefixes)}) to match your rig.`
    );
  }
  if (res.missingOptional.length) {
    console.info(
      `[Avatar3D] Optional finger end bones not found (last phalanx will reuse its parent direction): ${res.missingOptional.join(', ')}`
    );
  }
  return res.missing.length === 0;
}
