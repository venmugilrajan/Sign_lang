// Procedural T-pose humanoid used ONLY by the unit tests (not shipped to the browser).
// It is a plain Three.js bone hierarchy, not a model file. Rest bones carry non-trivial roll
// quaternions on purpose, so the retargeting is exercised on rigs whose local axes are not identity.
import { Object3D, Quaternion, Vector3 } from 'three';

const X = new Vector3(1, 0, 0);

/**
 * @param {{ prefix?: string, omit?: string[], withFace?: boolean }} [opts]
 */
export function buildTestRig({ prefix = '', omit = [], withFace = true } = {}) {
  const root = new Object3D();
  root.name = 'Armature';
  /** @type {Map<string, {obj: Object3D, wq: Quaternion}>} */
  const nodes = new Map();
  const add = (name, parentName, worldOffset, roll = 0) => {
    if (omit.includes(name)) return;
    const parent = parentName && nodes.get(parentName) ? nodes.get(parentName) : { obj: root, wq: new Quaternion() };
    const obj = new Object3D();
    obj.name = prefix + name;
    obj.position.copy(worldOffset).applyQuaternion(parent.wq.clone().invert());
    obj.quaternion.setFromAxisAngle(X, roll);
    parent.obj.add(obj);
    nodes.set(name, { obj, wq: parent.wq.clone().multiply(obj.quaternion) });
  };
  const v = (x, y, z) => new Vector3(x, y, z);

  add('Hips', null, v(0, 1, 0));
  add('Spine', 'Hips', v(0, 0.1, 0));
  add('Spine1', 'Spine', v(0, 0.12, 0));
  add('Spine2', 'Spine1', v(0, 0.12, 0));
  add('Neck', 'Spine2', v(0, 0.15, 0));
  add('Head', 'Neck', v(0, 0.1, 0));

  // Right arm = -x, left arm = +x (avatar faces +z, so avatar-right is world -x).
  for (const [S, sx] of [['Right', -1], ['Left', 1]]) {
    add(`${S}Shoulder`, 'Spine2', v(sx * 0.05, 0.12, 0));
    add(`${S}Arm`, `${S}Shoulder`, v(sx * 0.14, 0, 0), 0.2);
    add(`${S}ForeArm`, `${S}Arm`, v(sx * 0.3, 0, 0), 0.3);
    add(`${S}Hand`, `${S}ForeArm`, v(sx * 0.27, 0, 0), 0.4);
    // [name, per-bone world offsets from the previous joint (x is mirrored per side)]
    const fingers = {
      Index: [[-0.09, 0, 0.03], [-0.04, 0, 0], [-0.025, 0, 0], [-0.02, 0, 0]],
      Middle: [[-0.095, 0, 0.005], [-0.045, 0, 0], [-0.027, 0, 0], [-0.02, 0, 0]],
      Ring: [[-0.09, 0, -0.02], [-0.04, 0, 0], [-0.025, 0, 0], [-0.02, 0, 0]],
      Pinky: [[-0.08, 0, -0.04], [-0.03, 0, 0], [-0.018, 0, 0], [-0.015, 0, 0]],
      // Palm-down T-pose: thumb points forward (+z).
      Thumb: [[-0.03, -0.01, 0.03], [-0.012, -0.004, 0.03], [-0.01, -0.003, 0.025], [-0.008, -0.002, 0.02]],
    };
    for (const [F, offs] of Object.entries(fingers)) {
      offs.forEach((o, i) => {
        const parent = i === 0 ? `${S}Hand` : `${S}Hand${F}${i}`;
        // Offsets above are authored for the RIGHT hand (fingers toward -x); mirror for the left hand.
        add(`${S}Hand${F}${i + 1}`, parent, v(-sx * o[0], o[1], o[2]), 0.15);
      });
    }
  }

  if (withFace) {
    const head = nodes.get('Head');
    const face = new Object3D();
    face.name = 'Wolf3D_Head';
    face.morphTargetDictionary = { jawOpen: 0, eyeBlinkLeft: 1, eyeBlinkRight: 2, browInnerUp: 3, mouthSmileLeft: 4 };
    face.morphTargetInfluences = [0, 0, 0, 0, 0];
    if (head) head.obj.add(face);
  }

  root.updateMatrixWorld(true);
  return { root, nodes };
}
