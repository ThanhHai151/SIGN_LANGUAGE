const FINGERS = ['index', 'middle', 'ring', 'pinky'];
const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

function boundedGain(value, name) {
  if (!Number.isFinite(value) || value < 0 || value > 3) {
    throw new RangeError(`${name} phải nằm trong khoảng 0–3`);
  }
  return value;
}

/** Map a canonical right-hand pose to Mixamo local Euler targets. */
export function canonicalHandPoseToMixamoTargets(pose, {
  side = 'Right',
  flexionGain = 1,
  splayGain = 1,
  thumbAbductionGain = 1
} = {}) {
  if (pose?.schema !== 'canonical-hand-pose/v1' || pose?.unit !== 'radian') {
    throw new TypeError('Retargeter cần canonical-hand-pose/v1 ở radian');
  }
  if (side !== 'Right' && side !== 'Left') throw new RangeError('side phải là Right hoặc Left');
  const gains = {
    flexion: boundedGain(flexionGain, 'flexionGain'),
    splay: boundedGain(splayGain, 'splayGain'),
    thumb: boundedGain(thumbAbductionGain, 'thumbAbductionGain')
  };
  const sideSign = side === 'Right' ? -1 : 1;
  const targets = [];

  for (const finger of FINGERS) {
    const values = pose.fingers?.[finger];
    if (!Array.isArray(values?.flexion) || values.flexion.length !== 3 || !Number.isFinite(values.splay)) {
      throw new TypeError(`Pose thiếu dữ liệu ngón ${finger}`);
    }
    for (let joint = 0; joint < 3; joint += 1) {
      const flexion = clamp(values.flexion[joint] * gains.flexion, 0, 1.65);
      const splay = joint === 0 ? clamp(values.splay * gains.splay * (side === 'Right' ? 1 : -1), -0.45, 0.45) : 0;
      targets.push({ bone: `${side}Hand${finger[0].toUpperCase()}${finger.slice(1)}${joint + 1}`, rot: [flexion, 0, splay] });
    }
  }

  const thumb = pose.thumb;
  if (!Array.isArray(thumb?.flexion) || thumb.flexion.length !== 3 || !Number.isFinite(thumb.abduction)) {
    throw new TypeError('Pose thiếu dữ liệu ngón cái');
  }
  const thumbFlexion = thumb.flexion.map((value) => clamp(value * gains.flexion, 0, 1.65));
  const abduction = clamp(thumb.abduction * gains.thumb, -1.2, 1.2);
  targets.push({ bone: `${side}HandThumb1`, rot: [thumbFlexion[0], sideSign * 0.22, sideSign * abduction] });
  targets.push({ bone: `${side}HandThumb2`, rot: [thumbFlexion[1], 0, 0] });
  targets.push({ bone: `${side}HandThumb3`, rot: [thumbFlexion[2], 0, 0] });
  targets.push({ bone: `${side}HandThumb4`, rot: [0, 0, 0] });

  return targets;
}
