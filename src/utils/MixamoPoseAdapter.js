export function setFingers(targets, prefix, flexAngle) {
  const bend = typeof flexAngle === 'number'
    ? [flexAngle, flexAngle, flexAngle]
    : (Array.isArray(flexAngle) ? flexAngle : (flexAngle?.bend || [0, 0, 0]));
  const splay = typeof flexAngle === 'object' && !Array.isArray(flexAngle) ? (flexAngle.splay || 0) : 0;
  for (let segment = 1; segment <= 3; segment += 1) {
    targets.push({ bone: `${prefix}${segment}`, rot: [bend[segment - 1] ?? 0, 0, segment === 1 ? splay : 0] });
  }
}

export function setThumb(targets, side = 'Right', { abduction = 0.75, extension = 0.0, pitch = 0.3 } = {}) {
  const isRight = side === 'Right';
  const sign = isRight ? -1 : 1;
  const prefix = `${side}HandThumb`;

  targets.push({ bone: `${prefix}1`, rot: [pitch, sign * 0.22, sign * abduction] });
  targets.push({ bone: `${prefix}2`, rot: [extension, 0, 0] });
  targets.push({ bone: `${prefix}3`, rot: [extension, 0, 0] });
  targets.push({ bone: `${prefix}4`, rot: [extension * 0.5, 0, 0] });
}

export function createRestPose() {
  const targets = [
    { bone: 'RightArm', rot: [1.25, 0.05, -0.15] },
    { bone: 'RightForeArm', rot: [0.0, 0.0, -0.2] },
    { bone: 'RightHand', rot: [0.0, 0.0, 0.0] },
    { bone: 'LeftArm', rot: [1.25, -0.05, 0.15] },
    { bone: 'LeftForeArm', rot: [0.0, 0.0, 0.2] },
    { bone: 'LeftHand', rot: [0.0, 0.0, 0.0] }
  ];

  for (const side of ['Right', 'Left']) {
    for (const finger of ['Index', 'Middle', 'Ring', 'Pinky']) {
      setFingers(targets, `${side}Hand${finger}`, 0.15);
    }
    setThumb(targets, side, { abduction: 0.75 });
  }

  return targets;
}

function createLetterPose({
  index, middle, ring, pinky, thumb,
  armRot = [0.0, 0.0, -1.5],
  forearmRot = [-1.5, 0.0, -0.2],
  wristRot = [0.0, 0.0, 0.05]
}) {
  // Start from the natural rest rig on the left; hold the right signing arm at chest height.
  const targets = createRestPose().filter((target) => !target.bone.startsWith('Right'));
  targets.push(
    { bone: 'RightArm', rot: armRot },
    { bone: 'RightForeArm', rot: forearmRot },
    { bone: 'RightHand', rot: wristRot }
  );
  setFingers(targets, 'RightHandIndex', index);
  setFingers(targets, 'RightHandMiddle', middle);
  setFingers(targets, 'RightHandRing', ring);
  setFingers(targets, 'RightHandPinky', pinky);
  setThumb(targets, 'Right', thumb);
  return targets;
}

export const MIXAMO_TEST_POSES = {
  A: createLetterPose({ index: 1.57, middle: 1.57, ring: 1.57, pinky: 1.57, thumb: { abduction: 0.3, extension: 0, pitch: 0.1 } }),
  B: createLetterPose({ index: 0, middle: 0, ring: 0, pinky: 0, thumb: { abduction: -0.4, extension: 0.2, pitch: 0.05 } })
};

export const MIXAMO_ALPHABET = {
  // A: fist, thumb beside the fingers.
  A: createLetterPose({ index: 1.57, middle: 1.57, ring: 1.57, pinky: 1.57, thumb: { abduction: 0.3, extension: 0, pitch: 0.1 } }),
  // B: four fingers straight together; thumb folds across the palm.
  B: createLetterPose({ index: { bend: [0, 0, 0], splay: -0.025 }, middle: { bend: [0, 0, 0], splay: 0 }, ring: { bend: [0, 0, 0], splay: 0.025 }, pinky: { bend: [0, 0, 0], splay: 0.05 }, thumb: { abduction: -0.4, extension: 0.2, pitch: 0.05 } }),
  // C: all fingers and thumb form a curved C-shaped opening.
  C: createLetterPose({ index: [0.55, 0.85, 0.8], middle: [0.6, 0.9, 0.85], ring: [0.65, 0.9, 0.85], pinky: [0.7, 0.9, 0.85], thumb: { abduction: 0.75, extension: 0.35, pitch: 0.25 } }),
  // D: index points up; remaining fingers curve toward the thumb.
  D: createLetterPose({ index: [0, 0, 0], middle: [1.0, 1.35, 1.25], ring: [1.05, 1.4, 1.3], pinky: [1.1, 1.4, 1.3], thumb: { abduction: 0.55, extension: 0.35, pitch: 0.45 } }),
  // E: fingertips curl toward the thumb.
  E: createLetterPose({ index: [0.55, 1.1, 1.45], middle: [0.55, 1.1, 1.45], ring: [0.55, 1.1, 1.45], pinky: [0.55, 1.1, 1.45], thumb: { abduction: 0.05, extension: 0.9, pitch: 0.6 } }),
  // F: index and thumb make a small circle; the other three fingers extend.
  F: createLetterPose({ index: [0.45, 1.0, 1.2], middle: [0, 0, 0], ring: [0, 0, 0], pinky: [0, 0, 0], thumb: { abduction: 0.55, extension: 0.6, pitch: 0.75 } }),
  // G: index and thumb extend sideways; remaining fingers fold.
  G: createLetterPose({ index: [0, 0, 0], middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.8, extension: 0.15, pitch: 0.2 }, wristRot: [0, -0.55, 0.05] }),
  // H: index and middle extend together sideways.
  H: createLetterPose({ index: { bend: [0, 0, 0], splay: -0.04 }, middle: { bend: [0, 0, 0], splay: 0.04 }, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.75, extension: 0.15, pitch: 0.2 }, wristRot: [0, -0.55, 0.05] }),
  // I: only the pinky extends.
  I: createLetterPose({ index: 1.5, middle: 1.6, ring: 1.55, pinky: [0, 0, 0], thumb: { abduction: -0.35, extension: 0.6, pitch: 0.2 } }),
  // J: straight pinky, remaining fingers folded, thumb tucked to the folded fingers.
  J: createLetterPose({ index: 1.57, middle: 1.57, ring: 1.57, pinky: [0, 0, 0], thumb: { abduction: -0.05, extension: 0.85, pitch: 0.8 } }),
  // K: index and middle extend apart while the other fingers fold.
  K: createLetterPose({ index: [0, 0, 0], middle: { bend: [0, 0, 0], splay: 0.2 }, ring: 1.65, pinky: 1.65, thumb: { abduction: -0.05, extension: 0.1, pitch: 0.55 } }),
  // L: index up and thumb extended at a right angle.
  L: createLetterPose({ index: [0, 0, 0], middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { abduction: 1.05, extension: 0, pitch: 0.15 } }),
  // M: first three fingers fold over the tucked thumb.
  M: createLetterPose({ index: [1.05, 1.35, 1.45], middle: [1.05, 1.35, 1.45], ring: [1.05, 1.35, 1.45], pinky: 1.57, thumb: { abduction: 0.05, extension: 0.75, pitch: 0.75 } }),
  // N: index and middle fold over the tucked thumb.
  N: createLetterPose({ index: [1.05, 1.35, 1.45], middle: [1.05, 1.35, 1.45], ring: 1.57, pinky: 1.57, thumb: { abduction: 0.05, extension: 0.75, pitch: 0.7 } }),
  // O: fingertips curve toward the thumb to form a small opening.
  O: createLetterPose({ index: [0.65, 0.95, 1.05], middle: [0.7, 1.0, 1.05], ring: [0.75, 1.0, 1.0], pinky: [0.8, 0.95, 0.95], thumb: { abduction: 0.8, extension: 0.35, pitch: 0.55 } }),
  // P: K-like handshape; the fixed arm frame keeps the hand in signing space.
  P: createLetterPose({ index: [0, 0, 0], middle: { bend: [0, 0.55, 0.4], splay: 0.12 }, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.55, extension: 0.1, pitch: 0.4 }, wristRot: [-0.35, -0.25, 0.55] }),
  // Q: G-like handshape with the index and thumb extended.
  Q: createLetterPose({ index: [0, 0, 0], middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.8, extension: 0.15, pitch: 0.2 }, wristRot: [-0.4, -0.25, 0.55] }),
  // R: crossed index/middle approximation using opposing splay.
  R: createLetterPose({ index: { bend: [0, 0, 0], splay: 0.24 }, middle: { bend: [0, 0, 0], splay: -0.24 }, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.4, extension: 0.1, pitch: 0.25 } }),
  // S: closed fist with thumb across the front.
  S: createLetterPose({ index: 1.65, middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.25, extension: 0.15, pitch: 0.25 } }),
  // T: fist with thumb tucked between index and middle (approximation).
  T: createLetterPose({ index: 1.65, middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.1, extension: 0.55, pitch: 0.65 } }),
  // U: index and middle straight and close together.
  U: createLetterPose({ index: { bend: [0, 0, 0], splay: -0.015 }, middle: { bend: [0, 0, 0], splay: 0.015 }, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.1, extension: 0.35, pitch: 0.35 } }),
  // V: index and middle straight with a visible gap.
  V: createLetterPose({ index: { bend: [0, 0, 0], splay: -0.22 }, middle: { bend: [0, 0, 0], splay: 0.22 }, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.1, extension: 0.35, pitch: 0.35 } }),
  // W: three extended fingers spread; pinky folds.
  W: createLetterPose({ index: { bend: [0, 0, 0], splay: -0.14 }, middle: [0, 0, 0], ring: { bend: [0, 0, 0], splay: 0.14 }, pinky: 1.65, thumb: { abduction: 0.2, extension: 0.45, pitch: 0.4 } }),
  // X: index curls into a hook; remaining fingers fold.
  X: createLetterPose({ index: [0.05, 0.75, 1.35], middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.4, extension: 0.1, pitch: 0.35 } }),
  // Y: thumb and pinky extend while the middle three fingers fold.
  Y: createLetterPose({ index: 1.65, middle: 1.65, ring: 1.65, pinky: [0, 0, 0], thumb: { abduction: 1.1, extension: 0, pitch: 0.25 } }),
  // Z: index extends; trace the Z path separately with wrist/arm keyframes.
  Z: createLetterPose({ index: [0, 0, 0], middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { abduction: 0.35, extension: 0, pitch: 0.15 }, wristRot: [0.2, -0.1, 0.25] })
};

const SAVED_POSES_STORAGE_KEY = 'mixamo-hand-pose-overrides';

function getSavedPoseOverrides() {
  try {
    const value = JSON.parse(globalThis.localStorage?.getItem(SAVED_POSES_STORAGE_KEY) || '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

export function savePoseForLetter(letter, targets) {
  const normalizedLetter = typeof letter === 'string' ? letter.trim().toUpperCase() : '';
  if (!MIXAMO_ALPHABET[normalizedLetter] || !Array.isArray(targets)) return false;

  const overrides = getSavedPoseOverrides();
  overrides[normalizedLetter] = targets
    .filter((target) => /^RightHand(?:Thumb|Index|Middle|Ring|Pinky)\d+$/.test(target.bone))
    .map(({ bone, rot }) => ({ bone, rot: [...rot] }));
  try {
    globalThis.localStorage?.setItem(SAVED_POSES_STORAGE_KEY, JSON.stringify(overrides));
    return true;
  } catch {
    return false;
  }
}

const J_LETTER_HAND = {
  index: 1.57,
  middle: 1.57,
  ring: 1.57,
  pinky: [0, 0, 0],
  thumb: { abduction: -0.05, extension: 0.6, pitch: -0.3 }
};

// The extended pinky draws down, curls around, and sweeps back to form J.
export const MIXAMO_LETTER_ANIMATIONS = {
  J: {
    durationMs: 1050,
    keyframes: [
      { t: 0.0, targets: createLetterPose({ ...J_LETTER_HAND, wristRot: [0.15, 0, 0.05] }) },
      { t: 0.28, targets: createLetterPose({ ...J_LETTER_HAND, forearmRot: [-1.35, 0, -0.15], wristRot: [0.25, -0.12, 0.35] }) },
      { t: 0.58, targets: createLetterPose({ ...J_LETTER_HAND, armRot: [0.05, 0, -1.42], forearmRot: [-1.2, -0.08, -0.12], wristRot: [0.38, -0.42, 0.42] }) },
      { t: 0.82, targets: createLetterPose({ ...J_LETTER_HAND, armRot: [-0.08, 0.05, -1.52], forearmRot: [-1.42, 0.08, -0.2], wristRot: [-0.28, -0.35, -0.18] }) },
      { t: 1.0, targets: createLetterPose({ ...J_LETTER_HAND, wristRot: [-0.12, 0.02, -0.3] }) }
    ]
  }
};

export function getPoseForLetter(char) {
  const letter = typeof char === 'string' ? char.trim().toUpperCase() : '';
  const basePose = MIXAMO_ALPHABET[letter] || createRestPose();
  const savedTargets = getSavedPoseOverrides()[letter];
  if (!Array.isArray(savedTargets)) return basePose.map((target) => ({ ...target, rot: [...target.rot] }));

  const savedRotations = new Map(savedTargets.map((target) => [target.bone, target.rot]));
  return basePose.map((target) => ({
    ...target,
    rot: [...(savedRotations.get(target.bone) || target.rot)]
  }));
}

function findBone(bonesMap, name) {
  const candidates = [name, `mixamorig${name}`, name.replace(/^mixamorig/, '')];
  for (const candidate of candidates) {
    const bone = bonesMap.get(candidate);
    if (bone) return bone;
  }
  return null;
}

export function applyMixamoPose(bonesMap, targets) {
  if (!bonesMap || !Array.isArray(targets)) return;

  for (const target of targets) {
    const bone = findBone(bonesMap, target.bone);
    if (!bone?.rotation || !Array.isArray(target.rot)) continue;
    bone.rotation.set(target.rot[0], target.rot[1], target.rot[2]);
  }
}
