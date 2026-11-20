export function setFingers(targets, prefix, flexAngle) {
  if (flexAngle?.jointRotations) {
    flexAngle.jointRotations.forEach((rot, index) => targets.push({ bone: `${prefix}${index + 1}`, rot: [...rot] }));
    return;
  }
  const bend = typeof flexAngle === 'number'
    ? [flexAngle, flexAngle, flexAngle]
    : (Array.isArray(flexAngle) ? flexAngle : (flexAngle?.bend || [0, 0, 0]));
  const splay = typeof flexAngle === 'object' && !Array.isArray(flexAngle) ? (flexAngle.splay || 0) : 0;
  for (let segment = 1; segment <= 3; segment += 1) {
    targets.push({ bone: `${prefix}${segment}`, rot: [bend[segment - 1] ?? 0, 0, segment === 1 ? splay : 0] });
  }
}

export function setThumb(targets, side = 'Right', { abduction = 0.75, extension = 0.0, pitch = 0.3, jointRotations } = {}) {
  const isRight = side === 'Right';
  const sign = isRight ? -1 : 1;
  const prefix = `${side}HandThumb`;
  if (jointRotations) {
    jointRotations.forEach((rot, index) => targets.push({ bone: `${prefix}${index + 1}`, rot: [...rot] }));
    return;
  }

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
  shoulderRot,
  armRot = [0.0, 0.0, -1.5],
  forearmRot = [-1.5, 0.0, -0.2],
  wristRot = [0.0, 0.0, 0.05]
}) {
  // Start from the natural rest rig on the left; hold the right signing arm at chest height.
  const targets = createRestPose().filter((target) => !target.bone.startsWith('Right'));
  if (shoulderRot) targets.push({ bone: 'RightShoulder', rot: shoulderRot });
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

// export const MIXAMO_TEST_POSES = {
//   A: createLetterPose({ index: 1.57, middle: 1.57, ring: 1.57, pinky: 1.57, thumb: { abduction: 0.3, extension: 0, pitch: 0.1 } }),
//   B: createLetterPose({ index: 0, middle: 0, ring: 0, pinky: 0, thumb: { abduction: -0.4, extension: 0.2, pitch: 0.05 } })
// };

const O_FOUR_FINGERS = {
  middle: { jointRotations: [[0.7, 0, 0.038], [1, 0, 0], [1.05, 0, -0.232]] },
  ring: { jointRotations: [[0.75, 0, 0], [1, 0, 0], [1, 0, -0.322]] },
  pinky: { jointRotations: [[0.8, 0, 0], [0.95, 0, 0], [0.95, 0, -0.692]] },
  thumb: { jointRotations: [[0.218, -1.27, -0.782], [0.308, 0.678, 0.398], [0.35, 0, 0.678], [0.175, 0, 0]] }
};

const D_ARM_POSE = {
  shoulderRot: [1.66, 0.22, 1.37],
  armRot: [-0.04, -0.92, -1.45],
  forearmRot: [-0.67, 1.40, -1.56],
  wristRot: [-0.03, -0.12, -0.38]
};

const I_HAND_SHAPE = {
  index: 1.5,
  middle: 1.6,
  ring: 1.55,
  pinky: [0, 0, 0],
  thumb: { jointRotations: [[-1.632, -1.512, -2.312], [0.75, 0.49, -0.02], [0.41, 0.24, 0.41], [0.3, 0, 0]] }
};

const BREVE_HAND_SHAPE = {
  index: [-0.052, -0.052, -0.142],
  middle: [0.948, 1.948, 0.858],
  ring: [1.038, 1.948, 0.308],
  pinky: { jointRotations: [[1.128, 0, 0], [1.828, 0, -0.092], [0.698, 0, -0.092]] },
  thumb: { jointRotations: [[0.398, 0.218, -0.052], [0.308, -0.052, 0.398], [0.128, 0.128, 1.218], [0.125, 0, 0]] },
  wristRot: [-0.022, 0.148, 0.148]
};

export const MIXAMO_ALPHABET = {
  A: createLetterPose({ index: 1.57, middle: 1.57, ring: 1.57, pinky: 1.57, thumb: { abduction: 0.3, extension: 0, pitch: 0.1 } }),
  B: createLetterPose({ index: { bend: [0, 0, 0], splay: -0.025 }, middle: { bend: [0, 0, 0], splay: 0 }, ring: { bend: [0, 0, 0], splay: 0.025 }, pinky: { bend: [0, 0, 0], splay: 0.05 }, thumb: { abduction: -0.4, extension: 0.2, pitch: 0.05 } }),
  C: createLetterPose({
    shoulderRot: [1.66, -0.02, 1.39],
    armRot: [-0.05, -0.99, -1.40],
    forearmRot: [-0.67, 1.40, -1.56],
    wristRot: [0.06, -0.12, -0.36],
    index: [0.55, 0.85, 0.80],
    middle: [0.60, 0.90, 0.85],
    ring: [0.65, 0.90, 0.85],
    pinky: [0.70, 0.90, 0.85],
    thumb: { jointRotations: [[0.15, -1.20, -0.87], [0.35, 0.00, -0.19], [0.35, 0.00, 0.83], [0.17, 0.00, 0.15]] }
  }),
  D: createLetterPose({
    ...D_ARM_POSE,
    ...O_FOUR_FINGERS,
    index: [0, 0, 0]
  }),
  Đ: createLetterPose({
    ...D_ARM_POSE,
    index: [-0.28, 0.79, 1.24],
    middle: { jointRotations: [[1.05, 0, 0], [1.18, 0.19, 0], [1.25, 0, -0.05]] },
    ring: { jointRotations: [[0.98, -0.96, -0.05], [1.34, 0.15, -1], [0.85, 0, -0.14]] },
    pinky: { jointRotations: [[1.05, 0.36, -0.14], [1.40, 0, 0.11], [0.36, 0.52, 0.11]] },
    thumb: { jointRotations: [[0.75, -0.95, -0.44], [0.75, -0.36, 0.49], [0.49, 0.83, 0.83], [0.15, 0, 0]] }
  }),
  E: createLetterPose({ index: { jointRotations: [[-0.052, -0.05, 0], [1.668, 0, 0], [1.578, 0, 0]] }, middle: { jointRotations: [[0.218, 0.03, 0], [1.578, 0, 0], [1.45, 0, 0]] }, ring: { jointRotations: [[0.308, 0.03, 0], [1.398, 0, 0], [1.45, 0, 0]] }, pinky: { jointRotations: [[0.308, 0.19, 0], [1.398, 0, 0], [1.45, 0, 0]] }, thumb: { jointRotations: [[1.488, -1.412, 0.398], [-0.412, 0.398, 1.038], [-0.052, 1.038, 1.758], [0.45, 0, 0]] } }),
  F: createLetterPose({
    armRot: [0.2953, -1.2631, -0.8959],
    forearmRot: [-2.307, 0.4325, -0.3541],
    wristRot: [-1.0213, 1.1743, 2.0917],
    // Rotate the pinched fingers together about the line through their base joints.
    index: { jointRotations: [[0.321, 0.1556, 0.307], [1.17, -0.47, -0.34], [1.11, 0.16, -0.03]] },
    middle: { bend: [0, 0, 0], splay: -0.07 },
    ring: [0, 0, 0],
    pinky: { bend: [0, 0, 0], splay: 0.09 },
    thumb: { jointRotations: [[0.5783, -0.2025, -0.5823], [0.61, -0.82, 0.7], [0.58, 0.26, 0.61], [0.28, 0, 0]] }
  }),
  G: createLetterPose({
    shoulderRot: [1.03, 0.25, 2.25],
    armRot: [-0.81, -1.00, -1.87],
    forearmRot: [0.00, 1.37, -2.06],
    wristRot: [1.32, 1.44, 0.05],
    index: [0.00, 0.00, 0.00],
    middle: 1.65,
    ring: 1.65,
    pinky: 1.65,
    thumb: { jointRotations: [[-0.05, 0.31, -0.69], [0.27, -0.63, 0.13], [0.03, 0.19, 0.31], [0.07, 0.00, 0.00]] }
  }),
  H: createLetterPose({
    shoulderRot: [1.38, 0.24, 1.65],
    armRot: [-0.13, -0.82, -1.57],
    forearmRot: [-1.79, 0.01, -0.19],
    wristRot: [0.03, 1.26, -0.03],
    index: { bend: [-0.49, 0, 0], splay: 0.11 },
    middle: { bend: [0.38, 0, 0], splay: 0.04 },
    ring: { jointRotations: [[1.65, -0.05, 0], [1.65, 0, 0], [1.65, 0, 0]] },
    pinky: 1.65,
    thumb: { jointRotations: [[-0.30, -0.30, -0.41], [1.67, -0.69, 0.77], [0.40, 0.68, 0.03], [0.07, 0, 0]] }
  }),
  I: createLetterPose(I_HAND_SHAPE),
  J: createLetterPose(I_HAND_SHAPE),
  K: createLetterPose({
    shoulderRot: [1.55, 0.22, 1.66],
    armRot: [-0.10, -0.80, -1.45],
    forearmRot: [-1.97, 0.02, -0.18],
    wristRot: [0.24, 1.50, -0.36],
    index: { bend: [0, 0, 0], splay: -0.14 },
    middle: { bend: [1.40, 0, 0], splay: 0.11 },
    ring: { jointRotations: [[1.40, 0.19, 0], [1.91, 0.27, 0.03], [1.09, 0, 0]] },
    pinky: { jointRotations: [[1.40, 0, 0], [2.16, 0.27, -0.22], [0.93, 0, 0]] },
    thumb: { jointRotations: [[-0.05, -0.14, 0.03], [0.10, 0, 0], [0.10, 0, 0], [0.05, 0, 0]] }
  }),
  L: createLetterPose({ index: [0, 0, 0], middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { jointRotations: [[-0.02, -0.02, -1.232], [0, 0, 0], [0, 0, 0], [0, 0, 0]] } }),
  M: createLetterPose({ index: { jointRotations: [[1.398, -0.002, 0.078], [-0.052, 0, 0], [-0.052, 0, 0]] }, middle: { jointRotations: [[1.398, -0.262, -0.002], [-0.002, -0.002, 0], [-0.002, 0, -0.002]] }, ring: [1.398, -0.002, -0.002], pinky: 1.57, thumb: { jointRotations: [[-0.052, -1.142, 0.398], [0.678, -0.412, 0], [0.488, -0.052, 0.308], [0.375, 0, 0]] }, wristRot: [1.598, 0.068, 0.05] }),
  N: createLetterPose({ index: { jointRotations: [[1.398, -0.002, 0.078], [-0.052, 0, 0], [-0.052, 0, 0]] }, middle: { jointRotations: [[1.398, -0.262, -0.002], [-0.002, -0.002, 0], [-0.002, 0, -0.002]] }, ring: [1.398, 1.758, 1.218], pinky: 1.57, thumb: { jointRotations: [[-0.052, -1.142, 0.398], [0.678, -0.412, 0], [0.488, -0.052, 0.308], [0.375, 0, 0]] }, wristRot: [1.598, 0.068, 0.05] }),
  O: createLetterPose({
    ...O_FOUR_FINGERS,
    index: { jointRotations: [[0.678, 0.218, 0.038], [0.95, 0, 0.308], [1.05, 0, 0]] }
  }),
  P: createLetterPose({ index: { bend: [0, 0, 0], splay: -0.14 }, middle: { bend: [1.398, 0, 0], splay: 0.11 }, ring: { jointRotations: [[1.398, 0.19, 0], [1.91, 0.27, 0.03], [1.09, 0, 0]] }, pinky: { jointRotations: [[1.398, 0, 0], [2.16, 0.27, -0.22], [0.93, 0, 0]] }, thumb: { jointRotations: [[-0.05, -0.14, 0.03], [0.1, 0, 0], [0.1, 0, 0], [0.05, 0, 0]] }, wristRot: [1.848, 0.068, -0.612] }),
  Q: createLetterPose({
    shoulderRot: [1.57, 0.24, 1.58],
    armRot: [-0.08, -0.14, -1.50],
    forearmRot: [-1.50, 0.00, -0.20],
    wristRot: [1.44, -0.18, -0.79],
    index: { jointRotations: [[0.42, 0.23, -0.05], [1.05, 0, 0], [0.73, 0, 0]] },
    middle: 1.65,
    ring: 1.65,
    pinky: 1.65,
    thumb: { jointRotations: [[0.54, -0.97, -0.85], [-0.05, 0, 0.42], [0.15, 0, 0.48], [0.08, 0, 0]] }
  }),
  R: createLetterPose({
    shoulderRot: [1.57, 0.24, 1.58],
    armRot: [-0.01, -0.42, -1.51],
    forearmRot: [-2.02, 0.02, -0.18],
    wristRot: [1.56, 0.00, 0.00],
    index: { jointRotations: [[0, 1.04, 0.13], [0, 0, 0], [0, 0, 0]] },
    middle: { bend: [0, 0, 0], splay: -0.23 },
    ring: 1.65,
    pinky: 1.65,
    thumb: { jointRotations: [[0.31, -0.32, 0.40], [0.13, -0.41, 0.22], [0.10, 0.04, 0], [0.05, 0, 0]] }
  }),
  S: createLetterPose({
    shoulderRot: [1.57, 0.24, 1.58],
    armRot: [-0.09, -0.52, -1.57],
    forearmRot: [-1.79, 0.01, -0.19],
    wristRot: [0.72, 1.02, -0.67],
    index: { jointRotations: [[0.98, -0.14, -0.05], [1.17, 0, 0], [0.79, 0, 0]] },
    middle: [1.85, 1.93, 1.65],
    ring: [1.85, 1.80, 1.68],
    pinky: [1.80, 2.18, 1.65],
    thumb: { jointRotations: [[0.68, -1.37, -0.38], [-0.09, 0.04, 0], [0.15, 0, 0], [0.08, 0, 0]] }
  }),
  T: createLetterPose({
    shoulderRot: [1.57, 0.24, 1.58],
    index: { jointRotations: [[0.73, 0.79, -0.34], [1.17, -0.47, -0.34], [1.11, 0.16, -0.03]] },
    middle: { bend: [-0.05, -0.05, -0.05], splay: -0.05 },
    ring: { bend: [-0.14, 0.04, -0.05], splay: 0.04 },
    pinky: { jointRotations: [[-0.05, -0.14, 0.13], [-0.05, 0, 0], [0.04, 0, 0]] },
    thumb: { jointRotations: [[1.05, 0.35, -1.23], [0.49, -1.14, 0.68], [0.58, 0.22, 0.77], [0.28, 0, 0]] }
  }),
  U: createLetterPose({ index: { jointRotations: [[-0.142, -0.052, -0.052], [0, 0, 0], [0, 0, 0]] }, middle: { jointRotations: [[-0.142, -0.092, -0.092], [0, 0, 0], [0, 0, 0]] }, ring: 1.65, pinky: 1.65, thumb: { jointRotations: [[-0.232, 0, 0.128], [1.218, -0.322, 0.678], [0.578, 0.858, -0.692], [0.175, 0, 0]] } }),
  V: createLetterPose({ index: { jointRotations: [[-0.052, -0.052, -0.322], [0, 0, 0], [0, 0, 0]] }, middle: { jointRotations: [[-0.052, -0.142, 0.128], [0, 0, 0], [0, 0, 0]] }, ring: 1.65, pinky: 1.65, thumb: { jointRotations: [[-0.232, 0, 0.128], [1.218, -0.322, 0.678], [0.578, 0.858, -0.692], [0.175, 0, 0]] } }),
  W: createLetterPose({
    index: { bend: [0, 0, 0], splay: -0.14 },
    middle: [0, 0, 0],
    ring: { bend: [0, 0, 0], splay: 0.14 },
    pinky: { jointRotations: [[1.11, 0, -0.43], [1.29, 0, 0], [0.82, 0, 0]] },
    thumb: { jointRotations: [[0.28, -0.77, 0.12], [0.85, 0, 0], [1.22, 0, 0], [0, 0, 0]] }
  }),
  X: createLetterPose({
    shoulderRot: [1.57, 0.24, 1.58],
    armRot: [-0.01, -0.30, -1.50],
    forearmRot: [-1.84, 0.02, -0.19],
    wristRot: [0.00, 1.50, 0.05],
    index: { jointRotations: [[0.16, -0.09, -0.22], [1.30, 0.10, 0], [1.68, 0.54, -0.28]] },
    middle: 1.65,
    ring: 1.65,
    pinky: 1.65,
    thumb: { jointRotations: [[0.54, 0.42, -0.28], [-0.47, -0.66, -0.09], [0.10, 0, 0], [0.05, 0, 0]] }
  }),
  Y: createLetterPose({ index: 1.65, middle: 1.65, ring: 1.65, pinky: { jointRotations: [[0, -0.002, 0.348], [0, 0, 0], [0, 0, 0]] }, thumb: { abduction: 1.142, extension: 0, pitch: -0.232 }, wristRot: [0, 0, 0.408] }),
  Z: createLetterPose({ ...BREVE_HAND_SHAPE, armRot: [0, -0.35, -1.5] }),
};

// Các pose dưới đây là cấu hình khởi đầu theo đặc tả của pilot; cần được
// tinh chỉnh/duyệt trực quan trên rig Mixamo thực tế bằng Hand Pose Tuner.
function createMarkPose({ index, middle, ring, pinky, thumb, armRot, forearmRot, wristRot }) {
  return createLetterPose({ index, middle, ring, pinky, thumb, armRot, forearmRot, wristRot });
}

export const DIACRITIC_POSES = {
  // //Dấu Â
  HAT: createMarkPose({
    index: { jointRotations: [[-0.052, -0.052, -0.322], [0.678, 0, 0], [1.948, 0, 0]] },
    middle: { bend: [0.05, 0.768, 2.128], splay: 0.06 },
    ring: [1.218, 1.668, 1.578],
    pinky: [1.128, 1.758, 1.758],
    thumb: { jointRotations: [[0.678, -0.322, 0.398], [0.038, -0.592, 0.308], [0.35, 0, 0], [0.175, 0, 0]] },
    wristRot: [0.08, 0, 0.05]
  }),
  // Dấu Ơ
  HORN: createMarkPose({
    index: [1.488, 1.218, 0.308],
    middle: [1.488, 1.398, 0.308],
    ring: [1.488, 1.398, 0.398],
    pinky: [-0.092, 0.778, 1.738],
    thumb: { jointRotations: [[0.398, 0.218, -0.502], [0.308, -0.052, 0.398], [0.128, 0.128, 1.218], [0.125, 0, 0]] },
    wristRot: [-0.022, 0.148, 0.148]
  }),
  // Dấu Ă
  BREVE: createMarkPose(BREVE_HAND_SHAPE),
  POINTING_POSE: createMarkPose({
    index: [0, 0, 0], middle: [1.55, 1.6, 1.55],
    ring: [1.6, 1.6, 1.55], pinky: [1.55, 1.6, 1.55],
    thumb: { abduction: 0.25, extension: 0.15, pitch: 0.35 },
    wristRot: [0, 0, 0]
  })
};

const VIETNAMESE_LETTER_MARKS = {
  'Ă': { base: 'A', mark: 'BREVE' },
  'Â': { base: 'A', mark: 'HAT' },
  'Ê': { base: 'E', mark: 'HAT' },
  'Ô': { base: 'O', mark: 'HAT' },
  'Ơ': { base: 'O', mark: 'HORN' },
  'Ư': { base: 'U', mark: 'HORN' }
};

const HEAVY_TONE_POSE = createLetterPose({
  shoulderRot: [1.57, 0.24, 1.58],
  forearmRot: [-0.32, 1.36, -1.19],
  index: 1.57,
  middle: 1.57,
  ring: 1.57,
  pinky: 0,
  thumb: { jointRotations: [[0.25, 0.31, -0.61], [0.19, 0.44, 0.68], [0.25, -0.49, 0.93], [0.28, 0, 0]] },
  wristRot: [-0.02, 0.15, 1.44]
});

// Local rotation offsets calibrated against the Mixamo skeleton.
export const TONE_TRAJECTORIES = {
  // Longer diagonal down and outward, with wrist compensation to retain palm direction.
  HUYEN: [
    { t: 0, arm: [-0.07, 0.181, -0.06], forearm: [0.196, -0.051, -0.04], wrist: [0, 0, 0] },
    { t: 0.18, arm: [-0.07, 0.181, -0.06], forearm: [0.196, -0.051, -0.04], wrist: [0, 0, 0] },
    { t: 0.45, arm: [0.002, -0.012, 0.052], forearm: [-0.021, -0.003, 0.028], wrist: [0.031, 0.063, -0.147] },
    { t: 0.68, arm: [0.082, -0.145, 0.131], forearm: [-0.132, 0.017, 0.125], wrist: [0.029, 0.11, -0.306] },
    { t: 0.84, arm: [0.146, -0.214, 0.179], forearm: [-0.171, 0.019, 0.203], wrist: [0.02, 0.146, -0.432] },
    { t: 1, arm: [0.146, -0.214, 0.179], forearm: [-0.171, 0.019, 0.203], wrist: [0.02, 0.146, -0.432] }
  ],
  // QIPEDC D0525N, 4.15–4.67 s: the index points across the body, then turns up and out with the palm.
  SAC: [
    { t: 0, arm: [0, 0, 0], forearm: [0, 0, 0], wrist: [0, 0, 0] },
    { t: 0.18, arm: [0, 0, 0], forearm: [0, 0, 0], wrist: [0, 0, 0] },
    { t: 0.4, arm: [0.044, 0.128, -0.004], forearm: [-0.967, -0.367, 1.046], wrist: [-0.583, -0.435, 0.392] },
    { t: 0.62, arm: [0.067, 0.276, 0.038], forearm: [-0.993, -0.897, 1.177], wrist: [-1.13, -0.725, 0.433] },
    { t: 0.82, arm: [0.096, 0.421, 0.056], forearm: [-0.903, -1.39, 1.201], wrist: [-1.734, -0.885, 0.581] },
    { t: 1, arm: [0.096, 0.421, 0.056], forearm: [-0.903, -1.39, 1.201], wrist: [-1.734, -0.885, 0.581] }
  ],
  // D0526N, 1.19–2.35 s: rise, round the top, and drop.
  HOI: [
    { t: 0, arm: [-0.023, -0.062, -0.037], forearm: [-0.053, 0.013, -0.005] },
    { t: 0.16, arm: [-0.023, -0.062, -0.037], forearm: [-0.053, 0.013, -0.005] },
    { t: 0.34, arm: [-0.065, 0.148, -0.105], forearm: [0.171, -0.035, -0.017] },
    { t: 0.5, arm: [-0.038, 0.292, -0.064], forearm: [0.334, -0.073, -0.015] },
    { t: 0.65, arm: [0.016, 0.23, -0.01], forearm: [0.261, -0.056, 0.044] },
    { t: 0.84, arm: [0.035, -0.034, 0.044], forearm: [-0.032, 0, 0.016] },
    { t: 0.94, arm: [0.062, -0.178, 0.083], forearm: [-0.165, 0.025, 0.027] },
    { t: 1, arm: [0.062, -0.178, 0.083], forearm: [-0.165, 0.025, 0.027] }
  ],
  // D0527N, 1.20–2.22 s: small dip between two rising segments.
  NGA: [
    { t: 0, arm: [-0.035, -0.114, -0.037], forearm: [-0.1, 0.022, -0.022] },
    { t: 0.16, arm: [-0.035, -0.114, -0.037], forearm: [-0.1, 0.022, -0.022] },
    { t: 0.34, arm: [-0.068, 0.124, -0.107], forearm: [0.145, -0.039, -0.057] },
    { t: 0.51, arm: [-0.016, -0.063, -0.014], forearm: [-0.059, 0.013, -0.015] },
    { t: 0.67, arm: [0.015, 0.173, 0.009], forearm: [0.182, -0.038, -0.001] },
    { t: 0.84, arm: [0.044, 0.307, 0.058], forearm: [0.345, -0.07, 0.014] },
    { t: 1, arm: [0.044, 0.307, 0.058], forearm: [0.345, -0.07, 0.014] }
  ],
  // QIPEDC D0528, 1.07–1.89 s: the little fingertip leads the forward press.
  NANG: [
    { t: 0, arm: [0, 0, 0], forearm: [0, 0, 0], wrist: [0, 0, 0] },
    { t: 0.24, arm: [0, 0, 0], forearm: [0, 0, 0], wrist: [0, 0, 0] },
    { t: 0.44, arm: [-0.003, 0.007, -0.004], forearm: [0.05, -0.005, 0.05], wrist: [0.003, -0.008, -0.091] },
    { t: 0.55, arm: [-0.004, 0.018, -0.007], forearm: [0.087, -0.008, 0.088], wrist: [0.005, -0.013, -0.156] },
    { t: 0.68, arm: [-0.006, 0.032, -0.01], forearm: [0.124, -0.011, 0.126], wrist: [0.006, -0.019, -0.216] },
    { t: 1, arm: [-0.006, 0.032, -0.01], forearm: [0.124, -0.011, 0.126], wrist: [0.006, -0.019, -0.216] }
  ]
};

const TONE_DURATIONS = { HUYEN: 0.65, SAC: 0.62, HOI: 0.95, NGA: 0.85, NANG: 0.55 };

function armTrajectory(pose, frames) {
  const byBone = new Map(pose.map((target) => [target.bone, target.rot]));
  const offsetsFor = (bone, offset) => (
    offset ? [{ bone, rot: byBone.get(bone).map((angle, axis) => angle + offset[axis]) }] : []
  );
  return frames.map(({ t, arm, forearm, wrist }) => ({
    t,
    targets: [
      ...offsetsFor('RightArm', arm),
      ...offsetsFor('RightForeArm', forearm),
      ...offsetsFor('RightHand', wrist)
    ]
  }));
}

const TONE_NAMES = {
  TONE_HUYEN: 'HUYEN', TONE_SAC: 'SAC', TONE_HOI: 'HOI', TONE_NGA: 'NGA', TONE_NANG: 'NANG'
};

const TONE_LABELS = {
  'DẤU HUYỀN': 'TONE_HUYEN', 'DẤU SẮC': 'TONE_SAC', 'DẤU HỎI': 'TONE_HOI',
  'DẤU NGÃ': 'TONE_NGA', 'DẤU NẶNG': 'TONE_NANG'
};

const TONE_DESCRIPTIONS = {
  TONE_HUYEN: {
    handshape: 'Giữ hình dạng bàn tay và hướng lòng bàn tay trong suốt nét.',
    motion: 'Từ trước thân, kéo tay theo một nét chéo dài đi xuống và ra ngoài.'
  },
  TONE_SAC: {
    handshape: 'Ngón trỏ bắt đầu chỉ ngang về phía giữa thân, lòng bàn tay úp và hướng vào thân.',
    motion: 'Đưa tay chéo lên ra ngoài, đồng thời xoay lòng bàn tay ra trước và hướng ngón trỏ chéo lên theo nét.'
  },
  TONE_HOI: {
    handshape: 'Giữ hình dạng bàn tay khi vẽ nét móc.',
    motion: 'Đưa tay lên, lượn qua đỉnh rồi kéo đuôi nét dài xuống và hơi chếch ra ngoài.'
  },
  TONE_NGA: {
    handshape: 'Giữ hình dạng bàn tay trong suốt nét.',
    motion: 'Đưa tay lên thành một nét cong, hạ xuống rõ ở giữa rồi lượn lên lần nữa.'
  },
  TONE_NANG: {
    handshape: 'Ngón út duỗi thẳng. Ngón cái khép cong, ngón trỏ, giữa và áp út gập. Giữ nguyên hình dạng tay suốt nét.',
    motion: 'Dùng cánh tay đẩy đầu ngón út thẳng ra trước, vuông góc với thân. Cổ tay giữ hướng bàn tay.'
  }
};

export function getMixamoToneSymbol(token) {
  return TONE_LABELS[token] ?? token;
}

export function getMixamoToneDescription(token) {
  const symbol = getMixamoToneSymbol(token);
  const tone = splitTonedLetter(symbol)?.tone ?? symbol;
  return Object.hasOwn(TONE_DESCRIPTIONS, tone) ? TONE_DESCRIPTIONS[tone] : undefined;
}

const VIETNAMESE_SYMBOL_OPTIONS = {
  ...Object.fromEntries([...`ABCDEFGHIKLMNOPQRSTUVWXYZ`].map((letter) => [letter, letter])),
  'Ă': 'Ă', 'Â': 'Â', 'Đ': 'Đ', 'Ê': 'Ê', 'Ô': 'Ô', 'Ơ': 'Ơ', 'Ư': 'Ư',
  'Ắ': 'Ắ',
  'Dấu Mũ (^)': 'HAT', "Dấu Móc (')": 'HORN', 'Dấu Á (˘)': 'BREVE', 'Dấu Chỉ Nét': 'POINTING_POSE',
  'Dấu Sắc': 'TONE_SAC', 'Dấu Huyền': 'TONE_HUYEN', 'Dấu Hỏi': 'TONE_HOI', 'Dấu Ngã': 'TONE_NGA', 'Dấu Nặng': 'TONE_NANG'
};

const copyTargets = (targets) => targets.map((target) => ({
  ...target,
  rot: [...target.rot],
  ...(target.pos ? { pos: [...target.pos] } : {})
}));

const SAVED_POSES_STORAGE_KEY = 'mixamo-hand-pose-overrides';
const TUNABLE_SYMBOLS = new Set(Object.values(VIETNAMESE_SYMBOL_OPTIONS));

function isEditableTarget(target) {
  return /^(?:Right|Left)(?:Shoulder|Arm|ForeArm|Hand(?:(?:Thumb|Index|Middle|Ring|Pinky)\d+)?)$/.test(target?.bone)
    && Array.isArray(target.rot) && target.rot.length === 3 && target.rot.every(Number.isFinite);
}

function readSavedPoseOverrides(storage) {
  const value = JSON.parse(storage.getItem(SAVED_POSES_STORAGE_KEY) || '{}');
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new TypeError('Invalid saved pose overrides');
  return value;
}

export function getSavedPoseOverrides() {
  try {
    return readSavedPoseOverrides(globalThis.localStorage);
  } catch {
    return {};
  }
}

export const MIXAMO_LETTER_ANIMATIONS = {
  J: {
    durationMs: 1050,
    keyframes: [
      { t: 0.0, targets: createLetterPose({ ...I_HAND_SHAPE, wristRot: [0.15, 0, 0.05] }) },
      { t: 0.28, targets: createLetterPose({ ...I_HAND_SHAPE, forearmRot: [-1.35, 0, -0.15], wristRot: [0.25, -0.12, 0.35] }) },
      { t: 0.58, targets: createLetterPose({ ...I_HAND_SHAPE, armRot: [0.05, 0, -1.42], forearmRot: [-1.2, -0.08, -0.12], wristRot: [0.38, -0.42, 0.42] }) },
      { t: 0.82, targets: createLetterPose({ ...I_HAND_SHAPE, armRot: [-0.08, 0.05, -1.52], forearmRot: [-1.42, 0.08, -0.2], wristRot: [-0.28, -0.35, -0.18] }) },
      { t: 1.0, targets: createLetterPose({ ...I_HAND_SHAPE, wristRot: [-0.12, 0.02, -0.3] }) }
    ]
  },
  // Start toward the avatar's center and draw the top stroke outward.
  Z: {
    durationMs: 1500,
    keyframes: [
      { t: 0, arm: [-0.061, 0.136, -0.093], forearm: [0.148, -0.027, -0.058] },
      { t: 0.16, arm: [-0.061, 0.136, -0.093], forearm: [0.148, -0.027, -0.058] },
      { t: 0.28, arm: [-0.005, 0.153, -0.032], forearm: [0.157, -0.027, -0.002] },
      { t: 0.40, arm: [0.043, 0.204, 0.031], forearm: [0.212, -0.029, 0.052] },
      { t: 0.54, arm: [0, 0, 0], forearm: [0, 0, 0] },
      { t: 0.68, arm: [-0.062, -0.167, -0.004], forearm: [-0.151, 0.045, -0.078] },
      { t: 0.78, arm: [0.015, -0.154, 0.03], forearm: [-0.146, 0.023, 0.004] },
      { t: 0.88, arm: [0.086, -0.097, 0.063], forearm: [-0.09, 0.001, 0.079] },
      { t: 1, arm: [0.086, -0.097, 0.063], forearm: [-0.09, 0.001, 0.079] }
    ]
  },
  // Rotations relative to BREVE, calibrated on the Mixamo rig from the front.
  BREVE_ARC: {
    durationMs: 620,
    keyframes: [
      { t: 0, arm: [0, 0, 0], forearm: [0, 0, 0] },
      { t: 0.125, arm: [0.016, -0.01, 0], forearm: [-0.01, 0.001, 0] },
      { t: 0.25, arm: [0.064, -0.036, 0], forearm: [-0.036, 0.004, 0] },
      { t: 0.375, arm: [0.071, -0.055, 0.06], forearm: [-0.056, 0.029, 0] },
      { t: 0.5, arm: [0.08, -0.062, 0.06], forearm: [-0.062, 0, 0.06] },
      { t: 0.625, arm: [0.133, -0.023, 0.061], forearm: [-0.023, 0.001, 0.06] },
      { t: 0.75, arm: [0.096, 0.029, 0.12], forearm: [0.031, -0.02, 0.065] },
      { t: 0.875, arm: [0.113, 0.074, 0.121], forearm: [0.078, -0.033, 0.062] },
      { t: 1, arm: [0.12, 0.091, 0.12], forearm: [0.097, -0.032, 0.06] }
    ]
  }
};

function withSavedOverrides(symbol, basePose) {
  const savedTargets = getSavedPoseOverrides()[symbol];
  if (!Array.isArray(savedTargets)) return copyTargets(basePose);
  const targets = new Map(copyTargets(basePose).map((target) => [target.bone, target]));
  for (const target of savedTargets) {
    if (isEditableTarget(target)) {
      targets.set(target.bone, { ...targets.get(target.bone), bone: target.bone, rot: [...target.rot] });
    }
  }
  return [...targets.values()];
}

const TONE_BY_MARK = {
  '\u0300': 'TONE_HUYEN',
  '\u0301': 'TONE_SAC',
  '\u0309': 'TONE_HOI',
  '\u0303': 'TONE_NGA',
  '\u0323': 'TONE_NANG'
};

function splitTonedLetter(symbol) {
  const parts = [...symbol.normalize('NFD')];
  if (!/^[A-Z]$/.test(parts[0] ?? '') || parts.slice(1).some((part) => !/\p{M}/u.test(part))) return null;
  const tone = parts.find((part) => Object.hasOwn(TONE_BY_MARK, part));
  if (!tone) return null;
  return { letter: parts.filter((part) => part !== tone).join('').normalize('NFC'), tone: TONE_BY_MARK[tone] };
}

export function getPoseForSymbol(value) {
  const symbol = typeof value === 'string' ? getMixamoToneSymbol(value.trim().normalize('NFC').toUpperCase()) : '';
  if (MIXAMO_ALPHABET[symbol]) return withSavedOverrides(symbol, MIXAMO_ALPHABET[symbol]);
  if (DIACRITIC_POSES[symbol]) return withSavedOverrides(symbol, DIACRITIC_POSES[symbol]);
  if (symbol === 'Ă') return withSavedOverrides(symbol, getPoseForSymbol('A'));
  if (VIETNAMESE_LETTER_MARKS[symbol]) {
    return withSavedOverrides(symbol, MIXAMO_ALPHABET[VIETNAMESE_LETTER_MARKS[symbol].base]);
  }
  const toned = splitTonedLetter(symbol);
  if (toned) return withSavedOverrides(symbol, getPoseForSymbol(toned.letter));
  if (symbol === 'TONE_NANG') return withSavedOverrides(symbol, HEAVY_TONE_POSE);
  if (symbol === 'TONE_SAC') {
    const startRotations = {
      RightArm: [-0.043, -0.103, -1.538],
      RightForeArm: [-0.232, 1.33, -1.369],
      RightHand: [1.709, 1.119, -0.229]
    };
    const pose = getPoseForSymbol('BREVE').map((target) => ({
      ...target, rot: startRotations[target.bone] ?? target.rot
    }));
    return withSavedOverrides(symbol, pose);
  }
  if (TONE_NAMES[symbol]) return withSavedOverrides(symbol, getPoseForSymbol('BREVE'));
  return withSavedOverrides(symbol, createRestPose());
}

// Giữ tên cũ để các caller hiện có tương thích, đồng thời chấp nhận Unicode.
export function getPoseForLetter(char) {
  return getPoseForSymbol(char);
}

const poseAction = (data, duration = 0.4, label = 'pose') => ({ type: 'pose', label, data: copyTargets(data), duration });

export function resolveVietnameseSign(token, { hasNextStep = false } = {}) {
  const actions = resolveSignActions(token);
  return hasNextStep && actions.at(-1)?.label === 'REST' ? actions.slice(0, -1) : actions;
}

/** Expand only poses this Mixamo renderer can play, preserving Vietnamese marks. */
export function createMixamoPlayback(steps) {
  return steps.flatMap((step) => {
    const spellingSymbol = step.type === 'fingerspelling' ? step.token || step.gloss : '';
    const text = String(step.avatar_symbol || spellingSymbol || step.word || step.label || step.token || step.gloss || '')
      .normalize('NFC').trim().replace(/^ALPHA_/i, '');
    const symbols = resolveVietnameseSign(text).length ? [text] : [...text];
    return symbols.flatMap((value) => {
      const symbol = getMixamoToneSymbol(value.normalize('NFC').toUpperCase());
      if (!resolveVietnameseSign(symbol).length) return [];
      return [{
        token: symbol, gloss: symbol, avatar_symbol: symbol,
        label: symbol.startsWith('TONE_') ? step.label || symbol : `Chữ ${symbol}`,
        type: 'fingerspelling', word: text, has_pose: true,
        duration_ms: 800,
        description: `Đánh vần “${text}” bằng ký tự ${symbol}`
      }];
    });
  });
}

function resolveSignActions(token) {
  const normalized = typeof token === 'string' ? getMixamoToneSymbol(token.trim().normalize('NFC').toUpperCase()) : '';
  if (!normalized) return [];

  const toned = splitTonedLetter(normalized);
  if (toned) {
    const letterActions = resolveVietnameseSign(toned.letter);
    if (letterActions[0]?.type === 'pose') {
      letterActions[0].data = withSavedOverrides(normalized, letterActions[0].data);
    }
    return letterActions.length
      ? [...letterActions.filter(({ label }) => label !== 'REST'), ...resolveVietnameseSign(toned.tone)]
      : [];
  }

  const tone = TONE_NAMES[normalized];
  if (tone) {
    const pose = getPoseForSymbol(normalized);
    return [
      {
        type: 'motion', label: normalized, pose,
        trajectory: armTrajectory(pose, TONE_TRAJECTORIES[tone]), duration: TONE_DURATIONS[tone]
      },
      poseAction(createRestPose(), 0.35, 'REST')
    ];
  }

  if (normalized === 'Ă') {
    const pose = getPoseForSymbol('BREVE');
    const { durationMs, keyframes } = MIXAMO_LETTER_ANIMATIONS.BREVE_ARC;
    const trajectory = armTrajectory(pose, keyframes);
    const finalJoints = new Map(trajectory.at(-1).targets.map((target) => [target.bone, target]));
    return [
      poseAction(getPoseForSymbol(normalized), 0.65, 'A'),
      poseAction(pose, 0.22, 'BREVE_START'),
      { type: 'motion', label: 'BREVE', pose, duration: durationMs / 1000, trajectory },
      poseAction(pose.map((target) => finalJoints.get(target.bone) ?? target), 0.18, 'BREVE_END'),
      poseAction(createRestPose(), 0.12, 'REST')
    ];
  }

  const combined = VIETNAMESE_LETTER_MARKS[normalized];
  if (combined) {
    const baseDuration = normalized === 'Ô' || normalized === 'Ơ' ? 0.65 : 0.4;
    return [
      poseAction(getPoseForSymbol(normalized), baseDuration, combined.base),
      poseAction(getPoseForSymbol(combined.mark), 0.5, combined.mark),
      poseAction(createRestPose(), 0.35, 'REST')
    ];
  }

  if (normalized === 'Đ') {
    return [poseAction(getPoseForSymbol(normalized), 0.45, 'Đ')];
  }

  if (normalized === 'J') {
    const { durationMs, keyframes } = MIXAMO_LETTER_ANIMATIONS.J;
    return [{ type: 'keyframes', label: 'J', keyframes, duration: durationMs / 1000 }];
  }

  if (normalized === 'Z') {
    const pose = getPoseForSymbol(normalized);
    const { durationMs, keyframes } = MIXAMO_LETTER_ANIMATIONS.Z;
    return [{ type: 'motion', label: 'Z', pose, trajectory: armTrajectory(pose, keyframes), duration: durationMs / 1000 }];
  }

  if (MIXAMO_ALPHABET[normalized]) {
    return [poseAction(getPoseForLetter(normalized), 0.4, normalized)];
  }
  if (DIACRITIC_POSES[normalized]) {
    return [poseAction(getPoseForSymbol(normalized), 0.4, normalized)];
  }
  const saved = getSavedPoseOverrides()[normalized];
  if (TUNABLE_SYMBOLS.has(normalized) && Array.isArray(saved) && saved.some(isEditableTarget)) {
    return [poseAction(getPoseForSymbol(normalized), 0.4, normalized)];
  }
  return [];
}

export function getMixamoBoneName(name) {
  return name.split(/[|:]/).at(-1)
    .replace(/^mixamorig\d*_?/i, '')
    .replace(/^(Right|Left)(Clavicle|UpperArm|LowerArm)$/, (_, side, joint) => (
      side + { Clavicle: 'Shoulder', UpperArm: 'Arm', LowerArm: 'ForeArm' }[joint]
    ));
}

export function findMixamoBone(bonesMap, name) {
  return bonesMap?.get(name) ?? bonesMap?.get(getMixamoBoneName(name)) ?? bonesMap?.get(`mixamorig${name}`) ?? null;
}

export function applyMixamoPose(bonesMap, targets) {
  if (!bonesMap || !Array.isArray(targets)) return;

  for (const target of targets) {
    const bone = findMixamoBone(bonesMap, target.bone);
    if (!bone?.rotation || !Array.isArray(target.rot)) continue;
    bone.rotation.set(target.rot[0], target.rot[1], target.rot[2]);
  }
}
