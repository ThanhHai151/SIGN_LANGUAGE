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

// export const MIXAMO_TEST_POSES = {
//   A: createLetterPose({ index: 1.57, middle: 1.57, ring: 1.57, pinky: 1.57, thumb: { abduction: 0.3, extension: 0, pitch: 0.1 } }),
//   B: createLetterPose({ index: 0, middle: 0, ring: 0, pinky: 0, thumb: { abduction: -0.4, extension: 0.2, pitch: 0.05 } })
// };

export const MIXAMO_ALPHABET = {
  A: createLetterPose({ index: 1.57, middle: 1.57, ring: 1.57, pinky: 1.57, thumb: { abduction: 0.3, extension: 0, pitch: 0.1 } }),
  B: createLetterPose({ index: { bend: [0, 0, 0], splay: -0.025 }, middle: { bend: [0, 0, 0], splay: 0 }, ring: { bend: [0, 0, 0], splay: 0.025 }, pinky: { bend: [0, 0, 0], splay: 0.05 }, thumb: { abduction: -0.4, extension: 0.2, pitch: 0.05 } }),
  C: createLetterPose({ index: [0.55, 0.85, 0.8], middle: [0.6, 0.9, 0.85], ring: [0.65, 0.9, 0.85], pinky: [0.7, 0.9, 0.85], thumb: { jointRotations: [[0.15, -1.2, -0.87], [0.35, 0, -0.19], [0.35, 0, 0.83], [0.17, 0, 0.15]] } }),
  D: createLetterPose({ index: [0, 0, 0], middle: { jointRotations: [[0.77, 0, 0], [1.18, 0.19, 0], [1.25, 0, -0.05]] }, ring: { jointRotations: [[0.68, -0.96, -0.05], [1.34, 0.15, -1], [0.85, 0, -0.14]] }, pinky: { jointRotations: [[0.77, 0.36, -0.14], [1.4, 0, 0.11], [0.36, 0.52, 0.11]] }, thumb: { jointRotations: [[0.75, -0.95, -0.44], [0.75, -0.36, 0.49], [0.49, 0.83, 0.83], [0.15, 0, 0]] } }),
  Đ: createLetterPose({ index: [-0.052, 0.858, 1.128], middle: { jointRotations: [[1.048, 0, 0], [1.478, -0.092, 0], [1.568, 0, -0.05]] }, ring: { jointRotations: [[1.048, -0.96, -0.05], [1.308, 0.15, -1], [1.308, 0, -0.14]] }, pinky: { jointRotations: [[1.048, 0.36, -0.14], [1.308, 0.078, 0.11], [1.048, 0.52, 0.11]] }, thumb: { jointRotations: [[0.75, -0.95, -0.44], [0.75, -0.36, 0.49], [0.49, 0.83, 0.83], [0.15, 0, 0]] } }),
  E: createLetterPose({ index: { jointRotations: [[-0.052, -0.05, 0], [1.668, 0, 0], [1.578, 0, 0]] }, middle: { jointRotations: [[0.218, 0.03, 0], [1.578, 0, 0], [1.45, 0, 0]] }, ring: { jointRotations: [[0.308, 0.03, 0], [1.398, 0, 0], [1.45, 0, 0]] }, pinky: { jointRotations: [[0.308, 0.19, 0], [1.398, 0, 0], [1.45, 0, 0]] }, thumb: { jointRotations: [[1.488, -1.412, 0.398], [-0.412, 0.398, 1.038], [-0.052, 1.038, 1.758], [0.45, 0, 0]] } }),
  G: createLetterPose({ index: [-0.002, 0, 0], middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { jointRotations: [[-0.052, 0.308, -0.692], [0.27, -0.63, 0.128], [0.03, 0.19, 0.308], [0.07, 0, 0]] }, wristRot: [0, 0.408, 0.05] }),
  H: createLetterPose({ index: { bend: [-0.322, 0, 0], splay: 0.11 }, middle: { bend: [0.308, 0, 0], splay: 0.04 }, ring: { jointRotations: [[1.65, -0.052, 0], [1.65, 0, 0], [1.65, 0, 0]] }, pinky: 1.65, thumb: { jointRotations: [[0.398, -1.052, -0.412], [1.668, -0.692, 0.768], [0.398, 0.678, 0.03], [0.07, 0, 0]] }, wristRot: [0, -0.55, 0.05] }),
  I: createLetterPose({ index: 1.5, middle: 1.6, ring: 1.55, pinky: [0, 0, 0], thumb: { jointRotations: [[-1.632, -1.512, -2.312], [0.75, 0.49, -0.02], [0.41, 0.24, 0.41], [0.3, 0, 0]] } }),
  K: createLetterPose({ index: { bend: [0, 0, 0], splay: -0.14 }, middle: { bend: [1.398, 0, 0], splay: 0.11 }, ring: { jointRotations: [[1.398, 0.19, 0], [1.91, 0.27, 0.03], [1.09, 0, 0]] }, pinky: { jointRotations: [[1.398, 0, 0], [2.16, 0.27, -0.22], [0.93, 0, 0]] }, thumb: { jointRotations: [[-0.05, -0.14, 0.03], [0.1, 0, 0], [0.1, 0, 0], [0.05, 0, 0]] } }),
  L: createLetterPose({ index: [0, 0, 0], middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { jointRotations: [[-0.02, -0.02, -1.232], [0, 0, 0], [0, 0, 0], [0, 0, 0]] } }),
  M: createLetterPose({ index: { jointRotations: [[1.398, -0.002, 0.078], [-0.052, 0, 0], [-0.052, 0, 0]] }, middle: { jointRotations: [[1.398, -0.262, -0.002], [-0.002, -0.002, 0], [-0.002, 0, -0.002]] }, ring: [1.398, -0.002, -0.002], pinky: 1.57, thumb: { jointRotations: [[-0.052, -1.142, 0.398], [0.678, -0.412, 0], [0.488, -0.052, 0.308], [0.375, 0, 0]] }, wristRot: [1.598, 0.068, 0.05] }),
  N: createLetterPose({ index: { jointRotations: [[1.398, -0.002, 0.078], [-0.052, 0, 0], [-0.052, 0, 0]] }, middle: { jointRotations: [[1.398, -0.262, -0.002], [-0.002, -0.002, 0], [-0.002, 0, -0.002]] }, ring: [1.398, 1.758, 1.218], pinky: 1.57, thumb: { jointRotations: [[-0.052, -1.142, 0.398], [0.678, -0.412, 0], [0.488, -0.052, 0.308], [0.375, 0, 0]] }, wristRot: [1.598, 0.068, 0.05] }),
  O: createLetterPose({ index: { jointRotations: [[0.678, 0.218, 0.038], [0.95, 0, 0.308], [1.05, 0, 0]] }, middle: { jointRotations: [[0.7, 0, 0.038], [1, 0, 0], [1.05, 0, -0.232]] }, ring: { jointRotations: [[0.75, 0, 0], [1, 0, 0], [1, 0, -0.322]] }, pinky: { jointRotations: [[0.8, 0, 0], [0.95, 0, 0], [0.95, 0, -0.692]] }, thumb: { jointRotations: [[0.218, -1.27, -0.782], [0.308, 0.678, 0.398], [0.35, 0, 0.678], [0.175, 0, 0]] } }),
  P: createLetterPose({ index: { bend: [0, 0, 0], splay: -0.14 }, middle: { bend: [1.398, 0, 0], splay: 0.11 }, ring: { jointRotations: [[1.398, 0.19, 0], [1.91, 0.27, 0.03], [1.09, 0, 0]] }, pinky: { jointRotations: [[1.398, 0, 0], [2.16, 0.27, -0.22], [0.93, 0, 0]] }, thumb: { jointRotations: [[-0.05, -0.14, 0.03], [0.1, 0, 0], [0.1, 0, 0], [0.05, 0, 0]] }, wristRot: [1.848, 0.068, -0.612] }),
  Q: createLetterPose({ index: { jointRotations: [[0.308, -0.142, -0.052], [0.858, 0, 0], [0.948, 0, 0]] }, middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { jointRotations: [[0.398, -1.55, -0.692], [-0.052, 0, 0.218], [0.15, 0, 0.398], [0.075, 0, 0]] }, wristRot: [1.508, 0.488, -0.192] }),
  R: createLetterPose({ index: { jointRotations: [[0, 1.038, 0.128], [0, 0, 0], [0, 0, 0]] }, middle: { bend: [0, 0, 0], splay: -0.232 }, ring: 1.65, pinky: 1.65, thumb: { jointRotations: [[0.308, -0.322, 0.398], [0.128, -0.412, 0.218], [0.1, 0.038, 0], [0.05, 0, 0]] }, wristRot: [1.428, 0, 0.05] }),
  S: createLetterPose({ index: { jointRotations: [[1.128, -0.142, -0.052], [1.308, 0, 0], [0.768, 0, 0]] }, middle: [1.848, 1.65, 1.65], ring: [1.848, 1.65, 1.65], pinky: [1.848, 1.65, 1.65], thumb: { jointRotations: [[0.678, -1.37, -0.383], [-0.052, 0.038, 0], [0.15, 0, 0], [0.075, 0, 0]] }, wristRot: [0.578, 0, 0.05] }),
  T: createLetterPose({ index: { jointRotations: [[0.678, 0.868, -0.352], [1.218, 0.078, 0.608], [0.948, 0.428, -0.002]] }, middle: { bend: [-0.052, -0.052, -0.052], splay: -0.052 }, ring: { bend: [-0.142, 0.038, -0.052], splay: 0.038 }, pinky: { jointRotations: [[-0.052, -0.142, 0.128], [-0.052, 0, 0], [0.038, 0, 0]] }, thumb: { jointRotations: [[0.678, -0.412, -0.592], [0.488, -1.142, 0.678], [0.578, 0.218, 0.768], [0.275, 0, 0]] } }),
  U: createLetterPose({ index: { jointRotations: [[-0.142, -0.052, -0.052], [0, 0, 0], [0, 0, 0]] }, middle: { jointRotations: [[-0.142, -0.092, -0.092], [0, 0, 0], [0, 0, 0]] }, ring: 1.65, pinky: 1.65, thumb: { jointRotations: [[-0.232, 0, 0.128], [1.218, -0.322, 0.678], [0.578, 0.858, -0.692], [0.175, 0, 0]] } }),
  V: createLetterPose({ index: { jointRotations: [[-0.052, -0.052, -0.322], [0, 0, 0], [0, 0, 0]] }, middle: { jointRotations: [[-0.052, -0.142, 0.128], [0, 0, 0], [0, 0, 0]] }, ring: 1.65, pinky: 1.65, thumb: { jointRotations: [[-0.232, 0, 0.128], [1.218, -0.322, 0.678], [0.578, 0.858, -0.692], [0.175, 0, 0]] } }),
  X: createLetterPose({ index: { jointRotations: [[0.308, 0.038, -0.322], [1.398, 0, 0], [1.488, 0, 0]] }, middle: 1.65, ring: 1.65, pinky: 1.65, thumb: { jointRotations: [[-0.232, -1.142, -0.4], [-0.142, 1.398, 0], [0.1, 0, 0], [0.05, 0, 0]] } }),
  Y: createLetterPose({ index: 1.65, middle: 1.65, ring: 1.65, pinky: { jointRotations: [[0, -0.002, 0.348], [0, 0, 0], [0, 0, 0]] }, thumb: { abduction: 1.142, extension: 0, pitch: -0.232 }, wristRot: [0, 0, 0.408] }),
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
  BREVE: createMarkPose({
    index: [-0.052, -0.052, -0.142],
    middle: [0.948, 1.948, 0.858],
    ring: [1.038, 1.948, 0.308],
    pinky: { jointRotations: [[1.128, 0, 0], [1.828, 0, -0.092], [0.698, 0, -0.092]] },
    thumb: { jointRotations: [[0.398, 0.218, -0.052], [0.308, -0.052, 0.398], [0.128, 0.128, 1.218], [0.125, 0, 0]] },
    wristRot: [-0.022, 0.148, 0.148]
  }),
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

// Relative shoulder and elbow angles fitted to front-view motion on the Mixamo rig.
export const TONE_TRAJECTORIES = {
  // D0524N, 4.02–4.62 s: down and outward.
  HUYEN: [
    { t: 0, arm: [-0.07, 0.181, -0.06], forearm: [0.196, -0.051, -0.04] },
    { t: 0.18, arm: [-0.07, 0.181, -0.06], forearm: [0.196, -0.051, -0.04] },
    { t: 0.45, arm: [-0.044, 0.125, -0.016], forearm: [0.133, -0.037, -0.011] },
    { t: 0.68, arm: [0.013, 0.031, 0.037], forearm: [0.032, -0.017, 0.039] },
    { t: 0.84, arm: [0.08, -0.048, 0.087], forearm: [-0.043, -0.007, 0.089] },
    { t: 1, arm: [0.08, -0.048, 0.087], forearm: [-0.043, -0.007, 0.089] }
  ],
  // D0525N, 4.15–4.67 s: up and outward.
  SAC: [
    { t: 0, arm: [-0.043, -0.103, -0.038], forearm: [-0.09, 0.021, -0.036] },
    { t: 0.18, arm: [-0.043, -0.103, -0.038], forearm: [-0.09, 0.021, -0.036] },
    { t: 0.4, arm: [-0.006, 0.006, -0.046], forearm: [0.013, 0.007, 0.01] },
    { t: 0.62, arm: [0.022, 0.173, -0.002], forearm: [0.187, -0.028, 0.024] },
    { t: 0.82, arm: [0.053, 0.318, 0.018], forearm: [0.365, -0.06, 0.032] },
    { t: 1, arm: [0.053, 0.318, 0.018], forearm: [0.365, -0.06, 0.032] }
  ],
  // D0526N, 1.19–2.35 s: rise, round the top, and drop.
  HOI: [
    { t: 0, arm: [-0.023, -0.062, -0.037], forearm: [-0.053, 0.013, -0.005] },
    { t: 0.16, arm: [-0.023, -0.062, -0.037], forearm: [-0.053, 0.013, -0.005] },
    { t: 0.34, arm: [-0.065, 0.148, -0.105], forearm: [0.171, -0.035, -0.017] },
    { t: 0.5, arm: [-0.038, 0.292, -0.064], forearm: [0.334, -0.073, -0.015] },
    { t: 0.65, arm: [0.016, 0.23, -0.01], forearm: [0.261, -0.056, 0.044] },
    { t: 0.84, arm: [0.035, -0.034, 0.044], forearm: [-0.032, 0, 0.016] },
    { t: 1, arm: [0.035, -0.034, 0.044], forearm: [-0.032, 0, 0.016] }
  ],
  // D0527N, 1.20–2.22 s: small dip between two rising segments.
  NGA: [
    { t: 0, arm: [-0.035, -0.114, -0.037], forearm: [-0.1, 0.022, -0.022] },
    { t: 0.16, arm: [-0.035, -0.114, -0.037], forearm: [-0.1, 0.022, -0.022] },
    { t: 0.34, arm: [-0.072, 0.051, -0.091], forearm: [0.064, -0.017, -0.062] },
    { t: 0.51, arm: [-0.02, 0.014, -0.031], forearm: [0.019, -0.004, -0.014] },
    { t: 0.67, arm: [0.015, 0.173, 0.009], forearm: [0.182, -0.038, -0.001] },
    { t: 0.84, arm: [0.044, 0.307, 0.058], forearm: [0.345, -0.07, 0.014] },
    { t: 1, arm: [0.044, 0.307, 0.058], forearm: [0.345, -0.07, 0.014] }
  ],
  // D0528, 1.07–1.89 s: short downward tap and hold.
  NANG: [
    { t: 0, arm: [-0.009, 0.058, -0.008], forearm: [0.058, -0.016, 0] },
    { t: 0.24, arm: [-0.009, 0.058, -0.008], forearm: [0.058, -0.016, 0] },
    { t: 0.47, arm: [-0.003, 0.037, -0.008], forearm: [0.038, -0.013, 0] },
    { t: 0.62, arm: [0.007, -0.101, 0.043], forearm: [-0.1, 0.017, -0.015] },
    { t: 1, arm: [0.007, -0.101, 0.043], forearm: [-0.1, 0.017, -0.015] }
  ]
};

const TONE_DURATIONS = { HUYEN: 0.65, SAC: 0.62, HOI: 0.95, NGA: 0.85, NANG: 0.55 };

function armTrajectory(pose, frames) {
  const arm = pose.find(({ bone }) => bone === 'RightArm');
  const forearm = pose.find(({ bone }) => bone === 'RightForeArm');
  return frames.map(({ t, arm: armOffset, forearm: forearmOffset }) => ({
    t,
    targets: [
      { bone: arm.bone, rot: arm.rot.map((angle, axis) => angle + armOffset[axis]) },
      { bone: forearm.bone, rot: forearm.rot.map((angle, axis) => angle + forearmOffset[axis]) }
    ]
  }));
}

const TONE_NAMES = {
  TONE_HUYEN: 'HUYEN', TONE_SAC: 'SAC', TONE_HOI: 'HOI', TONE_NGA: 'NGA', TONE_NANG: 'NANG'
};

export const VIETNAMESE_SYMBOL_OPTIONS = {
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

export function getSavedPoseOverrides() {
  try {
    const value = JSON.parse(globalThis.localStorage?.getItem(SAVED_POSES_STORAGE_KEY) || '{}');
    return value && typeof value === 'object' && !Array.isArray(value) ? value : {};
  } catch {
    return {};
  }
}

export function savePoseForLetter(letter, targets) {
  const normalizedLetter = typeof letter === 'string' ? letter.trim().normalize('NFC').toUpperCase() : '';
  const isEditableSymbol = MIXAMO_ALPHABET[normalizedLetter]
    || Object.prototype.hasOwnProperty.call(DIACRITIC_POSES, normalizedLetter)
    || Object.prototype.hasOwnProperty.call(VIETNAMESE_LETTER_MARKS, normalizedLetter);
  if (!isEditableSymbol || !Array.isArray(targets)) return false;

  const overrides = getSavedPoseOverrides();
  overrides[normalizedLetter] = targets
    .filter((target) => /^(?:RightHand(?:Thumb|Index|Middle|Ring|Pinky)\d+|RightHand)$/.test(target.bone))
    .map(({ bone, rot }) => ({ bone, rot: [...rot] }));
  try {
    globalThis.localStorage?.setItem(SAVED_POSES_STORAGE_KEY, JSON.stringify(overrides));
    return true;
  } catch {
    return false;
  }
}

// Rotations relative to BREVE, calibrated on the Mixamo rig from the front.
export const MIXAMO_LETTER_ANIMATIONS = {
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

  const savedRotations = new Map(savedTargets.map((target) => [target.bone, target.rot]));
  return basePose.map((target) => ({
    ...target,
    rot: [...(savedRotations.get(target.bone) || target.rot)]
  }));
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
  const symbol = typeof value === 'string' ? value.trim().normalize('NFC').toUpperCase() : '';
  if (MIXAMO_ALPHABET[symbol]) return withSavedOverrides(symbol, MIXAMO_ALPHABET[symbol]);
  if (DIACRITIC_POSES[symbol]) return withSavedOverrides(symbol, DIACRITIC_POSES[symbol]);
  if (VIETNAMESE_LETTER_MARKS[symbol]) {
    return withSavedOverrides(symbol, MIXAMO_ALPHABET[VIETNAMESE_LETTER_MARKS[symbol].base]);
  }
  const toned = splitTonedLetter(symbol);
  if (toned) return getPoseForSymbol(toned.letter);
  if (TONE_NAMES[symbol]) return getPoseForSymbol('BREVE');
  return createRestPose();
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

function resolveSignActions(token) {
  const normalized = typeof token === 'string' ? token.trim().normalize('NFC').toUpperCase() : '';
  if (!normalized) return [];

  const toned = splitTonedLetter(normalized);
  if (toned) {
    const letterActions = resolveVietnameseSign(toned.letter);
    return letterActions.length
      ? [...letterActions.filter(({ label }) => label !== 'REST'), ...resolveVietnameseSign(toned.tone)]
      : [];
  }

  const tone = TONE_NAMES[normalized];
  if (tone) {
    const pose = getPoseForSymbol('BREVE');
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
      poseAction(getPoseForSymbol('A'), 0.65, 'A'),
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

  if (MIXAMO_ALPHABET[normalized]) {
    return [poseAction(getPoseForLetter(normalized), 0.4, normalized)];
  }
  if (DIACRITIC_POSES[normalized]) {
    return [poseAction(getPoseForSymbol(normalized), 0.4, normalized)];
  }
  return [];
}

export function getMixamoPoseConfig() {
  return {
    alphabet: Object.fromEntries(Object.entries(MIXAMO_ALPHABET).map(([key, value]) => [key, copyTargets(value)])),
    diacriticPoses: Object.fromEntries(Object.entries(DIACRITIC_POSES).map(([key, value]) => [key, copyTargets(value)])),
    toneTrajectories: TONE_TRAJECTORIES
  };
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
