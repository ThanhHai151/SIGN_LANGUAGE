const FINGER_JOINTS = {
  index: { mcp: 5, pip: 6, dip: 7, tip: 8 },
  middle: { mcp: 9, pip: 10, dip: 11, tip: 12 },
  ring: { mcp: 13, pip: 14, dip: 15, tip: 16 },
  pinky: { mcp: 17, pip: 18, dip: 19, tip: 20 }
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));
const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const cross = (a, b) => [
  a[1] * b[2] - a[2] * b[1],
  a[2] * b[0] - a[0] * b[2],
  a[0] * b[1] - a[1] * b[0]
];
const length = (v) => Math.sqrt(dot(v, v));

function normalize(v) {
  const magnitude = length(v);
  if (!Number.isFinite(magnitude) || magnitude < 1e-8) return null;
  return v.map((component) => component / magnitude);
}

function subtract(a, b) {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

function projectToRotationPlane(vector, axis) {
  const component = dot(vector, axis);
  return normalize(vector.map((value, index) => value - component * axis[index]));
}

function signedAngleAround(from, to, axis) {
  const start = projectToRotationPlane(from, axis);
  const end = projectToRotationPlane(to, axis);
  if (!start || !end) return 0;
  return Math.atan2(dot(axis, cross(start, end)), clamp(dot(start, end), -1, 1));
}

function jointFlexion(previous, joint, next) {
  const incoming = normalize(subtract(previous, joint));
  const outgoing = normalize(subtract(next, joint));
  if (!incoming || !outgoing) return 0;
  return clamp(Math.PI - Math.acos(clamp(dot(incoming, outgoing), -1, 1)), 0, Math.PI);
}

function readLandmarks(landmarks, width, height) {
  if (!Array.isArray(landmarks) || landmarks.length !== 21) {
    throw new TypeError('canonicalHandPose cần đúng 21 hand landmarks');
  }
  if (!(width > 0) || !(height > 0)) {
    throw new RangeError('Kích thước frame phải lớn hơn 0');
  }
  return landmarks.map((point, index) => {
    if (!Array.isArray(point) || point.length < 3 || !point.slice(0, 3).every(Number.isFinite)) {
      throw new TypeError(`Landmark ${index} không hợp lệ`);
    }
    // MediaPipe normalizes x/y separately; scale to pixels before 3D geometry.
    return [point[0] * width, point[1] * height, point[2] * width];
  });
}

/** Convert a MediaPipe Hands frame to rig-independent joint angles (radians). */
export function deriveCanonicalHandPose(landmarks, { width, height } = {}) {
  const points = readLandmarks(landmarks, width, height);
  const palmDirection = normalize(subtract(points[9], points[0]));
  const transverse = palmDirection
    ? normalize(subtract(subtract(points[5], points[17]), palmDirection.map((v) => v * dot(subtract(points[5], points[17]), palmDirection))))
    : null;
  if (!palmDirection || !transverse) throw new RangeError('Không xác lập được hệ trục lòng bàn tay');
  const palmNormal = normalize(cross(palmDirection, transverse));
  if (!palmNormal) throw new RangeError('Không xác lập được pháp tuyến lòng bàn tay');

  const fingers = {};
  for (const [name, joints] of Object.entries(FINGER_JOINTS)) {
    const proximal = normalize(subtract(points[joints.pip], points[joints.mcp]));
    fingers[name] = {
      flexion: [
        Math.abs(signedAngleAround(palmDirection, proximal, transverse)),
        jointFlexion(points[joints.mcp], points[joints.pip], points[joints.dip]),
        jointFlexion(points[joints.pip], points[joints.dip], points[joints.tip])
      ],
      splay: signedAngleAround(palmDirection, proximal, palmNormal)
    };
  }

  const thumbBase = normalize(subtract(points[2], points[1]));
  return {
    schema: 'canonical-hand-pose/v1',
    unit: 'radian',
    side: 'right',
    fingers,
    thumb: {
      flexion: [
        jointFlexion(points[0], points[1], points[2]),
        jointFlexion(points[1], points[2], points[3]),
        jointFlexion(points[2], points[3], points[4])
      ],
      abduction: thumbBase ? signedAngleAround(palmDirection, thumbBase, palmNormal) : 0
    }
  };
}

function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

/** Median aggregation reduces landmark jitter across a stable sign interval. */
export function medianCanonicalHandPose(poses) {
  if (!Array.isArray(poses) || poses.length === 0) {
    throw new TypeError('Cần ít nhất một canonical hand pose');
  }
  if (poses.some((pose) => pose?.schema !== 'canonical-hand-pose/v1')) {
    throw new TypeError('Mọi pose phải dùng schema canonical-hand-pose/v1');
  }
  const aggregate = (values) => median(values);
  const fingers = {};
  for (const name of Object.keys(FINGER_JOINTS)) {
    fingers[name] = {
      flexion: [0, 1, 2].map((joint) => aggregate(poses.map((pose) => pose.fingers[name].flexion[joint]))),
      splay: aggregate(poses.map((pose) => pose.fingers[name].splay))
    };
  }
  return {
    schema: 'canonical-hand-pose/v1',
    unit: 'radian',
    side: poses[0].side,
    fingers,
    thumb: {
      flexion: [0, 1, 2].map((joint) => aggregate(poses.map((pose) => pose.thumb.flexion[joint]))),
      abduction: aggregate(poses.map((pose) => pose.thumb.abduction))
    }
  };
}
