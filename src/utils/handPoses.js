/**
 * Thư viện Tư thế Khớp Bàn tay và Toàn thân Hai tay (Two-Handed & Whole Body VSL Poses)
 * Chuẩn giải phẫu:
 * - Body: { torso: [rx, ry, rz], head: [rx, ry, rz] }
 * - Right Hand: 15 khớp đốt (MCP, PIP, DIP) + Wrist + Forearm + Arm
 * - Left Hand: 15 khớp đốt (MCP, PIP, DIP) + Wrist + Forearm + Arm
 * - Keyframes đa khung hình diễn hoạt mượt mà cả 2 tay và cơ thể.
 */

// Tư thế nghỉ ngón tay tự nhiên
export const NATURAL_FINGERS_REST = {
  thumb: { cmc_x: 0.05, cmc_y: 0.1, cmc_z: 0.1, mcp_x: 0.15, ip_x: 0.15 },
  index: { mcp_x: 0.3, pip_x: 0.3, dip_x: 0.2, spread_z: 0.05 },
  middle: { mcp_x: 0.35, pip_x: 0.35, dip_x: 0.2, spread_z: 0.0 },
  ring: { mcp_x: 0.4, pip_x: 0.4, dip_x: 0.25, spread_z: -0.05 },
  pinky: { mcp_x: 0.45, pip_x: 0.45, dip_x: 0.3, spread_z: -0.1 }
};

// 4 ngón duỗi thẳng khép phẳng
export const FLAT_OPEN_FINGERS = {
  thumb: { cmc_x: 0.2, cmc_y: 0.8, cmc_z: 0.6, mcp_x: 1.1, ip_x: 0.8 },
  index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
  middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
  ring: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
  pinky: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 }
};

// 4 ngón nắm chặt
export const FIST_FINGERS = {
  thumb: { cmc_x: 0.0, cmc_y: -0.15, cmc_z: -0.15, mcp_x: 0.1, ip_x: 0.05 },
  index: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
  middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
  ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
  pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
};

// Góc xoay chuẩn cho cánh tay trong Không gian ký hiệu VSL (Signing Space) trước ngực
export const SIGNING_RIGHT_ARM = [-1.15, 0.15, -0.08];
export const SIGNING_RIGHT_FOREARM = [-1.10, 0.20, 0.08];
export const SIGNING_RIGHT_WRIST = [0.15, 0.10, 0.0];
export const PASSIVE_LEFT_ARM = [-0.22, -0.08, 0.05];
export const PASSIVE_LEFT_FOREARM = [-0.35, -0.08, -0.05];
export const PASSIVE_LEFT_WRIST = [0, 0, 0];

export const HAND_POSES = {
  // --- TƯ THẾ NGHỈ TỰ NHIÊN (REST POSE) ---
  REST: {
    name: 'Nghỉ tự nhiên',
    twoHanded: false,
    body: { torso: [0, 0, 0], head: [0, 0, 0] },
    rightArm: [-0.20, 0.05, 0.05],
    rightForearm: [-0.35, 0.05, 0.05],
    rightWrist: [0, 0, 0],
    rightFingers: NATURAL_FINGERS_REST,
    leftArm: [-0.20, -0.05, -0.05],
    leftForearm: [-0.35, -0.05, -0.05],
    leftWrist: [0, 0, 0],
    leftFingers: NATURAL_FINGERS_REST
  },

  // ==========================================
  // BẢNG CHỮ CÁI ĐÁNH VẦN NGÓN TAY (A - Z)
  // (Tay phải biểu diễn, Tay trái đặt tự nhiên hỗ trợ, Cơ thể có biểu cảm)
  // ==========================================
  A: {
    name: 'Chữ A',
    body: { torso: [0.03, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: -0.2, cmc_y: 0.25, cmc_z: 0.45, mcp_x: 0.05, ip_x: 0.05 },
      index: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  B: {
    name: 'Chữ B',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.25, cmc_y: 0.85, cmc_z: 0.65, mcp_x: 1.25, ip_x: 0.85 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
      middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
      ring: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
      pinky: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  C: {
    name: 'Chữ C',
    body: { torso: [0.03, -0.03, 0], head: [0.02, 0.03, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: [0, -0.4, 0],
    rightFingers: {
      thumb: { cmc_x: 0.3, cmc_y: 0.35, cmc_z: 0.2, mcp_x: 0.6, ip_x: 0.5 },
      index: { mcp_x: 0.7, pip_x: 0.7, dip_x: 0.5, spread_z: 0.0 },
      middle: { mcp_x: 0.75, pip_x: 0.75, dip_x: 0.5, spread_z: 0.0 },
      ring: { mcp_x: 0.75, pip_x: 0.75, dip_x: 0.5, spread_z: 0.0 },
      pinky: { mcp_x: 0.75, pip_x: 0.75, dip_x: 0.5, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  D: {
    name: 'Chữ D',
    body: { torso: [0.02, -0.02, 0], head: [0.01, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.35, cmc_y: 0.55, cmc_z: 0.35, mcp_x: 0.85, ip_x: 0.65 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
      middle: { mcp_x: 1.1, pip_x: 1.1, dip_x: 0.9, spread_z: 0.0 },
      ring: { mcp_x: 1.45, pip_x: 1.45, dip_x: 1.2, spread_z: 0.0 },
      pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.2, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  E: {
    name: 'Chữ E',
    body: { torso: [0.03, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.2, cmc_y: 0.7, cmc_z: 0.5, mcp_x: 1.2, ip_x: 0.85 },
      index: { mcp_x: 1.45, pip_x: 1.45, dip_x: 1.1, spread_z: 0.0 },
      middle: { mcp_x: 1.45, pip_x: 1.45, dip_x: 1.1, spread_z: 0.0 },
      ring: { mcp_x: 1.45, pip_x: 1.45, dip_x: 1.1, spread_z: 0.0 },
      pinky: { mcp_x: 1.45, pip_x: 1.45, dip_x: 1.1, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  F: {
    name: 'Chữ F',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.35, cmc_y: 0.45, cmc_z: 0.25, mcp_x: 0.75, ip_x: 0.55 },
      index: { mcp_x: 1.1, pip_x: 1.1, dip_x: 0.8, spread_z: 0.0 },
      middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.06 },
      ring: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.12 },
      pinky: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.18 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  G: {
    name: 'Chữ G',
    body: { torso: [0.02, -0.04, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: [0, -0.6, 0],
    rightFingers: {
      thumb: { cmc_x: 0.0, cmc_y: -0.2, cmc_z: 0.25, mcp_x: 0.05, ip_x: 0.05 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
      middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  H: {
    name: 'Chữ H',
    body: { torso: [0.02, -0.04, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: [0, -0.6, 0],
    rightFingers: {
      thumb: { cmc_x: 0.2, cmc_y: 0.5, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.5 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: -0.04 },
      middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.04 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  I: {
    name: 'Chữ I',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.2, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.4 },
      index: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  J: {
    name: 'Chữ J',
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
        rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
        rightFingers: {
          thumb: { cmc_x: 0.2, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.4 },
          index: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 }
        }
      },
      {
        t: 1.0,
        body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
        rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: [0.3, -0.4, -0.4],
        rightFingers: {
          thumb: { cmc_x: 0.2, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.4 },
          index: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.1 }
        }
      }
    ]
  },

  K: {
    name: 'Chữ K',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.25, cmc_y: 0.3, cmc_z: 0.2, mcp_x: 0.4, ip_x: 0.2 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
      middle: { mcp_x: 0.4, pip_x: 0.2, dip_x: 0.1, spread_z: 0.1 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  L: {
    name: 'Chữ L',
    body: { torso: [0.03, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.0, cmc_y: -0.5, cmc_z: -0.4, mcp_x: 0.0, ip_x: 0.0 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
      middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  M: {
    name: 'Chữ M',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.35, cmc_y: 0.65, cmc_z: 0.45, mcp_x: 0.7, ip_x: 0.3 },
      index: { mcp_x: 1.4, pip_x: 1.4, dip_x: 1.1, spread_z: -0.04 },
      middle: { mcp_x: 1.4, pip_x: 1.4, dip_x: 1.1, spread_z: 0.0 },
      ring: { mcp_x: 1.4, pip_x: 1.4, dip_x: 1.1, spread_z: 0.04 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  N: {
    name: 'Chữ N',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.35, cmc_y: 0.5, cmc_z: 0.35, mcp_x: 0.6, ip_x: 0.3 },
      index: { mcp_x: 1.4, pip_x: 1.4, dip_x: 1.1, spread_z: -0.04 },
      middle: { mcp_x: 1.4, pip_x: 1.4, dip_x: 1.1, spread_z: 0.04 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  O: {
    name: 'Chữ O',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: [0, -0.3, 0],
    rightFingers: {
      thumb: { cmc_x: 0.4, cmc_y: 0.4, cmc_z: 0.3, mcp_x: 0.7, ip_x: 0.5 },
      index: { mcp_x: 0.9, pip_x: 0.9, dip_x: 0.8, spread_z: -0.05 },
      middle: { mcp_x: 0.95, pip_x: 0.95, dip_x: 0.8, spread_z: 0.0 },
      ring: { mcp_x: 0.95, pip_x: 0.95, dip_x: 0.8, spread_z: 0.05 },
      pinky: { mcp_x: 0.9, pip_x: 0.9, dip_x: 0.8, spread_z: 0.1 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  P: {
    name: 'Chữ P',
    body: { torso: [0.04, -0.02, 0], head: [0.03, 0.02, 0] },
    rightArm: [-0.7, 0.2, 0.1], rightForearm: [-0.85, 0.15, 0.15], rightWrist: [0.65, 0, 0],
    rightFingers: {
      thumb: { cmc_x: 0.25, cmc_y: 0.3, cmc_z: 0.2, mcp_x: 0.4, ip_x: 0.2 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
      middle: { mcp_x: 0.4, pip_x: 0.2, dip_x: 0.1, spread_z: 0.1 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  Q: {
    name: 'Chữ Q',
    body: { torso: [0.04, -0.02, 0], head: [0.03, 0.02, 0] },
    rightArm: [-0.7, 0.2, 0.1], rightForearm: [-0.85, 0.15, 0.15], rightWrist: [0.65, -0.4, 0],
    rightFingers: {
      thumb: { cmc_x: 0.0, cmc_y: -0.2, cmc_z: 0.25, mcp_x: 0.05, ip_x: 0.05 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
      middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  R: {
    name: 'Chữ R',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.2, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.4 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.08 },
      middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: -0.08 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  S: {
    name: 'Chữ S',
    body: { torso: [0.03, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.4, cmc_y: 0.5, cmc_z: 0.35, mcp_x: 0.9, ip_x: 0.5 },
      index: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  T: {
    name: 'Chữ T',
    body: { torso: [0.03, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.25, cmc_y: 0.35, cmc_z: 0.25, mcp_x: 0.6, ip_x: 0.3 },
      index: { mcp_x: 1.4, pip_x: 1.4, dip_x: 1.1, spread_z: 0.0 },
      middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  U: {
    name: 'Chữ U',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.2, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.4 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: -0.02 },
      middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.02 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  V: {
    name: 'Chữ V',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.2, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.4 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: -0.22 },
      middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.22 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  W: {
    name: 'Chữ W',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.3, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.9, ip_x: 0.6 },
      index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: -0.22 },
      middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
      ring: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.22 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  X: {
    name: 'Chữ X',
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.2, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.4 },
      index: { mcp_x: 0.7, pip_x: 1.3, dip_x: 0.9, spread_z: 0.0 },
      middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  Y: {
    name: 'Chữ Y',
    body: { torso: [0.03, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: SIGNING_RIGHT_WRIST,
    rightFingers: {
      thumb: { cmc_x: 0.0, cmc_y: -0.6, cmc_z: -0.5, mcp_x: 0.0, ip_x: 0.0 },
      index: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
      pinky: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.35 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  Z: {
    name: 'Chữ Z',
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
        rightArm: [-0.75, 0.25, 0.1], rightForearm: [-0.85, 0.15, 0.15], rightWrist: [0, 0.15, 0],
        rightFingers: {
          thumb: { cmc_x: 0.2, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.4 },
          index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
          middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        }
      },
      {
        t: 0.5,
        body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
        rightArm: [-0.75, 0.1, 0.1], rightForearm: [-0.85, 0.15, 0.15], rightWrist: [0, -0.15, 0],
        rightFingers: {
          thumb: { cmc_x: 0.2, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.4 },
          index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
          middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        }
      },
      {
        t: 1.0,
        body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
        rightArm: [-0.75, 0.25, 0.1], rightForearm: [-0.85, 0.15, 0.15], rightWrist: [0, 0.15, 0],
        rightFingers: {
          thumb: { cmc_x: 0.2, cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.8, ip_x: 0.4 },
          index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
          middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        }
      }
    ]
  },

  // ==========================================
  // CÁC TỪ KÝ HIỆU SỬ DỤNG 2 TAY & TOÀN BỘ CƠ THỂ
  // ==========================================

  // XIN CHÀO (HELLO - Ảnh 2 & 3): Cả cơ thể hướng về phía đối thoại, 
  // Tay phải vẫy từ thái dương, Tay trái mở đón tiếp ngang ngực/bụng
  XIN_CHAO: {
    name: 'Xin chào (Hello)',
    isDynamic: true,
    twoHanded: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.06, -0.02, 0.0], head: [0.08, 0.02, 0.0] },
        rightArm: [-1.45, 0.20, -0.15], rightForearm: [-1.15, 0.25, 0.15], rightWrist: [0.10, 0.15, -0.20],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [0.09, 0.0, 0.0], head: [0.12, 0.0, 0.0] },
        rightArm: [-1.45, 0.20, -0.15], rightForearm: [-1.15, 0.25, 0.15], rightWrist: [0.10, 0.15, 0.20],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.06, 0.0, 0.0], head: [0.08, 0.0, 0.0] },
        rightArm: [-1.45, 0.20, -0.15], rightForearm: [-1.15, 0.25, 0.15], rightWrist: [0.10, 0.15, -0.10],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // CẢM ƠN (THANK YOU - Ảnh 2): Thân người cúi nhẹ gật đầu, 
  // Tay phải từ môi đưa thẳng ra trước, Tay trái nâng mở đỡ phía dưới lòng bàn tay
  CAM_ON: {
    name: 'Cảm ơn (Thank You)',
    isDynamic: true,
    twoHanded: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.10, 0.0, 0.0] },
        rightArm: [-1.25, 0.10, -0.05], rightForearm: [-1.30, 0.15, 0.0], rightWrist: [-0.45, 0.10, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.40, 0.15, 0.05], leftForearm: [-0.85, -0.15, 0.10], leftWrist: [0.15, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 0.6,
        body: { torso: [0.14, 0.0, 0.0], head: [0.15, 0.0, 0.0] },
        rightArm: [-0.95, 0.05, 0.0], rightForearm: [-0.60, 0.10, 0.0], rightWrist: [0.10, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.40, 0.15, 0.05], leftForearm: [-0.85, -0.15, 0.10], leftWrist: [0.15, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.08, 0.0, 0.0] },
        rightArm: [-0.80, 0.05, 0.0], rightForearm: [-0.45, 0.10, 0.0], rightWrist: [0.15, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.40, 0.15, 0.05], leftForearm: [-0.85, -0.15, 0.10], leftWrist: [0.15, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // XIN VUI LÒNG / LÀM ƠN (PLEASE - Ảnh 1):
  // Tay phải áp phẳng lên ngực xoa vòng tròn (chest circular rub), Thân thẳng, Đầu nghiêng nhẹ cầu thị, Tay trái khép tự nhiên
  XIN_VUI_LONG: {
    name: 'Xin vui lòng / Làm ơn (Please)',
    isDynamic: true,
    twoHanded: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.06, -0.04, 0.0], head: [0.08, -0.04, 0.04] },
        rightArm: [-0.62, -0.28, -0.15], rightForearm: [-0.95, 0.22, 0.0], rightWrist: [-0.15, 0.05, 0.25],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.35, -0.15, -0.1], leftForearm: [-0.65, 0.1, 0.05], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 0.33,
        body: { torso: [0.08, -0.02, 0.0], head: [0.10, -0.03, 0.04] },
        rightArm: [-0.55, -0.38, -0.22], rightForearm: [-0.85, 0.18, 0.0], rightWrist: [-0.22, 0.0, 0.30],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.35, -0.15, -0.1], leftForearm: [-0.65, 0.1, 0.05], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 0.66,
        body: { torso: [0.06, -0.04, 0.0], head: [0.08, -0.04, 0.04] },
        rightArm: [-0.68, -0.42, -0.18], rightForearm: [-0.92, 0.15, 0.0], rightWrist: [-0.18, -0.08, 0.25],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.35, -0.15, -0.1], leftForearm: [-0.65, 0.1, 0.05], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.06, -0.04, 0.0], head: [0.08, -0.04, 0.04] },
        rightArm: [-0.62, -0.28, -0.15], rightForearm: [-0.95, 0.22, 0.0], rightWrist: [-0.15, 0.05, 0.25],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.35, -0.15, -0.1], leftForearm: [-0.65, 0.1, 0.05], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  // XIN LỖI (SORRY): Nắm đấm áp vào ngực xoay nhẹ thể hiện sự hối lỗi
  XIN_LOI: {
    name: 'Xin lỗi (Sorry)',
    isDynamic: true,
    twoHanded: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.08, -0.06, 0.0], head: [0.1, -0.05, 0.04] },
        rightArm: [-0.6, -0.4, -0.2], rightForearm: [-0.9, 0.2, 0.0], rightWrist: [-0.2, 0.0, 0.3],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.35, -0.2, -0.1], leftForearm: [-0.75, 0.2, 0.1], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [0.10, -0.04, 0.0], head: [0.12, -0.04, 0.04] },
        rightArm: [-0.7, -0.3, -0.1], rightForearm: [-0.8, 0.1, 0.1], rightWrist: [-0.1, 0.1, 0.2],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.35, -0.2, -0.1], leftForearm: [-0.75, 0.2, 0.1], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.08, -0.06, 0.0], head: [0.1, -0.05, 0.04] },
        rightArm: [-0.6, -0.4, -0.2], rightForearm: [-0.9, 0.2, 0.0], rightWrist: [-0.2, 0.0, 0.3],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.35, -0.2, -0.1], leftForearm: [-0.75, 0.2, 0.1], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // YÊU THƯƠNG (LOVE / HUG): CẢ HAI TAY BẮT CHÉO ÔM TRƯỚC NGỰC, ĐẦU NGHIÊNG ÂU YẾM
  YEU_THUONG: {
    name: 'Yêu thương (Love / Hug)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.05, 0.0, 0.0], head: [0.04, 0.05, 0.08] },
        rightArm: [-0.5, 0.3, 0.1], rightForearm: [-0.6, 0.2, 0.1], rightWrist: [0, 0, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.5, -0.3, -0.1], leftForearm: [-0.6, -0.2, -0.1], leftWrist: [0, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.10, 0.0, 0.0], head: [0.08, 0.08, 0.12] }, // Đầu nghiêng âu yếm
        // Tay phải bắt chéo ôm vai trái
        rightArm: [-0.75, -0.35, -0.3], rightForearm: [-1.15, 0.2, 0.0], rightWrist: [-0.3, 0.0, 0.2],
        rightFingers: FLAT_OPEN_FINGERS,
        // Tay trái bắt chéo ôm vai phải
        leftArm: [-0.75, 0.35, 0.3], leftForearm: [-1.15, -0.2, 0.0], leftWrist: [-0.3, 0.0, -0.2],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // GIÚP ĐỠ (HELP): Tay trái mở ngửa đỡ dưới, Tay phải nắm đấm Like nâng lên
  GIUP_DO: {
    name: 'Giúp đỡ (Help)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.05, 0.0, 0.0], head: [0.05, 0.0, 0.0] },
        rightArm: [-0.55, -0.1, 0.0], rightForearm: [-0.85, 0.1, 0.0], rightWrist: [0.1, 0.0, 0.0],
        rightFingers: {
          thumb: { cmc_y: -0.1, cmc_z: 0.1, mcp_x: 0.0, ip_x: 0.0 }, // Thumbs up
          index: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        },
        leftArm: [-0.5, 0.1, 0.0], leftForearm: [-0.95, -0.1, 0.0], leftWrist: [0.4, 0.0, 0.0], // Ngửa lòng bàn tay
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.0, 0.0, 0.0], head: [0.0, 0.0, 0.0] },
        rightArm: [-0.75, -0.1, 0.0], rightForearm: [-0.65, 0.1, 0.0], rightWrist: [0.1, 0.0, 0.0],
        rightFingers: {
          thumb: { cmc_y: -0.1, cmc_z: 0.1, mcp_x: 0.0, ip_x: 0.0 },
          index: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        },
        leftArm: [-0.7, 0.1, 0.0], leftForearm: [-0.75, -0.1, 0.0], leftWrist: [0.4, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // BỆNH VIỆN (HOSPITAL): Tay trái đưa ngang, Tay phải vẽ chữ thập đỏ trên cẳng tay trái
  BENH_VIEN: {
    name: 'Bệnh viện (Hospital)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.06, 0.05, 0.0], head: [0.08, 0.06, 0.0] },
        leftArm: [-0.6, 0.2, 0.2], leftForearm: [-1.0, 0.2, 0.0], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS,
        rightArm: [-0.7, -0.2, -0.1], rightForearm: [-0.9, -0.1, 0.0], rightWrist: [0.2, 0.0, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.5, cmc_z: 0.4, mcp_x: 1.3, ip_x: 0.8 },
          index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
          middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        }
      },
      {
        t: 1.0,
        body: { torso: [0.06, 0.05, 0.0], head: [0.08, 0.06, 0.0] },
        leftArm: [-0.6, 0.2, 0.2], leftForearm: [-1.0, 0.2, 0.0], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS,
        rightArm: [-0.65, -0.15, -0.1], rightForearm: [-0.85, -0.1, 0.0], rightWrist: [-0.1, 0.0, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.5, cmc_z: 0.4, mcp_x: 1.3, ip_x: 0.8 },
          index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
          middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        }
      }
    ]
  },

  // BẠN: Ngón trỏ chỉ thẳng về người đối thoại, thân người hướng tới
  BAN: {
    name: 'Bạn (Friend / You)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.04, 0.0, 0.0], head: [0.03, 0.0, 0.0] },
        rightArm: [-1.10, -0.15, -0.08], rightForearm: [-1.15, 0.15, 0.08], rightWrist: [-0.15, 0.25, 0.0],
        rightFingers: {
          thumb: { cmc_x: 0.1, cmc_y: 0.3, cmc_z: 0.4, mcp_x: 0.2, ip_x: 0.1 },
          index: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          middle: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          ring: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          pinky: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 }
        },
        leftArm: [-1.10, 0.15, 0.08], leftForearm: [-1.15, -0.15, -0.08], leftWrist: [-0.15, -0.25, 0.0],
        leftFingers: {
          thumb: { cmc_x: 0.1, cmc_y: 0.3, cmc_z: 0.4, mcp_x: 0.2, ip_x: 0.1 },
          index: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          middle: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          ring: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          pinky: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 }
        }
      },
      {
        t: 0.5,
        body: { torso: [0.08, 0.0, 0.0], head: [0.07, 0.0, 0.0] },
        rightArm: [-1.15, -0.25, -0.05], rightForearm: [-1.20, 0.25, 0.05], rightWrist: [-0.10, 0.30, 0.05],
        rightFingers: {
          thumb: { cmc_x: 0.2, cmc_y: 0.4, cmc_z: 0.4, mcp_x: 0.4, ip_x: 0.3 },
          index: { mcp_x: 0.3, pip_x: 0.3, dip_x: 0.2, spread_z: 0.0 },
          middle: { mcp_x: 0.35, pip_x: 0.35, dip_x: 0.25, spread_z: 0.0 },
          ring: { mcp_x: 0.4, pip_x: 0.4, dip_x: 0.3, spread_z: 0.0 },
          pinky: { mcp_x: 0.4, pip_x: 0.4, dip_x: 0.3, spread_z: 0.0 }
        },
        leftArm: [-1.15, 0.25, 0.05], leftForearm: [-1.20, -0.25, -0.05], leftWrist: [-0.10, -0.30, -0.05],
        leftFingers: {
          thumb: { cmc_x: 0.2, cmc_y: 0.4, cmc_z: 0.4, mcp_x: 0.4, ip_x: 0.3 },
          index: { mcp_x: 0.3, pip_x: 0.3, dip_x: 0.2, spread_z: 0.0 },
          middle: { mcp_x: 0.35, pip_x: 0.35, dip_x: 0.25, spread_z: 0.0 },
          ring: { mcp_x: 0.4, pip_x: 0.4, dip_x: 0.3, spread_z: 0.0 },
          pinky: { mcp_x: 0.4, pip_x: 0.4, dip_x: 0.3, spread_z: 0.0 }
        }
      },
      {
        t: 1.0,
        body: { torso: [0.05, 0.0, 0.0], head: [0.04, 0.0, 0.0] },
        rightArm: [-1.10, -0.15, -0.08], rightForearm: [-1.15, 0.15, 0.08], rightWrist: [-0.15, 0.25, 0.0],
        rightFingers: {
          thumb: { cmc_x: 0.1, cmc_y: 0.3, cmc_z: 0.4, mcp_x: 0.2, ip_x: 0.1 },
          index: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          middle: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          ring: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          pinky: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 }
        },
        leftArm: [-1.10, 0.15, 0.08], leftForearm: [-1.15, -0.15, -0.08], leftWrist: [-0.15, -0.25, 0.0],
        leftFingers: {
          thumb: { cmc_x: 0.1, cmc_y: 0.3, cmc_z: 0.4, mcp_x: 0.2, ip_x: 0.1 },
          index: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          middle: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          ring: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 },
          pinky: { mcp_x: 0.1, pip_x: 0.1, dip_x: 0.05, spread_z: 0.0 }
        }
      }
    ]
  },

  // TÔI: Ngón trỏ chỉ vào giữa ngực mình chuẩn từ điển VSL
  TOI: {
    name: 'Tôi (Me / I)',
    twoHanded: false,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, 0.0, 0.0], head: [0.04, 0.0, 0.0] },
        rightArm: [-1.05, -0.15, -0.05], rightForearm: [-1.05, 0.15, 0.05], rightWrist: [0.10, 0.0, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.4, cmc_z: 0.3, mcp_x: 1.2, ip_x: 0.8 },
          index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
          middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.06, 0.0, 0.0], head: [0.10, 0.0, 0.0] },
        rightArm: [-0.75, -0.30, -0.05], rightForearm: [-1.35, 0.20, -0.10], rightWrist: [-0.30, 0.15, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.4, cmc_z: 0.3, mcp_x: 1.2, ip_x: 0.8 },
          index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.0 },
          middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  // GIA ĐÌNH (FAMILY): Hai bàn tay tạo hình vòng tròn từ trước ngực xoay ra ngoài rồi gặp lại nhau
  GIA_DINH: {
    name: 'Gia đình (Family)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.05, 0.0, 0.0], head: [0.04, 0.0, 0.0] },
        rightArm: [-0.65, -0.15, 0.0], rightForearm: [-0.85, 0.15, 0.0], rightWrist: [0.1, 0.0, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.9, ip_x: 0.7 },
          index: { mcp_x: 1.1, pip_x: 1.1, dip_x: 0.8, spread_z: 0.0 },
          middle: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.06 },
          ring: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.12 },
          pinky: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.18 }
        },
        leftArm: [-0.65, 0.15, 0.0], leftForearm: [-0.85, -0.15, 0.0], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: {
          thumb: { cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.9, ip_x: 0.7 },
          index: { mcp_x: 1.1, pip_x: 1.1, dip_x: 0.8, spread_z: 0.0 },
          middle: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.06 },
          ring: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.12 },
          pinky: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.18 }
        }
      },
      {
        t: 0.5,
        body: { torso: [0.08, 0.0, 0.0], head: [0.06, 0.0, 0.0] },
        rightArm: [-0.75, 0.25, 0.1], rightForearm: [-0.75, 0.2, 0.0], rightWrist: [0.0, 0.1, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.9, ip_x: 0.7 },
          index: { mcp_x: 1.1, pip_x: 1.1, dip_x: 0.8, spread_z: 0.0 },
          middle: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.06 },
          ring: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.12 },
          pinky: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.18 }
        },
        leftArm: [-0.75, -0.25, -0.1], leftForearm: [-0.75, -0.2, 0.0], leftWrist: [0.0, -0.1, 0.0],
        leftFingers: {
          thumb: { cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.9, ip_x: 0.7 },
          index: { mcp_x: 1.1, pip_x: 1.1, dip_x: 0.8, spread_z: 0.0 },
          middle: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.06 },
          ring: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.12 },
          pinky: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.18 }
        }
      },
      {
        t: 1.0,
        body: { torso: [0.05, 0.0, 0.0], head: [0.04, 0.0, 0.0] },
        rightArm: [-0.6, -0.1, 0.0], rightForearm: [-0.85, 0.15, 0.0], rightWrist: [0.1, 0.0, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.9, ip_x: 0.7 },
          index: { mcp_x: 1.1, pip_x: 1.1, dip_x: 0.8, spread_z: 0.0 },
          middle: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.06 },
          ring: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.12 },
          pinky: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.18 }
        },
        leftArm: [-0.6, 0.1, 0.0], leftForearm: [-0.85, -0.15, 0.0], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: {
          thumb: { cmc_y: 0.6, cmc_z: 0.4, mcp_x: 0.9, ip_x: 0.7 },
          index: { mcp_x: 1.1, pip_x: 1.1, dip_x: 0.8, spread_z: 0.0 },
          middle: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.06 },
          ring: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.12 },
          pinky: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.1, spread_z: 0.18 }
        }
      }
    ]
  },

  // HỌC BÀI (STUDY): Tay trái mở như sách, tay phải lấy tri thức chạm trán
  HOC_BAI: {
    name: 'Học bài (Study)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.10, 0.0, 0.0] },
        leftArm: [-0.5, 0.15, 0.1], leftForearm: [-0.95, 0.0, 0.1], leftWrist: [0.3, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS,
        rightArm: [-0.55, -0.15, 0.0], rightForearm: [-0.9, 0.0, 0.0], rightWrist: [-0.2, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [0.06, 0.0, 0.0], head: [0.06, 0.0, 0.0] },
        leftArm: [-0.5, 0.15, 0.1], leftForearm: [-0.95, 0.0, 0.1], leftWrist: [0.3, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS,
        rightArm: [-1.35, 0.25, 0.2], rightForearm: [-0.85, 0.45, 0.3], rightWrist: [-0.2, 0.1, -0.2],
        rightFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.10, 0.0, 0.0] },
        leftArm: [-0.5, 0.15, 0.1], leftForearm: [-0.95, 0.0, 0.1], leftWrist: [0.3, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS,
        rightArm: [-0.55, -0.15, 0.0], rightForearm: [-0.9, 0.0, 0.0], rightWrist: [-0.2, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // NẤU ĂN (COOK): Tay trái làm nồi/chảo khum nhẹ, tay phải đảo muôi xào nấu
  NAU_AN: {
    name: 'Nấu ăn (Cook)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.09, 0.0, 0.0], head: [0.12, 0.0, 0.0] },
        leftArm: [-0.55, 0.2, 0.1], leftForearm: [-0.95, 0.0, 0.1], leftWrist: [0.25, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS,
        rightArm: [-0.65, -0.15, -0.1], rightForearm: [-0.85, 0.1, 0.0], rightWrist: [-0.1, 0.2, 0.1],
        rightFingers: FIST_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [0.10, 0.0, 0.0], head: [0.12, 0.0, 0.0] },
        leftArm: [-0.55, 0.2, 0.1], leftForearm: [-0.95, 0.0, 0.1], leftWrist: [0.25, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS,
        rightArm: [-0.60, -0.05, 0.0], rightForearm: [-0.80, 0.2, 0.1], rightWrist: [0.1, -0.2, -0.1],
        rightFingers: FIST_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.09, 0.0, 0.0], head: [0.12, 0.0, 0.0] },
        leftArm: [-0.55, 0.2, 0.1], leftForearm: [-0.95, 0.0, 0.1], leftWrist: [0.25, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS,
        rightArm: [-0.65, -0.15, -0.1], rightForearm: [-0.85, 0.1, 0.0], rightWrist: [-0.1, 0.2, 0.1],
        rightFingers: FIST_FINGERS
      }
    ]
  },

  // LAU NHÀ (MOP / CLEAN): Hai tay cầm cán cây lau nhà đẩy tới lui
  LAU_NHA: {
    name: 'Lau nhà (Mop / Clean)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.14, 0.05, 0.0], head: [0.15, 0.05, 0.0] },
        rightArm: [-0.75, 0.1, 0.0], rightForearm: [-0.85, 0.1, 0.0], rightWrist: [0.1, 0.0, 0.0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.55, 0.05, 0.0], leftForearm: [-0.75, 0.0, 0.0], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: FIST_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [0.18, -0.05, 0.0], head: [0.18, -0.05, 0.0] },
        rightArm: [-0.95, -0.1, 0.0], rightForearm: [-0.65, -0.1, 0.0], rightWrist: [0.2, 0.0, 0.0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.75, -0.1, 0.0], leftForearm: [-0.55, -0.1, 0.0], leftWrist: [0.2, 0.0, 0.0],
        leftFingers: FIST_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.14, 0.05, 0.0], head: [0.15, 0.05, 0.0] },
        rightArm: [-0.75, 0.1, 0.0], rightForearm: [-0.85, 0.1, 0.0], rightWrist: [0.1, 0.0, 0.0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.55, 0.05, 0.0], leftForearm: [-0.75, 0.0, 0.0], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: FIST_FINGERS
      }
    ]
  },

  // ĐI NGỦ (SLEEP): Hai lòng bàn tay áp má, đầu nghiêng tựa
  DI_NGU: {
    name: 'Đi ngủ (Sleep)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.05, 0.0, 0.0], head: [0.05, 0.0, 0.0] },
        rightArm: [-0.5, 0.2, 0.1], rightForearm: [-0.6, 0.1, 0.1], rightWrist: [0, 0, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.5, -0.2, -0.1], leftForearm: [-0.6, -0.1, -0.1], leftWrist: [0, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.08, 0.1, 0.15], head: [0.12, 0.15, 0.25] }, // Nghiêng đầu ngủ
        rightArm: [-1.2, 0.35, 0.2], rightForearm: [-0.95, 0.3, 0.1], rightWrist: [-0.3, 0.1, 0.2],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-1.1, 0.40, 0.25], leftForearm: [-0.95, 0.25, 0.1], leftWrist: [-0.3, 0.1, 0.2],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // RỬA TAY (WASH HANDS): Hai tay xoa miết rửa tay
  RUA_TAY: {
    name: 'Rửa tay (Wash Hands)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.12, 0.0, 0.0] },
        rightArm: [-0.6, -0.1, 0.0], rightForearm: [-0.85, 0.1, 0.0], rightWrist: [0.1, 0.1, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.6, 0.1, 0.0], leftForearm: [-0.85, -0.1, 0.0], leftWrist: [0.1, -0.1, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [0.08, 0.0, 0.0], head: [0.12, 0.0, 0.0] },
        rightArm: [-0.65, -0.05, 0.0], rightForearm: [-0.8, 0.05, 0.1], rightWrist: [-0.1, -0.1, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.65, 0.05, 0.0], leftForearm: [-0.8, -0.05, -0.1], leftWrist: [-0.1, 0.1, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.12, 0.0, 0.0] },
        rightArm: [-0.6, -0.1, 0.0], rightForearm: [-0.85, 0.1, 0.0], rightWrist: [0.1, 0.1, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.6, 0.1, 0.0], leftForearm: [-0.85, -0.1, 0.0], leftWrist: [0.1, -0.1, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // ĐỌC SÁCH (READ BOOK): Hai bàn tay mở song song như trang sách mở trước mặt
  DOC_SACH: {
    name: 'Đọc sách (Read Book)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.14, 0.06, 0.0] },
        rightArm: [-0.6, -0.1, 0.0], rightForearm: [-0.95, 0.05, 0.1], rightWrist: [0.3, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.6, 0.1, 0.0], leftForearm: [-0.95, -0.05, -0.1], leftWrist: [0.3, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.14, -0.06, 0.0] }, // Đầu đọc từ trái qua phải
        rightArm: [-0.6, -0.1, 0.0], rightForearm: [-0.95, 0.05, 0.1], rightWrist: [0.3, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.6, 0.1, 0.0], leftForearm: [-0.95, -0.05, -0.1], leftWrist: [0.3, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // VIẾT BÀI (WRITE): Tay trái làm giấy, tay phải cầm bút viết
  VIET_BAI: {
    name: 'Viết bài (Write)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.10, 0.0, 0.0], head: [0.14, 0.0, 0.0] },
        leftArm: [-0.55, 0.1, 0.0], leftForearm: [-0.95, 0.0, 0.1], leftWrist: [0.3, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS,
        rightArm: [-0.60, -0.1, 0.0], rightForearm: [-0.90, 0.1, 0.0], rightWrist: [0.1, -0.1, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.5, cmc_z: 0.3, mcp_x: 0.8, ip_x: 0.6 },
          index: { mcp_x: 0.9, pip_x: 0.9, dip_x: 0.6, spread_z: 0.0 },
          middle: { mcp_x: 1.4, pip_x: 1.4, dip_x: 1.2, spread_z: 0.0 },
          ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.2, spread_z: 0.0 },
          pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.2, spread_z: 0.0 }
        }
      },
      {
        t: 1.0,
        body: { torso: [0.10, 0.0, 0.0], head: [0.14, 0.0, 0.0] },
        leftArm: [-0.55, 0.1, 0.0], leftForearm: [-0.95, 0.0, 0.1], leftWrist: [0.3, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS,
        rightArm: [-0.58, -0.05, 0.0], rightForearm: [-0.92, 0.15, 0.0], rightWrist: [0.15, 0.1, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.5, cmc_z: 0.3, mcp_x: 0.8, ip_x: 0.6 },
          index: { mcp_x: 0.9, pip_x: 0.9, dip_x: 0.6, spread_z: 0.0 },
          middle: { mcp_x: 1.4, pip_x: 1.4, dip_x: 1.2, spread_z: 0.0 },
          ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.2, spread_z: 0.0 },
          pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.2, spread_z: 0.0 }
        }
      }
    ]
  },

  // LÀM VIỆC (WORK): Hai bàn tay nắm gõ luân phiên
  LAM_VIEC: {
    name: 'Làm việc (Work)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.06, 0.0, 0.0], head: [0.06, 0.0, 0.0] },
        rightArm: [-0.65, 0.1, 0.0], rightForearm: [-0.85, 0.0, 0.0], rightWrist: [0.1, 0.0, 0.0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.55, -0.1, 0.0], leftForearm: [-0.75, 0.0, 0.0], leftWrist: [-0.1, 0.0, 0.0],
        leftFingers: FIST_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [0.06, 0.0, 0.0], head: [0.06, 0.0, 0.0] },
        rightArm: [-0.55, 0.1, 0.0], rightForearm: [-0.75, 0.0, 0.0], rightWrist: [-0.1, 0.0, 0.0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.65, -0.1, 0.0], leftForearm: [-0.85, 0.0, 0.0], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: FIST_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.06, 0.0, 0.0], head: [0.06, 0.0, 0.0] },
        rightArm: [-0.65, 0.1, 0.0], rightForearm: [-0.85, 0.0, 0.0], rightWrist: [0.1, 0.0, 0.0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.55, -0.1, 0.0], leftForearm: [-0.75, 0.0, 0.0], leftWrist: [-0.1, 0.0, 0.0],
        leftFingers: FIST_FINGERS
      }
    ]
  },

  // KHỎE MẠNH (STRONG): Hai cánh tay gập gồng cơ bắp 90 độ
  KHOE_MANH: {
    name: 'Khỏe mạnh (Strong)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [-0.04, 0.0, 0.0], head: [-0.02, 0.0, 0.0] }, // Ưỡn ngực
        rightArm: [-1.2, 0.5, 0.4], rightForearm: [-1.15, 0.2, 0.1], rightWrist: [0.2, 0.0, 0.0],
        rightFingers: FIST_FINGERS,
        leftArm: [-1.2, -0.5, -0.4], leftForearm: [-1.15, -0.2, -0.1], leftWrist: [0.2, 0.0, 0.0],
        leftFingers: FIST_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [-0.06, 0.0, 0.0], head: [-0.04, 0.0, 0.0] },
        rightArm: [-1.25, 0.55, 0.45], rightForearm: [-1.25, 0.2, 0.1], rightWrist: [0.3, 0.0, 0.0],
        rightFingers: FIST_FINGERS,
        leftArm: [-1.25, -0.55, -0.45], leftForearm: [-1.25, -0.2, -0.1], leftWrist: [0.3, 0.0, 0.0],
        leftFingers: FIST_FINGERS
      }
    ]
  },

  // MỆT MỎI (TIRED): Rũ người buông thõng
  MET_MOI: {
    name: 'Mệt mỏi (Tired)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.10, 0.0, 0.0] },
        rightArm: [-0.3, 0.1, 0.1], rightForearm: [-0.4, 0.0, 0.0], rightWrist: [0, 0, 0],
        rightFingers: NATURAL_FINGERS_REST,
        leftArm: [-0.3, -0.1, -0.1], leftForearm: [-0.4, 0.0, 0.0], leftWrist: [0, 0, 0],
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.18, 0.0, 0.0], head: [0.25, 0.0, 0.0] }, // Cúi rũ mệt mỏi
        rightArm: [-0.15, 0.05, 0.05], rightForearm: [-0.2, 0.0, 0.0], rightWrist: [0.2, 0, 0],
        rightFingers: NATURAL_FINGERS_REST,
        leftArm: [-0.15, -0.05, -0.05], leftForearm: [-0.2, 0.0, 0.0], leftWrist: [0.2, 0, 0],
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  // VUI VẺ (HAPPY): Hai tay vẫy rung nhẹ trước ngực, đầu ngẩng cao
  VUI_VE: {
    name: 'Vui vẻ (Happy)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [-0.02, 0.0, 0.0], head: [-0.04, 0.0, 0.0] },
        rightArm: [-0.75, 0.3, 0.1], rightForearm: [-0.65, 0.2, 0.1], rightWrist: [0.2, 0.1, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.75, -0.3, -0.1], leftForearm: [-0.65, -0.2, -0.1], leftWrist: [0.2, -0.1, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [-0.05, 0.0, 0.0], head: [-0.06, 0.0, 0.0] },
        rightArm: [-0.85, 0.35, 0.15], rightForearm: [-0.55, 0.25, 0.15], rightWrist: [-0.1, -0.1, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.85, -0.35, -0.15], leftForearm: [-0.55, -0.25, -0.15], leftWrist: [-0.1, 0.1, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [-0.02, 0.0, 0.0], head: [-0.04, 0.0, 0.0] },
        rightArm: [-0.75, 0.3, 0.1], rightForearm: [-0.65, 0.2, 0.1], rightWrist: [0.2, 0.1, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.75, -0.3, -0.1], leftForearm: [-0.65, -0.2, -0.1], leftWrist: [0.2, -0.1, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // BUỒN (SAD): Bàn tay vuốt ngực xuống
  BUON: {
    name: 'Buồn (Sad)',
    twoHanded: false,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.06, 0.0, 0.0], head: [0.1, 0.0, 0.0] },
        rightArm: [-0.6, -0.2, 0.0], rightForearm: [-0.95, 0.1, 0.0], rightWrist: [-0.2, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.35, -0.1, -0.1], leftForearm: [-0.5, 0.0, 0.0], leftWrist: [0, 0, 0],
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.12, 0.0, 0.0], head: [0.18, 0.0, 0.0] },
        rightArm: [-0.4, -0.2, 0.0], rightForearm: [-0.85, 0.1, 0.0], rightWrist: [0.2, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.35, -0.1, -0.1], leftForearm: [-0.5, 0.0, 0.0], leftWrist: [0, 0, 0],
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  // CHO / TẶNG (GIVE): Hai tay ngửa đưa ra trước
  CHO: {
    name: 'Cho / Tặng (Give)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.04, 0.0, 0.0], head: [0.04, 0.0, 0.0] },
        rightArm: [-0.5, 0.1, 0.0], rightForearm: [-0.95, 0.0, 0.0], rightWrist: [0.25, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.5, -0.1, 0.0], leftForearm: [-0.95, 0.0, 0.0], leftWrist: [0.25, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.06, 0.0, 0.0] }, // Nghiêng người trao
        rightArm: [-0.85, 0.1, 0.0], rightForearm: [-0.5, 0.0, 0.0], rightWrist: [0.35, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.85, -0.1, 0.0], leftForearm: [-0.5, 0.0, 0.0], leftWrist: [0.35, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // XIN / NHẬN (RECEIVE): Hai tay ngửa kéo về phía ngực
  XIN: {
    name: 'Xin / Nhận (Receive)',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.08, 0.0, 0.0], head: [0.06, 0.0, 0.0] },
        rightArm: [-0.85, 0.1, 0.0], rightForearm: [-0.5, 0.0, 0.0], rightWrist: [0.35, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.85, -0.1, 0.0], leftForearm: [-0.5, 0.0, 0.0], leftWrist: [0.35, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.04, 0.0, 0.0], head: [0.04, 0.0, 0.0] },
        rightArm: [-0.5, 0.1, 0.0], rightForearm: [-0.95, 0.0, 0.0], rightWrist: [0.25, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.5, -0.1, 0.0], leftForearm: [-0.95, 0.0, 0.0], leftWrist: [0.25, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  // ==========================================
  // CÁC CỬ CHỈ CÂU DÀI & ĐỜI SỐNG (LONG SENTENCE VSL)
  // ==========================================
  CUOI_TUAN: {
    name: 'Cuối tuần',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.02, 0.02, 0], head: [0.02, 0.02, 0] },
        rightArm: [-0.65, 0.2, 0.0], rightForearm: [-0.85, 0.1, 0.1], rightWrist: [0.0, 0.2, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.55, -0.2, 0.0], leftForearm: [-0.75, -0.1, -0.1], leftWrist: [0.0, -0.2, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [0.04, 0.0, 0], head: [0.04, 0.0, 0] },
        rightArm: [-0.75, -0.1, 0.1], rightForearm: [-0.75, 0.3, 0.0], rightWrist: [0.2, 0.0, 0.1],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.55, -0.2, 0.0], leftForearm: [-0.75, -0.1, -0.1], leftWrist: [0.0, -0.2, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.02, -0.02, 0], head: [0.02, -0.02, 0] },
        rightArm: [-0.5, 0.2, 0.1], rightForearm: [-0.9, 0.1, 0.0], rightWrist: [0.1, 0.0, 0.0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.5, -0.2, -0.1], leftForearm: [-0.9, -0.1, 0.0], leftWrist: [0.1, 0.0, 0.0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  LICH_TRINH: {
    name: 'Lịch trình / Kế hoạch',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.05, 0, 0], head: [0.08, 0, 0] },
        rightArm: [-0.7, 0.15, 0.1], rightForearm: [-0.85, 0.2, 0.1], rightWrist: [0.1, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.3, cmc_z: 0.2, mcp_x: 0.5, ip_x: 0.4 }, index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.05 }, middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: -0.05 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.5, -0.2, 0], leftForearm: [-0.85, -0.2, 0.1], leftWrist: [-0.2, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.06, 0, 0], head: [0.1, 0, 0] },
        rightArm: [-0.65, 0.1, 0.05], rightForearm: [-0.95, 0.2, 0.1], rightWrist: [0.3, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.3, cmc_z: 0.2, mcp_x: 0.5, ip_x: 0.4 }, index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.05 }, middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: -0.05 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.5, -0.2, 0], leftForearm: [-0.85, -0.2, 0.1], leftWrist: [-0.2, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  PHU_THUOC: {
    name: 'Phụ thuộc',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, 0, 0], head: [0.03, 0, 0] },
        rightArm: [-0.6, 0.1, 0], rightForearm: [-0.85, 0.15, 0], rightWrist: [0, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0.8, pip_x: 1.0, dip_x: 0.6, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.6, -0.1, 0], leftForearm: [-0.85, -0.15, 0], leftWrist: [0, 0, 0],
        leftFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0.8, pip_x: 1.0, dip_x: 0.6, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } }
      },
      {
        t: 1.0,
        body: { torso: [0.05, 0, 0], head: [0.05, 0, 0] },
        rightArm: [-0.65, 0.08, 0], rightForearm: [-0.8, 0.1, 0], rightWrist: [0.1, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0.8, pip_x: 1.0, dip_x: 0.6, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.65, -0.08, 0], leftForearm: [-0.8, -0.1, 0], leftWrist: [0.1, 0, 0],
        leftFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0.8, pip_x: 1.0, dip_x: 0.6, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } }
      }
    ]
  },

  BAN_RON: {
    name: 'Bận rộn',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, 0.03, 0], head: [0.02, 0.03, 0] },
        rightArm: [-0.6, 0.1, 0], rightForearm: [-0.9, 0.2, 0.1], rightWrist: [0, 0.3, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.6, -0.1, 0], leftForearm: [-0.9, -0.2, -0.1], leftWrist: [0, -0.3, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [0.03, -0.03, 0], head: [0.02, -0.03, 0] },
        rightArm: [-0.65, -0.05, 0], rightForearm: [-0.85, 0.1, 0.1], rightWrist: [0, -0.3, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.65, 0.05, 0], leftForearm: [-0.85, -0.1, -0.1], leftWrist: [0, 0.3, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.03, 0.03, 0], head: [0.02, 0.03, 0] },
        rightArm: [-0.6, 0.1, 0], rightForearm: [-0.9, 0.2, 0.1], rightWrist: [0, 0.3, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.6, -0.1, 0], leftForearm: [-0.9, -0.2, -0.1], leftWrist: [0, -0.3, 0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  TRONG_TUAN: {
    name: 'Trong tuần',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.02, 0, 0], head: [0.04, 0, 0] },
        rightArm: [-0.65, 0.1, 0], rightForearm: [-0.8, 0.15, 0], rightWrist: [0, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.5, -0.2, 0], leftForearm: [-0.85, -0.2, 0], leftWrist: [-0.2, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.02, 0, 0], head: [0.04, 0, 0] },
        rightArm: [-0.65, 0.0, 0], rightForearm: [-0.95, 0.1, 0], rightWrist: [0.2, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.5, -0.2, 0], leftForearm: [-0.85, -0.2, 0], leftWrist: [-0.2, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  THOI_GIAN: {
    name: 'Thời gian',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, -0.02, 0], head: [0.06, 0.02, 0] },
        rightArm: [-0.6, -0.1, 0], rightForearm: [-0.8, 0.2, 0], rightWrist: [0.1, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.45, -0.25, 0], leftForearm: [-0.85, -0.25, 0], leftWrist: [0, 0, 0],
        leftFingers: FIST_FINGERS
      },
      {
        t: 0.5,
        body: { torso: [0.04, -0.02, 0], head: [0.08, 0.02, 0] },
        rightArm: [-0.55, -0.15, 0], rightForearm: [-0.75, 0.25, 0], rightWrist: [0.2, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.45, -0.25, 0], leftForearm: [-0.85, -0.25, 0], leftWrist: [0, 0, 0],
        leftFingers: FIST_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.03, -0.02, 0], head: [0.06, 0.02, 0] },
        rightArm: [-0.6, -0.1, 0], rightForearm: [-0.8, 0.2, 0], rightWrist: [0.1, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.45, -0.25, 0], leftForearm: [-0.85, -0.25, 0], leftWrist: [0, 0, 0],
        leftFingers: FIST_FINGERS
      }
    ]
  },

  IT: {
    name: 'Một ít',
    twoHanded: false,
    body: { torso: [0.02, 0.02, 0], head: [0.02, 0.04, 0] },
    rightArm: [-0.7, 0.15, 0.1], rightForearm: [-0.85, 0.2, 0.1], rightWrist: [0.1, 0.2, 0],
    rightFingers: {
      thumb: { cmc_y: 0.4, cmc_z: 0.3, mcp_x: 0.4, ip_x: 0.3 },
      index: { mcp_x: 0.4, pip_x: 0.5, dip_x: 0.3, spread_z: 0 },
      middle: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0 },
      ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0 },
      pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  DU_AN: {
    name: 'Dự án',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, 0, 0], head: [0.03, 0, 0] },
        rightArm: [-0.7, 0.15, 0], rightForearm: [-0.85, 0.15, 0], rightWrist: [0, 0, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.7, -0.15, 0], leftForearm: [-0.85, -0.15, 0], leftWrist: [0, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.04, 0, 0], head: [0.04, 0, 0] },
        rightArm: [-0.55, 0.05, 0], rightForearm: [-0.95, 0.1, 0], rightWrist: [0.2, 0, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.55, -0.05, 0], leftForearm: [-0.95, -0.1, 0], leftWrist: [0.2, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  CA_NHAN: {
    name: 'Cá nhân',
    twoHanded: false,
    body: { torso: [0.04, 0, 0], head: [0.04, 0, 0] },
    rightArm: [-0.5, 0.0, 0.1], rightForearm: [-0.9, 0.1, 0.1], rightWrist: [0.2, 0, 0],
    rightFingers: FLAT_OPEN_FINGERS,
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  DAM_BAO: {
    name: 'Đảm bảo',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.02, 0, 0], head: [0.02, 0, 0] },
        rightArm: [-0.8, 0.1, 0], rightForearm: [-0.7, 0.2, 0], rightWrist: [0.1, 0, 0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.5, -0.2, 0], leftForearm: [-0.85, -0.2, 0], leftWrist: [-0.1, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.06, 0, 0], head: [0.08, 0, 0] },
        rightArm: [-0.55, 0.0, 0], rightForearm: [-0.95, 0.1, 0], rightWrist: [0.3, 0, 0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.5, -0.2, 0], leftForearm: [-0.85, -0.2, 0], leftWrist: [-0.1, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  DU: {
    name: 'Đủ',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, 0, 0], head: [0.03, 0, 0] },
        rightArm: [-0.65, 0.15, 0], rightForearm: [-0.85, 0.2, 0], rightWrist: [0, 0, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.5, -0.15, 0], leftForearm: [-0.85, -0.15, 0], leftWrist: [0, 0, 0],
        leftFingers: FIST_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.04, 0, 0], head: [0.04, 0, 0] },
        rightArm: [-0.65, -0.1, 0], rightForearm: [-0.85, 0.1, 0], rightWrist: [0.2, 0, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.5, -0.15, 0], leftForearm: [-0.85, -0.15, 0], leftWrist: [0, 0, 0],
        leftFingers: FIST_FINGERS
      }
    ]
  },

  THU_GIAN: {
    name: 'Thư giãn / Nghỉ ngơi',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [-0.02, 0, 0], head: [-0.04, 0, 0] },
        rightArm: [-0.5, 0.15, 0], rightForearm: [-0.7, 0.15, 0], rightWrist: [0, 0, 0],
        rightFingers: NATURAL_FINGERS_REST,
        leftArm: [-0.5, -0.15, 0], leftForearm: [-0.7, -0.15, 0], leftWrist: [0, 0, 0],
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [-0.05, 0, 0], head: [-0.08, 0, 0] },
        rightArm: [-0.35, 0.1, 0], rightForearm: [-0.5, 0.1, 0], rightWrist: [-0.2, 0, 0],
        rightFingers: NATURAL_FINGERS_REST,
        leftArm: [-0.35, -0.1, 0], leftForearm: [-0.5, -0.1, 0], leftWrist: [-0.2, 0, 0],
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  NEU: {
    name: 'Nếu',
    twoHanded: false,
    body: { torso: [0.03, 0.03, 0], head: [0.04, 0.05, 0] },
    rightArm: [-0.8, 0.2, 0.1], rightForearm: [-0.85, 0.2, 0.1], rightWrist: [0.2, 0.1, 0],
    rightFingers: {
      thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 },
      index: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 },
      middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 },
      ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 },
      pinky: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  CO_THE: {
    name: 'Có thể',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.02, 0, 0], head: [0.02, 0, 0] },
        rightArm: [-0.7, 0.15, 0], rightForearm: [-0.75, 0.15, 0], rightWrist: [-0.1, 0, 0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.7, -0.15, 0], leftForearm: [-0.75, -0.15, 0], leftWrist: [-0.1, 0, 0],
        leftFingers: FIST_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.05, 0, 0], head: [0.06, 0, 0] },
        rightArm: [-0.65, 0.15, 0], rightForearm: [-0.9, 0.15, 0], rightWrist: [0.2, 0, 0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.65, -0.15, 0], leftForearm: [-0.9, -0.15, 0], leftWrist: [0.2, 0, 0],
        leftFingers: FIST_FINGERS
      }
    ]
  },

  THICH: {
    name: 'Thích',
    twoHanded: false,
    body: { torso: [0.02, -0.02, 0], head: [0.02, 0.02, 0] },
    rightArm: [-0.55, 0.0, 0.1], rightForearm: [-0.9, 0.15, 0.1], rightWrist: [0.1, 0, 0],
    rightFingers: {
      thumb: { cmc_y: 0.4, cmc_z: 0.3, mcp_x: 0.3, ip_x: 0.2 },
      index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0.1 },
      middle: { mcp_x: 0.4, pip_x: 0.5, dip_x: 0.3, spread_z: 0 },
      ring: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: -0.1 },
      pinky: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: -0.15 }
    },
    leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
    leftFingers: NATURAL_FINGERS_REST
  },

  GAP_GO: {
    name: 'Gặp gỡ',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, 0, 0], head: [0.03, 0, 0] },
        rightArm: [-0.7, 0.25, 0.1], rightForearm: [-0.85, 0.2, 0], rightWrist: [0, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.7, -0.25, 0.1], leftForearm: [-0.85, -0.2, 0], leftWrist: [0, 0, 0],
        leftFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } }
      },
      {
        t: 1.0,
        body: { torso: [0.04, 0, 0], head: [0.05, 0, 0] },
        rightArm: [-0.65, 0.05, 0], rightForearm: [-0.9, 0.1, 0], rightWrist: [0.1, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.65, -0.05, 0], leftForearm: [-0.9, -0.1, 0], leftWrist: [0.1, 0, 0],
        leftFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } }
      }
    ]
  },

  BAN_BE: {
    name: 'Bạn bè',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, 0, 0], head: [0.03, 0, 0] },
        rightArm: [-0.6, 0.1, 0], rightForearm: [-0.85, 0.15, 0], rightWrist: [0.1, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0.7, pip_x: 0.8, dip_x: 0.5, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.6, -0.1, 0], leftForearm: [-0.85, -0.15, 0], leftWrist: [0.1, 0, 0],
        leftFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0.7, pip_x: 0.8, dip_x: 0.5, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } }
      },
      {
        t: 1.0,
        body: { torso: [0.04, 0, 0], head: [0.04, 0, 0] },
        rightArm: [-0.65, 0.08, 0], rightForearm: [-0.8, 0.1, 0], rightWrist: [-0.1, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0.7, pip_x: 0.8, dip_x: 0.5, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.65, -0.08, 0], leftForearm: [-0.8, -0.1, 0], leftWrist: [-0.1, 0, 0],
        leftFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0.7, pip_x: 0.8, dip_x: 0.5, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } }
      }
    ]
  },

  KHAM_PHA: {
    name: 'Khám phá',
    twoHanded: false,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.04, 0.04, 0], head: [0.04, 0.06, 0] },
        rightArm: [-0.85, 0.15, 0.1], rightForearm: [-0.95, 0.15, 0.1], rightWrist: [0, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.5, cmc_z: 0.4, mcp_x: 0.5, ip_x: 0.4 }, index: { mcp_x: 0.6, pip_x: 0.6, dip_x: 0.4, spread_z: 0 }, middle: { mcp_x: 0.6, pip_x: 0.6, dip_x: 0.4, spread_z: 0 }, ring: { mcp_x: 0.6, pip_x: 0.6, dip_x: 0.4, spread_z: 0 }, pinky: { mcp_x: 0.6, pip_x: 0.6, dip_x: 0.4, spread_z: 0 } },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.06, 0.06, 0], head: [0.06, 0.08, 0] },
        rightArm: [-0.7, 0.25, 0.1], rightForearm: [-0.7, 0.25, 0.1], rightWrist: [0.2, 0.2, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  DIA_DIEM: {
    name: 'Địa điểm',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.04, 0, 0], head: [0.06, 0, 0] },
        rightArm: [-0.6, 0.15, 0], rightForearm: [-0.85, 0.15, 0], rightWrist: [0, 0, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.6, -0.15, 0], leftForearm: [-0.85, -0.15, 0], leftWrist: [0, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.06, 0, 0], head: [0.08, 0, 0] },
        rightArm: [-0.55, 0.0, 0], rightForearm: [-0.95, 0.1, 0], rightWrist: [0.3, 0, 0],
        rightFingers: { thumb: { cmc_y: 0.2, cmc_z: 0.2, mcp_x: 0.3, ip_x: 0.2 }, index: { mcp_x: 0, pip_x: 0, dip_x: 0, spread_z: 0 }, middle: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, ring: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 }, pinky: { mcp_x: 1.5, pip_x: 1.5, dip_x: 1.3, spread_z: 0 } },
        leftArm: [-0.55, -0.1, 0], leftForearm: [-0.9, -0.15, 0], leftWrist: [0, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  MOI: {
    name: 'Mới',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.02, 0, 0], head: [0.02, 0, 0] },
        rightArm: [-0.6, 0.1, 0], rightForearm: [-0.85, 0.1, 0], rightWrist: [0, 0, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.5, -0.15, 0], leftForearm: [-0.85, -0.15, 0], leftWrist: [0, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.04, 0, 0], head: [0.05, 0, 0] },
        rightArm: [-0.8, 0.15, 0], rightForearm: [-0.7, 0.2, 0], rightWrist: [0.3, 0, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.5, -0.15, 0], leftForearm: [-0.85, -0.15, 0], leftWrist: [0, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      }
    ]
  },

  CO_HOI: {
    name: 'Cơ hội',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.04, 0, 0], head: [0.04, 0, 0] },
        rightArm: [-0.6, 0.15, 0], rightForearm: [-0.8, 0.15, 0], rightWrist: [-0.2, 0, 0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: [-0.6, -0.15, 0], leftForearm: [-0.8, -0.15, 0], leftWrist: [-0.2, 0, 0],
        leftFingers: FLAT_OPEN_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.05, 0, 0], head: [0.06, 0, 0] },
        rightArm: [-0.5, 0.1, 0], rightForearm: [-0.95, 0.1, 0], rightWrist: [0.2, 0, 0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.5, -0.1, 0], leftForearm: [-0.95, -0.1, 0], leftWrist: [0.2, 0, 0],
        leftFingers: FIST_FINGERS
      }
    ]
  },

  THOI_QUEN: {
    name: 'Thói quen',
    twoHanded: true,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, 0, 0], head: [0.04, 0, 0] },
        rightArm: [-0.65, 0.05, 0], rightForearm: [-0.8, 0.1, 0], rightWrist: [0.2, 0, 0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.55, -0.1, 0], leftForearm: [-0.85, -0.1, 0], leftWrist: [0, 0, 0],
        leftFingers: FIST_FINGERS
      },
      {
        t: 1.0,
        body: { torso: [0.06, 0, 0], head: [0.07, 0, 0] },
        rightArm: [-0.5, 0.0, 0], rightForearm: [-0.95, 0.1, 0], rightWrist: [0.3, 0, 0],
        rightFingers: FIST_FINGERS,
        leftArm: [-0.55, -0.1, 0], leftForearm: [-0.85, -0.1, 0], leftWrist: [0, 0, 0],
        leftFingers: FIST_FINGERS
      }
    ]
  },

  HANG_NGAY: {
    name: 'Hàng ngày',
    twoHanded: false,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.02, 0.02, 0], head: [0.02, 0.04, 0] },
        rightArm: SIGNING_RIGHT_ARM, rightForearm: SIGNING_RIGHT_FOREARM, rightWrist: [0.1, 0.2, 0],
        rightFingers: FIST_FINGERS,
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.04, 0.02, 0], head: [0.04, 0.04, 0] },
        rightArm: [-0.7, 0.15, 0.1], rightForearm: [-0.8, 0.15, 0.1], rightWrist: [0.2, 0.1, 0],
        rightFingers: FIST_FINGERS,
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  // ĂN (EAT): Tay phải chụm búp ngón tay đưa lên sát miệng (Chuẩn VSL)
  AN: {
    name: 'Ăn (Eat)',
    twoHanded: false,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.04, 0.0, 0.0], head: [0.06, 0.0, 0.0] },
        rightArm: [-1.15, 0.15, -0.05], rightForearm: [-1.20, 0.15, 0.05], rightWrist: [0.20, 0.10, 0.0],
        rightFingers: {
          thumb: { cmc_x: 0.35, cmc_y: 0.60, cmc_z: 0.40, mcp_x: 0.80, ip_x: 0.60 },
          index: { mcp_x: 0.85, pip_x: 0.85, dip_x: 0.60, spread_z: 0.0 },
          middle: { mcp_x: 0.90, pip_x: 0.90, dip_x: 0.65, spread_z: 0.0 },
          ring: { mcp_x: 0.95, pip_x: 0.95, dip_x: 0.70, spread_z: 0.0 },
          pinky: { mcp_x: 1.00, pip_x: 1.00, dip_x: 0.75, spread_z: 0.0 }
        },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 0.5,
        body: { torso: [0.06, 0.0, 0.0], head: [0.10, 0.0, 0.0] },
        rightArm: [-1.35, 0.20, -0.05], rightForearm: [-1.35, 0.20, 0.10], rightWrist: [0.35, 0.15, 0.0],
        rightFingers: {
          thumb: { cmc_x: 0.35, cmc_y: 0.60, cmc_z: 0.40, mcp_x: 0.80, ip_x: 0.60 },
          index: { mcp_x: 0.85, pip_x: 0.85, dip_x: 0.60, spread_z: 0.0 },
          middle: { mcp_x: 0.90, pip_x: 0.90, dip_x: 0.65, spread_z: 0.0 },
          ring: { mcp_x: 0.95, pip_x: 0.95, dip_x: 0.70, spread_z: 0.0 },
          pinky: { mcp_x: 1.00, pip_x: 1.00, dip_x: 0.75, spread_z: 0.0 }
        },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.04, 0.0, 0.0], head: [0.06, 0.0, 0.0] },
        rightArm: [-1.15, 0.15, -0.05], rightForearm: [-1.20, 0.15, 0.05], rightWrist: [0.20, 0.10, 0.0],
        rightFingers: {
          thumb: { cmc_x: 0.35, cmc_y: 0.60, cmc_z: 0.40, mcp_x: 0.80, ip_x: 0.60 },
          index: { mcp_x: 0.85, pip_x: 0.85, dip_x: 0.60, spread_z: 0.0 },
          middle: { mcp_x: 0.90, pip_x: 0.90, dip_x: 0.65, spread_z: 0.0 },
          ring: { mcp_x: 0.95, pip_x: 0.95, dip_x: 0.70, spread_z: 0.0 },
          pinky: { mcp_x: 1.00, pip_x: 1.00, dip_x: 0.75, spread_z: 0.0 }
        },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  // NGỦ (SLEEP): Áp bàn tay phải khum vào má phải, đầu nghiêng sang phải (Chuẩn VSL)
  NGU: {
    name: 'Ngủ (Sleep)',
    twoHanded: false,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.02, 0.0, 0.0], head: [0.02, 0.0, 0.0] },
        rightArm: [-0.80, 0.15, -0.05], rightForearm: [-0.80, 0.15, 0.05], rightWrist: [0.10, 0.0, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.05, 0.05, -0.10], head: [0.10, 0.10, -0.30] },
        rightArm: [-1.45, 0.25, -0.10], rightForearm: [-1.30, 0.30, 0.15], rightWrist: [-0.30, 0.20, -0.15],
        rightFingers: {
          thumb: { cmc_x: 0.1, cmc_y: 0.4, cmc_z: 0.3, mcp_x: 0.3, ip_x: 0.2 },
          index: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.15, spread_z: 0.0 },
          middle: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.15, spread_z: 0.0 },
          ring: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.15, spread_z: 0.0 },
          pinky: { mcp_x: 0.2, pip_x: 0.2, dip_x: 0.15, spread_z: 0.0 }
        },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  // UỐNG (DRINK): Tay phải hình chữ C đưa lên miệng như cầm cốc uống nước
  UONG: {
    name: 'Uống (Drink)',
    twoHanded: false,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, 0.0, 0.0], head: [0.04, 0.0, 0.0] },
        rightArm: [-1.05, 0.10, -0.05], rightForearm: [-1.10, 0.15, 0.05], rightWrist: [0.10, 0.0, 0.0],
        rightFingers: {
          thumb: { cmc_x: 0.3, cmc_y: 0.35, cmc_z: 0.2, mcp_x: 0.6, ip_x: 0.5 },
          index: { mcp_x: 0.7, pip_x: 0.7, dip_x: 0.5, spread_z: 0.0 },
          middle: { mcp_x: 0.75, pip_x: 0.75, dip_x: 0.5, spread_z: 0.0 },
          ring: { mcp_x: 0.75, pip_x: 0.75, dip_x: 0.5, spread_z: 0.0 },
          pinky: { mcp_x: 0.75, pip_x: 0.75, dip_x: 0.5, spread_z: 0.0 }
        },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.05, 0.0, 0.0], head: [-0.08, 0.0, 0.0] }, // Hơi ngửa đầu uống nước
        rightArm: [-1.35, 0.15, -0.05], rightForearm: [-1.30, 0.20, 0.10], rightWrist: [0.55, 0.10, 0.0],
        rightFingers: {
          thumb: { cmc_x: 0.3, cmc_y: 0.35, cmc_z: 0.2, mcp_x: 0.6, ip_x: 0.5 },
          index: { mcp_x: 0.7, pip_x: 0.7, dip_x: 0.5, spread_z: 0.0 },
          middle: { mcp_x: 0.75, pip_x: 0.75, dip_x: 0.5, spread_z: 0.0 },
          ring: { mcp_x: 0.75, pip_x: 0.75, dip_x: 0.5, spread_z: 0.0 },
          pinky: { mcp_x: 0.75, pip_x: 0.75, dip_x: 0.5, spread_z: 0.0 }
        },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  // ĐI (WALK / GO): Tay phải ngón trỏ và giữa tạo chữ V úp xuống bước đi
  DI: {
    name: 'Đi (Walk / Go)',
    twoHanded: false,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.05, 0.0, 0.0], head: [0.06, 0.0, 0.0] },
        rightArm: [-1.15, 0.10, -0.05], rightForearm: [-1.05, 0.15, 0.05], rightWrist: [-0.60, 0.0, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.4, cmc_z: 0.3, mcp_x: 1.2, ip_x: 0.8 },
          index: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: -0.15 },
          middle: { mcp_x: 0.6, pip_x: 0.6, dip_x: 0.4, spread_z: 0.15 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.07, 0.0, 0.0], head: [0.07, 0.0, 0.0] },
        rightArm: [-1.20, 0.10, -0.05], rightForearm: [-0.95, 0.15, 0.05], rightWrist: [-0.60, 0.0, 0.0],
        rightFingers: {
          thumb: { cmc_y: 0.4, cmc_z: 0.3, mcp_x: 1.2, ip_x: 0.8 },
          index: { mcp_x: 0.6, pip_x: 0.6, dip_x: 0.4, spread_z: -0.15 },
          middle: { mcp_x: 0.0, pip_x: 0.0, dip_x: 0.0, spread_z: 0.15 },
          ring: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 },
          pinky: { mcp_x: 1.57, pip_x: 1.57, dip_x: 1.4, spread_z: 0.0 }
        },
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  },

  // KHÔNG (NO): Lắc nhẹ bàn tay phải và lắc đầu từ chối
  KHONG: {
    name: 'Không (No)',
    twoHanded: false,
    isDynamic: true,
    keyframes: [
      {
        t: 0.0,
        body: { torso: [0.03, 0.06, 0.0], head: [0.02, 0.12, 0.0] },
        rightArm: [-1.15, 0.15, -0.05], rightForearm: [-1.10, 0.20, 0.05], rightWrist: [0.15, 0.25, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      },
      {
        t: 1.0,
        body: { torso: [0.03, -0.06, 0.0], head: [0.02, -0.12, 0.0] },
        rightArm: [-1.15, 0.15, -0.05], rightForearm: [-1.10, 0.20, 0.05], rightWrist: [0.15, -0.25, 0.0],
        rightFingers: FLAT_OPEN_FINGERS,
        leftArm: PASSIVE_LEFT_ARM, leftForearm: PASSIVE_LEFT_FOREARM, leftWrist: PASSIVE_LEFT_WRIST,
        leftFingers: NATURAL_FINGERS_REST
      }
    ]
  }
};

/**
 * Tìm tư thế chuẩn theo mã Token / Gloss (hỗ trợ cả tiếng Anh và tiếng Việt)
 */
export function getHandPoseByGloss(gloss) {
  if (!gloss) return HAND_POSES.REST;
  const clean = gloss.toUpperCase().replace('ALPHA_', '').trim();

  // Bảng ánh xạ bí danh linh hoạt
  const ALIAS_MAP = {
    'PLEASE': 'XIN_VUI_LONG',
    'LAM_ON': 'XIN_VUI_LONG',
    'VUI_LONG': 'XIN_VUI_LONG',
    'HELLO': 'XIN_CHAO',
    'CHAO': 'XIN_CHAO',
    'HI': 'XIN_CHAO',
    'THANK_YOU': 'CAM_ON',
    'THANKS': 'CAM_ON',
    'HELP': 'GIUP_DO',
    'LOVE': 'YEU_THUONG',
    'HUG': 'YEU_THUONG',
    'FAMILY': 'GIA_DINH',
    'NHA': 'GIA_DINH',
    // Sinh hoạt đời sống
    'HOC': 'HOC_BAI',
    'HOC_TAP': 'HOC_BAI',
    'STUDY': 'HOC_BAI',
    'NAU': 'NAU_AN',
    'COOK': 'NAU_AN',
    'LAU': 'LAU_NHA',
    'QUET_NHA': 'LAU_NHA',
    'MOP': 'LAU_NHA',
    'CLEAN': 'LAU_NHA',
    'AN': 'AN',
    'EAT': 'AN',
    'NGU': 'NGU',
    'DI_NGU': 'NGU',
    'SLEEP': 'NGU',
    'UONG': 'UONG',
    'DRINK': 'UONG',
    'DI': 'DI',
    'WALK': 'DI',
    'GO': 'DI',
    'BAN': 'BAN',
    'YOU': 'BAN',
    'BAN_BE': 'BAN',
    'FRIEND': 'BAN',
    'TOI': 'TOI',
    'ME': 'TOI',
    'MY': 'TOI',
    'I': 'TOI',
    'RUA': 'RUA_TAY',
    'WASH': 'RUA_TAY',
    'DOC': 'DOC_SACH',
    'READ': 'DOC_SACH',
    'VIET': 'VIET_BAI',
    'WRITE': 'VIET_BAI',
    'DI_LAM': 'LAM_VIEC',
    'WORK': 'LAM_VIEC',
    'KHOE': 'KHOE_MANH',
    'STRONG': 'KHOE_MANH',
    'MET': 'MET_MOI',
    'TIRED': 'MET_MOI',
    'VUI': 'VUI_VE',
    'HAPPY': 'VUI_VE',
    'SAD': 'BUON',
    'GIVE': 'CHO',
    'TANG': 'CHO',
    'RECEIVE': 'XIN',
    'GET': 'XIN',
    'YEU': 'YEU_THUONG',
    // Cụm từ câu dài
    'WEEKEND': 'CUOI_TUAN',
    'SCHEDULE': 'LICH_TRINH',
    'PLAN': 'LICH_TRINH',
    'DEPEND': 'PHU_THUOC',
    'BUSY': 'BAN_RON',
    'WEEKDAY': 'TRONG_TUAN',
    'TIME': 'THOI_GIAN',
    'FEW': 'IT',
    'LITTLE': 'IT',
    'PROJECT': 'DU_AN',
    'PERSONAL': 'CA_NHAN',
    'SELF': 'CA_NHAN',
    'SURE': 'DAM_BAO',
    'ENOUGH': 'DU',
    'RELAX': 'THU_GIAN',
    'REST': 'THU_GIAN',
    'IF': 'NEU',
    'CAN': 'CO_THE',
    'LIKE': 'THICH',
    'MEET': 'GAP_GO',
    'EXPLORE': 'KHAM_PHA',
    'PLACE': 'DIA_DIEM',
    'LOCATION': 'DIA_DIEM',
    'NEW': 'MOI',
    'CHANCE': 'CO_HOI',
    'HABIT': 'THOI_QUEN',
    'DAILY': 'HANG_NGAY',
    'EVERYDAY': 'HANG_NGAY',
    'NO': 'KHONG',
    'NOT': 'KHONG'
  };

  const resolved = ALIAS_MAP[clean] || clean;
  return HAND_POSES[resolved] || HAND_POSES[clean] || HAND_POSES.REST;
}


