/**
 * Real-time Camera Framing Validator & Low-End Adaptive Performance Controller.
 *
 * 1. `evaluateCameraFraming`: Zero-allocation check that required keypoints (both shoulders,
 *    head reference, and active fingertips/wrists) stay fully inside the camera frame without
 *    border clipping.
 * 2. `AdaptivePerfController`: Dynamically adjusts upper-body Pose cadence (1/1, 1/2, or 1/3)
 *    and processing resolution on weak/low-end machines while keeping Hand/Finger tracking at
 *    100% frame cadence (every frame) so rapid hand swings never drop finger landmarks.
 */

const EDGE_MARGIN = 0.028;
const SHOULDER_MIN_VIS = 0.35;
const HAND_CHECK_INDICES = [0, 4, 8, 12, 16, 20];

function isFinitePoint(pt) {
  return Boolean(pt && Number.isFinite(pt.x) && Number.isFinite(pt.y) && !(pt.x === 0 && pt.y === 0));
}

function findHandEdgeClip(hand) {
  if (!hand || hand.length < 21) return null;
  for (let k = 0; k < HAND_CHECK_INDICES.length; k++) {
    const pt = hand[HAND_CHECK_INDICES[k]];
    if (!isFinitePoint(pt)) continue;
    if (pt.y < EDGE_MARGIN) return 'top';
    if (pt.y > 1 - EDGE_MARGIN) return 'bottom';
    if (pt.x < EDGE_MARGIN || pt.x > 1 - EDGE_MARGIN) return 'side';
  }
  return null;
}

/**
 * Evaluate whether the user's hands and upper body are cleanly captured inside the camera frame.
 * @param {Array|null} pose 33 MediaPipe Pose landmarks (normalized [0, 1]) or null in hands-only mode
 * @param {Array|null} leftHand 21 Left hand landmarks or null
 * @param {Array|null} rightHand 21 Right hand landmarks or null
 * @param {'words'|'letters'|'digits'|string} mode Current recognition mode
 */
export function evaluateCameraFraming(pose, leftHand, rightHand, mode = 'words') {
  const leftClip = findHandEdgeClip(leftHand);
  const rightClip = findHandEdgeClip(rightHand);
  const handClip = rightClip || leftClip;

  if (handClip === 'top') {
    return {
      ok: false,
      status: 'hand_clipped_top',
      badge: 'Tay sát mép trên',
      message: 'Bàn tay chạm mép trên camera — hạ nhẹ tay hoặc nâng góc camera.',
    };
  }
  if (handClip === 'side') {
    return {
      ok: false,
      status: 'hand_clipped_side',
      badge: 'Tay sát mép ngang',
      message: 'Bàn tay sát rìa khung hình — lùi nhẹ để camera bắt trọn ngón tay.',
    };
  }
  if (handClip === 'bottom' && mode !== 'words') {
    return {
      ok: false,
      status: 'hand_clipped_bottom',
      badge: 'Tay quá thấp',
      message: 'Đưa bàn tay lên giữa khung hình để bắt rõ 21 khớp ngón.',
    };
  }

  if (mode === 'words') {
    if (!pose || pose.length < 13) {
      return {
        ok: false,
        status: 'missing_pose',
        badge: 'Tìm khung thân trên',
        message: 'Đặt đầu, hai vai và bàn tay vào giữa khung hình.',
      };
    }
    const lSh = pose[11];
    const rSh = pose[12];
    const nose = pose[0];
    const lOk =
      isFinitePoint(lSh) &&
      (lSh.visibility ?? 1) >= SHOULDER_MIN_VIS &&
      lSh.x >= 0.02 &&
      lSh.x <= 0.98 &&
      lSh.y >= 0.05 &&
      lSh.y <= 0.95;
    const rOk =
      isFinitePoint(rSh) &&
      (rSh.visibility ?? 1) >= SHOULDER_MIN_VIS &&
      rSh.x >= 0.02 &&
      rSh.x <= 0.98 &&
      rSh.y >= 0.05 &&
      rSh.y <= 0.95;

    if (!lOk || !rOk) {
      return {
        ok: false,
        status: 'missing_shoulders',
        badge: 'Thiếu hai vai',
        message: 'Hãy lùi nhẹ để cả hai vai nằm trọn trong khung hình.',
      };
    }

    if (isFinitePoint(nose) && nose.y < 0.025) {
      return {
        ok: false,
        status: 'head_clipped',
        badge: 'Đầu sát mép trên',
        message: 'Đầu quá sát mép trên — các ký hiệu vùng trán/mặt dễ bị khuất.',
      };
    }

    const shoulderWidth = Math.hypot(lSh.x - rSh.x, lSh.y - rSh.y);
    if (shoulderWidth > 0.64) {
      return {
        ok: false,
        status: 'too_close',
        badge: 'Ngồi quá gần',
        message: 'Ngồi hơi gần camera — lùi nhẹ để không mất dấu tay khi vung rộng.',
      };
    }
    if (shoulderWidth < 0.13) {
      return {
        ok: false,
        status: 'too_far',
        badge: 'Ngồi hơi xa',
        message: 'Ngồi hơi xa camera — tiến gần hơn để nhận rõ các đốt ngón tay.',
      };
    }
  } else if (!leftHand && !rightHand) {
    return {
      ok: true,
      status: 'waiting_hand',
      badge: 'Đưa tay vào khung',
      message: '',
    };
  }

  return {
    ok: true,
    status: 'optimal',
    badge: 'Khung chuẩn',
    message: '',
  };
}

/**
 * Adaptive frame budget scheduler for low-end / weak laptops.
 * Keeps Hand detection at 100% frame rate while scaling Pose cadence & canvas resolution.
 */
export class AdaptivePerfController {
  constructor() {
    this.reset();
  }

  reset() {
    this.emaMs = 16;
    this.samples = 0;
    this.tier = 'high'; // 'high' | 'balanced' | 'eco'
    this.frameCounter = 0;
  }

  recordFrameMs(durationMs) {
    if (!Number.isFinite(durationMs) || durationMs <= 0) return;
    const clamped = Math.min(250, durationMs);
    this.samples += 1;
    this.emaMs = this.samples === 1 ? clamped : 0.82 * this.emaMs + 0.18 * clamped;

    // Hysteresis transitions prevent resolution thrashing
    if (this.tier === 'high' && this.emaMs > 28) {
      this.tier = this.emaMs > 46 ? 'eco' : 'balanced';
    } else if (this.tier === 'balanced') {
      if (this.emaMs > 46) this.tier = 'eco';
      else if (this.emaMs < 21) this.tier = 'high';
    } else if (this.tier === 'eco' && this.emaMs < 34) {
      this.tier = 'balanced';
    }
  }

  nextFramePlan({ detailed = false } = {}) {
    this.frameCounter += 1;
    if (detailed) {
      return {
        tier: 'detailed',
        targetMaxDim: 640,
        poseStride: 1,
        runPose: true,
        minIntervalMs: 26,
      };
    }
    if (this.tier === 'eco') {
      return {
        tier: 'eco',
        targetMaxDim: 384,
        poseStride: 3,
        runPose: this.frameCounter % 3 === 1,
        minIntervalMs: 32,
      };
    }
    if (this.tier === 'balanced') {
      return {
        tier: 'balanced',
        targetMaxDim: 448,
        poseStride: 2,
        runPose: this.frameCounter % 2 === 1,
        minIntervalMs: 26,
      };
    }
    return {
      tier: 'high',
      targetMaxDim: 480,
      poseStride: 1,
      runPose: true,
      minIntervalMs: 22,
    };
  }
}
