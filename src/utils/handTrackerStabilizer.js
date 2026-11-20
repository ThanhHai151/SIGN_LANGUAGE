/**
 * Real-time Holistic Hand & Upper-Body Stabilizer:
 * 1. Speed-adaptive One-Euro Filter (rock-steady when still, near-zero lag on fast motion).
 * 2. Duplicate-hand separation using Holistic pose wrists (pose[15], pose[16]).
 * 3. Pose-guided occlusion bridging using pose[15..22] (wrists, pinkies, indices, thumbs)
 *    so hands stop flickering on/off during fast motion blur, edge-on views, or clasped signs
 *    like "xin lỗi" / "năn nỉ".
 */

function smoothingAlpha(dt, cutoff) {
  const tau = 1.0 / (2.0 * Math.PI * Math.max(cutoff, 1e-4));
  return 1.0 / (1.0 + tau / Math.max(dt, 1e-4));
}

class OneEuroPointSet {
  constructor(minCutoff = 1.8, beta = 0.08, dCutoff = 1.0) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.prevX = null;
    this.prevDx = null;
    this.prevTs = null;
  }

  reset() {
    this.prevX = null;
    this.prevDx = null;
    this.prevTs = null;
  }

  filter(landmarks, tsMs) {
    if (!landmarks || landmarks.length === 0) return null;
    if (!this.prevX || this.prevX.length !== landmarks.length || this.prevTs === null) {
      this.prevX = landmarks.map((lm) => ({
        x: Number.isFinite(lm?.x) ? lm.x : 0,
        y: Number.isFinite(lm?.y) ? lm.y : 0,
        z: Number.isFinite(lm?.z) ? lm.z : 0,
        visibility: lm?.visibility,
      }));
      this.prevDx = landmarks.map(() => ({ x: 0, y: 0, z: 0 }));
      this.prevTs = tsMs;
      return this.prevX;
    }

    const dt = Math.min(0.15, Math.max(1 / 120, (tsMs - this.prevTs) / 1000));
    this.prevTs = tsMs;
    const aD = smoothingAlpha(dt, this.dCutoff);

    const out = new Array(landmarks.length);
    for (let i = 0; i < landmarks.length; i++) {
      const lm = landmarks[i];
      const rx = Number.isFinite(lm?.x) ? lm.x : 0;
      const ry = Number.isFinite(lm?.y) ? lm.y : 0;
      const rz = Number.isFinite(lm?.z) ? lm.z : 0;
      const px = this.prevX[i];
      const pd = this.prevDx[i];

      const dxRaw = (rx - px.x) / dt;
      const dyRaw = (ry - px.y) / dt;
      const dzRaw = (rz - px.z) / dt;

      const dx = aD * dxRaw + (1 - aD) * pd.x;
      const dy = aD * dyRaw + (1 - aD) * pd.y;
      const dz = aD * dzRaw + (1 - aD) * pd.z;

      const speed = Math.hypot(dx, dy);
      const cutoff = this.minCutoff + this.beta * speed;
      const a = smoothingAlpha(dt, cutoff);

      const fx = a * rx + (1 - a) * px.x;
      const fy = a * ry + (1 - a) * px.y;
      const fz = a * rz + (1 - a) * px.z;

      this.prevDx[i] = { x: dx, y: dy, z: dz };
      this.prevX[i] = { x: fx, y: fy, z: fz, visibility: lm?.visibility };
      out[i] = this.prevX[i];
    }
    return out;
  }
}

function dist2D(a, b) {
  if (!a || !b) return Infinity;
  return Math.hypot((a.x || 0) - (b.x || 0), (a.y || 0) - (b.y || 0));
}

function isValidPoseJoint(lm, minVis = 0.28) {
  if (!lm || !Number.isFinite(lm.x) || !Number.isFinite(lm.y)) return false;
  if (lm.x === 0 && lm.y === 0) return false;
  if (typeof lm.visibility === 'number' && lm.visibility < minVis) return false;
  return true;
}

function cloneLandmarks(lms) {
  return lms.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z || 0 }));
}

function posePalmVector(pose, wristIdx, pinkyIdx, indexIdx) {
  const w = pose?.[wristIdx];
  const p = pose?.[pinkyIdx];
  const ix = pose?.[indexIdx];
  if (!isValidPoseJoint(w, 0.25)) return null;
  if (isValidPoseJoint(p, 0.25) && isValidPoseJoint(ix, 0.25)) {
    return { x: 0.5 * (p.x + ix.x) - w.x, y: 0.5 * (p.y + ix.y) - w.y };
  }
  if (isValidPoseJoint(ix, 0.25)) {
    return { x: ix.x - w.x, y: ix.y - w.y };
  }
  return null;
}

function reanchorHandToPose(lastHand, lastPoseWrist, curPoseWrist, lastPalmVec, curPalmVec) {
  const dx = curPoseWrist.x - lastPoseWrist.x;
  const dy = curPoseWrist.y - lastPoseWrist.y;
  const dz = ((curPoseWrist.z || 0) - (lastPoseWrist.z || 0)) * 0.5;

  // Mild 2D rotation alignment when wrist rotates edge-on (clamped to +-25 deg to prevent pose jitter)
  let cosA = 1;
  let sinA = 0;
  if (lastPalmVec && curPalmVec) {
    const len1 = Math.hypot(lastPalmVec.x, lastPalmVec.y);
    const len2 = Math.hypot(curPalmVec.x, curPalmVec.y);
    if (len1 > 0.018 && len2 > 0.018) {
      const a1 = Math.atan2(lastPalmVec.y, lastPalmVec.x);
      const a2 = Math.atan2(curPalmVec.y, curPalmVec.x);
      let delta = a2 - a1;
      while (delta > Math.PI) delta -= 2 * Math.PI;
      while (delta < -Math.PI) delta += 2 * Math.PI;
      const clamped = Math.max(-0.42, Math.min(0.42, delta * 0.55));
      cosA = Math.cos(clamped);
      sinA = Math.sin(clamped);
    }
  }

  const origin = lastHand[0];
  const anchorX = origin.x + dx;
  const anchorY = origin.y + dy;
  const anchorZ = (origin.z || 0) + dz;

  return lastHand.map((pt) => {
    const rx = pt.x - origin.x;
    const ry = pt.y - origin.y;
    return {
      x: anchorX + rx * cosA - ry * sinA,
      y: anchorY + rx * sinA + ry * cosA,
      z: anchorZ + ((pt.z || 0) - (origin.z || 0)),
    };
  });
}

const MAX_MODEL_BRIDGE_FRAMES = 3;
const MAX_MODEL_BRIDGE_MS = 120;
const MAX_VISUAL_BRIDGE_FRAMES = 8;
const MAX_VISUAL_BRIDGE_MS = 280;
const MAX_CLASP_VISUAL_BRIDGE_FRAMES = 16;
const MAX_CLASP_VISUAL_BRIDGE_MS = 560;

export class HandTrackerStabilizer {
  constructor() {
    this.poseFilter = new OneEuroPointSet(1.6, 0.06, 1.0);
    this.leftFilter = new OneEuroPointSet(1.9, 0.10, 1.0);
    this.rightFilter = new OneEuroPointSet(1.9, 0.10, 1.0);
    this.reset();
  }

  reset() {
    this.poseFilter.reset();
    this.leftFilter.reset();
    this.rightFilter.reset();
    this.sides = {
      left: { lastHand: null, lastPoseWrist: null, lastPalmVec: null, missFrames: 0, lastSeenTs: -Infinity },
      right: { lastHand: null, lastPoseWrist: null, lastPalmVec: null, missFrames: 0, lastSeenTs: -Infinity },
    };
  }

  /**
   * Process one MediaPipe Holistic frame.
   * Returns:
   * - pose: stabilized pose landmarks (33) or null
   * - modelLeft / modelRight: hands for 75-joint backend tensor (strict short bridge <= 3 frames)
   * - drawLeft / drawRight: hands for smooth canvas rendering (longer bridge during clasp/occlusion)
   * - leftBridged / rightBridged: whether the drawn hand is pose-reconstructed
   * - maxWristSpeed: instantaneous normalized wrist speed (for fast-motion & settle detection)
   */
  process(rawPose, rawLeft, rawRight, tsMs = performance.now()) {
    const pose = rawPose && rawPose.length >= 33 ? this.poseFilter.filter(rawPose, tsMs) : null;

    let inLeft = rawLeft && rawLeft.length >= 21 ? rawLeft : null;
    let inRight = rawRight && rawRight.length >= 21 ? rawRight : null;

    // 1. Separate duplicate-hand collapse (Holistic can lock both tracks onto the same clasped hand)
    if (inLeft && inRight && pose) {
      const wristGap = dist2D(inLeft[0], inRight[0]);
      const tipGap = (
        dist2D(inLeft[8], inRight[8]) +
        dist2D(inLeft[12], inRight[12]) +
        dist2D(inLeft[16], inRight[16])
      ) / 3;
      if (wristGap < 0.038 && tipGap < 0.045) {
        const lPose = pose[15];
        const rPose = pose[16];
        if (isValidPoseJoint(lPose, 0.25) && isValidPoseJoint(rPose, 0.25)) {
          const dLeft = dist2D(inLeft[0], lPose);
          const dRight = dist2D(inRight[0], rPose);
          // Drop the falsely cloned side so step 2 bridges it onto its real pose wrist
          if (dLeft > dRight + 0.015) inLeft = null;
          else if (dRight > dLeft + 0.015) inRight = null;
        }
      }
    }

    // Check upper-body geometry for clasp detection & raised-wrist guard
    const lShoulder = pose?.[11];
    const rShoulder = pose?.[12];
    const hasShoulders = isValidPoseJoint(lShoulder, 0.25) && isValidPoseJoint(rShoulder, 0.25);
    const shoulderY = hasShoulders ? 0.5 * (lShoulder.y + rShoulder.y) : 0.45;
    const shoulderWidth = hasShoulders ? Math.max(0.08, dist2D(lShoulder, rShoulder)) : 0.25;
    const maxRaisedY = shoulderY + 0.85 * shoulderWidth;

    const lPoseWrist = pose?.[15];
    const rPoseWrist = pose?.[16];
    const handsClasped = Boolean(
      isValidPoseJoint(lPoseWrist, 0.28) &&
      isValidPoseJoint(rPoseWrist, 0.28) &&
      lPoseWrist.y <= maxRaisedY &&
      rPoseWrist.y <= maxRaisedY &&
      dist2D(lPoseWrist, rPoseWrist) <= 0.72 * shoulderWidth
    );

    const resolveSide = (sideName, rawHand, filter, wristIdx, pinkyIdx, indexIdx) => {
      const state = this.sides[sideName];
      const curPoseWrist = pose?.[wristIdx];
      const curPalmVec = posePalmVector(pose, wristIdx, pinkyIdx, indexIdx);
      const poseValid = isValidPoseJoint(curPoseWrist, 0.28) && curPoseWrist.y <= maxRaisedY;

      if (rawHand) {
        // If hand just returned after a long absence, reset One-Euro filter to avoid lag jump
        if (state.missFrames > 6 || tsMs - state.lastSeenTs > 350) {
          filter.reset();
        }
        const prevWrist = state.lastHand ? state.lastHand[0] : null;
        const filtered = filter.filter(rawHand, tsMs);
        const speed = prevWrist ? dist2D(filtered[0], prevWrist) : 0;

        state.lastHand = cloneLandmarks(filtered);
        state.lastPoseWrist = poseValid ? { x: curPoseWrist.x, y: curPoseWrist.y, z: curPoseWrist.z || 0 } : null;
        state.lastPalmVec = curPalmVec;
        state.missFrames = 0;
        state.lastSeenTs = tsMs;
        return { modelHand: filtered, drawHand: filtered, bridged: false, speed };
      }

      // Hand dropped out: attempt pose-guided occlusion bridging
      state.missFrames += 1;
      const elapsed = tsMs - state.lastSeenTs;
      const maxVisFrames = handsClasped ? MAX_CLASP_VISUAL_BRIDGE_FRAMES : MAX_VISUAL_BRIDGE_FRAMES;
      const maxVisMs = handsClasped ? MAX_CLASP_VISUAL_BRIDGE_MS : MAX_VISUAL_BRIDGE_MS;

      if (
        state.lastHand &&
        state.lastPoseWrist &&
        poseValid &&
        state.missFrames <= maxVisFrames &&
        elapsed <= maxVisMs
      ) {
        const bridgedHand = reanchorHandToPose(
          state.lastHand,
          state.lastPoseWrist,
          curPoseWrist,
          state.lastPalmVec,
          curPalmVec
        );
        const speed = dist2D(curPoseWrist, state.lastPoseWrist) / Math.max(1, state.missFrames);
        const allowModelBridge =
          state.missFrames <= MAX_MODEL_BRIDGE_FRAMES && elapsed <= MAX_MODEL_BRIDGE_MS;
        return {
          modelHand: allowModelBridge ? bridgedHand : null,
          drawHand: bridgedHand,
          bridged: true,
          speed,
        };
      }

      if (state.missFrames > maxVisFrames || elapsed > maxVisMs) {
        state.lastHand = null;
        state.lastPoseWrist = null;
        state.lastPalmVec = null;
        filter.reset();
      }
      return { modelHand: null, drawHand: null, bridged: false, speed: 0 };
    };

    const left = resolveSide('left', inLeft, this.leftFilter, 15, 17, 19);
    const right = resolveSide('right', inRight, this.rightFilter, 16, 18, 20);

    return {
      pose,
      modelLeft: left.modelHand,
      modelRight: right.modelHand,
      drawLeft: left.drawHand,
      drawRight: right.drawHand,
      leftBridged: left.bridged,
      rightBridged: right.bridged,
      handsClasped,
      maxWristSpeed: Math.max(left.speed, right.speed),
    };
  }
}

