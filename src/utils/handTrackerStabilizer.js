/**
 * Real-time Hand & Upper-Body Stabilizer:
 * 1. High-responsiveness Speed-Adaptive One-Euro Filter (rock-steady when still, near-zero phase lag on rapid swings).
 * 2. Damped Constant-Velocity Kinematic Finger Extrapolation (`predictHandWithVelocity`) for 1-4 frame
 *    high-speed motion-blur dropouts across BOTH `processHands` (letters/digits) and `process` (words).
 * 3. Duplicate-hand separation & Pose-guided occlusion bridging using pose[15..22] (wrists, pinkies, indices, thumbs).
 */

function smoothingAlpha(dt, cutoff) {
  const tau = 1.0 / (2.0 * Math.PI * Math.max(cutoff, 1e-4));
  return 1.0 / (1.0 + tau / Math.max(dt, 1e-4));
}

class OneEuroPointSet {
  constructor(minCutoff = 1.8, beta = 0.08, dCutoff = 1.0, handRelative = false) {
    this.minCutoff = minCutoff;
    this.beta = beta;
    this.dCutoff = dCutoff;
    this.handRelative = handRelative;
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
    const palmSize = this.handRelative ? Math.max(0.015, dist2D(landmarks[0], landmarks[9])) : 1;
    if (this.handRelative) {
      const wrist = landmarks[0];
      // Use the observed wrist as a local origin for finger articulation.
      // A palm that translates should not stretch fingers through differing filter lag.
      landmarks = landmarks.map((point, index) => index === 0 ? wrist : ({
        x: point.x - wrist.x, y: point.y - wrist.y, z: (point.z || 0) - (wrist.z || 0),
      }));
    }
    if (!this.prevX || this.prevX.length !== landmarks.length || this.prevTs === null) {
      this.prevX = landmarks.map((lm) => ({
        x: Number.isFinite(lm?.x) ? lm.x : 0,
        y: Number.isFinite(lm?.y) ? lm.y : 0,
        z: Number.isFinite(lm?.z) ? lm.z : 0,
        visibility: lm?.visibility,
      }));
      this.prevDx = landmarks.map(() => ({ x: 0, y: 0, z: 0 }));
      this.prevTs = tsMs;
      return this.reconstruct();
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
      const relativeSpeed = this.handRelative && i > 0 ? speed * (0.14 / palmSize) : speed;
      const cutoff = this.minCutoff + this.beta * relativeSpeed;
      const a = smoothingAlpha(dt, cutoff);

      const fx = a * rx + (1 - a) * px.x;
      const fy = a * ry + (1 - a) * px.y;
      const fz = a * rz + (1 - a) * px.z;

      this.prevDx[i] = { x: dx, y: dy, z: dz };
      this.prevX[i] = { x: fx, y: fy, z: fz, visibility: lm?.visibility };
      out[i] = this.prevX[i];
    }
    return this.handRelative ? this.reconstruct() : out;
  }

  reconstruct() {
    if (!this.handRelative) return this.prevX;
    const wrist = this.prevX[0];
    return this.prevX.map((point, index) => index === 0 ? wrist : ({
      x: point.x + wrist.x, y: point.y + wrist.y, z: point.z + wrist.z,
    }));
  }
}

function dist2D(a, b) {
  if (!a || !b) return Infinity;
  return Math.hypot((a.x || 0) - (b.x || 0), (a.y || 0) - (b.y || 0));
}

function isValidPoseJoint(lm, minVis = 0.22) {
  if (!lm || !Number.isFinite(lm.x) || !Number.isFinite(lm.y)) return false;
  if (lm.x === 0 && lm.y === 0) return false;
  if (typeof lm.visibility === 'number' && lm.visibility < minVis) return false;
  return true;
}

function cloneLandmarks(lms) {
  return lms.map((lm) => ({ x: lm.x, y: lm.y, z: lm.z || 0 }));
}

function clamp01Margin(v) {
  return Math.max(0.005, Math.min(0.995, v));
}

/**
 * Predict 21 hand joints along damped velocity vector during brief high-speed motion blur.
 * Preserves rigid finger-to-wrist offsets while translating along wrist velocity so fingers never deform.
 */
function predictHandWithVelocity(lastHand, wristVel, dtSec, missFrames) {
  const damping = Math.pow(0.72, missFrames);
  const maxStep = 0.14;
  const dx = Math.max(-maxStep, Math.min(maxStep, wristVel.x * dtSec * damping));
  const dy = Math.max(-maxStep, Math.min(maxStep, wristVel.y * dtSec * damping));
  const dz = Math.max(-0.08, Math.min(0.08, (wristVel.z || 0) * dtSec * damping));

  // Keep rigid hand shape inside frame bounds
  const nextWx = clamp01Margin(lastHand[0].x + dx);
  const nextWy = clamp01Margin(lastHand[0].y + dy);
  const effDx = nextWx - lastHand[0].x;
  const effDy = nextWy - lastHand[0].y;

  return lastHand.map((pt) => ({
    x: clamp01Margin(pt.x + effDx),
    y: clamp01Margin(pt.y + effDy),
    z: (pt.z || 0) + dz,
  }));
}

function posePalmVector(pose, wristIdx, pinkyIdx, indexIdx) {
  const w = pose?.[wristIdx];
  const p = pose?.[pinkyIdx];
  const ix = pose?.[indexIdx];
  if (!isValidPoseJoint(w, 0.22)) return null;
  if (isValidPoseJoint(p, 0.22) && isValidPoseJoint(ix, 0.22)) {
    return { x: 0.5 * (p.x + ix.x) - w.x, y: 0.5 * (p.y + ix.y) - w.y };
  }
  if (isValidPoseJoint(ix, 0.22)) {
    return { x: ix.x - w.x, y: ix.y - w.y };
  }
  return null;
}

function reanchorHandToPose(lastHand, lastPoseWrist, curPoseWrist, lastPalmVec, curPalmVec) {
  const dx = curPoseWrist.x - lastPoseWrist.x;
  const dy = curPoseWrist.y - lastPoseWrist.y;
  const dz = ((curPoseWrist.z || 0) - (lastPoseWrist.z || 0)) * 0.5;

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

const MAX_MODEL_BRIDGE_FRAMES = 5;
const MAX_MODEL_BRIDGE_MS = 260;
const MAX_KINEMATIC_SWING_FRAMES = 3;
const MAX_KINEMATIC_SWING_MS = 140;
const MAX_VISUAL_BRIDGE_FRAMES = 10;
const MAX_VISUAL_BRIDGE_MS = 350;
const MAX_CLASP_VISUAL_BRIDGE_FRAMES = 16;
const MAX_CLASP_VISUAL_BRIDGE_MS = 560;

export class HandTrackerStabilizer {
  constructor() {
    // High beta (0.58) gives near-zero lag during rapid hand swings while minCutoff (2.0) kills static jitter
    this.poseFilter = new OneEuroPointSet(1.8, 0.25, 1.2);
    this.leftFilter = new OneEuroPointSet(2.0, 0.58, 1.4, true);
    this.rightFilter = new OneEuroPointSet(2.0, 0.58, 1.4, true);
    this.reset();
  }

  reset() {
    this.poseFilter.reset();
    this.leftFilter.reset();
    this.rightFilter.reset();
    const emptySide = () => ({
      lastHand: null,
      lastPoseWrist: null,
      lastPalmVec: null,
      wristVel: { x: 0, y: 0, z: 0 },
      missFrames: 0,
      lastSeenTs: -Infinity,
      lastStepTs: -Infinity,
    });
    this.sides = { left: emptySide(), right: emptySide() };
    this.handsOnlySides = { left: emptySide(), right: emptySide() };
  }

  /**
   * Zero-lag One-Euro + short motion-blur velocity bridge for Hands-only mode (letters / digits).
   */
  processHands(rawLeft, rawRight, tsMs = performance.now()) {
    const stepSide = (sideName, rawHand, filter) => {
      const st = this.handsOnlySides[sideName];
      if (rawHand && rawHand.length >= 21) {
        if (st.missFrames > 4 || tsMs - st.lastSeenTs > 220) {
          filter.reset();
        }
        const filtered = filter.filter(rawHand, tsMs);
        const wVel = filter.prevDx?.[0] || { x: 0, y: 0, z: 0 };
        st.lastHand = cloneLandmarks(filtered);
        st.wristVel = { x: wVel.x, y: wVel.y, z: wVel.z };
        st.missFrames = 0;
        st.lastSeenTs = tsMs;
        st.lastStepTs = tsMs;
        return { hand: filtered, bridged: false };
      }

      st.missFrames += 1;
      const elapsed = tsMs - st.lastSeenTs;
      const dtSec = Math.min(0.08, Math.max(1 / 120, (tsMs - st.lastStepTs) / 1000));
      st.lastStepTs = tsMs;

      if (
        st.lastHand &&
        st.missFrames <= MAX_KINEMATIC_SWING_FRAMES &&
        elapsed <= MAX_KINEMATIC_SWING_MS
      ) {
        const predicted = predictHandWithVelocity(st.lastHand, st.wristVel, dtSec, st.missFrames);
        st.lastHand = predicted;
        return { hand: predicted, bridged: true };
      }

      st.lastHand = null;
      st.wristVel = { x: 0, y: 0, z: 0 };
      filter.reset();
      return { hand: null, bridged: false };
    };

    const lRes = stepSide('left', rawLeft, this.leftFilter);
    const rRes = stepSide('right', rawRight, this.rightFilter);
    return {
      left: lRes.hand,
      right: rRes.hand,
      leftBridged: lRes.bridged,
      rightBridged: rRes.bridged,
    };
  }

  /**
   * Process one 75-joint Holistic / Dual-Tracker frame (words mode).
   */
  process(rawPose, rawLeft, rawRight, tsMs = performance.now()) {
    const pose = rawPose && rawPose.length >= 33 ? this.poseFilter.filter(rawPose, tsMs) : null;

    let inLeft = rawLeft && rawLeft.length >= 21 ? rawLeft : null;
    let inRight = rawRight && rawRight.length >= 21 ? rawRight : null;

    // 1. Separate duplicate-hand collapse
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
          if (dLeft > dRight + 0.015) inLeft = null;
          else if (dRight > dLeft + 0.015) inRight = null;
        }
      }
    }

    const lShoulder = pose?.[11];
    const rShoulder = pose?.[12];
    const hasShoulders = isValidPoseJoint(lShoulder, 0.25) && isValidPoseJoint(rShoulder, 0.25);
    const shoulderY = hasShoulders ? 0.5 * (lShoulder.y + rShoulder.y) : 0.45;
    const shoulderWidth = hasShoulders ? Math.max(0.08, dist2D(lShoulder, rShoulder)) : 0.25;
    const maxRaisedY = shoulderY + 0.88 * shoulderWidth;

    const lPoseWrist = pose?.[15];
    const rPoseWrist = pose?.[16];
    const handsClasped = Boolean(
      isValidPoseJoint(lPoseWrist, 0.25) &&
      isValidPoseJoint(rPoseWrist, 0.25) &&
      lPoseWrist.y <= maxRaisedY &&
      rPoseWrist.y <= maxRaisedY &&
      dist2D(lPoseWrist, rPoseWrist) <= 0.72 * shoulderWidth
    );

    const resolveSide = (sideName, rawHand, filter, wristIdx, pinkyIdx, indexIdx) => {
      const state = this.sides[sideName];
      const curPoseWrist = pose?.[wristIdx];
      const curPalmVec = posePalmVector(pose, wristIdx, pinkyIdx, indexIdx);
      const poseValid = isValidPoseJoint(curPoseWrist, 0.22) && curPoseWrist.y <= maxRaisedY;

      if (rawHand) {
        if (state.missFrames > 6 || tsMs - state.lastSeenTs > 350) {
          filter.reset();
        }
        const prevWrist = state.lastHand ? state.lastHand[0] : null;
        const filtered = filter.filter(rawHand, tsMs);
        const speed = prevWrist ? dist2D(filtered[0], prevWrist) : 0;
        let fingerSpeed = 0;
        if (prevWrist && state.missFrames === 0) {
          for (const tip of [4, 8, 12, 16, 20]) {
            const deltaX = (filtered[tip].x - filtered[0].x) - (state.lastHand[tip].x - prevWrist.x);
            const deltaY = (filtered[tip].y - filtered[0].y) - (state.lastHand[tip].y - prevWrist.y);
            fingerSpeed = Math.max(fingerSpeed, Math.hypot(deltaX, deltaY));
          }
        }
        const wVel = filter.prevDx?.[0] || { x: 0, y: 0, z: 0 };

        state.lastHand = cloneLandmarks(filtered);
        state.lastPoseWrist = poseValid ? { x: curPoseWrist.x, y: curPoseWrist.y, z: curPoseWrist.z || 0 } : null;
        state.lastPalmVec = curPalmVec;
        state.wristVel = { x: wVel.x, y: wVel.y, z: wVel.z };
        state.missFrames = 0;
        state.lastSeenTs = tsMs;
        state.lastStepTs = tsMs;
        return { modelHand: filtered, drawHand: filtered, bridged: false, speed, fingerSpeed };
      }

      state.missFrames += 1;
      const elapsed = tsMs - state.lastSeenTs;
      const dtSec = Math.min(0.08, Math.max(1 / 120, (tsMs - state.lastStepTs) / 1000));
      state.lastStepTs = tsMs;
      const maxVisFrames = handsClasped ? MAX_CLASP_VISUAL_BRIDGE_FRAMES : MAX_VISUAL_BRIDGE_FRAMES;
      const maxVisMs = handsClasped ? MAX_CLASP_VISUAL_BRIDGE_MS : MAX_VISUAL_BRIDGE_MS;

      // Priority A: Pose-guided occlusion/swing bridge when pose wrist is tracked in signing space
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

      // Priority B: Kinematic velocity bridge when ultra-fast swing temporarily outruns pose wrist too!
      const velMag = Math.hypot(state.wristVel.x, state.wristVel.y);
      if (
        state.lastHand &&
        velMag >= 0.35 &&
        state.missFrames <= MAX_KINEMATIC_SWING_FRAMES &&
        elapsed <= MAX_KINEMATIC_SWING_MS
      ) {
        const predicted = predictHandWithVelocity(state.lastHand, state.wristVel, dtSec, state.missFrames);
        const speed = dist2D(predicted[0], state.lastHand[0]);
        state.lastHand = predicted;
        return {
          modelHand: predicted,
          drawHand: predicted,
          bridged: true,
          speed,
        };
      }

      if (state.missFrames > maxVisFrames || elapsed > maxVisMs) {
        state.lastHand = null;
        state.lastPoseWrist = null;
        state.lastPalmVec = null;
        state.wristVel = { x: 0, y: 0, z: 0 };
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
      maxFingerSpeed: Math.max(left.fingerSpeed ?? 0, right.fingerSpeed ?? 0),
    };
  }
}
