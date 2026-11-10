import test from 'node:test';
import assert from 'node:assert/strict';
import {
  deriveCanonicalHandPose,
  medianCanonicalHandPose
} from '../src/services/canonicalHandPose.js';
import { canonicalHandPoseToMixamoTargets } from '../src/services/mixamoHandRetargeter.js';

function extendedHand() {
  const points = Array.from({ length: 21 }, () => [0.5, 0.5, 0]);
  points[0] = [0.5, 0.8, 0];
  const fingers = [
    [5, 6, 7, 8, 0.32],
    [9, 10, 11, 12, 0.44],
    [13, 14, 15, 16, 0.56],
    [17, 18, 19, 20, 0.68]
  ];
  for (const [mcp, pip, dip, tip, x] of fingers) {
    points[mcp] = [x, 0.6, 0];
    points[pip] = [x, 0.4, 0];
    points[dip] = [x, 0.25, 0];
    points[tip] = [x, 0.1, 0];
  }
  points[1] = [0.43, 0.68, 0];
  points[2] = [0.35, 0.57, 0];
  points[3] = [0.32, 0.49, 0];
  points[4] = [0.30, 0.42, 0];
  return points;
}

test('canonical pose gives near-zero finger flexion for an extended hand', () => {
  const pose = deriveCanonicalHandPose(extendedHand(), { width: 1280, height: 720 });
  for (const finger of Object.values(pose.fingers)) {
    assert.ok(finger.flexion.every((angle) => angle < 0.01));
  }
});

test('canonical pose detects PIP bend and median suppresses an outlier', () => {
  const straight = deriveCanonicalHandPose(extendedHand(), { width: 1280, height: 720 });
  const bentPoints = extendedHand();
  bentPoints[7] = [0.40, 0.4, 0];
  bentPoints[8] = [0.48, 0.4, 0];
  const bent = deriveCanonicalHandPose(bentPoints, { width: 1280, height: 720 });
  assert.ok(bent.fingers.index.flexion[1] > 1);
  const median = medianCanonicalHandPose([straight, bent, straight]);
  assert.ok(median.fingers.index.flexion[1] < 0.01);
});

test('retargeter emits Mixamo right finger targets with bounded local rotations', () => {
  const canonical = deriveCanonicalHandPose(extendedHand(), { width: 1280, height: 720 });
  const targets = canonicalHandPoseToMixamoTargets(canonical);
  assert.equal(targets.length, 16);
  assert.ok(targets.some((target) => target.bone === 'RightHandIndex1'));
  assert.ok(targets.some((target) => target.bone === 'RightHandThumb4'));
  assert.ok(targets.every(({ rot }) => rot.length === 3 && rot.every(Number.isFinite)));
});

test('retargeter rejects unsupported schema and invalid calibration gains', () => {
  assert.throws(() => canonicalHandPoseToMixamoTargets({}), /canonical-hand-pose/);
  const pose = deriveCanonicalHandPose(extendedHand(), { width: 1280, height: 720 });
  assert.throws(() => canonicalHandPoseToMixamoTargets(pose, { flexionGain: 4 }), /flexionGain/);
});
