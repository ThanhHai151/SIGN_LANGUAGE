import test from 'node:test';
import assert from 'node:assert/strict';
import { evaluateCameraFraming, AdaptivePerfController } from '../src/utils/cameraFraming.js';

function validPose(shoulderLeftX = 0.64, shoulderRightX = 0.36, noseY = 0.22) {
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.95 }));
  pose[0] = { x: 0.5, y: noseY, z: 0, visibility: 0.98 };
  pose[11] = { x: shoulderLeftX, y: 0.46, z: 0, visibility: 0.95 };
  pose[12] = { x: shoulderRightX, y: 0.46, z: 0, visibility: 0.95 };
  return pose;
}

function handAt(x, y) {
  return Array.from({ length: 21 }, () => ({ x, y, z: 0 }));
}

test('evaluateCameraFraming approves well-framed shoulders and hands and flags edge clipping or bad distance', () => {
  const ok = evaluateCameraFraming(validPose(), null, handAt(0.42, 0.48), 'words');
  assert.equal(ok.ok, true);
  assert.equal(ok.status, 'optimal');

  const clippedTop = evaluateCameraFraming(validPose(), null, handAt(0.45, 0.01), 'words');
  assert.equal(clippedTop.ok, false);
  assert.equal(clippedTop.status, 'hand_clipped_top');

  const missingShoulder = validPose();
  missingShoulder[11].visibility = 0.10;
  const noSh = evaluateCameraFraming(missingShoulder, null, handAt(0.45, 0.50), 'words');
  assert.equal(noSh.ok, false);
  assert.equal(noSh.status, 'missing_shoulders');

  const tooClose = evaluateCameraFraming(validPose(0.86, 0.14), null, handAt(0.45, 0.50), 'words');
  assert.equal(tooClose.ok, false);
  assert.equal(tooClose.status, 'too_close');
});

test('AdaptivePerfController scales down pose cadence and resolution on weak machines while keeping hands active', () => {
  const ctrl = new AdaptivePerfController();
  assert.equal(ctrl.nextFramePlan().tier, 'high');

  // Simulate weak CPU taking 58ms per frame
  for (let i = 0; i < 12; i++) ctrl.recordFrameMs(58);
  const plans = [ctrl.nextFramePlan(), ctrl.nextFramePlan(), ctrl.nextFramePlan()];
  assert.equal(plans[0].tier, 'eco');
  assert.equal(plans[0].targetMaxDim, 384);
  assert.equal(plans.filter((p) => p.runPose).length, 1);

  // Simulate recovery on GPU (10ms per frame)
  for (let i = 0; i < 20; i++) ctrl.recordFrameMs(10);
  const fast = ctrl.nextFramePlan();
  assert.equal(fast.tier, 'high');
  assert.equal(fast.runPose, true);
});
