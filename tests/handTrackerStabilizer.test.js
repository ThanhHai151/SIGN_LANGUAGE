import test from 'node:test';
import assert from 'node:assert/strict';
import { HandTrackerStabilizer } from '../src/utils/handTrackerStabilizer.js';
import { assignHandsToAnatomicalSides } from '../src/utils/fastVisionTracker.js';

function makeHand(wx, wy, wz = 0) {
  return Array.from({ length: 21 }, (_, i) => ({
    x: wx + i * 0.004,
    y: wy - i * 0.005,
    z: wz,
  }));
}

function makePose({ rx = 0.35, ry = 0.45, rVis = 0.9, lx = 0.65, ly = 0.72, lVis = 0.9 } = {}) {
  const pose = Array.from({ length: 33 }, () => ({ x: 0.5, y: 0.5, z: 0, visibility: 0.95 }));
  pose[11] = { x: 0.64, y: 0.44, z: 0, visibility: 0.98 };
  pose[12] = { x: 0.36, y: 0.44, z: 0, visibility: 0.98 };
  pose[15] = { x: lx, y: ly, z: 0, visibility: lVis };
  pose[16] = { x: rx, y: ry, z: 0, visibility: rVis };
  pose[19] = { x: lx, y: ly - 0.05, z: 0, visibility: lVis };
  pose[20] = { x: rx, y: ry - 0.05, z: 0, visibility: rVis };
  return pose;
}

test('bridges foreshortened/pointing hand across 5 consecutive dropout frames when pose wrist stays raised', () => {
  const stab = new HandTrackerStabilizer();
  let ts = 1000;

  const first = stab.process(makePose({ rx: 0.41, ry: 0.46 }), null, makeHand(0.41, 0.46), ts);
  assert.ok(first.modelRight);
  assert.equal(first.rightBridged, false);

  for (let step = 1; step <= 5; step++) {
    ts += 40;
    const rx = 0.41 + step * 0.004;
    const out = stab.process(makePose({ rx, ry: 0.45, rVis: 0.25 }), null, null, ts);
    assert.ok(out.modelRight, `expected modelRight bridged at dropout frame ${step}`);
    assert.ok(out.drawRight, `expected drawRight bridged at dropout frame ${step}`);
    assert.equal(out.rightBridged, true);
  }

  ts += 40;
  const sixth = stab.process(makePose({ rx: 0.44, ry: 0.45, rVis: 0.25 }), null, null, ts);
  assert.equal(sixth.modelRight, null);
  assert.ok(sixth.drawRight);
});

test('retains >95% finger tracking across rapid high-speed hand swings with intermittent motion blur', () => {
  const stab = new HandTrackerStabilizer();
  const totalFrames = 60;
  let trackedHolistic = 0;
  let trackedHandsOnly = 0;
  const stabHands = new HandTrackerStabilizer();

  for (let f = 0; f < totalFrames; f++) {
    const ts = 1000 + f * 33;
    // Fast sinusoidal swing across the screen (~1.8 screen-widths/sec peak velocity)
    const wx = 0.50 + 0.28 * Math.sin(f * 0.38);
    const wy = 0.46 + 0.14 * Math.cos(f * 0.38);
    // Simulate severe motion blur dropping raw hand detection on 25% of peak-speed frames AND lagging pose wrist
    const isBlurDropout = f >= 4 && (f % 7 === 0 || f % 7 === 1);
    const rawRight = isBlurDropout ? null : makeHand(wx, wy);
    const pose = makePose({ rx: wx, ry: wy, rVis: isBlurDropout ? 0.10 : 0.90 });

    const resHolistic = stab.process(pose, null, rawRight, ts);
    if (resHolistic.modelRight && resHolistic.modelRight.length === 21) {
      trackedHolistic += 1;
    }

    const resHands = stabHands.processHands(null, rawRight, ts);
    if (resHands.right && resHands.right.length === 21) {
      trackedHandsOnly += 1;
    }
  }

  const retentionHolistic = trackedHolistic / totalFrames;
  const retentionHandsOnly = trackedHandsOnly / totalFrames;
  assert.ok(retentionHolistic >= 0.96, `expected >=96% holistic retention, got ${retentionHolistic}`);
  assert.ok(retentionHandsOnly >= 0.96, `expected >=96% hands-only retention, got ${retentionHandsOnly}`);
});

test('assignHandsToAnatomicalSides aligns detected palms to pose[15]/pose[16] and drops duplicate overlaps', () => {
  const pose = makePose({ lx: 0.66, ly: 0.50, rx: 0.34, ry: 0.48 });
  const physRight = makeHand(0.35, 0.48);
  const physLeft = makeHand(0.65, 0.50);

  const both = assignHandsToAnatomicalSides(
    [physRight, physLeft],
    [[{ categoryName: 'Left' }], [{ categoryName: 'Right' }]],
    pose,
    null,
    null
  );
  assert.equal(both.rightHandLandmarks, physRight);
  assert.equal(both.leftHandLandmarks, physLeft);

  // Duplicate palm box on the same physical right hand is collapsed cleanly
  const dupRight = makeHand(0.355, 0.482);
  const dedup = assignHandsToAnatomicalSides(
    [physRight, dupRight],
    [[{ categoryName: 'Left' }], [{ categoryName: 'Right' }]],
    pose,
    null,
    null
  );
  assert.ok(dedup.rightHandLandmarks);
  assert.equal(dedup.leftHandLandmarks, null);

  for (const index of [8, 20]) dupRight[index] = { ...dupRight[index], x: dupRight[index].x + 0.06 };
  const tipsDisagree = assignHandsToAnatomicalSides(
    [physRight, dupRight], [], pose, null, null
  );
  assert.equal(tipsDisagree.rightHandLandmarks, physRight);
  assert.equal(tipsDisagree.leftHandLandmarks, null);
});

test('a missing opposite pose wrist cannot win over the visible wrist of a single hand', () => {
  const hand = makeHand(0.58, 0.51);
  const visibleLeft = makePose({ lx: 0.59, ly: 0.52, rx: 0.15, ry: 0.9, rVis: 0.08 });
  const visibleRight = makePose({ rx: 0.59, ry: 0.52, lx: 0.15, ly: 0.9, lVis: 0.08 });

  const left = assignHandsToAnatomicalSides([hand], [[{ categoryName: 'Left' }]], visibleLeft, null, hand);
  assert.equal(left.leftHandLandmarks, hand);
  assert.equal(left.rightHandLandmarks, null);

  const right = assignHandsToAnatomicalSides([hand], [[{ categoryName: 'Right' }]], visibleRight, hand, null);
  assert.equal(right.rightHandLandmarks, hand);
  assert.equal(right.leftHandLandmarks, null);
});

test('moving the whole hand preserves all finger offsets while tracking a real bend', () => {
  const stabilizer = new HandTrackerStabilizer();
  const initial = makeHand(0.31, 0.48);
  const ended = initial.map((point) => ({ ...point, x: point.x + 0.22, y: point.y + 0.09 }));
  stabilizer.process(makePose(), null, initial, 1000);
  const moved = stabilizer.process(makePose(), null, ended, 1033).modelRight;
  for (let index = 1; index < initial.length; index += 1) {
    const dx = initial[index].x - initial[0].x;
    const dy = initial[index].y - initial[0].y;
    assert.ok(Math.abs((moved[index].x - moved[0].x) - dx) < 1e-8);
    assert.ok(Math.abs((moved[index].y - moved[0].y) - dy) < 1e-8);
  }
  ended[8] = { ...ended[8], y: ended[8].y + 0.06 };
  const bent = stabilizer.process(makePose(), null, ended, 1066);
  assert.ok(bent.maxFingerSpeed > 0.015);
  assert.ok(bent.modelRight[8].y - bent.modelRight[0].y > moved[8].y - moved[0].y + 0.015);
});
