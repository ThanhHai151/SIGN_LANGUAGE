import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { HandTrackerStabilizer } from './pilot_pinky_480x360/src/utils/handTrackerStabilizer.js';
import { createMonotonicTimestampGenerator } from './pilot_pinky_480x360/src/components/pilotProtocol.js';

console.log('=== AUTOMATED PILOT VERIFICATION SUITE ===');

// 1. Verify UI Freeze & File Integrity between Production and Pilot
const prodSrc = path.resolve('src');
const pilotSrc = path.resolve('scratch/pilot_pinky_480x360/src');

function listFiles(dir, base = dir) {
  let out = [];
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) out = out.concat(listFiles(full, base));
    else out.push(path.relative(base, full));
  }
  return out.sort();
}

const prodFiles = listFiles(prodSrc);
const pilotFiles = listFiles(pilotSrc);
assert.deepEqual(prodFiles, pilotFiles, 'File tree between production and pilot must match 100%');

for (const rel of prodFiles) {
  const cProd = fs.readFileSync(path.join(prodSrc, rel), 'utf8');
  const cPilot = fs.readFileSync(path.join(pilotSrc, rel), 'utf8');
  if (rel !== path.join('components', 'CameraSignRecognizer.jsx')) {
    assert.equal(cProd, cPilot, `UI Freeze violation: ${rel} must be 100% identical to production`);
  }
}
console.log('[PASS] 1. UI Freeze & File Tree Integrity (all CSS, services, utils, and other components 100% untouched)');

// 2. Verify Pilot MediaPipe Hands Configuration in CameraSignRecognizer.jsx
const pilotRecognizerCode = fs.readFileSync(path.join(pilotSrc, 'components', 'CameraSignRecognizer.jsx'), 'utf8');
assert.match(pilotRecognizerCode, /maxNumHands:\s*2/, 'maxNumHands must be 2');
assert.match(pilotRecognizerCode, /modelComplexity:\s*1,\s*\/\/\s*Pilot Full model/, 'Hands modelComplexity must be 1');
assert.match(pilotRecognizerCode, /minDetectionConfidence:\s*0\.45/, 'minDetectionConfidence must be 0.45');
assert.match(pilotRecognizerCode, /minTrackingConfidence:\s*0\.35/, 'minTrackingConfidence must be 0.35');
assert.match(pilotRecognizerCode, /pCanvas\.width\s*=\s*480;\s*\r?\n\s*pCanvas\.height\s*=\s*360;/, 'pCanvas must be 480x360');
assert.match(pilotRecognizerCode, /pCtx\.drawImage\(videoRef\.current,\s*0,\s*0,\s*480,\s*360\)/, 'drawImage must downsample to 480x360');
assert.match(pilotRecognizerCode, /processingResolution:\s*'480x360'/, 'Diagnostic metadata must report 480x360');
console.log('[PASS] 2. MediaPipe Hands Pilot Config (modelComplexity=1, 480x360 downsampler, metadata synced)');

// 3. Verify 21-Landmark & Pinky Bone Mapping (17-20)
assert.match(
  pilotRecognizerCode,
  /\[13,\s*17\],\s*\[0,\s*17\],\s*\[17,\s*18\],\s*\[18,\s*19\],\s*\[19,\s*20\]/,
  'Pinky HAND_BONES mapping [13,17], [0,17], [17,18], [18,19], [19,20] must be intact'
);
console.log('[PASS] 3. 21-Landmark & Pinky (17-20) Bone Connectivity Mapping intact');

// 4. Verify HandTrackerStabilizer (One-Euro Filter) on 21 Landmarks including Pinky 17-20
const stabilizer = new HandTrackerStabilizer();
const mockHand = Array.from({ length: 21 }, (_, idx) => ({
  x: 0.4 + idx * 0.01,
  y: 0.5 - idx * 0.005,
  z: -0.01 * (idx % 4),
}));

const step1 = stabilizer.processHands(null, mockHand, 1000);
assert.equal(step1.left, null);
assert.equal(step1.right.length, 21);
for (let i = 17; i <= 20; i++) {
  assert.ok(Number.isFinite(step1.right[i].x) && Number.isFinite(step1.right[i].y) && Number.isFinite(step1.right[i].z));
}

// Simulate folding pinky on next frame (33ms later)
const foldedPinkyHand = mockHand.map((pt, idx) =>
  idx >= 18 ? { x: pt.x + 0.01, y: pt.y + 0.06, z: pt.z } : pt
);
const step2 = stabilizer.processHands(null, foldedPinkyHand, 1033.3);
assert.equal(step2.right.length, 21);
assert.ok(step2.right[20].y > step1.right[20].y, 'Smoothed pinky TIP (20) must follow downward fold motion');
assert.equal(step2.right[8].y, step1.right[8].y, 'Stationary index TIP (8) must remain rock-steady');
console.log('[PASS] 4. HandTrackerStabilizer (Zero-Lag One-Euro Filter) 21-landmark & pinky fold response');

// 5. Verify Monotonic Timestamp Generator
const tsGen = createMonotonicTimestampGenerator();
const t1 = tsGen.next(100.0);
const t2 = tsGen.next(100.0);
assert.ok(t2 > t1, 'Timestamps must be strictly monotonic');
console.log('[PASS] 5. Monotonic Timestamp Generator & Pilot Protocol');
console.log('=== ALL FRONTEND PILOT CHECKS PASSED ===');
