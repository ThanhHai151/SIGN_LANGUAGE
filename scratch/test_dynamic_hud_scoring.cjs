const assert = require('assert');

// Dynamic evaluation logic mirroring src/components/pilotProtocol.js
async function runTests() {
  const { evaluateTrialFrames, DYNAMIC_DIACRITICS } = await import('../src/components/pilotProtocol.js');

  console.log("=== STARTING DYNAMIC HUD SCORING UNIT TEST SUITE ===");

  // Test 1: Static letter (Q) >= 60% match -> PASS
  const staticPassFrames = [
    { predicted_sign: 'Q', confidence: 0.85, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Q', confidence: 0.82, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Q', confidence: 0.80, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Q', confidence: 0.81, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'P', confidence: 0.70, hand_detected: true, frame_status: 'VALID' },
  ];
  const r1 = evaluateTrialFrames(staticPassFrames, 'Q');
  assert.strictEqual(r1.status, 'PASS');
  assert.strictEqual(r1.isPassed, true);
  assert.strictEqual(r1.matchPct, 80.0);
  console.log("PASS: Test 1 - Static letter Q with 80% match passes.");

  // Test 2: Static letter (Q) < 60% match -> NEEDS_FIX
  const staticFailFrames = [
    { predicted_sign: 'Q', confidence: 0.80, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'P', confidence: 0.75, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'P', confidence: 0.75, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'P', confidence: 0.75, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'P', confidence: 0.75, hand_detected: true, frame_status: 'VALID' },
  ];
  const r2 = evaluateTrialFrames(staticFailFrames, 'Q');
  assert.strictEqual(r2.status, 'NEEDS_FIX');
  assert.strictEqual(r2.isPassed, false);
  console.log("PASS: Test 2 - Static letter Q with 20.0% match receives NEEDS_FIX.");

  // Test 3: Static letter with 0 active frames -> NO_HAND
  const staticNoHand = [
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: false, frame_status: 'VALID' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: false, frame_status: 'VALID' },
  ];
  const r3 = evaluateTrialFrames(staticNoHand, 'Q');
  assert.strictEqual(r3.status, 'NO_HAND');
  console.log("PASS: Test 3 - Static letter with no active frames receives NO_HAND.");

  // Test 4: Dynamic letter (Â) with base A (5 frames) + Â confirmed (2 frames) -> PASS
  const dynamicPassFrames = [
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.92, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.88, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.91, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Â', confidence: 0.90, hand_detected: true, frame_status: 'VALID', confirmed: true, prediction_source: 'diacritic_fsm', new_token: 'Â' },
    { predicted_sign: 'Â', confidence: 0.90, hand_detected: true, frame_status: 'VALID', confirmed: true, prediction_source: 'diacritic_fsm' },
  ];
  const r4 = evaluateTrialFrames(dynamicPassFrames, 'Â');
  assert.strictEqual(r4.status, 'PASS');
  assert.strictEqual(r4.isPassed, true);
  assert.strictEqual(r4.isDynamic, true);
  assert.strictEqual(r4.hasBaseEvidence, true);
  assert.strictEqual(r4.hasDiacriticConfirmation, true);
  assert.strictEqual(r4.mismatchFrames, 0); // Base A is NOT counted as a mismatch for dynamic Â!
  console.log("PASS: Test 4 - Dynamic letter Â with base hold + diacritic confirmation passes.");

  // Test 5: Dynamic letter (Đ) with base D + Đ confirmed -> PASS
  const dynamicPassD = [
    { predicted_sign: 'D', confidence: 0.85, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'D', confidence: 0.88, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'D', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Đ', confidence: 0.95, hand_detected: true, frame_status: 'VALID', confirmed: true, prediction_source: 'diacritic_fsm', new_token: 'Đ' },
    { predicted_sign: 'Đ', confidence: 0.95, hand_detected: true, frame_status: 'VALID', confirmed: true, prediction_source: 'diacritic_fsm' },
  ];
  const r5 = evaluateTrialFrames(dynamicPassD, 'Đ');
  assert.strictEqual(r5.status, 'PASS');
  assert.strictEqual(r5.isPassed, true);
  console.log("PASS: Test 5 - Dynamic letter Đ with base D + Đ confirmation passes.");

  // Test 6: Dynamic letter with diacritic but NO base evidence -> NEEDS_FIX
  const dynamicNoBase = [
    { predicted_sign: 'B', confidence: 0.80, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'C', confidence: 0.80, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Â', confidence: 0.90, hand_detected: true, frame_status: 'VALID', confirmed: true, prediction_source: 'diacritic_fsm', new_token: 'Â' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: true, frame_status: 'VALID' },
  ];
  const r6 = evaluateTrialFrames(dynamicNoBase, 'Â');
  assert.strictEqual(r6.status, 'NEEDS_FIX');
  assert.strictEqual(r6.hasBaseEvidence, false);
  console.log("PASS: Test 6 - Dynamic letter without base evidence receives NEEDS_FIX.");

  // Test 7: Backend error frame cannot confirm diacritic
  const backendErrorConfirm = [
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Â', confidence: 0.90, hand_detected: true, frame_status: 'BACKEND_ERROR', backend_error: true, confirmed: true, new_token: 'Â' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: true, frame_status: 'VALID' },
  ];
  const r7 = evaluateTrialFrames(backendErrorConfirm, 'Â');
  assert.strictEqual(r7.status, 'NEEDS_FIX');
  assert.strictEqual(r7.hasDiacriticConfirmation, false);
  console.log("PASS: Test 7 - Backend error frames are excluded and cannot trigger confirmation.");

  // Test 8: Insufficient valid frames (< 5 frames) -> INSUFFICIENT_DATA
  const tooFewFrames = [
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Â', confidence: 0.90, hand_detected: true, frame_status: 'VALID', confirmed: true, new_token: 'Â', prediction_source: 'diacritic_fsm' },
  ];
  const r8 = evaluateTrialFrames(tooFewFrames, 'Â');
  assert.strictEqual(r8.status, 'INSUFFICIENT_DATA');
  console.log("PASS: Test 8 - Fewer than 5 valid frames marked INSUFFICIENT_DATA.");

  // Test 9: No hand detected across all frames -> NO_HAND
  const noHandFrames = [
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: false, frame_status: 'VALID' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: false, frame_status: 'VALID' },
  ];
  const r9 = evaluateTrialFrames(noHandFrames, 'Â');
  assert.strictEqual(r9.status, 'NO_HAND');
  console.log("PASS: Test 9 - Zero hand detection marked NO_HAND.");

  // Test 10: Dynamic letter where diacritic stroke was not recognized -> NEEDS_FIX
  const unconfirmedDynamic = [
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: true, frame_status: 'VALID' },
  ];
  const r10 = evaluateTrialFrames(unconfirmedDynamic, 'Â');
  assert.strictEqual(r10.status, 'NEEDS_FIX');
  assert.strictEqual(r10.hasBaseEvidence, true);
  assert.strictEqual(r10.hasDiacriticConfirmation, false);
  console.log("PASS: Test 10 - Unconfirmed dynamic letter receives NEEDS_FIX.");

  console.log("\n========================================================");
  console.log("ALL 10 DYNAMIC HUD SCORING UNIT TESTS PASSED RIGOROUSLY!");
  console.log("========================================================");
}

runTests().catch(err => {
  console.error("Test failed:", err);
  process.exit(1);
});
