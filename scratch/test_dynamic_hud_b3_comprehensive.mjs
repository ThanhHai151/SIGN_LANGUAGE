import assert from 'node:assert';
import {
  evaluateTrialFrames,
  buildEvaluationReport,
  DYNAMIC_DIACRITICS
} from '../src/components/pilotProtocol.js';

console.log("=== RUNNING CHECKPOINT B3 COMPREHENSIVE TEST SUITE ===\n");

// 1. Đúng base, chưa vẽ dấu -> NEEDS_FIX
console.log("--- 1. Đúng base (A), chưa vẽ dấu Â ---");
{
  const frames = [
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.91, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.89, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.92, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
  ];
  const r = evaluateTrialFrames(frames, 'Â');
  assert.strictEqual(r.status, 'NEEDS_FIX', 'Đúng base nhưng chưa vẽ dấu phải là NEEDS_FIX');
  assert.strictEqual(r.hasBaseEvidence, true);
  assert.strictEqual(r.hasDiacriticConfirmation, false);
  console.log("PASS: Đúng base, chưa vẽ dấu -> NEEDS_FIX");
}

// 2. Đúng base, vẽ sai dấu (A vẽ râu thành Ơ/Ă hay dấu khác không khớp Â) -> NEEDS_FIX
console.log("\n--- 2. Đúng base, vẽ sai dấu (Ă thay vì Â) ---");
{
  const frames = [
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.91, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.89, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Ă', confidence: 0.95, hand_detected: true, frame_status: 'VALID', confirmed: true, new_token: 'Ă' },
    { predicted_sign: 'Ă', confidence: 0.95, hand_detected: true, frame_status: 'VALID', confirmed: true },
  ];
  const r = evaluateTrialFrames(frames, 'Â');
  assert.strictEqual(r.status, 'NEEDS_FIX', 'Vẽ sai dấu (Ă thay vì Â) phải là NEEDS_FIX');
  assert.strictEqual(r.hasBaseEvidence, true);
  assert.strictEqual(r.hasDiacriticConfirmation, false, 'Dấu Ă không được tính xác nhận cho Â');
  console.log("PASS: Đúng base, vẽ sai dấu -> NEEDS_FIX");
}

// 3. Nhận nhầm Â thành Ê -> FAIL / NEEDS_FIX
console.log("\n--- 3. Nhận nhầm Â thành Ê ---");
{
  const frames = [
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.91, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Ê', confidence: 0.92, hand_detected: true, frame_status: 'VALID', confirmed: true, new_token: 'Ê', diacritic_fsm: { state: 'DIACRITIC_CONFIRMED', diacritic_detected: 'Ê' } },
    { predicted_sign: 'Ê', confidence: 0.92, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: true, frame_status: 'VALID' },
  ];
  const r = evaluateTrialFrames(frames, 'Â');
  assert.strictEqual(r.status, 'NEEDS_FIX', 'Xác nhận Ê khi mục tiêu là Â phải là NEEDS_FIX');
  assert.strictEqual(r.hasDiacriticConfirmation, false);
  console.log("PASS: Nhận nhầm Â thành Ê -> NEEDS_FIX");
}

// 4. Có source FSM nhưng chưa confirmed (fsmSource && isTarget alone) -> KHÔNG ĐƯỢC PASS
console.log("\n--- 4. Có source FSM nhưng chưa confirmed (unconfirmed transitional) ---");
{
  const frames = [
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    // Frame mang source diacritic_fsm nhưng confirmed=false, new_token=null, fsm state=COOLDOWN
    {
      predicted_sign: 'Â',
      confidence: 0.70,
      hand_detected: true,
      frame_status: 'VALID',
      prediction_source: 'diacritic_fsm',
      source: 'diacritic_fsm',
      confirmed: false,
      new_token: null,
      classifier_details: { diacritic_fsm: { state: 'COOLDOWN' } }
    },
    {
      predicted_sign: 'Â',
      confidence: 0.70,
      hand_detected: true,
      frame_status: 'VALID',
      prediction_source: 'diacritic_fsm',
      source: 'diacritic_fsm',
      confirmed: false,
      new_token: null,
      classifier_details: { diacritic_fsm: { state: 'BASE_STABLE' } }
    },
  ];
  const r = evaluateTrialFrames(frames, 'Â');
  assert.strictEqual(r.status, 'NEEDS_FIX', 'fsmSource alone without genuine confirmation must NOT pass');
  assert.strictEqual(r.hasDiacriticConfirmation, false);
  console.log("PASS: Có source FSM nhưng chưa confirmed -> NEEDS_FIX (Đã bẻ gãy lỗ hổng fsmSource && isTarget)");
}

// 5. Confirmation hợp lệ (tokenMatch + base evidence) -> PASS
console.log("\n--- 5. Confirmation hợp lệ (A hold + Â confirmed) ---");
{
  const frames = [
    { predicted_sign: 'A', confidence: 0.92, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.91, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    {
      predicted_sign: 'Â',
      confidence: 0.95,
      hand_detected: true,
      frame_status: 'VALID',
      prediction_source: 'diacritic_fsm',
      confirmed: true,
      new_token: 'Â',
      classifier_details: { diacritic_fsm: { state: 'DIACRITIC_CONFIRMED', diacritic_detected: 'Â' } }
    },
    { predicted_sign: 'Â', confidence: 0.95, hand_detected: true, frame_status: 'VALID', confirmed: true },
  ];
  const r = evaluateTrialFrames(frames, 'Â');
  assert.strictEqual(r.status, 'PASS');
  assert.strictEqual(r.isPassed, true);
  assert.strictEqual(r.hasBaseEvidence, true);
  assert.strictEqual(r.hasDiacriticConfirmation, true);
  console.log("PASS: Confirmation hợp lệ -> PASS");
}

// 6. Backend error frame không thể trigger confirmation -> NEEDS_FIX
console.log("\n--- 6. Backend error frame ---");
{
  const frames = [
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Â', confidence: 0.95, hand_detected: true, frame_status: 'BACKEND_ERROR', backend_error: true, confirmed: true, new_token: 'Â' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: true, frame_status: 'VALID' },
  ];
  const r = evaluateTrialFrames(frames, 'Â');
  assert.strictEqual(r.status, 'NEEDS_FIX');
  assert.strictEqual(r.hasDiacriticConfirmation, false);
  console.log("PASS: Backend error frame excluded from confirmation");
}

// 7. Thiếu frame (< 5 frames) -> INSUFFICIENT_DATA
console.log("\n--- 7. Thiếu frame (< 5 frames) ---");
{
  const frames = [
    { predicted_sign: 'A', confidence: 0.90, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Â', confidence: 0.95, hand_detected: true, frame_status: 'VALID', confirmed: true, new_token: 'Â' },
  ];
  const r = evaluateTrialFrames(frames, 'Â');
  assert.strictEqual(r.status, 'INSUFFICIENT_DATA');
  console.log("PASS: Thiếu frame -> INSUFFICIENT_DATA");
}

// 8. Không có tay (hand_detected = false on all frames) -> NO_HAND
console.log("\n--- 8. Không có tay (NO_HAND) ---");
{
  const frames = [
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: false, frame_status: 'VALID' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: false, frame_status: 'VALID' },
  ];
  const r = evaluateTrialFrames(frames, 'Â');
  assert.strictEqual(r.status, 'NO_HAND');
  console.log("PASS: Không có tay -> NO_HAND");
}

// 9. Báo cáo đánh giá tổng thể (buildEvaluationReport) chữ động dùng Dynamic Pass Rate
console.log("\n--- 9. Dynamic Accuracy & Pass Rate in buildEvaluationReport ---");
{
  const mockResults = {
    'Â': [
      { rep: 1, status: 'PASS', isDynamic: true, matchFrames: 2, activeFrames: 10 },
      { rep: 2, status: 'PASS', isDynamic: true, matchFrames: 3, activeFrames: 10 },
      { rep: 3, status: 'PASS', isDynamic: true, matchFrames: 2, activeFrames: 10 },
      { rep: 4, status: 'PASS', isDynamic: true, matchFrames: 2, activeFrames: 10 },
      { rep: 5, status: 'PASS', isDynamic: true, matchFrames: 2, activeFrames: 10 },
      { rep: 6, status: 'PASS', isDynamic: true, matchFrames: 2, activeFrames: 10 },
      { rep: 7, status: 'PASS', isDynamic: true, matchFrames: 2, activeFrames: 10 },
      { rep: 8, status: 'NEEDS_FIX', isDynamic: true, matchFrames: 0, activeFrames: 10 },
      { rep: 9, status: 'PASS', isDynamic: true, matchFrames: 2, activeFrames: 10 },
      { rep: 10, status: 'PASS', isDynamic: true, matchFrames: 2, activeFrames: 10 },
    ]
  };
  const rep = buildEvaluationReport('sess_dyn_01', mockResults, { trialsPerLetter: 10 });
  const aHat = rep.letters['Â'];
  assert.strictEqual(aHat.totalTrials, 10);
  assert.strictEqual(aHat.passedTrials, 9);
  assert.strictEqual(aHat.failedTrials, 1);
  assert.strictEqual(aHat.dynamicPassRate, 90.0, 'Dynamic pass rate must be 90%');
  assert.strictEqual(aHat.accuracyPct, 90.0, 'accuracyPct for dynamic letter must reflect pass rate, not frame pct (which would be ~20%)');
  assert.strictEqual(aHat.status, 'PASS', '9/10 passed reps >= 70% required -> PASS');
  console.log("PASS: buildEvaluationReport evaluates dynamic letter Â with 90% dynamic pass rate -> PASS");
}

// 10. Không công nhận nghiệm thu khi mới xong 1/10 reps (Chữ động)
console.log("\n--- 10. Không công nhận nghiệm thu khi mới chạy 1/10 reps ---");
{
  const mockResults1Rep = {
    'Â': [
      { rep: 1, status: 'PASS', isDynamic: true, matchFrames: 2, activeFrames: 10 }
    ]
  };
  const rep = buildEvaluationReport('sess_dyn_02', mockResults1Rep, { trialsPerLetter: 10 });
  const aHat = rep.letters['Â'];
  assert.strictEqual(aHat.status, 'NEEDS_FIX', '1/1 rep cannot pass when 10 reps are required');
  assert.strictEqual(rep.summary.letters_passed.includes('Â'), false);
  console.log("PASS: 1/10 rep completed does NOT prematurely pass letter Â");
}

// 11. Static scoring regression check: Chữ tĩnh Q vẫn dùng mẫu số active frames (match + mismatch + neutral)
console.log("\n--- 11. Static scoring regression check (Letter Q) ---");
{
  const staticFrames = [
    { predicted_sign: 'Q', confidence: 0.85, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Q', confidence: 0.82, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'Q', confidence: 0.80, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'NEUTRAL', confidence: 0.0, hand_detected: true, frame_status: 'VALID' },
    { predicted_sign: 'P', confidence: 0.70, hand_detected: true, frame_status: 'VALID' },
  ];
  const r = evaluateTrialFrames(staticFrames, 'Q');
  assert.strictEqual(r.activeFrames, 5, 'Static letter activeFrames includes neutral hand frame');
  assert.strictEqual(r.matchFrames, 3);
  assert.strictEqual(r.matchPct, 60.0, '3/5 = 60.0% match');
  assert.strictEqual(r.status, 'PASS', '60.0% >= 60.0% -> PASS');
  console.log("PASS: Static scoring regression preserved for Q");
}

console.log("\n========================================================");
console.log("ALL 11 CHECKPOINT B3 TESTS PASSED RIGOROUSLY!");
console.log("========================================================");
