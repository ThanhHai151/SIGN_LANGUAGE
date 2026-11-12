import assert from 'assert';
import {
  evaluateTrialFrames,
  buildEvaluationReport,
  isHandDetectedInFrame,
  DYNAMIC_DIACRITICS
} from '../src/components/pilotProtocol.js';

console.log("=== EXECUTING PHASE A SCORING EDGE CASES AUDIT ===");

// 1. Toàn bộ frames là BACKEND_ERROR → không được trả NO_HAND (trả INSUFFICIENT_DATA)
const ec1_frames = [
  { frame_status: 'BACKEND_ERROR', backend_error: true, hand_detected: true },
  { frame_status: 'BACKEND_ERROR', backend_error: true, hand_detected: true },
  { frame_status: 'BACKEND_ERROR', backend_error: true, hand_detected: false }
];
const r1 = evaluateTrialFrames(ec1_frames, 'H');
assert.strictEqual(r1.status, 'INSUFFICIENT_DATA', "EC1: All backend errors must return INSUFFICIENT_DATA, NOT NO_HAND");
assert.notStrictEqual(r1.status, 'NO_HAND', "EC1: All backend errors must NOT return NO_HAND");
console.log("PASS: EC 1 - All frames BACKEND_ERROR returns INSUFFICIENT_DATA, not NO_HAND.");

// 2. Không có frame nào (empty array) → INSUFFICIENT_DATA, không tự suy luận NO_HAND
const ec2_frames = [];
const r2 = evaluateTrialFrames(ec2_frames, 'H');
assert.strictEqual(r2.status, 'INSUFFICIENT_DATA', "EC2: Empty frames must return INSUFFICIENT_DATA, NOT NO_HAND");
assert.notStrictEqual(r2.status, 'NO_HAND', "EC2: Empty frames must NOT return NO_HAND");
console.log("PASS: EC 2 - Empty frames array returns INSUFFICIENT_DATA, not NO_HAND.");

// 3. Có hand_detected=false trên toàn bộ frames hợp lệ → NO_HAND
const ec3_frames = [
  { frame_status: 'VALID', hand_detected: false, predicted_sign: 'NEUTRAL' },
  { frame_status: 'VALID', hand_detected: false, predicted_sign: 'NEUTRAL' },
  { frame_status: 'VALID', hand_detected: false, predicted_sign: 'NEUTRAL' }
];
const r3 = evaluateTrialFrames(ec3_frames, 'H');
assert.strictEqual(r3.status, 'NO_HAND', "EC3: All valid frames with hand_detected=false must return NO_HAND");
console.log("PASS: EC 3 - hand_detected=false on all valid frames correctly returns NO_HAND.");

// 4. Có tay và toàn NEUTRAL → NEEDS_FIX nếu đủ dữ liệu (>= 5 frames)
const ec4_frames = Array(8).fill({
  frame_status: 'VALID',
  hand_detected: true,
  predicted_sign: 'NEUTRAL',
  confidence: 0.0
});
const r4 = evaluateTrialFrames(ec4_frames, 'H');
assert.strictEqual(r4.status, 'NEEDS_FIX', "EC4: All NEUTRAL with hands (>= 5 frames) must return NEEDS_FIX");
assert.strictEqual(r4.matchPct, 0.0);
console.log("PASS: EC 4 - Hand detected and all NEUTRAL with sufficient data returns NEEDS_FIX.");

// 5. hand_detected=null hoặc undefined → xử lý an toàn và tương thích ngược
const legacyHandWithLandmarks = {
  frame_status: 'VALID',
  landmarks: Array(21).fill([0.5, 0.5, 0.0]),
  predicted_sign: 'H',
  confidence: 0.85
};
assert.strictEqual(isHandDetectedInFrame(legacyHandWithLandmarks), true, "EC5.1: Landmarks present implies hand detected");

const legacyNoHandNoLandmarks = {
  frame_status: 'VALID',
  landmarks: null,
  predicted_sign: 'NEUTRAL',
  confidence: 0.0
};
assert.strictEqual(isHandDetectedInFrame(legacyNoHandNoLandmarks), false, "EC5.2: No landmarks and NEUTRAL implies no hand");

const legacyLegacySignNoField = {
  frame_status: 'VALID',
  predicted_sign: 'Q',
  confidence: 0.90
};
assert.strictEqual(isHandDetectedInFrame(legacyLegacySignNoField), true, "EC5.3: Valid predicted sign implies hand detected");

// Evaluate trial with 5 legacy frames
const ec5_trial_frames = Array(6).fill(legacyHandWithLandmarks);
const r5 = evaluateTrialFrames(ec5_trial_frames, 'H');
assert.strictEqual(r5.status, 'PASS');
console.log("PASS: EC 5 - hand_detected=null/undefined handled safely with backwards compatibility.");

// 6. 10 reps có một số lượt INSUFFICIENT_DATA (6 PASS, 2 INSUFFICIENT_DATA, 2 NEEDS_FIX) → không được nghiệm thu PASS sai
const ec6_trials = [
  ...Array(6).fill({ rep: 1, status: 'PASS', matchFrames: 8, activeFrames: 10 }),
  { rep: 7, status: 'INSUFFICIENT_DATA', matchFrames: 0, activeFrames: 2 },
  { rep: 8, status: 'INSUFFICIENT_DATA', matchFrames: 0, activeFrames: 3 },
  { rep: 9, status: 'NEEDS_FIX', matchFrames: 2, activeFrames: 10 },
  { rep: 10, status: 'NEEDS_FIX', matchFrames: 1, activeFrames: 10 }
];
const r6 = buildEvaluationReport('session_ec6', { 'H': ec6_trials }, { trialsPerLetter: 10 });
assert.strictEqual(r6.letters['H'].status, 'NEEDS_FIX', "EC6: 6 PASS out of 10 must NOT pass letter");
assert.strictEqual(r6.letters['H'].passedTrials, 6);
assert.strictEqual(r6.summary.letters_passed.includes('H'), false);
console.log("PASS: EC 6 - 6 PASS + 2 INSUFFICIENT_DATA + 2 NEEDS_FIX correctly results in letter NEEDS_FIX.");

// 7. 1/1 rep PASS → chưa đạt letter-level acceptance
const ec7_trials = [{ rep: 1, status: 'PASS', matchFrames: 10, activeFrames: 10 }];
const r7 = buildEvaluationReport('session_ec7', { 'H': ec7_trials }, { trialsPerLetter: 10 });
assert.strictEqual(r7.letters['H'].status, 'NEEDS_FIX', "EC7: 1/1 trial PASS must NOT pass letter when 10 reps required");
console.log("PASS: EC 7 - 1/1 trial PASS does not achieve letter-level acceptance.");

// 8. 10 reps, tối thiểu 7 PASS hợp lệ (7 PASS, 3 INSUFFICIENT_DATA) → đạt PASS theo SSOT
const ec8_trials = [
  ...Array(7).fill({ rep: 1, status: 'PASS', matchFrames: 9, activeFrames: 10 }),
  { rep: 8, status: 'INSUFFICIENT_DATA', matchFrames: 0, activeFrames: 2 },
  { rep: 9, status: 'INSUFFICIENT_DATA', matchFrames: 0, activeFrames: 3 },
  { rep: 10, status: 'NEEDS_FIX', matchFrames: 2, activeFrames: 10 }
];
const r8 = buildEvaluationReport('session_ec8', { 'H': ec8_trials }, { trialsPerLetter: 10 });
assert.strictEqual(r8.letters['H'].status, 'PASS', "EC8: 7 PASS out of 10 meets SSOT >= 70% requirement");
assert.strictEqual(r8.summary.letters_passed.includes('H'), true);
console.log("PASS: EC 8 - 10 reps with 7 PASS reaches letter-level acceptance per SSOT.");

// 9. Backend errors, stale responses và wrong trial_id không được làm tăng điểm
const ec9_frames = [
  ...Array(5).fill({ frame_status: 'BACKEND_ERROR', backend_error: true, predicted_sign: 'H', confidence: 0.99, hand_detected: true }),
  ...Array(2).fill({ frame_status: 'VALID', predicted_sign: 'H', confidence: 0.85, hand_detected: true })
];
const r9 = evaluateTrialFrames(ec9_frames, 'H');
assert.strictEqual(r9.matchFrames, 2, "EC9: Backend error frames must not count as match");
assert.strictEqual(r9.status, 'INSUFFICIENT_DATA', "EC9: Only 2 valid frames is INSUFFICIENT_DATA (< 5)");
console.log("PASS: EC 9 - Backend errors cannot inflate score.");

// 10. Không làm thay đổi hợp đồng dynamic confirmation
const ec10_dynamic_frames = [
  ...Array(4).fill({ frame_status: 'VALID', predicted_sign: 'A', confidence: 0.90, hand_detected: true }),
  { frame_status: 'VALID', predicted_sign: 'Â', confidence: 0.95, hand_detected: true, confirmed: true, new_token: 'Â', prediction_source: 'diacritic_fsm' }
];
const r10 = evaluateTrialFrames(ec10_dynamic_frames, 'Â');
assert.strictEqual(r10.status, 'PASS');
assert.strictEqual(r10.isDynamic, true);
assert.strictEqual(r10.hasBaseEvidence, true);
assert.strictEqual(r10.hasDiacriticConfirmation, true);

// Dynamic without diacritic stroke
const ec10_fail_frames = Array(6).fill({ frame_status: 'VALID', predicted_sign: 'A', confidence: 0.90, hand_detected: true });
const r10_fail = evaluateTrialFrames(ec10_fail_frames, 'Â');
assert.strictEqual(r10_fail.status, 'NEEDS_FIX');
assert.strictEqual(r10_fail.hasBaseEvidence, true);
assert.strictEqual(r10_fail.hasDiacriticConfirmation, false);
console.log("PASS: EC 10 - Dynamic confirmation contract is rigorously preserved.");

console.log("\n=======================================================");
console.log("ALL 10 SCORING EDGE CASES AUDITED AND VERIFIED PASSED!");
console.log("=======================================================");

