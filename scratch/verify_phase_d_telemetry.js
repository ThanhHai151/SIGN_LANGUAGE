// scratch/verify_phase_d_telemetry.js
// Verification of TASK 1.3C Phase D telemetry logic

import assert from 'node:assert';
import { phaseAtElapsedSeconds } from '../src/components/pilotProtocol.js';

// Logic under test: exactly mirrors onFrameResult from PilotTestingHUD.jsx
function simulateOnFrameResult({
  data,
  pilotMeta,
  selectedLetter = 'Q',
  recordingStartTime = 1000.0, // ms
  currentTime = 3500.0, // ms (elapsed = 2.5s -> STABLE_HOLD)
  trialFrames = []
}) {
  const expected = selectedLetter;
  const predicted = data?.sign || 'NEUTRAL';
  const conf = Number(data?.confidence || 0);
  const isMatch = (predicted === expected);

  // Tính toán thời gian trôi qua và phase cho lượt thử hiện tại (đo từ thời điểm bắt đầu lượt)
  const elapsedSec = recordingStartTime > 0
    ? (currentTime - recordingStartTime) / 1000
    : 0.0;
  const phase = phaseAtElapsedSeconds(elapsedSec);

  // Tracking bàn tay từ MediaPipe (true / false / null)
  let handDetected = null;
  if (typeof pilotMeta?.hand_detected === 'boolean') {
    handDetected = pilotMeta.hand_detected;
  } else if (typeof pilotMeta?.has_hands === 'boolean') {
    handDetected = pilotMeta.has_hands;
  }

  // Nguồn dự đoán và trạng thái phân loại
  const predictionSource = data?.prediction_source ?? data?.source ?? null;
  const gatingStatus = data?.gating_status ?? data?.classifier_details?.gating_status ?? null;
  const classifierStatus = data?.classifier_details?.status ?? null;
  const rejectReason = data?.classifier_details?.reject_reason ?? null;

  // Độ tin cậy thô của static head nếu có
  let headConfidence = null;
  if (typeof data?.classifier_details?.head_confidence === 'number') {
    headConfidence = data.classifier_details.head_confidence;
  } else if (predictionSource === 'static_head' || data?.classifier_details?.source === 'static_head') {
    if (typeof data?.classifier_details?.confidence === 'number') {
      headConfidence = data.classifier_details.confidence;
    } else if (typeof data?.confidence === 'number') {
      headConfidence = data.confidence;
    }
  }

  // Sự kiện xác nhận token từ backend
  const confirmed = Boolean(data?.confirmed);
  const newToken = data?.new_token ?? null;

  const frameRecord = {
    frame_idx: pilotMeta?.frame_idx || trialFrames.length + 1,
    expected_sign: expected,
    predicted_sign: predicted,
    confidence: conf,
    timestamp: currentTime / 1000,
    is_match: isMatch,
    landmarks: data?.landmarks || null,
    hand_detected: handDetected,
    prediction_source: predictionSource,
    gating_status: gatingStatus,
    classifier_status: classifierStatus,
    reject_reason: rejectReason,
    head_confidence: headConfidence,
    confirmed: confirmed,
    new_token: newToken,
    phase: phase,
    elapsed_sec: Number(elapsedSec.toFixed(3))
  };

  trialFrames.push(frameRecord);
  return frameRecord;
}

console.log('=== RUNNING PHASE D TELEMETRY VERIFICATION ===\n');

// CASE A: Hand detected + correct Q
console.log('--- TEST CASE A: Hand detected + correct Q ---');
{
  const pilotMeta = { frame_idx: 1, hand_detected: true, has_hands: true };
  const data = {
    sign: 'Q',
    confidence: 0.78,
    prediction_source: 'static_head',
    source: 'static_head',
    confirmed: false,
    new_token: null,
    classifier_details: {
      status: 'MATCH',
      letter: 'Q',
      confidence: 0.78,
      source: 'static_head',
      reject_reason: null
    }
  };
  const record = simulateOnFrameResult({ data, pilotMeta, selectedLetter: 'Q', currentTime: 3000.0, recordingStartTime: 1000.0 });
  assert.strictEqual(record.hand_detected, true, 'Case A: hand_detected must be true');
  assert.strictEqual(record.prediction_source, 'static_head', 'Case A: prediction_source must match API');
  assert.strictEqual(record.confidence, 0.78, 'Case A: confidence must be preserved');
  assert.strictEqual(record.predicted_sign, 'Q', 'Case A: predicted_sign must be Q');
  assert.strictEqual(record.is_match, true, 'Case A: is_match must be true');
  assert.strictEqual(record.classifier_status, 'MATCH', 'Case A: classifier_status must be MATCH');
  assert.strictEqual(record.head_confidence, 0.78, 'Case A: head_confidence must match');
  assert.strictEqual(record.reject_reason, null, 'Case A: reject_reason must be null');
  console.log('PASS: Case A verified.');
}

// CASE B: Hand detected + NEUTRAL (e.g. static head rejected below 0.75)
console.log('\n--- TEST CASE B: Hand detected + NEUTRAL ---');
{
  const pilotMeta = { frame_idx: 2, hand_detected: true, has_hands: true };
  const data = {
    sign: 'NEUTRAL',
    confidence: 0.0,
    prediction_source: 'none',
    source: 'none',
    confirmed: false,
    new_token: null,
    classifier_details: {
      status: 'NO_MATCH',
      letter: 'UNKNOWN',
      confidence: 0.0,
      source: 'static_head_rejected',
      head_letter: 'Q',
      head_confidence: 0.72,
      reject_reason: 'Confidence 0.72 < threshold 0.75'
    }
  };
  const record = simulateOnFrameResult({ data, pilotMeta, selectedLetter: 'Q', currentTime: 3200.0, recordingStartTime: 1000.0 });
  assert.strictEqual(record.hand_detected, true, 'Case B: hand_detected must be true even if NEUTRAL');
  assert.strictEqual(record.predicted_sign, 'NEUTRAL', 'Case B: predicted_sign must be NEUTRAL');
  assert.strictEqual(record.is_match, false, 'Case B: is_match must be false');
  assert.strictEqual(record.prediction_source, 'none', 'Case B: prediction_source is none from API');
  assert.strictEqual(record.classifier_status, 'NO_MATCH', 'Case B: classifier_status must be NO_MATCH');
  assert.strictEqual(record.reject_reason, 'Confidence 0.72 < threshold 0.75', 'Case B: reject_reason preserved');
  assert.strictEqual(record.head_confidence, 0.72, 'Case B: head_confidence preserved');
  console.log('PASS: Case B verified.');
}

// CASE C: No hand (hand_detected = false)
console.log('\n--- TEST CASE C: No hand (hand_detected = false) ---');
{
  const pilotMeta = { frame_idx: 3, hand_detected: false, has_hands: false };
  const data = {
    sign: 'NEUTRAL',
    label: 'Chờ cử chỉ...',
    type: 'unknown',
    confidence: 0.0,
    confirmed: false,
    new_token: null,
    sentence: ''
  };
  const record = simulateOnFrameResult({ data, pilotMeta, selectedLetter: 'Q', currentTime: 3400.0, recordingStartTime: 1000.0 });
  assert.strictEqual(record.hand_detected, false, 'Case C: hand_detected must be false from MediaPipe');
  assert.strictEqual(record.predicted_sign, 'NEUTRAL', 'Case C: predicted_sign must be NEUTRAL');
  assert.strictEqual(record.prediction_source, null, 'Case C: prediction_source is null when omitted by API');
  assert.strictEqual(record.classifier_status, null, 'Case C: classifier_status must be null');
  assert.strictEqual(record.reject_reason, null, 'Case C: reject_reason must be null');
  assert.strictEqual(record.head_confidence, null, 'Case C: head_confidence must be null');
  console.log('PASS: Case C verified.');
}

// CASE D: No tracking evidence (hand_detected = null)
console.log('\n--- TEST CASE D: No tracking evidence (hand_detected = null) ---');
{
  const pilotMeta = { frame_idx: 4 }; // no hand_detected or has_hands provided
  const data = {
    sign: 'NEUTRAL',
    confidence: 0.0,
    confirmed: false,
    new_token: null
  };
  const record = simulateOnFrameResult({ data, pilotMeta, selectedLetter: 'Q', currentTime: 3600.0, recordingStartTime: 1000.0 });
  assert.strictEqual(record.hand_detected, null, 'Case D: hand_detected must be null when tracking metadata is missing');
  console.log('PASS: Case D verified.');
}

// CASE E: Backend reports confirmed token
console.log('\n--- TEST CASE E: Confirmed token ---');
{
  const pilotMeta = { frame_idx: 5, hand_detected: true, has_hands: true };
  const data = {
    sign: 'Q',
    confidence: 0.81,
    prediction_source: 'static_head',
    source: 'static_head',
    confirmed: true,
    new_token: 'Q',
    classifier_details: {
      status: 'MATCH',
      letter: 'Q',
      confidence: 0.81,
      source: 'static_head'
    }
  };
  const record = simulateOnFrameResult({ data, pilotMeta, selectedLetter: 'Q', currentTime: 3800.0, recordingStartTime: 1000.0 });
  assert.strictEqual(record.confirmed, true, 'Case E: confirmed must be true');
  assert.strictEqual(record.new_token, 'Q', 'Case E: new_token must be Q');
  console.log('PASS: Case E verified.');
}

// CASE F: Phase check
console.log('\n--- TEST CASE F: Phase calculation ---');
{
  // Test per-trial elapsed time phases
  const t0 = 1000.0;
  const f_pre = simulateOnFrameResult({ data: {}, pilotMeta: {}, recordingStartTime: t0, currentTime: t0 + 200 }); // 0.2s
  assert.strictEqual(f_pre.phase, 'PRE_IDLE', 'Phase < 0.5s must be PRE_IDLE');

  const f_entry = simulateOnFrameResult({ data: {}, pilotMeta: {}, recordingStartTime: t0, currentTime: t0 + 1000 }); // 1.0s
  assert.strictEqual(f_entry.phase, 'ENTRY', 'Phase 0.5s-1.5s must be ENTRY');

  const f_hold = simulateOnFrameResult({ data: {}, pilotMeta: {}, recordingStartTime: t0, currentTime: t0 + 2500 }); // 2.5s
  assert.strictEqual(f_hold.phase, 'STABLE_HOLD', 'Phase 1.5s-4.5s must be STABLE_HOLD');

  const f_exit = simulateOnFrameResult({ data: {}, pilotMeta: {}, recordingStartTime: t0, currentTime: t0 + 4800 }); // 4.8s
  assert.strictEqual(f_exit.phase, 'EXIT', 'Phase 4.5s-5.5s must be EXIT');

  const f_post = simulateOnFrameResult({ data: {}, pilotMeta: {}, recordingStartTime: t0, currentTime: t0 + 5600 }); // 5.6s
  assert.strictEqual(f_post.phase, 'POST_IDLE', 'Phase >= 5.5s must be POST_IDLE');

  // Verify that a trial started late in the session (e.g. at session time 600s = 10 min) calculates phase from trial start
  const rep2_start = 600000.0;
  const rep2_frame = simulateOnFrameResult({ data: {}, pilotMeta: {}, recordingStartTime: rep2_start, currentTime: rep2_start + 2500 }); // 2.5s into Rep 2
  assert.strictEqual(rep2_frame.phase, 'STABLE_HOLD', 'Rep 2 phase must be STABLE_HOLD, not affected by session start');
  assert.strictEqual(rep2_frame.elapsed_sec, 2.5, 'elapsed_sec must be 2.5');
  console.log('PASS: Case F verified.');
}

// CASE G: Saved JSONL persistence & backward compatibility
console.log('\n--- TEST CASE G: Saved JSONL persistence & backward compatibility ---');
{
  const pilotMeta = { frame_idx: 1, hand_detected: true, has_hands: true };
  const data = {
    sign: 'Q',
    confidence: 0.77,
    prediction_source: 'static_head',
    landmarks: [[0.1, 0.2, 0.3]]
  };
  const record = simulateOnFrameResult({ data, pilotMeta, selectedLetter: 'Q', currentTime: 3500.0, recordingStartTime: 1000.0 });
  const serialized = JSON.stringify(record);
  const parsed = JSON.parse(serialized);

  // Check all legacy fields preserved
  const legacyFields = ['frame_idx', 'expected_sign', 'predicted_sign', 'confidence', 'timestamp', 'is_match', 'landmarks'];
  for (const field of legacyFields) {
    assert.ok(field in parsed, `Legacy field ${field} must exist`);
  }

  // Check all new fields exist
  const newFields = [
    'hand_detected', 'prediction_source', 'gating_status',
    'classifier_status', 'reject_reason', 'head_confidence',
    'confirmed', 'new_token', 'phase', 'elapsed_sec'
  ];
  for (const field of newFields) {
    assert.ok(field in parsed, `New field ${field} must exist in JSONL`);
  }
  console.log('PASS: Case G verified.');
}

console.log('\n=== ALL PHASE D TEST CASES A-G PASSED SUCCESSFULLY! ===');

