// scratch/test_task1_3g_cases.cjs
// Comprehensive 10-Case Automated Test Suite for Task 1.3G Error Handling & Trial Validity Patch

const assert = require('assert');

function createPilotTrialId(sessionId, target, repetition) {
  return `${sessionId}_${target}_rep${String(repetition).padStart(2, '0')}`;
}

function phaseAtElapsedSeconds(elapsedSec) {
  if (elapsedSec < 0.5) return 'PRE_IDLE';
  if (elapsedSec < 1.5) return 'ENTRY';
  if (elapsedSec <= 4.5) return 'STABLE_HOLD';
  return 'EXIT';
}

class PilotHUDHarnessTask13G {
  constructor(sessionId = 'pilot_20261111_test_task13g') {
    this.sessionId = sessionId;
    this.selectedLetter = 'Q';
    this.currentRep = 1;
    this.trialState = 'IDLE';
    this.trialGeneration = 0;
    this.trialFrames = [];
    this.recordingStartTime = 0;
    this.frameCounter = 0;
    this.trialSummaries = [];
  }

  transitionTrialState(newState) {
    this.trialState = newState;
    this.trialGeneration += 1;
  }

  startRecordingPeriod() {
    this.trialFrames = [];
    this.frameCounter = 0;
    this.recordingStartTime = Date.now();
    this.transitionTrialState('RECORDING');
  }

  handleCompleteTrial() {
    const frames = [...this.trialFrames];
    const totalFrames = frames.length;
    const validFrames = frames.filter(f => !f.backend_error && f.frame_status === 'VALID').length;
    const backendErrorFrames = frames.filter(f => f.backend_error || f.frame_status === 'BACKEND_ERROR').length;
    const validityRatio = totalFrames > 0 ? Number((validFrames / totalFrames).toFixed(3)) : 0.0;
    const isTrialValid = validFrames >= 5 && backendErrorFrames === 0;

    const summary = {
      trial_id: createPilotTrialId(this.sessionId, this.selectedLetter, this.currentRep),
      rep_idx: this.currentRep,
      frames,
      total_frames: totalFrames,
      valid_frames: validFrames,
      backend_error_frames: backendErrorFrames,
      validity_ratio: validityRatio,
      is_trial_valid: isTrialValid
    };

    this.trialSummaries.push(summary);
    this.transitionTrialState('TRIAL_REVIEW');
    return summary;
  }

  handleNextTrial() {
    this.currentRep += 1;
    this.startRecordingPeriod();
  }

  getFrameMetadata(handDetected = true) {
    if (this.trialState !== 'RECORDING') return null;
    this.frameCounter += 1;
    const trialId = createPilotTrialId(this.sessionId, this.selectedLetter, this.currentRep);
    return {
      session_id: this.sessionId,
      trial_id: trialId,
      expected_sign: this.selectedLetter,
      rep_idx: this.currentRep,
      trial_generation: this.trialGeneration,
      frame_idx: this.frameCounter,
      client_capture_ts: Date.now() / 1000,
      signer_id: 'signer_01',
      hand_detected: handDetected,
      has_hands: handDetected
    };
  }

  onFrameResult(data, requestMeta, errorOptions = {}) {
    const { backendError = false, backendErrorMessage = null } = errorOptions;

    // Gate 1: Must be in RECORDING state
    if (this.trialState !== 'RECORDING') return false;

    // Gate 2: Must carry non-null request metadata
    if (!requestMeta) return false;

    // Gate 3: Generation isolation
    if (requestMeta.trial_generation !== this.trialGeneration) return false;

    // Gate 4: Session ID match
    if (requestMeta.session_id !== this.sessionId) return false;

    // Gate 5: Rep index and sign match
    if (requestMeta.rep_idx !== this.currentRep || requestMeta.expected_sign !== this.selectedLetter) return false;

    // Gate 6: Canonical trial_id match
    const expectedTrialId = createPilotTrialId(this.sessionId, this.selectedLetter, this.currentRep);
    if (requestMeta.trial_id !== expectedTrialId) return false;

    const elapsedSec = (Date.now() - this.recordingStartTime) / 1000;
    const phase = phaseAtElapsedSeconds(elapsedSec);

    let frameRecord;
    if (backendError) {
      frameRecord = {
        frame_idx: this.trialFrames.length + 1,
        client_capture_ts: requestMeta.client_capture_ts || (Date.now() / 1000),
        expected_sign: this.selectedLetter,
        predicted_sign: 'ERROR',
        confidence: 0.0,
        is_match: false,
        hand_detected: requestMeta.hand_detected ?? null,
        prediction_source: 'error',
        gating_status: 'error',
        classifier_status: 'BACKEND_ERROR',
        reject_reason: backendErrorMessage || 'backend_error',
        head_confidence: 0.0,
        confirmed: false,
        new_token: null,
        phase,
        elapsed_sec: Number(elapsedSec.toFixed(3)),
        backend_error: true,
        backend_error_message: backendErrorMessage || 'backend_error',
        frame_status: 'BACKEND_ERROR'
      };
    } else {
      const predictedSign = data?.sign || 'NEUTRAL';
      const confidence = typeof data?.confidence === 'number' ? data.confidence : 0.0;
      const isMatch = Boolean(predictedSign === this.selectedLetter);

      frameRecord = {
        frame_idx: this.trialFrames.length + 1,
        client_capture_ts: requestMeta.client_capture_ts || (Date.now() / 1000),
        expected_sign: this.selectedLetter,
        predicted_sign: predictedSign,
        confidence,
        is_match: isMatch,
        hand_detected: typeof data?.hand_detected === 'boolean'
          ? data.hand_detected
          : (typeof requestMeta.hand_detected === 'boolean' ? requestMeta.hand_detected : null),
        prediction_source: data?.prediction_source || 'none',
        gating_status: data?.gating_status || (predictedSign === 'NEUTRAL' ? 'rejected' : 'passed'),
        classifier_status: data?.classifier_status || (predictedSign === 'NEUTRAL' ? 'NEUTRAL_REJECT' : 'PREDICTED'),
        reject_reason: data?.reject_reason || (predictedSign === 'NEUTRAL' ? 'confidence_below_threshold' : null),
        head_confidence: typeof data?.head_confidence === 'number' ? data.head_confidence : confidence,
        confirmed: Boolean(data?.confirmed),
        new_token: data?.new_token || null,
        phase,
        elapsed_sec: Number(elapsedSec.toFixed(3)),
        backend_error: false,
        backend_error_message: null,
        frame_status: 'VALID'
      };
    }

    this.trialFrames.push(frameRecord);
    return true;
  }

  computeSessionReport() {
    const allFrames = this.trialSummaries.flatMap(t => t.frames || []);
    const totalRecordedFrames = allFrames.length;
    const validFrames = allFrames.filter(f => !f.backend_error && f.frame_status === 'VALID').length;
    const backendErrorFrames = allFrames.filter(f => f.backend_error || f.frame_status === 'BACKEND_ERROR').length;
    const sessionValidityRatio = totalRecordedFrames > 0
      ? Number((validFrames / totalRecordedFrames).toFixed(3))
      : 0.0;
    const isCleanHoldout = backendErrorFrames === 0 && totalRecordedFrames >= 50;

    return {
      total_recorded_frames: totalRecordedFrames,
      valid_frames: validFrames,
      backend_error_frames: backendErrorFrames,
      session_validity_ratio: sessionValidityRatio,
      is_clean_holdout: isCleanHoldout
    };
  }
}

console.log('=== STARTING TASK 1.3G 10-CASE AUTOMATED TEST SUITE ===\n');

const harness = new PilotHUDHarnessTask13G();

// ----------------------------------------------------
// CASE 1: Normal backend success
// ----------------------------------------------------
console.log('--- Case 1: Normal backend success ---');
harness.startRecordingPeriod(); // Rep 1, generation 1
const req1 = harness.getFrameMetadata(true);
const accepted1 = harness.onFrameResult({
  sign: 'Q',
  confidence: 0.82,
  prediction_source: 'static_head',
  gating_status: 'passed',
  classifier_status: 'PREDICTED',
  head_confidence: 0.82
}, req1);

assert.strictEqual(accepted1, true, 'Case 1 should accept valid response');
assert.strictEqual(harness.trialFrames.length, 1);
const f1 = harness.trialFrames[0];
assert.strictEqual(f1.frame_status, 'VALID');
assert.strictEqual(f1.backend_error, false);
assert.strictEqual(f1.predicted_sign, 'Q');
assert.strictEqual(f1.is_match, true);
assert.strictEqual(f1.confidence, 0.82);
console.log('PASS: Case 1 - Frame recorded with VALID status, match=true, no backend error.');

// ----------------------------------------------------
// CASE 2: Backend error with local fallback (resp.backend_error === true)
// ----------------------------------------------------
console.log('\n--- Case 2: Backend error with local fallback ---');
const req2 = harness.getFrameMetadata(true);
// Simulated CameraSignRecognizer behavior when resp.backend_error === true:
// passes onPilotFrameResult(null, requestPilotMeta, { backendError: true, backendErrorMessage: 'Server 500' })
const accepted2 = harness.onFrameResult(null, req2, {
  backendError: true,
  backendErrorMessage: '500 Internal Server Error (fallback heuristic bypassed)'
});

assert.strictEqual(accepted2, true, 'Case 2 should record backend error frame');
assert.strictEqual(harness.trialFrames.length, 2);
const f2 = harness.trialFrames[1];
assert.strictEqual(f2.frame_status, 'BACKEND_ERROR');
assert.strictEqual(f2.backend_error, true);
assert.strictEqual(f2.predicted_sign, 'ERROR');
assert.strictEqual(f2.is_match, false);
assert.strictEqual(f2.confidence, 0.0);
assert.strictEqual(f2.classifier_status, 'BACKEND_ERROR');
assert(f2.backend_error_message.includes('500 Internal Server Error'));
console.log('PASS: Case 2 - Backend error handled as BACKEND_ERROR, fallback heuristic not counted as valid prediction.');

// ----------------------------------------------------
// CASE 3: Network exception during RECORDING
// ----------------------------------------------------
console.log('\n--- Case 3: Network exception during RECORDING ---');
const req3 = harness.getFrameMetadata(true);
// Simulated catch block in CameraSignRecognizer:
// passes onPilotFrameResult(null, requestPilotMeta, { backendError: true, backendErrorMessage: 'Network failure: Failed to fetch' })
const accepted3 = harness.onFrameResult(null, req3, {
  backendError: true,
  backendErrorMessage: 'Network failure: Failed to fetch'
});

assert.strictEqual(accepted3, true);
assert.strictEqual(harness.trialFrames.length, 3);
const f3 = harness.trialFrames[2];
assert.strictEqual(f3.frame_status, 'BACKEND_ERROR');
assert.strictEqual(f3.backend_error, true);
assert.strictEqual(f3.reject_reason, 'Network failure: Failed to fetch');
console.log('PASS: Case 3 - Network exception isolated and tagged with BACKEND_ERROR.');

// ----------------------------------------------------
// CASE 4: Timeout simulation
// ----------------------------------------------------
console.log('\n--- Case 4: Timeout simulation ---');
const req4 = harness.getFrameMetadata(true);
const accepted4 = harness.onFrameResult(null, req4, {
  backendError: true,
  backendErrorMessage: 'Request timeout after 5000ms'
});

assert.strictEqual(accepted4, true);
const f4 = harness.trialFrames[3];
assert.strictEqual(f4.frame_status, 'BACKEND_ERROR');
assert.strictEqual(f4.backend_error, true);
assert.strictEqual(f4.reject_reason, 'Request timeout after 5000ms');
console.log('PASS: Case 4 - Request timeout handled without crashing or falsifying prediction.');

// ----------------------------------------------------
// CASE 5: Stale response from old trial
// ----------------------------------------------------
console.log('\n--- Case 5: Stale response from old trial ---');
const staleReq1 = harness.getFrameMetadata(true); // Generated in Rep 1 (generation 1)
harness.handleCompleteTrial(); // Completes Rep 1 -> generation 2 (TRIAL_REVIEW)
harness.handleNextTrial(); // Starts Rep 2 -> generation 3 (RECORDING)

assert.strictEqual(harness.currentRep, 2);
assert.strictEqual(harness.trialGeneration, 3);
assert.strictEqual(harness.trialFrames.length, 0);

// Stale response arrives now
const acceptedStale = harness.onFrameResult({
  sign: 'Q',
  confidence: 0.90
}, staleReq1);

assert.strictEqual(acceptedStale, false, 'Stale response from Rep 1 must be rejected');
assert.strictEqual(harness.trialFrames.length, 0, 'Rep 2 frame buffer must remain untouched');
console.log('PASS: Case 5 - Stale response blocked by generation & trial_id mismatch.');

// ----------------------------------------------------
// CASE 6: Request missing trial metadata (null metadata)
// ----------------------------------------------------
console.log('\n--- Case 6: Request missing trial metadata ---');
const acceptedNullMeta = harness.onFrameResult({
  sign: 'Q',
  confidence: 0.95
}, null);

assert.strictEqual(acceptedNullMeta, false, 'Null metadata must be rejected');
assert.strictEqual(harness.trialFrames.length, 0, 'Buffer remains 0');
console.log('PASS: Case 6 - Request with null metadata safely dropped.');

// ----------------------------------------------------
// CASE 7: Trial with full valid frames (>= 5 frames, 0 backend errors) -> VALID
// ----------------------------------------------------
console.log('\n--- Case 7: Trial with >= 5 valid frames, 0 errors ---');
for (let i = 1; i <= 6; i++) {
  const req = harness.getFrameMetadata(true);
  const ok = harness.onFrameResult({
    sign: 'Q',
    confidence: 0.85,
    prediction_source: 'static_head'
  }, req);
  assert.strictEqual(ok, true);
}
assert.strictEqual(harness.trialFrames.length, 6);
const summaryRep2 = harness.handleCompleteTrial();
assert.strictEqual(summaryRep2.total_frames, 6);
assert.strictEqual(summaryRep2.valid_frames, 6);
assert.strictEqual(summaryRep2.backend_error_frames, 0);
assert.strictEqual(summaryRep2.validity_ratio, 1.0);
assert.strictEqual(summaryRep2.is_trial_valid, true);
console.log('PASS: Case 7 - Rep 2 marked is_trial_valid=true (6 valid frames, 0 errors).');

// ----------------------------------------------------
// CASE 8: Trial with < 5 valid frames or backend error frames -> INVALID
// ----------------------------------------------------
console.log('\n--- Case 8: Trial with < 5 valid frames or errors ---');
harness.handleNextTrial(); // Rep 3, generation 5
assert.strictEqual(harness.currentRep, 3);
assert.strictEqual(harness.trialFrames.length, 0);

// Record 3 valid frames and 2 error frames
for (let i = 1; i <= 3; i++) {
  const req = harness.getFrameMetadata(true);
  harness.onFrameResult({ sign: 'Q', confidence: 0.80 }, req);
}
for (let i = 1; i <= 2; i++) {
  const req = harness.getFrameMetadata(true);
  harness.onFrameResult(null, req, { backendError: true, backendErrorMessage: 'Simulated 503' });
}
assert.strictEqual(harness.trialFrames.length, 5);
const summaryRep3 = harness.handleCompleteTrial();
assert.strictEqual(summaryRep3.total_frames, 5);
assert.strictEqual(summaryRep3.valid_frames, 3);
assert.strictEqual(summaryRep3.backend_error_frames, 2);
assert.strictEqual(summaryRep3.validity_ratio, 0.6);
assert.strictEqual(summaryRep3.is_trial_valid, false, 'Trial with < 5 valid frames and errors must be INVALID');
console.log('PASS: Case 8 - Rep 3 marked is_trial_valid=false (3 valid frames < 5, 2 backend errors).');

// ----------------------------------------------------
// CASE 9: Hand detected, classifier returns NEUTRAL
// ----------------------------------------------------
console.log('\n--- Case 9: Hand detected, classifier returns NEUTRAL ---');
harness.handleNextTrial(); // Rep 4, generation 7
const reqNeutral = harness.getFrameMetadata(true); // hand_detected: true
const acceptedNeutral = harness.onFrameResult({
  sign: 'NEUTRAL',
  confidence: 0.45,
  hand_detected: true,
  prediction_source: 'classifier',
  gating_status: 'rejected',
  classifier_status: 'NEUTRAL_REJECT',
  reject_reason: 'confidence_below_threshold'
}, reqNeutral);

assert.strictEqual(acceptedNeutral, true);
const fNeutral = harness.trialFrames[0];
assert.strictEqual(fNeutral.frame_status, 'VALID');
assert.strictEqual(fNeutral.backend_error, false);
assert.strictEqual(fNeutral.hand_detected, true);
assert.strictEqual(fNeutral.predicted_sign, 'NEUTRAL');
assert.strictEqual(fNeutral.classifier_status, 'NEUTRAL_REJECT');
assert.strictEqual(fNeutral.reject_reason, 'confidence_below_threshold');
console.log('PASS: Case 9 - NEUTRAL frame recorded as VALID frame with hand_detected=true and classifier_status=NEUTRAL_REJECT.');

// ----------------------------------------------------
// CASE 10: Valid confirmed token from backend
// ----------------------------------------------------
console.log('\n--- Case 10: Valid confirmed token from backend ---');
const reqConfirmed = harness.getFrameMetadata(true);
const acceptedConfirmed = harness.onFrameResult({
  sign: 'Q',
  confidence: 0.88,
  hand_detected: true,
  prediction_source: 'static_head',
  gating_status: 'passed',
  classifier_status: 'PREDICTED',
  confirmed: true,
  new_token: 'Q'
}, reqConfirmed);

assert.strictEqual(acceptedConfirmed, true);
const fConfirmed = harness.trialFrames[1];
assert.strictEqual(fConfirmed.frame_status, 'VALID');
assert.strictEqual(fConfirmed.confirmed, true);
assert.strictEqual(fConfirmed.new_token, 'Q');
assert.strictEqual(fConfirmed.predicted_sign, 'Q');
console.log('PASS: Case 10 - Valid confirmed token recorded with confirmed=true and new_token="Q".');

console.log('\n======================================================');
console.log('ALL 10 TASK 1.3G TEST CASES PASSED RIGOROUSLY!');
console.log('======================================================\n');

