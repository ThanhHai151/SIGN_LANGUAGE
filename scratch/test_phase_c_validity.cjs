// scratch/test_phase_c_validity.cjs
// Automated simulation of Phase C test cases for Task 1.3F

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

class PilotHUDHarness {
  constructor(sessionId = 'pilot_20261111_test') {
    this.sessionId = sessionId;
    this.selectedLetter = 'Q';
    this.currentRep = 1;
    this.trialState = 'IDLE';
    this.trialGeneration = 0;
    this.trialFrames = [];
    this.recordingStartTime = 0;
    this.frameCounter = 0;
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
    this.transitionTrialState('TRIAL_REVIEW');
  }

  handleNextTrial() {
    this.currentRep += 1;
    this.startRecordingPeriod();
  }

  getFrameMetadata() {
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
      hand_detected: true,
      has_hands: true
    };
  }

  onFrameResult(data, requestMeta) {
    // Gate 1: Must be RECORDING
    if (this.trialState !== 'RECORDING') return false;

    // Gate 2: Must carry non-null metadata
    if (!requestMeta) return false;

    // Gate 3: Generation match
    if (requestMeta.trial_generation !== this.trialGeneration) return false;

    // Gate 4: Session match
    if (requestMeta.session_id !== this.sessionId) return false;

    // Gate 5: Rep & letter match
    if (requestMeta.rep_idx !== this.currentRep) return false;
    if (requestMeta.expected_sign !== this.selectedLetter) return false;

    // Gate 6: Trial ID match
    const expectedTrialId = createPilotTrialId(this.sessionId, this.selectedLetter, this.currentRep);
    if (requestMeta.trial_id !== expectedTrialId) return false;

    const elapsedSec = (Date.now() - this.recordingStartTime) / 1000;
    const phase = phaseAtElapsedSeconds(elapsedSec);

    const frameRecord = {
      frame_idx: this.trialFrames.length + 1,
      expected_sign: this.selectedLetter,
      predicted_sign: data?.sign || 'NEUTRAL',
      confidence: data?.confidence || 0.0,
      hand_detected: requestMeta.hand_detected,
      prediction_source: data?.prediction_source || 'none',
      confirmed: Boolean(data?.confirmed),
      new_token: data?.new_token || null,
      phase,
      elapsed_sec: Number(elapsedSec.toFixed(3))
    };

    this.trialFrames.push(frameRecord);
    return true;
  }
}

console.log('=== STARTING PHASE C VALIDITY TESTS ===\n');

// CASE 1: Normal Trial
console.log('--- Case 1: Normal Trial ---');
const hud = new PilotHUDHarness();
hud.startRecordingPeriod(); // Rep 1, generation 1
const req1 = Object.freeze(hud.getFrameMetadata());
assert(req1 !== null, 'Metadata should not be null during recording');
assert.strictEqual(req1.rep_idx, 1);
assert.strictEqual(req1.trial_generation, 1);

const accepted1 = hud.onFrameResult({ sign: 'Q', confidence: 0.81, prediction_source: 'static_head' }, req1);
assert.strictEqual(accepted1, true, 'Normal request should be accepted');
assert.strictEqual(hud.trialFrames.length, 1);
assert.strictEqual(hud.trialFrames[0].predicted_sign, 'Q');
console.log('PASS: Case 1 - Frame recorded correctly into Trial A');

// CASE 2: Late Response from Trial A returning during Trial B
console.log('\n--- Case 2: Late Response ---');
const lateReqA = Object.freeze(hud.getFrameMetadata()); // Sent during Rep 1 (gen 1)
hud.handleCompleteTrial(); // Gen increments to 2 ('TRIAL_REVIEW')
hud.handleNextTrial(); // Rep 2, Gen increments to 3 ('RECORDING')
assert.strictEqual(hud.currentRep, 2);
assert.strictEqual(hud.trialGeneration, 3);

// Now late response from Rep 1 returns while Trial B is recording
const acceptedLate = hud.onFrameResult({ sign: 'Q', confidence: 0.82, prediction_source: 'static_head' }, lateReqA);
assert.strictEqual(acceptedLate, false, 'Late response from Trial A must be rejected in Trial B');
assert.strictEqual(hud.trialFrames.length, 0, 'Trial B frame buffer must remain empty');
console.log('PASS: Case 2 - Late response A rejected from Trial B (generation & trial_id mismatch)');

// CASE 3: Missing Metadata from Review Transition
console.log('\n--- Case 3: Missing Metadata ---');
// Suppose a request was dispatched during TRIAL_REVIEW where metadata is null
const unassignedMeta = null;
// Response returns when Trial B is RECORDING
const acceptedMissing = hud.onFrameResult({ sign: 'Q', confidence: 0.80, confirmed: true, new_token: 'Q' }, unassignedMeta);
assert.strictEqual(acceptedMissing, false, 'Missing metadata request must be rejected');
assert.strictEqual(hud.trialFrames.length, 0, 'Trial B frame buffer must still be empty');
console.log('PASS: Case 3 - Response with null metadata rejected from Trial B');

// CASE 4: First Frame Isolation of Trial B
console.log('\n--- Case 4: First Frame Isolation ---');
const reqB1 = Object.freeze(hud.getFrameMetadata()); // First legitimate frame of Trial B
assert(reqB1 !== null);
assert.strictEqual(reqB1.rep_idx, 2);
assert.strictEqual(reqB1.trial_generation, 3);
assert.strictEqual(reqB1.trial_id, 'pilot_20261111_test_Q_rep02');

// Backend returns first frame (in isolated state, confirmed is FALSE)
const acceptedB1 = hud.onFrameResult({ sign: 'Q', confidence: 0.81, confirmed: false, new_token: null, prediction_source: 'static_head' }, reqB1);
assert.strictEqual(acceptedB1, true);
assert.strictEqual(hud.trialFrames.length, 1);
assert.strictEqual(hud.trialFrames[0].frame_idx, 1);
assert.strictEqual(hud.trialFrames[0].confirmed, false);
console.log('PASS: Case 4 - First frame of Trial B isolated cleanly with confirmed=false');

// CASE 5: Valid Confirmation inside Trial B after accumulating votes
console.log('\n--- Case 5: Valid Confirmation ---');
for (let i = 2; i <= 4; i++) {
  const req = Object.freeze(hud.getFrameMetadata());
  hud.onFrameResult({ sign: 'Q', confidence: 0.81, confirmed: false, new_token: null, prediction_source: 'static_head' }, req);
}
// 5th frame reaches smoothing threshold and cooldown expired
const reqB5 = Object.freeze(hud.getFrameMetadata());
const acceptedB5 = hud.onFrameResult({ sign: 'Q', confidence: 0.83, confirmed: true, new_token: 'Q', prediction_source: 'static_head' }, reqB5);
assert.strictEqual(acceptedB5, true);
assert.strictEqual(hud.trialFrames.length, 5);
assert.strictEqual(hud.trialFrames[4].frame_idx, 5);
assert.strictEqual(hud.trialFrames[4].confirmed, true);
assert.strictEqual(hud.trialFrames[4].new_token, 'Q');
console.log('PASS: Case 5 - Legitimate confirmed token accepted on Frame 5');

// CASE 6: Tracking and Phase Validation
console.log('\n--- Case 6: Tracking and Phase ---');
assert.strictEqual(typeof hud.trialFrames[0].hand_detected, 'boolean');
assert.strictEqual(hud.trialFrames[0].hand_detected, true);
assert(['PRE_IDLE', 'ENTRY', 'STABLE_HOLD', 'EXIT'].includes(hud.trialFrames[0].phase));
assert(hud.trialFrames[0].elapsed_sec >= 0.0);
console.log('PASS: Case 6 - hand_detected is boolean, phase and elapsed_sec valid');

console.log('\n=== ALL 6 PHASE C TEST CASES PASSED SUCCESSFULLY ===');

