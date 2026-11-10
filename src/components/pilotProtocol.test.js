import test from 'node:test';
import assert from 'node:assert/strict';
import {
  phaseAtElapsedSeconds,
  PHASE_LABELS_VI,
  pilotTrialOutcome,
  pilotHudStatus,
  canStartPilotSequence,
  canChangePilotTarget,
  pilotTargetFields,
  addPilotTargetResult,
  latestPilotSequenceForTarget,
  VSL_ALPHABET_29,
  DEFAULT_PILOT_TARGET,
  PILOT_SEQUENCE_INITIAL_ACTIVE,
  createPilotTrialId,
  createPilotSequenceId,
  createPilotSequenceTrialId,
  pilotSequenceCanAdvance,
  createPilotSessionId,
  summarizeStableFrames,
  summarizeSessionTrials,
  trialHasConfirmedTarget,
  trialIsValid,
  candidateDConfigMatches,
  pilotLoggingConfirmed,
  resolvePilotSession,
  pilotLoggingEnableAcknowledged,
  pilotFlushVerified,
  pilotLoggingInterrupted,
  PILOT_DURATION_SEC,
  PILOT_REPETITIONS,
} from './pilotProtocol.js';

test('phase boundaries follow the fixed six second protocol', () => {
  assert.equal(phaseAtElapsedSeconds(0), 'PRE_IDLE');
  assert.equal(phaseAtElapsedSeconds(0.4999), 'PRE_IDLE');
  assert.equal(phaseAtElapsedSeconds(0.5), 'ENTRY');
  assert.equal(phaseAtElapsedSeconds(1.5), 'STABLE_HOLD');
  assert.equal(phaseAtElapsedSeconds(4.5), 'EXIT');
  assert.equal(phaseAtElapsedSeconds(5.5), 'POST_IDLE');
  assert.equal(phaseAtElapsedSeconds(6.0), 'POST_IDLE');
  assert.equal(PILOT_DURATION_SEC, 6);
  assert.equal(PILOT_REPETITIONS, 10);
});

test('ten O sequence trial IDs are unique within one session and invalid trials cannot advance', () => {
  const ids = Array.from({ length: PILOT_REPETITIONS }, (_, index) => createPilotTrialId('pilot_session', 'O', index + 1));
  assert.equal(new Set(ids).size, 10);
  const sequenceId = createPilotSequenceId(() => 0.5);
  const runIds = Array.from({ length: PILOT_REPETITIONS }, (_, index) => createPilotSequenceTrialId('pilot_session', 'O', index + 1, sequenceId));
  assert.equal(new Set([...runIds, ...runIds.map(id => id.replace(sequenceId, 'seq000001'))]).size, 20);
  assert.ok(runIds.every(id => id.length <= 64));
  assert.ok(ids.every(id => id.startsWith('pilot_session_O_')));
  assert.equal(pilotSequenceCanAdvance({ active: true, completed: true, valid: true, repetition: 1 }), true);
  assert.equal(pilotSequenceCanAdvance({ active: true, completed: true, valid: false, repetition: 1 }), false);
  assert.equal(pilotSequenceCanAdvance({ active: false, completed: true, valid: true, repetition: 1 }), false);
  assert.equal(pilotSequenceCanAdvance({ active: true, completed: true, valid: true, repetition: 10 }), false);
  let completedRep = 1;
  let advanceCount = 0;
  while (pilotSequenceCanAdvance({ active: true, completed: true, valid: true, repetition: completedRep })) {
    completedRep += 1;
    advanceCount += 1;
  }
  assert.equal(advanceCount, 9);
  assert.equal(completedRep, PILOT_REPETITIONS);
  // Countdown time is not added to trial elapsed time, so the existing phase boundaries stay fixed.
  assert.equal(phaseAtElapsedSeconds(0), 'PRE_IDLE');
  assert.equal(phaseAtElapsedSeconds(1.5), 'STABLE_HOLD');
  assert.equal(phaseAtElapsedSeconds(5.5), 'POST_IDLE');
});

test('stable metrics exclude all transition and idle phases', () => {
  const result = summarizeStableFrames([
    { phase: 'ENTRY', sign: 'O' },
    { phase: 'STABLE_HOLD', sign: 'O' },
    { phase: 'STABLE_HOLD', sign: 'A' },
    { phase: 'EXIT', sign: 'O' },
  ], 'O');
  assert.equal(result.stableFrameCorrect, 1);
  assert.equal(result.stableFrameTotal, 2);
  assert.equal(result.stableFalsePositiveCount, 1);
  assert.deepEqual(result.phaseFrameCounts, { ENTRY: 1, STABLE_HOLD: 2, EXIT: 1 });
});

test('Candidate D runtime provenance is strict and logging must match session', () => {
  const config = {
    success: true,
    classifier_rule_version: 'Candidate-D',
    schema_version: '2.1-kinematic-phase',
    feature_extractor_version: '2.0-kinematic',
    classifier_config: { O_D_TI_MAX: 0.23, O_IDX_STRAIGHT_MAX: 0.65 },
  };
  assert.equal(candidateDConfigMatches(config), true);
  assert.equal(candidateDConfigMatches({ ...config, classifier_config: { ...config.classifier_config, O_D_TI_MAX: 0.3 } }), false);
  assert.equal(pilotLoggingConfirmed({ success: true, session_id: 'session1', pilot_logging_enabled: true }, 'session1'), true);
  assert.equal(pilotLoggingConfirmed({ success: true, session_id: 'other', pilot_logging_enabled: true }, 'session1'), false);
  const baseline = { dropped_queue_full: 0, dropped_io_error: 0 };
  assert.equal(pilotLoggingInterrupted({ success: true, session_id: 'session1', pilot_logging_enabled: true, stats: baseline }, config, baseline, 'session1'), false);
  assert.equal(pilotLoggingInterrupted({ success: true, session_id: 'session1', pilot_logging_enabled: true, stats: { ...baseline, dropped_io_error: 1 } }, config, baseline, 'session1'), true);
});

test('trial validity requires a completed six second trial without logging interruption', () => {
  const complete = { completed: true, durationSec: 6, loggingInterrupted: false, stableHoldFrameCount: 12, queuedFrameCount: 30, loggedFrameCount: 30, submittedFrameCount: 30 };
  assert.equal(trialIsValid(complete), true);
  assert.equal(trialIsValid({ ...complete, completed: false }), false);
  assert.equal(trialIsValid({ ...complete, durationSec: 5 }), false);
  assert.equal(trialIsValid({ ...complete, loggingInterrupted: true }), false);
  assert.equal(trialIsValid({ ...complete, stableHoldFrameCount: 0 }), false);
  assert.deepEqual(pilotTrialOutcome({ completed: true, trialValid: false }), { status: 'INVALID', duration: 'COMPLETED' });
  assert.deepEqual(pilotTrialOutcome({ completed: false, trialValid: false }), { status: 'INVALID', duration: 'STOPPED' });
  assert.deepEqual(pilotTrialOutcome({ completed: true, trialValid: true }), { status: 'VALID', duration: 'COMPLETED' });
});

test('phase guidance has explicit Vietnamese labels for every protocol phase', () => {
  assert.deepEqual(Object.values(PHASE_LABELS_VI), ['Chưa đưa tay', 'Đưa tay vào', 'Giữ ký hiệu', 'Rút tay ra', 'Không có tay']);
});

test('HUD status distinguishes lifecycle, saving, trial run, preparation, completion, invalid, and error', () => {
  const base = { sessionStatus: 'active', trialState: 'READY', sequenceActive: false, sequenceCountdown: null, sequenceComplete: false, lastTrialSummary: null, lifecycleAllowsStart: true };
  assert.equal(pilotHudStatus(base), 'READY');
  assert.equal(pilotHudStatus({ ...base, sessionStatus: 'checking', lifecycleAllowsStart: false }), 'PREPARING');
  assert.equal(pilotHudStatus({ ...base, trialState: 'RECORDING' }), 'RUNNING');
  assert.equal(pilotHudStatus({ ...base, trialState: 'FINALIZING' }), 'SAVING');
  assert.equal(pilotHudStatus({ ...base, sequenceActive: true }), 'PREPARING');
  assert.equal(pilotHudStatus({ ...base, sequenceCountdown: 2 }), 'PREPARING');
  assert.equal(pilotHudStatus({ ...base, sequenceComplete: true }), 'COMPLETED');
  assert.equal(pilotHudStatus({ ...base, lastTrialSummary: { completed: true, trialValid: false } }), 'INVALID');
  assert.equal(pilotHudStatus({ ...base, sessionStatus: 'unavailable' }), 'ERROR');
});

test('sequence Start is enabled only with camera and verified available session', () => {
  const ready = { sequenceActive: false, isRecording: false, sessionOperationBusy: false, cameraActive: true, lifecycleAllowsStart: true };
  assert.equal(canStartPilotSequence(ready), true);
  assert.equal(canStartPilotSequence({ ...ready, sequenceActive: true }), false);
  assert.equal(canStartPilotSequence({ ...ready, isRecording: true }), false);
  assert.equal(canStartPilotSequence({ ...ready, cameraActive: false }), false);
  assert.equal(canStartPilotSequence({ ...ready, lifecycleAllowsStart: false }), false);
});

test('VSL target selector contains exactly the 29 unique letters and defaults to O', () => {
  assert.equal(VSL_ALPHABET_29.length, 29);
  assert.equal(new Set(VSL_ALPHABET_29).size, 29);
  assert.deepEqual(VSL_ALPHABET_29, ['A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y']);
  assert.equal(DEFAULT_PILOT_TARGET, 'O');
  assert.equal(PILOT_SEQUENCE_INITIAL_ACTIVE, false, 'a HUD reload must not resume or start a sequence');
});

test('changing target is allowed only while idle with a valid session', () => {
  const idle = { trialState: 'READY', sequenceActive: false, sequenceCountdown: null, sessionOperationBusy: false, sessionStatus: 'active' };
  assert.equal(canChangePilotTarget(idle), true);
  assert.equal(canChangePilotTarget({ ...idle, trialState: 'RECORDING' }), false);
  assert.equal(canChangePilotTarget({ ...idle, trialState: 'FINALIZING' }), false);
  assert.equal(canChangePilotTarget({ ...idle, sequenceActive: true }), false);
  assert.equal(canChangePilotTarget({ ...idle, sequenceCountdown: 2 }), false);
  assert.equal(canChangePilotTarget({ ...idle, sessionOperationBusy: true }), false);
  assert.equal(canChangePilotTarget({ ...idle, sessionStatus: 'checking' }), false);
});

test('target metadata, trial IDs, and result history remain separated by selected letter', () => {
  assert.deepEqual(pilotTargetFields('Ă'), { expected_sign: 'Ă', target_letter: 'Ă' });
  assert.match(createPilotSequenceTrialId('session1', 'Ă', 1, 'seq000001'), /_Ă_seq000001_rep01$/);
  const summaries = {};
  const afterO = addPilotTargetResult(summaries, { targetLetter: 'O', sequenceId: 'seq_o1', rep: 1, trialValid: true });
  const afterA = addPilotTargetResult(afterO, { targetLetter: 'A', sequenceId: 'seq_a1', rep: 1, trialValid: true });
  const afterONext = addPilotTargetResult(afterA, { targetLetter: 'O', sequenceId: 'seq_o1', rep: 2, trialValid: false });
  assert.deepEqual(latestPilotSequenceForTarget(afterONext, 'O').map(item => item.rep), [1, 2]);
  assert.deepEqual(latestPilotSequenceForTarget(afterONext, 'A').map(item => item.rep), [1]);
  const afterOSecondRun = addPilotTargetResult(afterONext, { targetLetter: 'O', sequenceId: 'seq_o2', rep: 1, trialValid: true });
  assert.deepEqual(latestPilotSequenceForTarget(afterOSecondRun, 'O').map(item => item.sequenceId), ['seq_o2']);
  assert.deepEqual(latestPilotSequenceForTarget(afterOSecondRun, 'A').map(item => item.sequenceId), ['seq_a1']);
  assert.equal(trialHasConfirmedTarget([{ sign: 'A', confirmed: true }], 'A'), true);
  const aMetrics = summarizeStableFrames([{ phase: 'STABLE_HOLD', sign: 'A' }, { phase: 'STABLE_HOLD', sign: 'O' }], 'A');
  assert.equal(aMetrics.stableFrameCorrect, 1);
  assert.equal(aMetrics.stableFrameTotal, 2);
  assert.equal(aMetrics.stableFalsePositiveCount, 1);
});

test('session metrics report numerators and valid trial denominator only', () => {
  const result = summarizeSessionTrials([
    { trialValid: true, stableFrameCorrect: 7, stableFrameTotal: 10, stableFalsePositiveCount: 1, trialCorrectConfirmed: true },
    { trialValid: false, stableFrameCorrect: 100, stableFrameTotal: 100, stableFalsePositiveCount: 0, trialCorrectConfirmed: true },
  ]);
  assert.equal(result.validTrialCount, 1);
  assert.equal(result.stableFrameCorrect, 7);
  assert.equal(result.stableFrameTotal, 10);
  assert.equal(result.trialCorrect, 1);
  assert.equal(result.trialAccuracyDenominator, 1);
  assert.equal(result.stableFalsePositiveCount, 1);
});

test('trial accuracy confirmation is isolated from the preceding trial', () => {
  const previousTrial = [{ sign: 'O', confirmed: true }];
  assert.equal(trialHasConfirmedTarget(previousTrial, 'O'), true);
  const nextTrialResponses = [];
  assert.equal(trialHasConfirmedTarget(nextTrialResponses, 'O'), false);
  nextTrialResponses.push({ sign: 'O', confirmed: false });
  assert.equal(trialHasConfirmedTarget(nextTrialResponses, 'O'), false);
});

test('fresh session ID is unique-shaped and accepted by backend session rules', () => {
  const id = createPilotSessionId(new Date(2026, 10, 9, 6, 45, 55), () => 0.5);
  assert.equal(id, 'pilot_20261109_064555_800000');
  assert.match(id, /^[a-zA-Z0-9_-]{3,64}$/);
});

test('reload adopts an active backend session only when Candidate D runtime is valid', () => {
  const config = { success: true, classifier_rule_version: 'Candidate-D', schema_version: '2.1-kinematic-phase', feature_extractor_version: '2.0-kinematic', classifier_config: { O_D_TI_MAX: 0.23, O_IDX_STRAIGHT_MAX: 0.65 } };
  const active = { success: true, pilot_logging_enabled: true, session_id: 'pilot_backend', has_static_head: true };
  assert.equal(resolvePilotSession(active, config, 'pilot_backend').status, 'active');
  assert.equal(resolvePilotSession(active, config, 'pilot_new_mount_id').status, 'session_mismatch');
  assert.equal(resolvePilotSession({ success: true, pilot_logging_enabled: false }, config, 'pilot_new').status, 'ready');
  assert.equal(resolvePilotSession({ success: false, error: 'offline' }, config, 'pilot_new').status, 'unavailable');
  assert.equal(resolvePilotSession(active, { ...config, classifier_rule_version: 'unknown' }, 'pilot_backend').status, 'config_invalid');
});

test('logging enable acknowledgement must match the requested session and file', () => {
  assert.equal(pilotLoggingEnableAcknowledged({ success: true, enabled: true, session_id: 'pilot_a', filepath: 'pilot_a.jsonl' }, 'pilot_a'), true);
  assert.equal(pilotLoggingEnableAcknowledged({ success: true, enabled: true, session_id: 'pilot_b', filepath: 'pilot_b.jsonl' }, 'pilot_a'), false);
  assert.equal(pilotLoggingEnableAcknowledged({ success: false, error: '409 conflict' }, 'pilot_a'), false);
});

test('safe session rotation requires a verified flush with no pending, mismatch, or new drops', () => {
  const before = { success: true, pilot_logging_enabled: true, session_id: 'pilot_a', filepath: 'pilot_a.jsonl', stats: { queued_count: 10, logged_count: 9, pending_in_queue: 1, dropped_queue_full: 0, dropped_io_error: 0 } };
  const flush = { success: true, session_id: 'pilot_a', filepath: 'pilot_a.jsonl', stats: { queued_count: 10, logged_count: 10, pending_in_queue: 0, dropped_queue_full: 0, dropped_io_error: 0 } };
  const after = { ...before, stats: flush.stats };
  assert.equal(pilotFlushVerified(before, flush, after, 'pilot_a'), true);
  assert.equal(pilotFlushVerified(before, { ...flush, session_id: 'pilot_b' }, after, 'pilot_a'), false);
  assert.equal(pilotFlushVerified(before, { ...flush, stats: { ...flush.stats, dropped_io_error: 1 } }, after, 'pilot_a'), false);
  assert.equal(pilotFlushVerified(before, { ...flush, stats: { ...flush.stats, pending_in_queue: 1 } }, after, 'pilot_a'), false);
  assert.equal(pilotFlushVerified(before, flush, { ...after, pilot_logging_enabled: false }, 'pilot_a'), false);
});
