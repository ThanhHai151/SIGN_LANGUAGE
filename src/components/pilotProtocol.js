export const PILOT_DURATION_SEC = 6;
export const PILOT_REPETITIONS = 10;
export const DEFAULT_PILOT_TARGET = 'O';
export const PILOT_SEQUENCE_INITIAL_ACTIVE = false;
export const VSL_ALPHABET_29 = Object.freeze([
  'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 'H', 'I', 'K', 'L', 'M',
  'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y',
]);

export function canChangePilotTarget({ trialState, sequenceActive, sequenceCountdown, sessionOperationBusy, sessionStatus }) {
  return !['RECORDING', 'FINALIZING'].includes(trialState) && !sequenceActive && sequenceCountdown === null &&
    !sessionOperationBusy && ['ready', 'active'].includes(sessionStatus);
}

export function pilotTargetFields(target) {
  return { expected_sign: target, target_letter: target };
}

export function addPilotTargetResult(resultsByTarget, summary) {
  const target = summary?.targetLetter;
  if (!target) return resultsByTarget;
  return { ...resultsByTarget, [target]: [...(resultsByTarget[target] || []), summary] };
}

export function latestPilotSequenceForTarget(resultsByTarget, target) {
  const history = resultsByTarget?.[target] || [];
  if (!history.length) return [];
  const latestSequenceId = history[history.length - 1].sequenceId;
  return latestSequenceId ? history.filter(item => item.sequenceId === latestSequenceId) : history;
}

export const PHASE_LABELS_VI = Object.freeze({
  PRE_IDLE: 'Chưa đưa tay',
  ENTRY: 'Đưa tay vào',
  STABLE_HOLD: 'Giữ ký hiệu',
  EXIT: 'Rút tay ra',
  POST_IDLE: 'Không có tay',
});

export function pilotTrialOutcome(summary) {
  if (summary?.trialValid) return { status: 'VALID', duration: 'COMPLETED' };
  return { status: 'INVALID', duration: summary?.completed ? 'COMPLETED' : 'STOPPED' };
}

export function pilotHudStatus({ sessionStatus, trialState, sequenceActive, sequenceCountdown, sequenceComplete, lastTrialSummary, lifecycleAllowsStart }) {
  if (['unavailable', 'session_mismatch', 'runtime_invalid', 'config_invalid', 'session_invalid'].includes(sessionStatus)) return 'ERROR';
  if (trialState === 'FINALIZING') return 'SAVING';
  if (sessionStatus === 'checking') return 'PREPARING';
  if (sequenceCountdown !== null || (sequenceActive && trialState !== 'RECORDING')) return 'PREPARING';
  if (trialState === 'RECORDING') return 'RUNNING';
  if (sequenceComplete) return 'COMPLETED';
  if (lastTrialSummary && !lastTrialSummary.trialValid) return 'INVALID';
  return lifecycleAllowsStart ? 'READY' : 'ERROR';
}

export function canStartPilotSequence({ sequenceActive, isRecording, sessionOperationBusy, cameraActive, lifecycleAllowsStart }) {
  return !sequenceActive && !isRecording && !sessionOperationBusy && cameraActive && lifecycleAllowsStart;
}

export function createPilotTrialId(sessionId, target, repetition) {
  return `${sessionId}_${target}_rep${String(repetition).padStart(2, '0')}`;
}

export function createPilotSequenceId(random = Math.random) {
  return `seq${Math.floor(random() * 0x1000000).toString(16).padStart(6, '0')}`;
}

export function createPilotSequenceTrialId(sessionId, target, repetition, sequenceId) {
  return `${sessionId}_${target}_${sequenceId}_rep${String(repetition).padStart(2, '0')}`;
}

export function pilotSequenceCanAdvance({ active, completed, valid, repetition, repetitions = PILOT_REPETITIONS }) {
  return Boolean(active && completed && valid && repetition < repetitions);
}

export function createPilotSessionId(now = new Date(), random = Math.random) {
  const pad = value => String(value).padStart(2, '0');
  const timestamp = `${now.getFullYear()}${pad(now.getMonth() + 1)}${pad(now.getDate())}_${pad(now.getHours())}${pad(now.getMinutes())}${pad(now.getSeconds())}`;
  const suffix = Math.floor(random() * 0x1000000).toString(16).padStart(6, '0');
  return `pilot_${timestamp}_${suffix}`;
}

export function candidateDConfigMatches(config) {
  return Boolean(config?.success && config.classifier_rule_version === 'Candidate-D' &&
    config.classifier_config?.O_D_TI_MAX === 0.23 && config.classifier_config?.O_IDX_STRAIGHT_MAX === 0.65 &&
    config.schema_version === '2.1-kinematic-phase' && config.feature_extractor_version === '2.0-kinematic');
}

export function pilotLoggingConfirmed(stats, sessionId) {
  return Boolean(stats?.success && stats.pilot_logging_enabled === true && stats.session_id === sessionId);
}

export function resolvePilotSession(stats, config, hudSessionId) {
  if (!stats?.success) return { status: 'unavailable', reason: stats?.error || 'Backend stats unavailable' };
  if (!candidateDConfigMatches(config)) return { status: 'config_invalid', reason: 'Candidate D runtime config mismatch' };
  if (stats.pilot_logging_enabled) {
    if (!stats.session_id) return { status: 'session_invalid', reason: 'Backend reports logging enabled without session_id' };
    if (stats.session_id !== hudSessionId) {
      return { status: 'session_mismatch', reason: `HUD session ${hudSessionId} differs from backend session ${stats.session_id}` };
    }
    return { status: 'active', reason: null };
  }
  return { status: 'ready', reason: null };
}

export function pilotLoggingEnableAcknowledged(response, sessionId) {
  return Boolean(response?.success && response.enabled === true && response.session_id === sessionId && response.filepath);
}

export function pilotFlushVerified(before, flush, after, expectedSessionId) {
  const beforeStats = before?.stats;
  const flushStats = flush?.stats;
  const afterStats = after?.stats;
  if (!beforeStats) return false;
  if (!pilotLoggingConfirmed(before, expectedSessionId) || !before.filepath) return false;
  if (!flush?.success || flush.session_id !== expectedSessionId || flush.filepath !== before.filepath || !flushStats) return false;
  if (!after?.success || !pilotLoggingConfirmed(after, expectedSessionId) || after.filepath !== before.filepath || !afterStats) return false;
  const noNewDrops = flushStats.dropped_queue_full === beforeStats.dropped_queue_full &&
    flushStats.dropped_io_error === beforeStats.dropped_io_error &&
    afterStats.dropped_queue_full === flushStats.dropped_queue_full &&
    afterStats.dropped_io_error === flushStats.dropped_io_error;
  const flushed = stats => stats.pending_in_queue === 0 && stats.queued_count === stats.logged_count;
  return noNewDrops && flushed(flushStats) && flushed(afterStats) &&
    afterStats.queued_count === flushStats.queued_count && afterStats.logged_count === flushStats.logged_count;
}

export function pilotLoggingInterrupted(stats, config, baseline, sessionId) {
  return !pilotLoggingConfirmed(stats, sessionId) || !candidateDConfigMatches(config) ||
    (stats?.stats?.dropped_queue_full || 0) > baseline.dropped_queue_full ||
    (stats?.stats?.dropped_io_error || 0) > baseline.dropped_io_error;
}

export function phaseAtElapsedSeconds(elapsedSec) {
  if (elapsedSec < 0.5) return 'PRE_IDLE';
  if (elapsedSec < 1.5) return 'ENTRY';
  if (elapsedSec < 4.5) return 'STABLE_HOLD';
  if (elapsedSec < 5.5) return 'EXIT';
  return 'POST_IDLE';
}

export function summarizeStableFrames(frames, expectedSign) {
  const stable = frames.filter(frame => frame.phase === 'STABLE_HOLD');
  const correctFrames = stable.filter(frame => frame.sign === expectedSign).length;
  const falsePositivePairs = new Set(['A:O', 'O:S', 'C:E', 'G:Q', 'M:N', 'H:R', 'H:V']);
  const falsePositives = stable.filter(frame => {
    if (!frame.sign || frame.sign === 'NEUTRAL' || frame.sign === expectedSign) return false;
    const pair = [expectedSign, frame.sign].sort().join(':');
    return falsePositivePairs.has(pair);
  }).length;
  return {
    phaseFrameCounts: frames.reduce((counts, frame) => {
      counts[frame.phase] = (counts[frame.phase] || 0) + 1;
      return counts;
    }, {}),
    stableFrameCorrect: correctFrames,
    stableFrameTotal: stable.length,
    stableFrameAccuracy: stable.length ? correctFrames / stable.length : null,
    stableFalsePositiveCount: falsePositives,
  };
}

export function summarizeSessionTrials(trials) {
  const valid = trials.filter(trial => trial.trialValid);
  const stableCorrect = valid.reduce((sum, trial) => sum + (trial.stableFrameCorrect || 0), 0);
  const stableTotal = valid.reduce((sum, trial) => sum + (trial.stableFrameTotal || 0), 0);
  const trialCorrect = valid.filter(trial => trial.trialCorrectConfirmed).length;
  const falsePositives = valid.reduce((sum, trial) => sum + (trial.stableFalsePositiveCount || 0), 0);
  return {
    validTrialCount: valid.length,
    totalTrialCount: trials.length,
    stableFrameCorrect: stableCorrect,
    stableFrameTotal: stableTotal,
    stableFrameAccuracy: stableTotal ? stableCorrect / stableTotal : null,
    trialCorrect,
    trialAccuracyDenominator: valid.length,
    stableFalsePositiveCount: falsePositives,
  };
}

export function trialHasConfirmedTarget(responses, expectedSign) {
  return responses.some(response => response.confirmed && response.sign === expectedSign);
}

export function trialIsValid({ completed, durationSec, loggingInterrupted, stableHoldFrameCount, queuedFrameCount, loggedFrameCount, submittedFrameCount }) {
  return completed && durationSec === PILOT_DURATION_SEC && !loggingInterrupted &&
    stableHoldFrameCount > 0 && queuedFrameCount > 0 &&
    queuedFrameCount === loggedFrameCount && queuedFrameCount === submittedFrameCount;
}
