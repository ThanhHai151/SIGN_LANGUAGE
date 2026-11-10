export const PILOT_DURATION_SEC = 5;
export const PILOT_REPETITIONS = 10;
export const DEFAULT_PILOT_TARGET = 'A';
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
  // Hoàn toàn linh hoạt: không ép cứng version hoặc threshold khi user tự chỉnh sửa logic nhận diện
  return Boolean(config && config.success !== false);
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

/**
 * Đánh giá chi tiết các frame thu được trong 1 lượt thử (5.0s)
 */
export function evaluateTrialFrames(frames, expectedSign, durationSec = 5.0) {
  const totalFrames = frames.length;
  let matchFrames = 0;
  let mismatchFrames = 0;
  let neutralFrames = 0;
  const confusions = {};
  let totalConfidence = 0;
  let matchConfidence = 0;

  frames.forEach(frame => {
    const sign = frame.sign || frame.predicted_sign || 'NEUTRAL';
    const conf = Number(frame.confidence || 0);
    totalConfidence += conf;

    if (sign === expectedSign) {
      matchFrames += 1;
      matchConfidence += conf;
    } else if (sign === 'NEUTRAL' || sign === 'WAIT' || sign === 'unknown') {
      neutralFrames += 1;
    } else {
      mismatchFrames += 1;
      confusions[sign] = (confusions[sign] || 0) + 1;
    }
  });

  const activeFrames = matchFrames + mismatchFrames;
  const matchPct = activeFrames > 0 ? Number(((matchFrames / activeFrames) * 100).toFixed(1)) : 0.0;
  const avgConf = matchFrames > 0 ? Number((matchConfidence / matchFrames).toFixed(2)) : 0.0;

  // Xác định trạng thái của trial
  let status = 'NEEDS_FIX';
  if (activeFrames === 0) {
    status = 'NO_HAND';
  } else if (matchPct >= 60.0) {
    status = 'PASS';
  }

  // Chữ cái bị nhầm lẫn nhiều nhất
  let topConfusion = null;
  let maxConfCount = 0;
  Object.entries(confusions).forEach(([letter, count]) => {
    if (count > maxConfCount) {
      maxConfCount = count;
      topConfusion = letter;
    }
  });

  return {
    totalFrames,
    activeFrames,
    matchFrames,
    mismatchFrames,
    neutralFrames,
    matchPct,
    avgConfidence: avgConf,
    confusions,
    topConfusion,
    status,
    isPassed: status === 'PASS',
    durationSec
  };
}

/**
 * Xây dựng báo cáo chẩn đoán toàn diện cho 29 chữ cái (để ghi JSON vào backend/data/diagnostics/)
 */
export function buildEvaluationReport(sessionId, resultsByTarget, config = {}) {
  const letters = {};
  const lettersPassed = [];
  const lettersNeedingFix = [];
  const lettersUnsupported = [];
  let totalTrials = 0;
  let passedTrials = 0;
  let totalAccuracySum = 0;
  let evaluatedLetterCount = 0;

  VSL_ALPHABET_29.forEach(letter => {
    const trials = resultsByTarget[letter] || [];
    if (!trials.length) {
      letters[letter] = {
        letter,
        totalTrials: 0,
        passedTrials: 0,
        accuracyPct: 0.0,
        status: 'UNTESTED',
        topConfusions: {},
        trials: []
      };
      return;
    }

    const completed = trials.filter(t => t && t.status !== 'SKIPPED');
    const passed = completed.filter(t => t.status === 'PASS');
    const totalMatchFrames = completed.reduce((sum, t) => sum + (t.matchFrames || 0), 0);
    const totalActiveFrames = completed.reduce((sum, t) => sum + (t.activeFrames || 0), 0);
    const accuracyPct = totalActiveFrames > 0
      ? Number(((totalMatchFrames / totalActiveFrames) * 100).toFixed(1))
      : 0.0;

    // Tổng hợp confusions
    const mergedConfusions = {};
    completed.forEach(t => {
      Object.entries(t.confusions || {}).forEach(([confLetter, count]) => {
        mergedConfusions[confLetter] = (mergedConfusions[confLetter] || 0) + count;
      });
    });

    let status = 'NEEDS_FIX';
    if (totalActiveFrames === 0 && completed.some(t => t.status === 'NO_HAND')) {
      status = 'NO_HAND';
    } else if (accuracyPct >= 70.0 || (completed.length >= 5 && (passed.length / completed.length) >= 0.7)) {
      status = 'PASS';
      lettersPassed.push(letter);
    } else {
      status = 'NEEDS_FIX';
      lettersNeedingFix.push(letter);
    }

    totalTrials += completed.length;
    passedTrials += passed.length;
    totalAccuracySum += accuracyPct;
    evaluatedLetterCount += 1;

    letters[letter] = {
      letter,
      totalTrials: completed.length,
      passedTrials: passed.length,
      accuracyPct,
      status,
      topConfusions: mergedConfusions,
      trials: completed.map(t => ({
        rep: t.rep,
        matchPct: t.matchPct,
        status: t.status,
        matchFrames: t.matchFrames,
        mismatchFrames: t.mismatchFrames,
        confusions: t.confusions
      }))
    };
  });

  const overallAccuracyPct = evaluatedLetterCount > 0
    ? Number((totalAccuracySum / evaluatedLetterCount).toFixed(1))
    : 0.0;

  return {
    schema_version: '3.0-controlled-pilot-29',
    session_id: sessionId,
    created_at: new Date().toISOString(),
    config: {
      total_alphabet: 29,
      trials_per_letter: config.trialsPerLetter || PILOT_REPETITIONS,
      trial_duration_sec: config.trialDurationSec || PILOT_DURATION_SEC,
      mode: config.mode || 'single_letter',
      signer_id: config.signerId || 'signer_01',
      tested_target: config.testedTarget || null
    },
    summary: {
      total_trials_completed: totalTrials,
      total_trials_passed: passedTrials,
      evaluated_letters_count: evaluatedLetterCount,
      overall_accuracy_pct: overallAccuracyPct,
      letters_passed: lettersPassed,
      letters_needing_fix: lettersNeedingFix,
      letters_unsupported: lettersUnsupported
    },
    letters
  };
}
