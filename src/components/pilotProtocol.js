import { SIGN_CONVENTIONS } from './signConventions.js';

export const PILOT_DURATION_SEC = 5;
export const PILOT_REPETITIONS = 10;
export const DEFAULT_PILOT_TARGET = 'A';
export const PILOT_SEQUENCE_INITIAL_ACTIVE = false;
export const VSL_ALPHABET_29 = Object.freeze([
  'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 'H', 'I', 'K', 'L', 'M',
  'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y',
]);

export function calculate3DDistance(p1, p2) {
  if (!p1 || !p2 || p1.length < 3 || p2.length < 3) return 0;
  const dx = (p1[0] || 0) - (p2[0] || 0);
  const dy = (p1[1] || 0) - (p2[1] || 0);
  const dz = (p1[2] || 0) - (p2[2] || 0);
  return Math.hypot(dx, dy, dz);
}

export function calculate3DAngle(a, b, c) {
  if (!a || !b || !c || a.length < 3 || b.length < 3 || c.length < 3) return 0;
  const ba = [(a[0] || 0) - (b[0] || 0), (a[1] || 0) - (b[1] || 0), (a[2] || 0) - (b[2] || 0)];
  const bc = [(c[0] || 0) - (b[0] || 0), (c[1] || 0) - (b[1] || 0), (c[2] || 0) - (b[2] || 0)];
  const dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2];
  const normBa = Math.hypot(ba[0], ba[1], ba[2]);
  const normBc = Math.hypot(bc[0], bc[1], bc[2]);
  if (normBa < 1e-6 || normBc < 1e-6) return 0;
  const cosAngle = Math.max(-1, Math.min(1, dot / (normBa * normBc)));
  return Math.round((Math.acos(cosAngle) * 180) / Math.PI);
}

export function computePinkyMetrics(landmarks, classifierFeatures = null) {
  const defaultRes = {
    hasData: false,
    lm17: [0, 0, 0],
    lm18: [0, 0, 0],
    lm19: [0, 0, 0],
    lm20: [0, 0, 0],
    pipAngle: 0,
    dipAngle: 0,
    curl: classifierFeatures?.pinky_curl ?? 0.0,
    d_pw: classifierFeatures?.d_PW ?? 0.0,
    status: 'NO_DATA'
  };

  if (!landmarks || landmarks.length < 21) {
    if (classifierFeatures?.pinky_curl !== undefined) {
      const c = Number(classifierFeatures.pinky_curl || 0);
      return {
        ...defaultRes,
        curl: c,
        d_pw: Number(classifierFeatures.d_PW || 0),
        status: (c <= 0.20) ? 'STRAIGHT' : 'CURLED'
      };
    }
    return defaultRes;
  }

  const p0 = landmarks[0];
  const p9 = landmarks[9];
  const p17 = landmarks[17];
  const p18 = landmarks[18];
  const p19 = landmarks[19];
  const p20 = landmarks[20];

  const pipAngle = calculate3DAngle(p17, p18, p19);
  const dipAngle = calculate3DAngle(p18, p19, p20);

  const palmScale = calculate3DDistance(p0, p9) || 1.0;
  const d_pw = calculate3DDistance(p20, p0) / palmScale;

  const d20_17 = calculate3DDistance(p20, p17);
  const d20_18 = calculate3DDistance(p20, p18);
  const d18_17 = calculate3DDistance(p18, p17);
  const strLen = d20_18 + d18_17;
  const calcCurl = strLen > 1e-6 ? Math.max(0, Math.min(1, 1.0 - (d20_17 / strLen))) : 0.0;
  const curl = classifierFeatures?.pinky_curl !== undefined ? Number(classifierFeatures.pinky_curl) : calcCurl;

  const isStraight = (curl <= 0.20 && pipAngle >= 140) || (d_pw >= 1.30 && pipAngle >= 135);
  const status = isStraight ? 'STRAIGHT' : 'CURLED';

  return {
    hasData: true,
    lm17: [Number((p17[0] || 0).toFixed(4)), Number((p17[1] || 0).toFixed(4)), Number((p17[2] || 0).toFixed(4))],
    lm18: [Number((p18[0] || 0).toFixed(4)), Number((p18[1] || 0).toFixed(4)), Number((p18[2] || 0).toFixed(4))],
    lm19: [Number((p19[0] || 0).toFixed(4)), Number((p19[1] || 0).toFixed(4)), Number((p19[2] || 0).toFixed(4))],
    lm20: [Number((p20[0] || 0).toFixed(4)), Number((p20[1] || 0).toFixed(4)), Number((p20[2] || 0).toFixed(4))],
    pipAngle,
    dipAngle,
    curl,
    d_pw,
    status
  };
}

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

export const DYNAMIC_DIACRITICS = Object.freeze({
  'Ă': { base: 'A' },
  'Â': { base: 'A' },
  'Ê': { base: 'E' },
  'Đ': { base: 'D' },
  'Ơ': { base: 'O' },
  'Ô': { base: 'O' },
  'Ư': { base: 'U' }
});

/**
 * Kiểm tra xem một frame có phát hiện bàn tay hay không
 * Đảm bảo tương thích ngược: boolean hand_detected -> has_hands -> landmarks -> predicted letter
 */
export function isHandDetectedInFrame(frame) {
  if (!frame) return false;
  if (typeof frame.hand_detected === 'boolean') {
    return frame.hand_detected;
  }
  if (typeof frame.has_hands === 'boolean') {
    return frame.has_hands;
  }
  if (Array.isArray(frame.landmarks) && frame.landmarks.length > 0) {
    return true;
  }
  const sign = frame.sign || frame.predicted_sign;
  if (sign && sign !== 'NEUTRAL' && sign !== 'WAIT' && sign !== 'unknown' && Number(frame.confidence || 0) > 0) {
    return true;
  }
  return false;
}

/**
 * Đánh giá chi tiết các frame thu được trong 1 lượt thử (5.0s)
 * - 22 chữ cái tĩnh: Đánh giá theo tỷ lệ khớp khung hình matchPct >= 60.0% (giữ nguyên tuyệt đối)
 * - 7 chữ cái động có dấu (Ă, Â, Ê, Đ, Ơ, Ư, Ô): Đánh giá dựa trên bằng chứng chữ nền + xác nhận dấu FSM
 */
export function evaluateTrialFrames(frames, expectedSign, durationSec = 5.0) {
  const isDynamic = Boolean(DYNAMIC_DIACRITICS[expectedSign]);
  const expectedBase = isDynamic ? DYNAMIC_DIACRITICS[expectedSign].base : null;

  const totalFrames = frames.length;
  const validFrames = frames.filter(f => !f.backend_error && f.frame_status !== 'BACKEND_ERROR');
  const handFrames = validFrames.filter(isHandDetectedInFrame);

  let matchFrames = 0;
  let mismatchFrames = 0;
  let neutralFrames = 0;
  const confusions = {};
  let totalConfidence = 0;
  let matchConfidence = 0;

  handFrames.forEach(frame => {
    const sign = frame.sign || frame.predicted_sign || 'NEUTRAL';
    const conf = Number(frame.confidence || 0);
    totalConfidence += conf;

    if (sign === expectedSign) {
      matchFrames += 1;
      matchConfidence += conf;
    } else if (sign === 'NEUTRAL' || sign === 'WAIT' || sign === 'unknown') {
      neutralFrames += 1;
    } else {
      // Đối với chữ động: frame thể hiện chữ nền hợp lệ không tính là mismatch nhầm lẫn
      if (isDynamic && sign === expectedBase) {
        // Frame chuẩn bị chữ nền hợp lệ
      } else {
        mismatchFrames += 1;
        confusions[sign] = (confusions[sign] || 0) + 1;
      }
    }
  });

  let status = 'NEEDS_FIX';
  let hasBaseEvidence = false;
  let hasDiacriticConfirmation = false;
  let activeFrames = 0;
  let matchPct = 0.0;
  const avgConf = matchFrames > 0 ? Number((matchConfidence / matchFrames).toFixed(2)) : 0.0;

  if (isDynamic) {
    // --- ĐÁNH GIÁ CHỮ ĐỘNG (Ă, Â, Ê, Đ, Ơ, Ư, Ô) ---
    activeFrames = matchFrames + mismatchFrames;
    matchPct = activeFrames > 0 ? Number(((matchFrames / activeFrames) * 100).toFixed(1)) : 0.0;

    if (frames.length === 0 || validFrames.length === 0) {
      status = 'INSUFFICIENT_DATA';
    } else if (handFrames.length === 0) {
      status = 'NO_HAND';
    } else if (validFrames.length < 5) {
      status = 'INSUFFICIENT_DATA';
    } else {
      // 1. Kiểm tra bằng chứng chữ nền hợp lệ (ít nhất 2 frames chữ nền hoặc FSM base)
      const baseFrames = validFrames.filter(f => {
        const sign = f.sign || f.predicted_sign;
        const fsmBase = f.diacritic_fsm?.base_label;
        return sign === expectedBase || fsmBase === expectedBase;
      });
      hasBaseEvidence = baseFrames.length >= 2;

      // 2. Kiểm tra sự kiện xác nhận dấu động học chuẩn xác (token hoặc FSM confirmation)
      const diacriticConfirmFrames = validFrames.filter(f => {
        const sign = f.sign || f.predicted_sign;
        const isTarget = (sign === expectedSign);
        const tokenMatch = (f.new_token === expectedSign);
        const fsmConfirmed = Boolean(f.confirmed && isTarget);
        const fsmDetails = f.classifier_details?.diacritic_fsm || f.diacritic_fsm;
        const fsmStateConfirmed = Boolean(
          fsmDetails &&
          fsmDetails.state === 'DIACRITIC_CONFIRMED' &&
          (isTarget || fsmDetails.diacritic_detected === expectedSign)
        );

        // Chốt chặn an toàn: Tuyệt đối KHÔNG chấp nhận chỉ riêng fsmSource && isTarget nếu FSM chưa thực sự xác nhận
        return !f.backend_error && (tokenMatch || fsmConfirmed || fsmStateConfirmed);
      });
      hasDiacriticConfirmation = diacriticConfirmFrames.length >= 1;

      // 3. Phân định PASS vs NEEDS_FIX vs INSUFFICIENT_DATA vs BLOCKED_SOURCE
      // Tuyệt đối không cho phép fake PASS nếu chữ thiếu video nguồn Toàn quốc / Miền Nam hợp lệ
      if (SIGN_CONVENTIONS[expectedSign]?.status === 'BLOCKED_SOURCE') {
        status = 'BLOCKED_SOURCE';
      } else if (hasDiacriticConfirmation && hasBaseEvidence) {
        status = 'PASS';
      } else {
        status = 'NEEDS_FIX';
      }
    }
  } else {
    // --- ĐÁNH GIÁ 22 CHỮ TĨNH (CP7-HN-FIX: MẪU SỐ LÀ TOÀN BỘ HAND FRAMES) ---
    activeFrames = matchFrames + mismatchFrames + neutralFrames;
    matchPct = activeFrames > 0 ? Number(((matchFrames / activeFrames) * 100).toFixed(1)) : 0.0;

    if (frames.length === 0 || validFrames.length === 0) {
      status = 'INSUFFICIENT_DATA';
    } else if (handFrames.length === 0) {
      status = 'NO_HAND';
    } else if (activeFrames < 5) {
      status = 'INSUFFICIENT_DATA';
    } else if (matchPct >= 60.0) {
      status = 'PASS';
    } else {
      status = 'NEEDS_FIX';
    }
  }

  // Phân định nguyên nhân lỗi (Root Cause) theo 6 nhóm lỗi bắt buộc (Task Master Section C)
  let rootCause = 'NONE';
  if (status !== 'PASS') {
    if (status === 'NO_HAND') {
      rootCause = 'NO_HAND_DETECTED';
    } else if (status === 'INSUFFICIENT_DATA') {
      rootCause = 'INSUFFICIENT_DATA';
    } else if (status === 'BLOCKED_SOURCE') {
      rootCause = 'BLOCKED_SOURCE';
    } else if (isDynamic) {
      if (!hasBaseEvidence) {
        rootCause = 'BASE_LETTER_NOT_RECOGNIZED';
      } else {
        const fsmStates = validFrames.map(f => (f.classifier_details?.diacritic_fsm?.state || f.diacritic_fsm?.state || 'IDLE'));
        const reachedStable = fsmStates.some(s => s === 'BASE_STABLE' || s === 'DIACRITIC_CANDIDATE' || s === 'DIACRITIC_CONFIRMED');
        const reachedConfirmed = fsmStates.some(s => s === 'DIACRITIC_CONFIRMED');

        if (!reachedStable) {
          rootCause = 'BASE_DETECTED_BUT_FSM_NOT_STABILIZED';
        } else if (!reachedConfirmed) {
          rootCause = 'FSM_STABLE_BUT_NO_DIACRITIC_MOTION';
        } else {
          const hasTargetSign = validFrames.some(f => (f.sign || f.predicted_sign) === expectedSign);
          if (!hasTargetSign) {
            rootCause = 'FSM_CONFIRMED_BUT_RECOGNIZER_NEUTRAL';
          } else {
            rootCause = 'FRONTEND_DISPLAY_MISMATCH';
          }
        }
      }
    } else {
      rootCause = 'STATIC_CLASSIFICATION_MISMATCH';
    }
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
    durationSec,
    isDynamic,
    hasBaseEvidence: isDynamic ? hasBaseEvidence : undefined,
    hasDiacriticConfirmation: isDynamic ? hasDiacriticConfirmation : undefined,
    root_cause: rootCause
  };
}

/**
 * Xây dựng báo cáo chẩn đoán toàn diện cho 29 chữ cái (để ghi JSON vào backend/data/diagnostics/)
 */
export function buildEvaluationReport(sessionId, resultsByTarget, config = {}) {
  const letters = {};
  const lettersPassed = [];
  const lettersNeedingFix = [];
  const lettersBlockedSource = [];
  const lettersUnsupported = [];
  let totalTrials = 0;
  let passedTrials = 0;
  let totalAccuracySum = 0;
  let evaluatedLetterCount = 0;
  const requiredTrials = config.trialsPerLetter || PILOT_REPETITIONS;

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
    const failed = completed.filter(t => t.status === 'NEEDS_FIX');
    const insufficient = completed.filter(t => t.status === 'INSUFFICIENT_DATA');
    const noHand = completed.filter(t => t.status === 'NO_HAND');
    const valid = completed.filter(t => t.status !== 'INSUFFICIENT_DATA' && t.status !== 'NO_HAND');

    const isDynamic = Boolean(DYNAMIC_DIACRITICS[letter]);
    const dynamicPassRate = completed.length > 0
      ? Number(((passed.length / completed.length) * 100).toFixed(1))
      : 0.0;

    const totalMatchFrames = completed.reduce((sum, t) => sum + (t.matchFrames || 0), 0);
    const totalActiveFrames = completed.reduce((sum, t) => sum + (t.activeFrames || 0), 0);
    const frameAccuracyPct = totalActiveFrames > 0
      ? Number(((totalMatchFrames / totalActiveFrames) * 100).toFixed(1))
      : 0.0;

    // Đối với 7 chữ động: Thước đo độ chính xác cốt lõi là tỷ lệ số lượt PASS (Dynamic Pass Rate)
    // Đối với 22 chữ tĩnh: Giữ nguyên tỷ lệ frame khớp trên tổng số active frames
    const accuracyPct = isDynamic ? dynamicPassRate : frameAccuracyPct;

    // Tổng hợp confusions
    const mergedConfusions = {};
    completed.forEach(t => {
      Object.entries(t.confusions || {}).forEach(([confLetter, count]) => {
        mergedConfusions[confLetter] = (mergedConfusions[confLetter] || 0) + count;
      });
    });

    let status = 'NEEDS_FIX';
    if (SIGN_CONVENTIONS[letter]?.status === 'BLOCKED_SOURCE') {
      status = 'BLOCKED_SOURCE';
      lettersBlockedSource.push(letter);
    } else if (completed.length === 0) {
      status = 'UNTESTED';
    } else if (completed.every(t => t.status === 'NO_HAND')) {
      status = 'NO_HAND';
    } else if (completed.length >= requiredTrials && (passed.length / completed.length) >= 0.7) {
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
      isDynamic,
      totalTrials: completed.length,
      passedTrials: passed.length,
      failedTrials: failed.length,
      validTrials: valid.length,
      insufficientDataTrials: insufficient.length,
      noHandTrials: noHand.length,
      dynamicPassRate: isDynamic ? dynamicPassRate : undefined,
      accuracyPct,
      status,
      topConfusions: mergedConfusions,
      trials: completed.map(t => ({
        rep: t.rep,
        matchPct: t.matchPct,
        status: t.status,
        matchFrames: t.matchFrames,
        mismatchFrames: t.mismatchFrames,
        confusions: t.confusions,
        isDynamic: t.isDynamic,
        hasBaseEvidence: t.hasBaseEvidence,
        hasDiacriticConfirmation: t.hasDiacriticConfirmation
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
      letters_blocked_source: lettersBlockedSource,
      letters_unsupported: lettersUnsupported
    },
    letters
  };
}

/**
 * Trả về timestamp chuẩn Unix-epoch giây nhất quán giữa Frontend và Backend.
 * Ưu tiên performance.timeOrigin + performance.now() để có độ phân giải sub-ms cùng gốc thời gian với time.time().
 */
export function getEpochSecondsNow() {
  if (
    typeof performance !== 'undefined' &&
    typeof performance.now === 'function' &&
    Number.isFinite(performance.timeOrigin) &&
    performance.timeOrigin > 0
  ) {
    return (performance.timeOrigin + performance.now()) / 1000;
  }
  return Date.now() / 1000;
}

/**
 * Tạo bộ sinh timestamp đơn điệu tăng (epoch giây) cho mỗi phiên / lượt thử,
 * chống trùng hoặc lệch vi mô giữa các lần gửi request tuần tự.
 */
export function createMonotonicTimestampGenerator(initialLastTs = 0) {
  let lastTs = Number.isFinite(initialLastTs) ? initialLastTs : 0;
  return {
    next(candidateTs = getEpochSecondsNow()) {
      const raw = Number(candidateTs);
      const safe = Number.isFinite(raw) ? raw : getEpochSecondsNow();
      const mono = safe > lastTs ? safe : lastTs + 1e-4;
      lastTs = mono;
      return mono;
    },
    reset(newBaseline = 0) {
      lastTs = Number.isFinite(newBaseline) ? newBaseline : 0;
    },
    current() {
      return lastTs;
    }
  };
}
