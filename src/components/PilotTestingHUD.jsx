import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  FlaskConical, Play, Square, RotateCcw, Save, AlertTriangle
} from 'lucide-react';
import {
  setPilotLogging,
  flushPilotLogs,
  fetchPilotStats,
  fetchPilotRuntimeConfig,
  savePilotSessionManifest,
  savePilotSidecarSync
} from '../services/api';
import { phaseAtElapsedSeconds, PHASE_LABELS_VI, VSL_ALPHABET_29, DEFAULT_PILOT_TARGET, PILOT_SEQUENCE_INITIAL_ACTIVE, pilotTrialOutcome, pilotHudStatus, canStartPilotSequence, canChangePilotTarget, pilotTargetFields, addPilotTargetResult, latestPilotSequenceForTarget, summarizeStableFrames, summarizeSessionTrials, trialHasConfirmedTarget, trialIsValid, candidateDConfigMatches, pilotLoggingConfirmed, pilotLoggingInterrupted, createPilotSessionId, createPilotTrialId, createPilotSequenceId, createPilotSequenceTrialId, pilotSequenceCanAdvance, resolvePilotSession, pilotLoggingEnableAcknowledged, pilotFlushVerified, PILOT_DURATION_SEC, PILOT_REPETITIONS } from './pilotProtocol';

export { VSL_ALPHABET_29 };

// Danh sách 22 chữ cái tĩnh (tương thích ngược cho các module cũ)
export const STATIC_LETTERS_22 = [
  'A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
  'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y'
];

// 8 chữ cái trọng tâm trong kế hoạch FIX 8 TASK
export const FOCUS_LETTERS_8 = new Set(['P', 'X', 'O', 'Q', 'M', 'N', 'E', 'H']);

// 7 chữ cái có dấu (động/phức)
export const DIACRITIC_LETTERS_7 = new Set(['Ă', 'Â', 'Đ', 'Ê', 'Ô', 'Ơ', 'Ư']);

// Thời lượng trial tự động chuẩn theo cơ chế Pilot VSL (6.0s tương đương ~180 frames ở 30 FPS theo TASK 3.9/3.9.1)
export const DEFAULT_TRIAL_DURATION_SEC = PILOT_DURATION_SEC;
export const DURATION_PRESETS = [PILOT_DURATION_SEC];

function MetricCard({ label, value, detail }) {
  return <div style={{ display: 'grid', gap: 4, borderRadius: 8, padding: 12, background: 'rgba(15,23,42,.82)', border: '1px solid rgba(148,163,184,.22)' }}>
    <span style={{ fontSize: 11, color: '#cbd5e1' }}>{label}</span>
    <strong style={{ fontSize: 22, fontVariantNumeric: 'tabular-nums' }}>{value}</strong>
    <span style={{ fontSize: 11, color: '#94a3b8' }}>{detail}</span>
  </div>;
}

export default function PilotTestingHUD({
  isActive,
  onToggleActive,
  onMetadataUpdate,
  cameraActive,
  latestRecognition,
  videoElementRef
}) {
  // Session Configuration
  const [sessionId, setSessionId] = useState(() => createPilotSessionId());

  const [selectedLetter, setSelectedLetter] = useState(DEFAULT_PILOT_TARGET);
  const [currentRep, setCurrentRep] = useState(1);
  const totalReps = PILOT_REPETITIONS;

  // Cấu hình thời lượng tự động kết thúc trial (giây)
  const trialDurationSec = PILOT_DURATION_SEC;
  const [countdownSeconds, setCountdownSeconds] = useState(DEFAULT_TRIAL_DURATION_SEC);
  const [trialFrameCount, setTrialFrameCount] = useState(0);

  // Trial State: 'IDLE' | 'READY' | 'RECORDING'
  const [trialState, setTrialState] = useState('IDLE');

  // Video RGB Recording & Synchronization (Task 3.10)
  const [recordVideoRGB, setRecordVideoRGB] = useState(true);
  const [isRecordingVideo, setIsRecordingVideo] = useState(false);
  const [mediaRecorderSupported, setMediaRecorderSupported] = useState(true);
  const [videoRecordError, setVideoRecordError] = useState(null);
  const [recordedVideoBlob, setRecordedVideoBlob] = useState(null);
  const [videoBlobUrl, setVideoBlobUrl] = useState(null);
  const [videoDownloadName, setVideoDownloadName] = useState(null);
  const [syncSidecarData, setSyncSidecarData] = useState(null);
  const [syncSidecarUrl, setSyncSidecarUrl] = useState(null);
  const [sidecarDownloadName, setSidecarDownloadName] = useState(null);
  const [signerId, setSignerId] = useState('signer_01');
  const [syncAnomalies, setSyncAnomalies] = useState({ jumps: 0, stalls: 0 });
  const [currentKinematicPhase, setCurrentKinematicPhase] = useState('PRE_IDLE');
  const [trialError, setTrialError] = useState(null);
  const [sessionLifecycle, setSessionLifecycle] = useState({ status: 'checking', backendSessionId: null, reason: null });
  const [sessionOperationBusy, setSessionOperationBusy] = useState(false);
  const [sequenceActive, setSequenceActive] = useState(PILOT_SEQUENCE_INITIAL_ACTIVE);
  const [sequenceCountdown, setSequenceCountdown] = useState(null);
  const [sequenceStartPending, setSequenceStartPending] = useState(false);
  const [sequenceResults, setSequenceResults] = useState([]);
  const [sequenceResultsByTarget, setSequenceResultsByTarget] = useState({});

  const mediaRecorderRef = useRef(null);
  const videoChunksRef = useRef([]);
  const syncMappingRef = useRef([]);
  const lastSendTsRef = useRef(null);
  const lastFrameIdxRef = useRef(null);
  const trialStartPerfRef = useRef(null);
  const stableFramesRef = useRef([]);
  const phaseFrameCountsRef = useRef({});
  const trialResponsesRef = useRef([]);
  const loggingInterruptedRef = useRef(false);
  const loggingBaselineRef = useRef({ dropped_queue_full: 0, dropped_io_error: 0, queued_count: 0, logged_count: 0 });
  const pendingPilotFramesRef = useRef(0);
  const pendingPilotDrainResolverRef = useRef(null);
  const handleEndTrialRef = useRef(null);
  const sessionCheckInitializedRef = useRef(false);
  const knownBackendSessionRef = useRef(null);
  const startInProgressRef = useRef(false);
  const sequenceActiveRef = useRef(PILOT_SEQUENCE_INITIAL_ACTIVE);
  const sequenceRunIdRef = useRef(null);
  const sequenceTrialIdsRef = useRef(new Set());
  const lastTelemetryUiUpdateRef = useRef(0);
  const sequenceTimerRef = useRef(null);
  const sequenceStartTimerRef = useRef(null);
  const handleStartTrialRef = useRef(null);

  const waitForPendingPilotFrames = () => new Promise(resolve => {
    if (pendingPilotFramesRef.current === 0) resolve();
    else pendingPilotDrainResolverRef.current = resolve;
  });

  const settlePendingPilotFrame = () => {
    pendingPilotFramesRef.current = Math.max(0, pendingPilotFramesRef.current - 1);
    if (pendingPilotFramesRef.current === 0 && pendingPilotDrainResolverRef.current) {
      pendingPilotDrainResolverRef.current();
      pendingPilotDrainResolverRef.current = null;
    }
  };
  const recordVideoRGBRef = useRef(recordVideoRGB);
  recordVideoRGBRef.current = recordVideoRGB;
  const signerIdRef = useRef(signerId);
  signerIdRef.current = signerId;

  // Telemetry & Metrics thu thập tự động theo thời gian thực (KHÔNG yêu cầu tester đánh dấu thủ công)
  const [realtimeMetrics, setRealtimeMetrics] = useState({
    total: 0,
    matchCount: 0,
    mismatchCount: 0,
    neutralCount: 0,
    confusions: {},
    latestSign: 'NEUTRAL',
    latestConfidence: 0.0,
    latestSource: 'none'
  });

  // Tóm tắt kết quả Trial vừa hoàn tất tự động
  const [lastTrialSummary, setLastTrialSummary] = useState(null);
  const [resultArtifactTarget, setResultArtifactTarget] = useState(null);

  // Lịch sử các Trial đã hoàn thành trong Session này
  const [completedTrials, setCompletedTrials] = useState([]);

  // Backend Logging Telemetry & Info
  const [loggingEnabled, setLoggingEnabled] = useState(false);
  const [logFilePath, setLogFilePath] = useState('');
  const [hasStaticHead, setHasStaticHead] = useState(false);
  const [backendStats, setBackendStats] = useState({
    queued_count: 0,
    logged_count: 0,
    dropped_queue_full: 0,
    dropped_io_error: 0,
    total_dropped: 0
  });
  const [flushStatus, setFlushStatus] = useState(null);

  // Refs bảo vệ ràng buộc vi giây (Task 7F.5D) cho 60 FPS loop
  const trialStateRef = useRef(trialState);
  if (trialStateRef.current !== 'FINALIZING') trialStateRef.current = trialState;

  const selectedLetterRef = useRef(selectedLetter);
  selectedLetterRef.current = selectedLetter;

  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  const trialDurationRef = useRef(trialDurationSec);

  const frameIdxRef = useRef(0);
  const timerIntervalRef = useRef(null);
  const metricsRef = useRef({ total: 0, matchCount: 0, mismatchCount: 0, neutralCount: 0, confusions: {} });

  // Current Unique Trial ID
  const trialId = sequenceRunIdRef.current
    ? createPilotSequenceTrialId(sessionId, selectedLetter, currentRep, sequenceRunIdRef.current)
    : createPilotTrialId(sessionId, selectedLetter, currentRep);
  const trialIdRef = useRef(trialId);
  trialIdRef.current = trialId;

  // Cập nhật trạng thái IDLE / READY dựa trên cameraActive
  useEffect(() => {
    if (!cameraActive) {
      if (trialStateRef.current === 'RECORDING') {
        handleEndTrial({ completed: false, reason: 'camera_disconnected' });
      }
      if (trialStateRef.current !== 'FINALIZING') setTrialState('IDLE');
    } else if (trialStateRef.current === 'IDLE') {
      setTrialState('READY');
    }
  }, [cameraActive]);

  // Xử lý frame kết quả trả về từ Backend theo thời gian thực (Hoàn toàn tự động, không yêu cầu xác nhận thủ công)
  const handleIncomingFrameResult = useCallback((data, metadata = null, status = {}) => {
    const belongsToCurrentTrial = metadata?.trial_id && metadata.trial_id === trialIdRef.current;
    if (belongsToCurrentTrial) settlePendingPilotFrame();
    if (!['RECORDING', 'FINALIZING'].includes(trialStateRef.current)) return;
    if (!belongsToCurrentTrial) return;

    if (status.backendError) {
      loggingInterruptedRef.current = true;
      setTrialError(`Backend logging/interpreting failed: ${status.backendErrorMessage || 'connection lost'}`);
        handleEndTrialRef.current?.({ completed: false, reason: 'backend_interrupted' });
      return;
    }

    const predSign = data?.sign || 'NEUTRAL';
    const conf = typeof data?.confidence === 'number' ? data.confidence : 0.0;
    const source = data?.source || data?.prediction_source || 'none';
    const target = selectedLetterRef.current;
    trialResponsesRef.current.push({ sign: predSign, confirmed: Boolean(data?.confirmed) });
    const framePhase = metadata?.phase;
    if (framePhase) phaseFrameCountsRef.current[framePhase] = (phaseFrameCountsRef.current[framePhase] || 0) + 1;
    if (framePhase === 'STABLE_HOLD') {
      stableFramesRef.current.push({ sign: predSign, phase: framePhase, confirmed: Boolean(data?.confirmed) });
    }

    metricsRef.current.total += 1;
    if (predSign === target) {
      metricsRef.current.matchCount += 1;
    } else if (predSign === 'NEUTRAL') {
      metricsRef.current.neutralCount += 1;
    } else {
      metricsRef.current.mismatchCount += 1;
      metricsRef.current.confusions[predSign] = (metricsRef.current.confusions[predSign] || 0) + 1;
    }

    const uiNow = performance.now();
    if (uiNow - lastTelemetryUiUpdateRef.current >= 250) {
      lastTelemetryUiUpdateRef.current = uiNow;
      setRealtimeMetrics({
        total: metricsRef.current.total,
        matchCount: metricsRef.current.matchCount,
        mismatchCount: metricsRef.current.mismatchCount,
        neutralCount: metricsRef.current.neutralCount,
        confusions: { ...metricsRef.current.confusions },
        latestSign: predSign,
        latestConfidence: conf,
        latestSource: source
      });
    }
  }, []);

  // Lắng nghe latestRecognition prop nếu có
  useEffect(() => {
    if (latestRecognition && trialState === 'RECORDING') {
      handleIncomingFrameResult(latestRecognition);
    }
  }, [latestRecognition, trialState, handleIncomingFrameResult]);

  // MediaRecorder Helpers (Task 3.10 Implementation B)
  const getSupportedMimeType = () => {
    if (typeof window === 'undefined' || typeof window.MediaRecorder === 'undefined') return null;
    const types = [
      'video/webm;codecs=vp9',
      'video/webm;codecs=vp8',
      'video/webm',
      'video/mp4'
    ];
    for (const t of types) {
      if (MediaRecorder.isTypeSupported(t)) return t;
    }
    return '';
  };

  const startMediaRecorder = () => {
    setVideoRecordError(null);
    if (!videoElementRef?.current?.srcObject) {
      setVideoRecordError('Không tìm thấy luồng camera để ghi hình.');
      return false;
    }
    const stream = videoElementRef.current.srcObject;
    const mimeType = getSupportedMimeType();
    if (mimeType === null) {
      setMediaRecorderSupported(false);
      setVideoRecordError('Trình duyệt không hỗ trợ MediaRecorder API.');
      return false;
    }
    try {
      videoChunksRef.current = [];
      const options = mimeType ? { mimeType } : undefined;
      const mr = new MediaRecorder(stream, options);
      mr.ondataavailable = (e) => {
        if (e.data && e.data.size > 0) {
          videoChunksRef.current.push(e.data);
        }
      };
      mr.onerror = (e) => {
        setVideoRecordError(`Lỗi MediaRecorder: ${e.error?.message || 'Không xác định'}`);
      };
      mr.start(100);
      mediaRecorderRef.current = mr;
      setIsRecordingVideo(true);
      return true;
    } catch (err) {
      setVideoRecordError(`Không thể khởi tạo MediaRecorder: ${err.message}`);
      return false;
    }
  };

  const stopMediaRecorder = () => {
    return new Promise((resolve) => {
      if (!mediaRecorderRef.current || mediaRecorderRef.current.state === 'inactive') {
        setIsRecordingVideo(false);
        resolve(null);
        return;
      }
      const mr = mediaRecorderRef.current;
      mr.onstop = () => {
        try {
          const blob = new Blob(videoChunksRef.current, { type: mr.mimeType || 'video/webm' });
          const url = URL.createObjectURL(blob);
          setRecordedVideoBlob(blob);
          setVideoBlobUrl(url);
          setIsRecordingVideo(false);
          resolve({ blob, url });
        } catch (e) {
          setIsRecordingVideo(false);
          resolve(null);
        }
      };
      try {
        mr.stop();
      } catch (err) {
        setIsRecordingVideo(false);
        resolve(null);
      }
    });
  };

  // Đồng bộ metadata ra ngoài CameraSignRecognizer qua callback
  useEffect(() => {
    if (trialState === 'RECORDING') {
      onMetadataUpdate({
        getFrameMetadata: () => {
          // BẢO VỆ RÀNG BUỘC VI GIÂY: Kiểm tra ref đồng bộ ngay bên trong callback
          if (trialStateRef.current !== 'RECORDING') {
            return null;
          }
          frameIdxRef.current += 1;
          const currFidx = frameIdxRef.current;
          if (currFidx === 1 || currFidx % 8 === 0) setTrialFrameCount(currFidx);

          const nowSec = performance.now() / 1000;
          const elapsedSec = trialStartPerfRef.current === null ? 0 : Math.max(0, nowSec - trialStartPerfRef.current);
          const phase = phaseAtElapsedSeconds(elapsedSec);
          const targetFields = pilotTargetFields(selectedLetterRef.current);
          pendingPilotFramesRef.current += 1;
          setCurrentKinematicPhase(phase);
          if (lastFrameIdxRef.current !== null && currFidx - lastFrameIdxRef.current > 1) {
            setSyncAnomalies(prev => ({ ...prev, jumps: prev.jumps + (currFidx - lastFrameIdxRef.current - 1) }));
          }
          if (lastSendTsRef.current !== null && (nowSec - lastSendTsRef.current) * 1000 > 70) {
            setSyncAnomalies(prev => ({ ...prev, stalls: prev.stalls + 1 }));
          }
          lastFrameIdxRef.current = currFidx;
          lastSendTsRef.current = nowSec;

          const videoPts = videoElementRef?.current && Number.isFinite(videoElementRef.current.currentTime)
            ? Number(videoElementRef.current.currentTime.toFixed(4))
            : 'UNKNOWN';

          const syncEntry = {
            frame_idx: currFidx,
            client_capture_ts: Number(nowSec.toFixed(4)),
            video_pts_sec: videoPts,
            video_frame_idx: currFidx,
            signer_id: signerIdRef.current,
            trial_id: trialIdRef.current,
            ...targetFields,
            phase
          };
          syncMappingRef.current.push(syncEntry);

          return {
            expected_sign: targetFields.expected_sign,
            session_id: sessionIdRef.current,
            trial_id: trialIdRef.current,
            frame_idx: currFidx,
            signer_id: signerIdRef.current,
            client_capture_ts: Number(nowSec.toFixed(4)),
            video_pts_sec: videoPts,
            video_frame_idx: currFidx,
            phase
          };
        },
        onFrameResult: handleIncomingFrameResult
      });
    } else {
      // Khi không trong trial, metadata là null tuyệt đối (Task 7F.5D)
      onMetadataUpdate({
        getFrameMetadata: () => null,
        onFrameResult: null
      });
    }
  }, [trialState, handleIncomingFrameResult, onMetadataUpdate, videoElementRef]);

  // Quản lý Bật/Tắt chế độ Pilot Logging trên Backend
  const handleTogglePilotLogging = async (enable, requestedSessionId = sessionIdRef.current) => {
    const res = await setPilotLogging(enable, null, requestedSessionId);
    if (!res?.success) return res || { success: false, error: 'Backend did not acknowledge pilot logging.' };
    if (enable && !pilotLoggingEnableAcknowledged(res, requestedSessionId)) {
      return { success: false, error: `Backend did not confirm session ${requestedSessionId}.`, response: res };
    }
    setLoggingEnabled(enable);
    if (res.filepath) setLogFilePath(res.filepath);
    if (res.stats) setBackendStats(res.stats);
    return res;
  };

  const refreshBackendSession = useCallback(async () => {
    const [stats, config] = await Promise.all([fetchPilotStats(), fetchPilotRuntimeConfig()]);
    if (!stats?.success) {
      const reason = stats?.error || 'Không kết nối được backend stats.';
      setSessionLifecycle({ status: 'unavailable', backendSessionId: knownBackendSessionRef.current, reason });
      setTrialError(`Không xác minh được session backend: ${reason}`);
      if (sequenceActiveRef.current && trialStateRef.current !== 'RECORDING') {
        sequenceActiveRef.current = false;
        setSequenceActive(false);
        setSequenceCountdown(null);
        setSequenceStartPending(false);
        clearInterval(sequenceTimerRef.current);
        sequenceTimerRef.current = null;
      }
      if (trialStateRef.current === 'RECORDING') {
        loggingInterruptedRef.current = true;
        handleEndTrialRef.current?.({ completed: false, reason: 'logging_status_unavailable' });
      }
      return { stats, config, status: 'unavailable' };
    }

    if (stats.stats) setBackendStats(stats.stats);
    if (stats.filepath) setLogFilePath(stats.filepath);
    if (stats.has_static_head !== undefined) setHasStaticHead(Boolean(stats.has_static_head));
    setLoggingEnabled(Boolean(stats.pilot_logging_enabled));

    if (stats.pilot_logging_enabled) {
      const backendSessionId = stats.session_id || null;
      if (!backendSessionId) {
        const reason = 'Backend báo logging đang bật nhưng không có session_id.';
        setSessionLifecycle({ status: 'session_invalid', backendSessionId: null, reason });
        setTrialError(reason);
        if (sequenceActiveRef.current && trialStateRef.current !== 'RECORDING') {
          sequenceActiveRef.current = false;
          setSequenceActive(false);
          setSequenceCountdown(null);
          setSequenceStartPending(false);
          clearInterval(sequenceTimerRef.current);
          sequenceTimerRef.current = null;
        }
        return { stats, config, status: 'session_invalid' };
      }

      const isFirstSync = !sessionCheckInitializedRef.current && !knownBackendSessionRef.current;
      if (isFirstSync) setSessionId(backendSessionId);
      knownBackendSessionRef.current = backendSessionId;
      const hudSessionId = isFirstSync ? backendSessionId : sessionIdRef.current;
      const decision = resolvePilotSession(stats, config, hudSessionId);
      const validRuntime = decision.status === 'active' && stats.has_static_head === true;
      const status = decision.status === 'session_mismatch'
        ? 'session_mismatch'
        : validRuntime ? 'active' : decision.status === 'active' ? 'runtime_invalid' : decision.status;
      const reason = decision.reason || (stats.has_static_head === true ? null : 'Static head chưa được backend xác nhận loaded.');
      setSessionLifecycle({ status, backendSessionId, reason, filepath: stats.filepath || null });
      if (sequenceActiveRef.current && status !== 'active' && trialStateRef.current !== 'RECORDING') {
        sequenceActiveRef.current = false;
        setSequenceActive(false);
        setSequenceCountdown(null);
        setSequenceStartPending(false);
        clearInterval(sequenceTimerRef.current);
        sequenceTimerRef.current = null;
        setTrialError(`Chuỗi chữ ${selectedLetterRef.current} dừng vì session/runtime không còn hợp lệ: ${reason || status}`);
      }
      if (status === 'session_mismatch') {
        setTrialError(`Session HUD (${sessionIdRef.current}) khác session backend (${backendSessionId}). Hãy đóng session backend an toàn trước khi tiếp tục.`);
      } else if (status !== 'active') {
        setTrialError(reason || 'Runtime backend chưa hợp lệ.');
      } else {
        setTrialError(null);
      }
      if (trialStateRef.current === 'RECORDING' && (
        status !== 'active' ||
        pilotLoggingInterrupted(stats, config, loggingBaselineRef.current, sessionIdRef.current)
      )) {
        loggingInterruptedRef.current = true;
        handleEndTrialRef.current?.({ completed: false, reason: 'session_or_runtime_invalid' });
      }
      sessionCheckInitializedRef.current = true;
      return { stats, config, status };
    }

    const stoppedSessionId = knownBackendSessionRef.current;
    knownBackendSessionRef.current = null;
    const backendStoppedDuringTrial = trialStateRef.current === 'RECORDING';
    if (stoppedSessionId && sessionIdRef.current === stoppedSessionId && !backendStoppedDuringTrial) {
      let freshId = createPilotSessionId();
      while (freshId === stoppedSessionId) freshId = createPilotSessionId();
      setSessionId(freshId);
      setTrialError(`Backend session ${stoppedSessionId} đã dừng hoặc backend vừa khởi động lại. Đã tạo ID mới; file cũ được giữ nguyên.`);
    }
    const configReady = candidateDConfigMatches(config) && stats.has_static_head === true;
    const readyStatus = configReady ? 'ready' : candidateDConfigMatches(config) ? 'runtime_invalid' : 'config_invalid';
    const readyReason = configReady ? null : !candidateDConfigMatches(config)
      ? (config?.error || 'Candidate-D runtime config chưa được xác minh.')
      : 'Static head chưa được backend xác nhận loaded.';
    const status = backendStoppedDuringTrial ? 'unavailable' : readyStatus;
    const reason = backendStoppedDuringTrial ? 'Backend đã tắt hoặc restart trong lúc Trial; Trial hiện tại bị loại.' : readyReason;
    setSessionLifecycle({ status, backendSessionId: null, reason });
    if (sequenceActiveRef.current && status !== 'active' && trialStateRef.current !== 'RECORDING') {
      sequenceActiveRef.current = false;
      setSequenceActive(false);
      setSequenceCountdown(null);
      setSequenceStartPending(false);
      clearInterval(sequenceTimerRef.current);
      sequenceTimerRef.current = null;
      setTrialError(`Chuỗi chữ ${selectedLetterRef.current} dừng vì logging/session không còn hợp lệ: ${reason || status}`);
    }
    if (backendStoppedDuringTrial) {
      loggingInterruptedRef.current = true;
      setTrialError(reason);
      Promise.resolve(handleEndTrialRef.current?.({ completed: false, reason: 'backend_session_stopped' })).finally(() => {
        if (stoppedSessionId && sessionIdRef.current === stoppedSessionId) {
          let freshId = createPilotSessionId();
          while (freshId === stoppedSessionId) freshId = createPilotSessionId();
          setSessionId(freshId);
          setSessionLifecycle({ status: readyStatus, backendSessionId: null, reason: readyReason });
        }
      });
    } else if (status !== 'ready') setTrialError(reason);
    else if (!stoppedSessionId) setTrialError(null);
    sessionCheckInitializedRef.current = true;
    return { stats, config, status };
  }, []);

  // Định kỳ cập nhật stats từ Backend mỗi 2 giây khi mở HUD
  useEffect(() => {
    if (!isActive) return;
    refreshBackendSession();
    const interval = setInterval(refreshBackendSession, 2000);
    return () => clearInterval(interval);
  }, [isActive, refreshBackendSession]);

  const flushAndStopPilotSession = async (expectedSessionId) => {
    const before = await fetchPilotStats();
    if (!pilotLoggingConfirmed(before, expectedSessionId)) {
      return { success: false, error: `Backend session không còn khớp (${before?.session_id || 'inactive'}). Không dừng logger.` };
    }
    const flush = await flushPilotLogs(expectedSessionId);
    const afterFlush = await fetchPilotStats();
    if (!pilotFlushVerified(before, flush, afterFlush, expectedSessionId)) {
      return { success: false, error: 'Flush chưa được xác minh: session/path thay đổi, còn record pending, có record drop mới hoặc queued/logged không khớp. Logger vẫn được giữ bật.' };
    }
    const stopped = await handleTogglePilotLogging(false, expectedSessionId);
    if (!stopped?.success || stopped.session_id !== expectedSessionId) {
      return { success: false, error: stopped?.error || 'Backend không xác nhận đã đóng session hiện tại.' };
    }
    const afterStop = await fetchPilotStats();
    if (!afterStop?.success || afterStop.pilot_logging_enabled || afterStop.session_id) {
      return { success: false, error: 'Backend chưa xác nhận trạng thái logging đã tắt.' };
    }
    return { success: true, filepath: before.filepath, stats: afterFlush.stats };
  };

  const handleRotatePilotSession = async () => {
    if (sessionOperationBusy || trialStateRef.current === 'RECORDING') return;
    const backendSessionId = sessionLifecycle.backendSessionId;
    if (!backendSessionId) {
      setTrialError('Không có session backend đã xác minh để đóng.');
      return;
    }
    setSessionOperationBusy(true);
    try {
      const closed = await flushAndStopPilotSession(backendSessionId);
      if (!closed.success) {
        await refreshBackendSession();
        setTrialError(closed.error);
        return;
      }
      knownBackendSessionRef.current = null;
      let nextSessionId = createPilotSessionId();
      while (nextSessionId === backendSessionId) nextSessionId = createPilotSessionId();
      setSessionId(nextSessionId);
      setSessionLifecycle({ status: 'checking', backendSessionId: null, reason: null });
      const enabled = await handleTogglePilotLogging(true, nextSessionId);
      if (!pilotLoggingEnableAcknowledged(enabled, nextSessionId)) {
        setSessionLifecycle({ status: 'unavailable', backendSessionId: null, reason: enabled?.error || 'Logging enable failed.' });
        setTrialError(`Session cũ đã flush và đóng. Không bật được session mới: ${enabled?.error || 'backend không xác nhận session mới'}`);
        await refreshBackendSession();
        return;
      }
      const [stats, config] = await Promise.all([fetchPilotStats(), fetchPilotRuntimeConfig()]);
      const decision = resolvePilotSession(stats, config, nextSessionId);
      if (decision.status !== 'active' || stats.has_static_head !== true) {
        const reason = decision.reason || 'Static head chưa được xác minh.';
        setSessionLifecycle({ status: decision.status, backendSessionId: stats?.session_id || null, reason });
        setTrialError(`Session cũ đã đóng; session mới chưa sẵn sàng: ${reason}`);
        return;
      }
      knownBackendSessionRef.current = nextSessionId;
      sessionCheckInitializedRef.current = true;
      setSessionLifecycle({ status: 'active', backendSessionId: nextSessionId, reason: null, filepath: stats.filepath || null });
      setBackendStats(stats.stats);
      setLogFilePath(stats.filepath || '');
      setTrialError(null);
      setFlushStatus(`Session cũ đã flush/đóng an toàn (${closed.filepath}); session mới ${nextSessionId} đang ghi.`);
      setCurrentRep(1);
      setCompletedTrials([]);
    } catch (error) {
      setSessionLifecycle({ status: 'unavailable', backendSessionId: knownBackendSessionRef.current, reason: error?.message || String(error) });
      setTrialError(`Không hoàn tất được đổi session an toàn: ${error?.message || String(error)}`);
    } finally {
      setSessionOperationBusy(false);
    }
  };

  // Bắt đầu một Trial ghi nhận có hẹn giờ tự động
  const runStartTrial = async () => {
    const startedAsSequenceTrial = sequenceActiveRef.current;
    if (!cameraActive) {
      alert('Vui lòng bật Camera trước khi bắt đầu Trial kiểm thử!');
      return;
    }
    if (!startedAsSequenceTrial) {
      sequenceRunIdRef.current = null;
      trialIdRef.current = createPilotTrialId(sessionIdRef.current, selectedLetterRef.current, currentRep);
    }
    if (!sequenceActiveRef.current && completedTrials.filter(trial => trial.trialValid).length >= PILOT_REPETITIONS) {
      setTrialError('Session đã đủ 10 trial hợp lệ.');
      return;
    }

    const expectedSessionId = sessionIdRef.current;
    let [runtime, pilotStats] = await Promise.all([fetchPilotRuntimeConfig(), fetchPilotStats()]);
    if (startedAsSequenceTrial && !sequenceActiveRef.current) return;
    if (!pilotStats?.success) {
      const reason = pilotStats?.error || 'Backend stats unavailable.';
      setSessionLifecycle({ status: 'unavailable', backendSessionId: null, reason });
      setTrialError(`Không xác minh được backend session: ${reason}`);
      return;
    }
    if (!candidateDConfigMatches(runtime)) {
      const reason = runtime?.error || 'Runtime config không khớp Candidate-D.';
      setSessionLifecycle({ status: 'config_invalid', backendSessionId: pilotStats.session_id || null, reason });
      setTrialError(`Trial bị chặn: ${reason}`);
      return;
    }

    let decision = resolvePilotSession(pilotStats, runtime, expectedSessionId);
    if (decision.status === 'session_mismatch') {
      setSessionLifecycle({ status: 'session_mismatch', backendSessionId: pilotStats.session_id, reason: decision.reason });
      setTrialError(`Trial bị chặn: HUD session '${expectedSessionId}' khác backend session '${pilotStats.session_id}'. Dùng thao tác đóng session an toàn.`);
      return;
    }
    if (decision.status === 'unavailable' || decision.status === 'session_invalid') {
      setSessionLifecycle({ status: decision.status, backendSessionId: pilotStats.session_id || null, reason: decision.reason });
      setTrialError(`Trial bị chặn: ${decision.reason}`);
      return;
    }

    if (decision.status === 'ready') {
      const enabled = await handleTogglePilotLogging(true, expectedSessionId);
      if (!pilotLoggingEnableAcknowledged(enabled, expectedSessionId)) {
        const reason = enabled?.error || 'Backend không xác nhận enable logging.';
        setTrialError(`Không bật được logging cho session '${expectedSessionId}': ${reason}`);
        await refreshBackendSession();
        return;
      }
      [runtime, pilotStats] = await Promise.all([fetchPilotRuntimeConfig(), fetchPilotStats()]);
      if (startedAsSequenceTrial && !sequenceActiveRef.current) return;
      decision = resolvePilotSession(pilotStats, runtime, expectedSessionId);
    }

    if (decision.status !== 'active' || pilotStats.has_static_head !== true) {
      const reason = decision.reason || (pilotStats?.has_static_head === true ? 'Backend session chưa active.' : 'Static head chưa được xác nhận loaded.');
      setSessionLifecycle({ status: decision.status === 'active' ? 'runtime_invalid' : decision.status, backendSessionId: pilotStats?.session_id || null, reason });
      setTrialError(`Backend/session/configuration không hợp lệ; trial bị chặn: ${reason}`);
      return;
    }
    knownBackendSessionRef.current = expectedSessionId;
    sessionCheckInitializedRef.current = true;
    setLoggingEnabled(true);
    setLogFilePath(pilotStats.filepath || '');
    setBackendStats(pilotStats.stats || {});
    setSessionLifecycle({ status: 'active', backendSessionId: expectedSessionId, reason: null, filepath: pilotStats.filepath || null });
    loggingBaselineRef.current = {
      dropped_queue_full: pilotStats.stats?.dropped_queue_full || 0,
      dropped_io_error: pilotStats.stats?.dropped_io_error || 0,
      queued_count: pilotStats.stats?.queued_count || 0,
      logged_count: pilotStats.stats?.logged_count || 0,
    };
    if (trialDurationRef.current !== PILOT_DURATION_SEC || totalReps !== PILOT_REPETITIONS) {
      setTrialError('Protocol khóa: 6 giây, 10 repetitions.');
      return;
    }
    if (startedAsSequenceTrial && !sequenceActiveRef.current) return;

    // Reset frame counter, real-time metrics, sync refs
    frameIdxRef.current = 0;
    metricsRef.current = { total: 0, matchCount: 0, mismatchCount: 0, neutralCount: 0, confusions: {} };
    syncMappingRef.current = [];
    lastSendTsRef.current = null;
    lastFrameIdxRef.current = null;
    setSyncAnomalies({ jumps: 0, stalls: 0 });
    setVideoRecordError(null);
    if (videoBlobUrl) {
      URL.revokeObjectURL(videoBlobUrl);
      setVideoBlobUrl(null);
    }
    if (syncSidecarUrl) {
      URL.revokeObjectURL(syncSidecarUrl);
      setSyncSidecarUrl(null);
    }
    setRecordedVideoBlob(null);
    setTrialFrameCount(0);
    setRealtimeMetrics({
      total: 0,
      matchCount: 0,
      mismatchCount: 0,
      neutralCount: 0,
      confusions: {},
      latestSign: 'NEUTRAL',
      latestConfidence: 0.0,
      latestSource: 'none'
    });
    setFlushStatus(null);
    setLastTrialSummary(null);
    setTrialError(null);
    stableFramesRef.current = [];
    phaseFrameCountsRef.current = {};
    trialResponsesRef.current = [];
    loggingInterruptedRef.current = false;

    // Kích hoạt trạng thái RECORDING
    if (startedAsSequenceTrial) sequenceTrialIdsRef.current.add(trialIdRef.current);
    trialStateRef.current = 'RECORDING';
    setTrialState('RECORDING');
    setCurrentKinematicPhase('PRE_IDLE');

    if (recordVideoRGBRef.current) {
      startMediaRecorder();
    }

    const durationSec = Math.max(1.0, Math.min(30.0, trialDurationRef.current || DEFAULT_TRIAL_DURATION_SEC));
    setCountdownSeconds(durationSec);
    trialStartPerfRef.current = performance.now() / 1000;
    const durationMs = durationSec * 1000;

    clearInterval(timerIntervalRef.current);
    timerIntervalRef.current = setInterval(() => {
      const elapsedSec = Math.max(0, performance.now() / 1000 - trialStartPerfRef.current);
      const elapsed = elapsedSec * 1000;
      const remaining = Math.max(0, (durationMs - elapsed) / 1000);
      setCountdownSeconds(Math.ceil(remaining));

      // Phân định 5 pha động học theo đúng protocol 6.0s đã duyệt (TASK 3.9 & TASK 3.9.1)
      // Các mốc thời gian tuyệt đối tính từ khi bắt đầu trial (không co giãn theo durationSec):
      // - PRE_IDLE: [0.0, 0.5) (0.5 giây)
      // - ENTRY: [0.5, 1.5) (1.0 giây)
      // - STABLE_HOLD: [1.5, 4.5) (3.0 giây - Đánh giá tĩnh)
      // - EXIT: [4.5, 5.5) (1.0 giây)
      // - POST_IDLE: [5.5, 6.0] (0.5 giây)
      setCurrentKinematicPhase(phaseAtElapsedSeconds(elapsedSec));

      // KHI HẾT THỜI LƯỢNG: TỰ ĐỘNG KẾT THÚC TRIAL VÀ LƯU DỮ LIỆU
      if (elapsed >= durationMs) {
        clearInterval(timerIntervalRef.current);
        handleEndTrial({ completed: true, isAuto: true });
      }
    }, 50);
  };

  const handleStartTrial = async () => {
    if (startInProgressRef.current || sessionOperationBusy) return;
    startInProgressRef.current = true;
    setSessionOperationBusy(true);
    try {
      await runStartTrial();
      if (sequenceActiveRef.current && trialStateRef.current !== 'RECORDING') {
        sequenceActiveRef.current = false;
        setSequenceActive(false);
        setSequenceCountdown(null);
        setSequenceStartPending(false);
        setTrialError(current => current || 'Chuỗi dừng: backend/session/config không cho phép bắt đầu trial tiếp theo.');
      }
    } catch (error) {
      const reason = error?.message || String(error);
      setSessionLifecycle({ status: 'unavailable', backendSessionId: knownBackendSessionRef.current, reason });
      setTrialError(`Không bắt đầu được Trial: ${reason}`);
    } finally {
      startInProgressRef.current = false;
      setSessionOperationBusy(false);
    }
  };
  handleStartTrialRef.current = handleStartTrial;

  useEffect(() => {
    if (!sequenceStartPending || !sequenceActiveRef.current) return undefined;
    sequenceStartTimerRef.current = setTimeout(() => {
      sequenceStartTimerRef.current = null;
      setSequenceStartPending(false);
      if (sequenceActiveRef.current) handleStartTrialRef.current?.();
    }, 0);
    return () => {
      if (sequenceStartTimerRef.current) clearTimeout(sequenceStartTimerRef.current);
      sequenceStartTimerRef.current = null;
    };
  }, [sequenceStartPending]);

  const handleStartSequence = () => {
    if (!canStartPilotSequence({ sequenceActive: sequenceActiveRef.current, isRecording, sessionOperationBusy, cameraActive, lifecycleAllowsStart })) return;
    sequenceActiveRef.current = true;
    sequenceRunIdRef.current = createPilotSequenceId();
    setSequenceActive(true);
    setSequenceCountdown(null);
    setSequenceResults([]);
    setCurrentRep(1);
    setLastTrialSummary(null);
    setTrialError(null);
    setSequenceStartPending(true);
  };

  const stopSequence = async (reason = 'user_stopped') => {
    sequenceActiveRef.current = false;
    setSequenceActive(false);
    setSequenceCountdown(null);
    setSequenceStartPending(false);
    clearInterval(sequenceTimerRef.current);
    sequenceTimerRef.current = null;
    clearTimeout(sequenceStartTimerRef.current);
    sequenceStartTimerRef.current = null;
    if (trialStateRef.current === 'RECORDING') {
      await handleEndTrial({ completed: false, reason: `sequence_${reason}` });
      setFlushStatus('Chuỗi đã dừng an toàn; trial hiện tại bị đánh dấu không hợp lệ và đã flush phần dữ liệu nhận được.');
      return;
    }
    try {
      const before = await fetchPilotStats();
      if (!pilotLoggingConfirmed(before, sessionIdRef.current)) throw new Error('Session logging không còn khớp; không thể xác minh flush.');
      const flush = await flushPilotLogs(sessionIdRef.current);
      const after = await fetchPilotStats();
      if (!pilotFlushVerified(before, flush, after, sessionIdRef.current)) throw new Error('Flush không được xác minh (pending/drop/session mismatch).');
      setFlushStatus('Chuỗi đã dừng; dữ liệu đã flush và được bảo toàn. Logger/session vẫn mở.');
    } catch (error) {
      setTrialError(`Đã dừng chuỗi, cần kiểm tra flush: ${error?.message || String(error)}`);
    }
  };

  const scheduleNextSequenceTrial = (nextRep) => {
    setCurrentRep(nextRep);
    setSequenceCountdown(3);
    let remaining = 3;
    clearInterval(sequenceTimerRef.current);
    sequenceTimerRef.current = setInterval(() => {
      if (!sequenceActiveRef.current) {
        clearInterval(sequenceTimerRef.current);
        sequenceTimerRef.current = null;
        return;
      }
      remaining -= 1;
      if (remaining <= 0) {
        clearInterval(sequenceTimerRef.current);
        sequenceTimerRef.current = null;
        setSequenceCountdown(null);
        handleStartTrialRef.current?.();
      } else {
        setSequenceCountdown(remaining);
      }
    }, 1000);
  };

  // Kết thúc một Trial (tự động hết giờ hoặc bấm Dừng sớm thủ công)
  const handleEndTrial = async ({ completed = true, isAuto = false, reason = null } = {}) => {
    if (trialStateRef.current === 'FINALIZING') return;
    // VÔ HIỆU HÓA METADATA NGAY LẬP TỨC Ở CẤP VI GIÂY (Task 7F.5D)
    trialStateRef.current = 'FINALIZING';
    setTrialState('FINALIZING');
    clearInterval(timerIntervalRef.current);
    setCurrentKinematicPhase('PRE_IDLE');

    // Dừng MediaRecorder và thu thập blob video
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      await stopMediaRecorder();
    }
    await waitForPendingPilotFrames();

    const totalF = metricsRef.current.total || frameIdxRef.current;
    const matchF = metricsRef.current.matchCount;
    const misF = metricsRef.current.mismatchCount;
    const neutF = metricsRef.current.neutralCount;
    const matchPct = totalF > 0 ? ((matchF / totalF) * 100).toFixed(1) : '0.0';
    const misPct = totalF > 0 ? ((misF / totalF) * 100).toFixed(1) : '0.0';
    const neutPct = totalF > 0 ? ((neutF / totalF) * 100).toFixed(1) : '0.0';
    const stableMetrics = summarizeStableFrames(stableFramesRef.current, selectedLetterRef.current);
    const trialConfirmedTarget = trialHasConfirmedTarget(trialResponsesRef.current, selectedLetterRef.current);

    // Flush ngay lập tức toàn bộ buffer trial xuống file JSONL
    let statsBeforeFlush = null;
    let flushRes = null;
    let statsAfterFlush = null;
    try { statsBeforeFlush = await fetchPilotStats(); } catch { loggingInterruptedRef.current = true; }
    try { flushRes = await flushPilotLogs(sessionIdRef.current); } catch { loggingInterruptedRef.current = true; }
    try { statsAfterFlush = await fetchPilotStats(); } catch { loggingInterruptedRef.current = true; }
    const flushVerified = pilotFlushVerified(statsBeforeFlush, flushRes, statsAfterFlush, sessionIdRef.current);
    const flushStats = flushRes?.stats;
    const flushDropped = (flushStats?.dropped_queue_full || 0) > loggingBaselineRef.current.dropped_queue_full ||
      (flushStats?.dropped_io_error || 0) > loggingBaselineRef.current.dropped_io_error;
    const queuedFrameCount = Math.max(0, (flushStats?.queued_count || 0) - loggingBaselineRef.current.queued_count);
    const loggedFrameCount = Math.max(0, (flushStats?.logged_count || 0) - loggingBaselineRef.current.logged_count);
    if (!flushVerified || !flushRes?.success || flushRes.session_id !== sessionIdRef.current || flushDropped) {
      loggingInterruptedRef.current = true;
      reason = reason || 'logging_or_record_accounting_invalid';
      setTrialError('Flush/session logging không xác minh được, còn record pending hoặc có record bị rơi; trial bị đánh dấu không hợp lệ.');
    }
    const validTrial = trialIsValid({
      completed,
      durationSec: trialDurationRef.current,
      loggingInterrupted: loggingInterruptedRef.current || !flushVerified || !flushRes?.success || flushRes.session_id !== sessionIdRef.current || flushDropped,
      stableHoldFrameCount: stableMetrics.stableFrameTotal,
      queuedFrameCount,
      loggedFrameCount,
      submittedFrameCount: frameIdxRef.current,
    });
    let flushedFile = logFilePath;
    if (flushRes && flushRes.success) {
      if (flushRes.stats) setBackendStats(flushRes.stats);
      if (flushRes.filepath) {
        setLogFilePath(flushRes.filepath);
        flushedFile = flushRes.filepath;
      }
    }

    const currentTrialId = trialIdRef.current;
    const isSequenceTrial = sequenceTrialIdsRef.current.has(currentTrialId);
    const currentTarget = selectedLetterRef.current;
    const videoFileName = isSequenceTrial ? `pilot_${currentTrialId}.webm` : `pilot_${sessionIdRef.current}_${currentTarget}_rep${String(currentRep).padStart(2, '0')}.webm`;
    const sidecarFileName = isSequenceTrial ? `pilot_${currentTrialId}_sync.json` : `pilot_${sessionIdRef.current}_${currentTarget}_rep${String(currentRep).padStart(2, '0')}_sync.json`;
    setVideoDownloadName(videoFileName);
    setSidecarDownloadName(sidecarFileName);

    // Xây dựng sidecar synchronization data (Task 3.10 Implementation C)
    const sidecarData = {
      schema_version: '2.1-kinematic-phase',
      session_id: sessionIdRef.current,
      trial_id: currentTrialId,
      signer_id: signerIdRef.current,
      ...pilotTargetFields(currentTarget),
      rep: currentRep,
      video_file: recordVideoRGBRef.current ? videoFileName : null,
      jsonl_file: flushedFile ? flushedFile.split(/[\/\\]/).pop() : `pilot_${sessionIdRef.current}.jsonl`,
      total_frames: totalF,
      phase_frame_counts: { ...phaseFrameCountsRef.current },
      stable_hold_metrics: stableMetrics,
      trial_valid: validTrial,
      sync_anomalies: { ...syncAnomalies },
      frame_sync_table: [...syncMappingRef.current]
    };
    setSyncSidecarData(sidecarData);

    try {
      const sidecarBlob = new Blob([JSON.stringify(sidecarData, null, 2)], { type: 'application/json' });
      const scUrl = URL.createObjectURL(sidecarBlob);
      setSyncSidecarUrl(scUrl);
      const sidecarSaveResult = await savePilotSidecarSync(sessionIdRef.current, sidecarData);
      const savedSidecarPath = sidecarSaveResult?.success ? sidecarSaveResult.sidecar_path : null;
      const savedSidecarName = savedSidecarPath ? savedSidecarPath.split(/[\/\\]/).pop() : null;

      const manifest = {
        schema_version: '2.1-kinematic-phase',
        session_id: sessionIdRef.current,
        signer_id: signerIdRef.current,
        created_at: new Date().toISOString(),
        trial_summary: {
          trialId: currentTrialId,
          targetLetter: currentTarget,
          rep: currentRep,
          durationSec: trialDurationRef.current,
          completed,
          trialValid: validTrial,
          stableFrameCorrect: stableMetrics.stableFrameCorrect,
          stableFrameTotal: stableMetrics.stableFrameTotal,
          stableFrameAccuracy: stableMetrics.stableFrameAccuracy,
          stableFalsePositiveCount: stableMetrics.stableFalsePositiveCount,
          trialCorrectConfirmed: trialConfirmedTarget,
          isAuto,
          reason,
          totalFrames: totalF,
          matchFrames: matchF,
          matchPct,
          mismatchFrames: misF,
          misPct,
          neutralFrames: neutF,
          neutPct,
          confusions: { ...metricsRef.current.confusions }
        },
        files: {
          jsonl: { name: sidecarData.jsonl_file },
          video: { name: videoFileName, storage: 'browser_local_download' },
          sidecar: {
            name: savedSidecarName || sidecarFileName,
            storage: 'backend',
            saved: Boolean(savedSidecarName),
            error: sidecarSaveResult?.success ? null : (sidecarSaveResult?.error || 'sidecar_save_unconfirmed')
          }
        }
      };
      await savePilotSessionManifest(sessionIdRef.current, manifest);
    } catch (e) {
      console.warn('Lỗi lưu sync/manifest:', e);
    }

    // Tạo bản tóm tắt Trial hoàn tất
    const summary = {
      trialId: currentTrialId,
      targetLetter: currentTarget,
      rep: currentRep,
      durationSec: trialDurationRef.current,
      completed,
      trialValid: validTrial,
      stableFrameCorrect: stableMetrics.stableFrameCorrect,
      stableFrameTotal: stableMetrics.stableFrameTotal,
      stableFrameAccuracy: stableMetrics.stableFrameAccuracy,
      stableFalsePositiveCount: stableMetrics.stableFalsePositiveCount,
      trialCorrectConfirmed: trialConfirmedTarget,
      isAuto,
      reason,
      totalFrames: totalF,
      matchFrames: matchF,
      matchPct,
      mismatchFrames: misF,
      misPct,
      neutralFrames: neutF,
      neutPct,
      confusions: { ...metricsRef.current.confusions },
      timestamp: new Date().toLocaleTimeString(),
      flushedFile: flushedFile ? flushedFile.split(/[\/\\]/).pop() : 'JSONL',
      videoFile: recordVideoRGBRef.current ? videoFileName : null,
      sequenceId: isSequenceTrial ? sequenceRunIdRef.current : null,
    };

    trialStateRef.current = cameraActive ? 'READY' : 'IDLE';
    setTrialState(cameraActive ? 'READY' : 'IDLE');
    setLastTrialSummary(summary);
    setResultArtifactTarget(currentTarget);
    setCompletedTrials(prev => [...prev, summary]);
    if (isSequenceTrial) {
      setSequenceResults(prev => [...prev, summary]);
      setSequenceResultsByTarget(prev => addPilotTargetResult(prev, summary));
    }

    if (completed && validTrial) {
      setFlushStatus(`Đã tự động hoàn thành trial, lưu ${totalF} frames vào JSONL và đồng bộ Video.`);
      // Tự động tăng repetition cho lần thử tiếp theo nếu chưa đạt tối đa
      if (!sequenceActiveRef.current && currentRep < totalReps) {
        setCurrentRep(prev => prev + 1);
      }
      if (pilotSequenceCanAdvance({ active: sequenceActiveRef.current, completed, valid: validTrial, repetition: currentRep })) {
        scheduleNextSequenceTrial(currentRep + 1);
      } else if (sequenceActiveRef.current && currentRep >= totalReps) {
        sequenceActiveRef.current = false;
        setSequenceActive(false);
        setSequenceCountdown(null);
        setSequenceStartPending(false);
        setFlushStatus(`Chuỗi 10 trial chữ ${selectedLetterRef.current} đã hoàn tất; toàn bộ trial đều hợp lệ và đã flush.`);
      }
    } else {
      setFlushStatus(`Trial không hợp lệ (${reason || (completed ? 'logging_or_flush_error' : 'dừng sớm')}). Không tính vào metrics; đã lưu bảo toàn ${totalF} frames nếu backend nhận được.`);
      if (sequenceActiveRef.current) {
        sequenceActiveRef.current = false;
        setSequenceActive(false);
        setSequenceCountdown(null);
        setSequenceStartPending(false);
        clearInterval(sequenceTimerRef.current);
        sequenceTimerRef.current = null;
        setTrialError(`Chuỗi dừng tại trial ${currentRep}/${PILOT_REPETITIONS}: ${reason || (completed ? 'trial_invalid_or_logging_error' : 'trial_stopped_early')}. Không tự bỏ qua trial.`);
      }
    }
  };
  handleEndTrialRef.current = handleEndTrial;

  // Hủy trial đang chạy mà không tính vào hoàn thành
  const handleCancelTrial = () => {
    trialStateRef.current = cameraActive ? 'READY' : 'IDLE';
    clearInterval(timerIntervalRef.current);
    setTrialState(cameraActive ? 'READY' : 'IDLE');
    setCurrentKinematicPhase('PRE_IDLE');
    if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
      try {
        mediaRecorderRef.current.stop();
      } catch (e) {}
    }
    frameIdxRef.current = 0;
    setTrialFrameCount(0);
    setFlushStatus('Trial đã bị hủy bỏ');
  };

  // Dọn dẹp an toàn khi đóng HUD giữa chừng (Bảo toàn dữ liệu đã ghi nhận)
  useEffect(() => {
    return () => {
      clearInterval(timerIntervalRef.current);
      clearInterval(sequenceTimerRef.current);
      clearTimeout(sequenceStartTimerRef.current);
      if (mediaRecorderRef.current && mediaRecorderRef.current.state !== 'inactive') {
        try {
          mediaRecorderRef.current.stop();
        } catch (e) {}
      }
      if (trialStateRef.current === 'RECORDING') {
        flushPilotLogs(sessionIdRef.current);
      }
    };
  }, []);

  // Chọn chữ cái mới
  const handleSelectLetter = (letter) => {
    if (!VSL_ALPHABET_29.includes(letter) || !canChangePilotTarget({
      trialState: trialStateRef.current,
      sequenceActive: sequenceActiveRef.current,
      sequenceCountdown,
      sessionOperationBusy,
      sessionStatus: sessionLifecycle.status,
    })) return;
    sequenceRunIdRef.current = null;
    setSelectedLetter(letter);
    setCurrentRep(1);
    setSequenceResults([]);
    setTrialState(cameraActive ? 'READY' : 'IDLE');
    const targetHistory = latestPilotSequenceForTarget(sequenceResultsByTarget, letter);
    setLastTrialSummary(targetHistory[targetHistory.length - 1] || null);
    setTrialError(null);
    setFlushStatus(null);
  };

  // Kết thúc toàn bộ Session và Flush dữ liệu
  const handleFinishSession = async () => {
    if (sessionOperationBusy) return;
    setSessionOperationBusy(true);
    try {
    if (trialState === 'RECORDING') {
      await handleEndTrial({ completed: false, reason: 'session_closed' });
    }
    const currentStats = await fetchPilotStats();
    if (!currentStats?.success) {
      setSessionLifecycle({ status: 'unavailable', backendSessionId: null, reason: currentStats?.error || 'Backend stats unavailable.' });
      setTrialError(`Không thể đóng session: ${currentStats?.error || 'backend không phản hồi'}`);
      return;
    }
    let closedFile = logFilePath;
    let finalStats = currentStats.stats;
    if (currentStats.pilot_logging_enabled) {
      const closingId = sessionLifecycle.backendSessionId || currentStats.session_id;
      const closed = await flushAndStopPilotSession(closingId);
      if (!closed.success) {
        setTrialError(closed.error);
        await refreshBackendSession();
        return;
      }
      closedFile = closed.filepath;
      finalStats = closed.stats;
    }
    knownBackendSessionRef.current = null;
    const nextSessionId = createPilotSessionId();
    setSessionId(nextSessionId);
    setLoggingEnabled(false);
    setFlushStatus(`Session đã flush/đóng an toàn: ${closedFile || 'không có file đang mở'}.`);
    if (finalStats) setBackendStats(finalStats);
    const [nextStats, nextConfig] = await Promise.all([fetchPilotStats(), fetchPilotRuntimeConfig()]);
    if (!nextStats?.success) {
      setSessionLifecycle({ status: 'unavailable', backendSessionId: null, reason: nextStats?.error || 'Backend unavailable after closing session.' });
      setTrialError(`Session cũ đã đóng; chưa xác minh được session tiếp theo: ${nextStats?.error || 'backend offline'}`);
    } else if (nextStats.pilot_logging_enabled) {
      setSessionLifecycle({ status: 'session_mismatch', backendSessionId: nextStats.session_id || null, reason: 'Một session backend khác đã được bật.' });
      setTrialError(`Session cũ đã đóng; backend hiện đang giữ session '${nextStats.session_id}'. Trial bị khóa cho đến khi xác minh session.`);
    } else {
      const nextReady = candidateDConfigMatches(nextConfig) && nextStats.has_static_head === true;
      setSessionLifecycle({ status: nextReady ? 'ready' : candidateDConfigMatches(nextConfig) ? 'runtime_invalid' : 'config_invalid', backendSessionId: null, reason: nextReady ? null : nextConfig?.error || 'Candidate-D/static head chưa sẵn sàng.' });
    }
    alert(`Đã đóng session an toàn. File: ${closedFile || 'không có file đang mở'}\nRecord đã ghi: ${finalStats?.logged_count ?? 0}\nRecord thất thoát: ${finalStats?.total_dropped ?? 0}\nSession tiếp theo: ${nextSessionId}`);
    setTrialState(cameraActive ? 'READY' : 'IDLE');
    } catch (error) {
      setTrialError(`Không hoàn tất được việc đóng session: ${error?.message || String(error)}`);
    } finally {
      setSessionOperationBusy(false);
    }
  };

  if (!isActive) {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '-8px' }}>
        <button
          onClick={onToggleActive}
          className="btn btn-secondary"
          style={{
            fontSize: '11px',
            padding: '5px 12px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            background: 'rgba(30, 41, 59, 0.85)',
            borderColor: 'rgba(56, 189, 248, 0.4)',
            color: '#38bdf8',
            fontWeight: '600'
          }}
          title="Bật giao diện kiểm thử Controlled Webcam Pilot 29 Chữ Cái VSL"
        >
          <FlaskConical size={14} style={{ color: '#38bdf8' }} />
          Mở Pilot Testing HUD (29 Chữ)
        </button>
      </div>
    );
  }

  const isRecording = trialState === 'RECORDING';
  const lifecycleAllowsStart = sessionLifecycle.status === 'ready' || sessionLifecycle.status === 'active';
  const isReady = trialState === 'READY' && lifecycleAllowsStart && !sessionOperationBusy;

  // Tính phần trăm đếm ngược cho Progress Bar
  const progressPct = trialDurationSec > 0
    ? Math.min(100, Math.max(0, ((trialDurationSec - countdownSeconds) / trialDurationSec) * 100))
    : 100;

  const phaseText = `${currentKinematicPhase}: ${PHASE_LABELS_VI[currentKinematicPhase] || 'Đang chờ'}`;
  const targetRunResults = sequenceResults.filter(item => item.targetLetter === selectedLetter);
  const targetHistory = latestPilotSequenceForTarget(sequenceResultsByTarget, selectedLetter);
  const displayedSequenceResults = sequenceActive ? targetRunResults : targetRunResults.length ? targetRunResults : targetHistory;
  const sequenceComplete = sequenceResults.length === PILOT_REPETITIONS && sequenceResults.every(item => item.trialValid);
  const lastOutcome = pilotTrialOutcome(lastTrialSummary);
  const hudStatus = pilotHudStatus({ sessionStatus: sessionLifecycle.status, trialState, sequenceActive, sequenceCountdown, sequenceComplete, lastTrialSummary, lifecycleAllowsStart: lifecycleAllowsStart || isReady });
  const statusError = hudStatus === 'ERROR';
  const timerValue = isRecording ? Math.ceil(countdownSeconds)
    : sequenceCountdown !== null ? sequenceCountdown
    : trialState === 'FINALIZING' || lastTrialSummary ? '—' : 6;
  const sequenceMetrics = summarizeSessionTrials(displayedSequenceResults);
  const invalidCount = displayedSequenceResults.filter(item => !item.trialValid).length;
  const trialCards = Array.from({ length: PILOT_REPETITIONS }, (_, index) => {
    const rep = index + 1;
    return displayedSequenceResults.find(item => item.rep === rep) || null;
  });

  return (
    <div style={{ background: 'linear-gradient(145deg, #0f172a, #172554)', border: '1px solid rgba(96,165,250,.28)', borderRadius: 12, padding: 'clamp(12px, 2vw, 22px)', display: 'grid', gap: 16, color: '#e2e8f0', boxShadow: '0 12px 36px rgba(0,0,0,.28)', width: '100%', boxSizing: 'border-box' }}>
      <header style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, borderBottom: '1px solid rgba(255,255,255,.1)', paddingBottom: 12 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
          <FlaskConical size={22} aria-hidden="true" />
          <div>
            <h2 style={{ fontSize: 18, margin: 0 }}>Controlled Pilot — Chữ {selectedLetter}</h2>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, marginTop: 6, fontSize: 12 }}>
              <span style={{ padding: '3px 8px', borderRadius: 20, background: '#1e3a8a' }}>Candidate D</span>
              <span aria-label={`Trạng thái ${hudStatus}`} style={{ padding: '3px 8px', borderRadius: 20, background: statusError || hudStatus === 'INVALID' ? '#7f1d1d' : hudStatus === 'COMPLETED' ? '#14532d' : '#334155' }}>
                Trạng thái: {hudStatus}
              </span>
              <span aria-live="polite">{sessionLifecycle.status === 'active' ? 'Session backend khớp' : sessionLifecycle.status === 'ready' ? 'Backend sẵn sàng' : sessionLifecycle.status === 'checking' ? 'Đang kiểm tra kết nối' : 'Backend cần kiểm tra'}</span>
            </div>
          </div>
        </div>
        <button type="button" onClick={onToggleActive} className="btn btn-secondary" aria-label="Đóng HUD">Đóng HUD</button>
      </header>

      {(trialError || (lastTrialSummary && !lastTrialSummary.trialValid)) && (
        <div role="alert" style={{ border: '1px solid #f87171', background: 'rgba(127,29,29,.35)', borderRadius: 8, padding: 10, display: 'flex', gap: 8, alignItems: 'flex-start' }}>
          <AlertTriangle size={18} aria-hidden="true" />
          <div><strong>Trial INVALID — dữ liệu không được tính là PASS.</strong><div style={{ fontSize: 12, marginTop: 3 }}>{trialError || lastTrialSummary?.reason || 'Không đạt điều kiện validity; chuỗi đã dừng.'}</div></div>
        </div>
      )}

      <section aria-labelledby="pilot-control-heading" style={{ display: 'grid', gap: 12, background: 'rgba(2,6,23,.38)', borderRadius: 10, padding: 'clamp(12px, 2vw, 20px)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
          <div><h3 id="pilot-control-heading" style={{ fontSize: 15, margin: 0 }}>Kiểm thử có kiểm soát</h3><div style={{ fontSize: 12, color: '#cbd5e1', marginTop: 4 }}>Target {selectedLetter} · 10 lần · 6 giây mỗi trial · 5 pha · Stable Hold 1.5–4.5 giây</div></div>
          <div aria-label={`Tiến độ ${displayedSequenceResults.length} trên ${PILOT_REPETITIONS}`} style={{ fontSize: 14, fontWeight: 700 }}>Tiến độ {displayedSequenceResults.length}/{PILOT_REPETITIONS}</div>
        </div>

        <div aria-label="Chọn chữ cái VSL" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(38px, 1fr))', gap: 6 }}>
          {VSL_ALPHABET_29.map(letter => <button
            key={letter}
            type="button"
            aria-pressed={selectedLetter === letter}
            aria-label={`Chọn chữ ${letter}`}
            onClick={() => handleSelectLetter(letter)}
            disabled={!canChangePilotTarget({ trialState, sequenceActive, sequenceCountdown, sessionOperationBusy, sessionStatus: sessionLifecycle.status })}
            style={{ minHeight: 38, borderRadius: 7, border: selectedLetter === letter ? '2px solid #7dd3fc' : '1px solid #475569', background: selectedLetter === letter ? '#075985' : '#1e293b', color: '#f8fafc', fontWeight: selectedLetter === letter ? 800 : 600, cursor: 'pointer' }}
          >{letter}</button>)}
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12, alignItems: 'stretch' }}>
          <div style={{ display: 'grid', placeItems: 'center', minHeight: 112, borderRadius: 10, background: '#020617', border: '1px solid #334155' }}>
            <div style={{ color: '#94a3b8', fontSize: 11 }}>THỜI GIAN TRIAL</div>
            <div aria-live="off" style={{ fontSize: 'clamp(40px, 8vw, 62px)', lineHeight: 1.1, fontVariantNumeric: 'tabular-nums', fontWeight: 800 }}>{timerValue}{typeof timerValue === 'number' && <span style={{ fontSize: 20, marginLeft: 6 }}>giây</span>}</div>
            <div style={{ fontSize: 12 }}>{isRecording ? `Đang ghi Trial ${currentRep}/10` : sequenceCountdown !== null ? 'Chuẩn bị — chưa ghi dữ liệu' : sequenceActive ? 'Đang kiểm tra trial kế tiếp' : 'Sẵn sàng khi backend hợp lệ'}</div>
          </div>

          <div style={{ display: 'grid', alignContent: 'center', gap: 8, borderRadius: 10, background: 'rgba(15,23,42,.8)', padding: 14 }}>
            <div style={{ fontSize: 11, color: '#94a3b8' }}>PHA HIỆN TẠI</div>
            <div aria-live="polite" style={{ fontSize: 19, fontWeight: 700 }}>{isRecording ? phaseText : sequenceCountdown !== null ? 'Chuẩn bị trial tiếp theo' : `${currentKinematicPhase}: ${PHASE_LABELS_VI[currentKinematicPhase] || 'Chờ bắt đầu'}`}</div>
            <div style={{ height: 8, background: '#334155', borderRadius: 8, overflow: 'hidden' }}><div style={{ height: '100%', width: `${progressPct}%`, background: '#38bdf8', transition: 'width .2s ease' }} /></div>
            <div style={{ display: 'flex', justifyContent: 'space-between', gap: 4, flexWrap: 'wrap', fontSize: 10, color: '#cbd5e1' }}>
              {Object.entries(PHASE_LABELS_VI).map(([phase, label]) => <span key={phase} aria-current={isRecording && currentKinematicPhase === phase ? 'step' : undefined} style={{ fontWeight: isRecording && currentKinematicPhase === phase ? 800 : 400 }}>{label}</span>)}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'center' }}>
          {!sequenceActive ? <button type="button" onClick={handleStartSequence} disabled={!canStartPilotSequence({ sequenceActive, isRecording, sessionOperationBusy: sessionOperationBusy || trialState === 'FINALIZING', cameraActive, lifecycleAllowsStart })} className="btn btn-primary" style={{ minHeight: 46, padding: '10px 20px', fontWeight: 800, fontSize: 15, opacity: cameraActive && lifecycleAllowsStart && !sessionOperationBusy ? 1 : .55 }}><Play size={17} /> Bắt đầu kiểm thử 10 lần</button>
            : <button type="button" onClick={() => stopSequence('user_stopped')} disabled={trialState === 'FINALIZING'} className="btn btn-danger" style={{ minHeight: 46, padding: '10px 18px', fontWeight: 800 }}><Square size={16} /> Dừng an toàn</button>}
          <span style={{ fontSize: 12 }}>{sequenceActive ? `Chữ ${selectedLetter} · Trial ${Math.min(currentRep, PILOT_REPETITIONS)} / ${PILOT_REPETITIONS}` : `Mục tiêu: chữ ${selectedLetter}`}</span>
          {sequenceCountdown !== null && <span role="status" style={{ fontSize: 13, fontWeight: 700 }}>Chuẩn bị trong {sequenceCountdown} giây; countdown không thuộc trial.</span>}
        </div>
        {trialError && <div style={{ color: '#fecaca', fontSize: 12 }}>Cần chú ý: {trialError}</div>}
        {!cameraActive && <div role="status" style={{ color: '#fde68a', fontSize: 12 }}>Bật camera trước khi bắt đầu. Không có trial tự khởi chạy.</div>}
      </section>

      <section aria-labelledby="pilot-results-heading" style={{ display: 'grid', gap: 12 }}>
        <h3 id="pilot-results-heading" style={{ fontSize: 15, margin: 0 }}>Kết quả chữ {selectedLetter}</h3>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(145px, 1fr))', gap: 8 }}>
          <MetricCard label="Stable Frame Accuracy" value={`${sequenceMetrics.stableFrameCorrect}/${sequenceMetrics.stableFrameTotal}`} detail={sequenceMetrics.stableFrameAccuracy === null ? 'Chưa có dữ liệu' : `${(100 * sequenceMetrics.stableFrameAccuracy).toFixed(1)}%`} />
          <MetricCard label="Trial Accuracy" value={`${sequenceMetrics.trialCorrect}/${sequenceMetrics.trialAccuracyDenominator}`} detail="Đúng / trial hợp lệ" />
          <MetricCard label="Trial hợp lệ" value={`${sequenceMetrics.validTrialCount}/10`} detail={`Invalid: ${invalidCount}`} />
          <MetricCard label="False Positive" value={`${sequenceMetrics.stableFalsePositiveCount}`} detail="Trong Stable Hold" />
        </div>
        <div aria-label="Trạng thái từng trial" style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(112px, 1fr))', gap: 7 }}>
          {trialCards.map((summary, index) => {
            const rep = index + 1;
            const outcome = summary ? pilotTrialOutcome(summary) : null;
            const current = sequenceActive && currentRep === rep && !summary;
            return <div key={rep} style={{ border: `1px solid ${summary?.trialValid ? '#22c55e' : summary ? '#f87171' : current ? '#38bdf8' : '#475569'}`, borderRadius: 8, padding: 8, minHeight: 56, background: 'rgba(2,6,23,.3)' }}>
              <strong>Trial {rep}</strong><div style={{ fontSize: 11, marginTop: 5 }}>{summary ? `${outcome.duration === 'COMPLETED' ? '6 giây hoàn tất' : 'Dừng sớm'} · ${outcome.status}` : current ? (isRecording ? 'Đang chạy' : 'Đang chuẩn bị') : 'Chưa chạy'}</div>
              {summary && !summary.trialValid && <div style={{ fontSize: 10, color: '#fecaca', marginTop: 3 }}>{summary.reason || 'Không đạt điều kiện validity'}</div>}
            </div>;
          })}
        </div>
      </section>

      <details style={{ borderTop: '1px solid rgba(255,255,255,.12)', paddingTop: 10 }}>
        <summary style={{ cursor: 'pointer', fontWeight: 700, padding: '6px 0' }}>Chi tiết nâng cao — session, logging và diagnostics</summary>
        <div style={{ display: 'grid', gap: 12, paddingTop: 12, fontSize: 12 }}>
          <div role="status" aria-live="polite" style={{ padding: 10, borderRadius: 8, background: statusError ? 'rgba(127,29,29,.35)' : 'rgba(15,23,42,.8)', border: '1px solid #475569' }}>
            <strong>Backend/session: {sessionLifecycle.status}</strong><div>{sessionLifecycle.reason || (sessionLifecycle.status === 'active' ? `Session khớp: ${sessionLifecycle.backendSessionId}` : sessionLifecycle.status === 'ready' ? 'Candidate D đã xác minh; logging sẽ được bật khi bắt đầu.' : '')}</div>
            <div style={{ overflowWrap: 'anywhere', marginTop: 4 }}>Session ID: <code>{sessionId}</code> · Trial ID hiện tại: <code>{trialId}</code></div>
            <div style={{ overflowWrap: 'anywhere' }}>JSONL: <code>{logFilePath || sessionLifecycle.filepath || 'Chưa có file'}</code></div>
            <div>Runtime: Candidate D · O_D_TI_MAX 0.23 · O_IDX_STRAIGHT_MAX 0.65 · Static head {hasStaticHead ? 'loaded' : 'chưa xác minh'}</div>
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, alignItems: 'center' }}>
            <label>Signer <select value={signerId} onChange={event => setSignerId(event.target.value)} disabled={isRecording || sequenceActive}>{['signer_01', 'signer_02', 'signer_03', 'signer_04', 'signer_05', 'signer_06', 'signer_07', 'signer_08'].map(item => <option key={item}>{item}</option>)}</select></label>
            <label style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}><input type="checkbox" checked={recordVideoRGB} onChange={event => setRecordVideoRGB(event.target.checked)} disabled={isRecording || sequenceActive || !mediaRecorderSupported} /> Ghi video RGB cục bộ</label>
            {videoRecordError && <span role="alert">Video: {videoRecordError}</span>}
            <button type="button" onClick={handleFinishSession} disabled={isRecording || trialState === 'FINALIZING' || sequenceActive || sessionOperationBusy} className="btn btn-secondary"><Save size={14} /> Flush và đóng session an toàn</button>
            {sessionLifecycle.backendSessionId && <button type="button" onClick={handleRotatePilotSession} disabled={isRecording || trialState === 'FINALIZING' || sequenceActive || sessionOperationBusy} className="btn btn-secondary"><RotateCcw size={14} /> Đóng session & tạo session mới</button>}
          </div>

          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 14 }}>
            <span>Frames trial: {trialFrameCount}</span><span>Queued: {backendStats.queued_count || 0}</span><span>Logged: {backendStats.logged_count || 0}</span><span>Dropped: {backendStats.total_dropped || 0}</span>
            <span>Stable Hold frames: {lastTrialSummary?.stableFrameTotal ?? 0}</span><span>Phase: {currentKinematicPhase}</span><span>Prediction: {realtimeMetrics.latestSign} ({(realtimeMetrics.latestConfidence * 100).toFixed(0)}%) · {realtimeMetrics.latestSource}</span>
          </div>
          {lastTrialSummary && <div style={{ display: 'grid', gap: 5, border: '1px solid #475569', borderRadius: 8, padding: 10 }}>
            <strong>Kết quả lần gần nhất: {lastOutcome.duration === 'COMPLETED' ? 'đã chạy đủ 6 giây' : 'dừng trước 6 giây'} · {lastOutcome.status}</strong>
            <span>Trial ID: <code>{lastTrialSummary.trialId}</code> · {lastTrialSummary.totalFrames} frame · correct {lastTrialSummary.matchFrames} · neutral {lastTrialSummary.neutralFrames} · mismatch {lastTrialSummary.mismatchFrames}</span>
            <span>Stable Hold: {lastTrialSummary.stableFrameCorrect}/{lastTrialSummary.stableFrameTotal} · false positive {lastTrialSummary.stableFalsePositiveCount} · lý do: {lastTrialSummary.reason || (lastTrialSummary.trialValid ? 'Không có' : 'Không đạt điều kiện logging/record accounting')}</span>
            <span>Phase counts: {Object.entries(lastTrialSummary.phaseFrameCounts || {}).map(([phase, count]) => `${phase} ${count}`).join(' · ') || 'Chưa có'}</span>
            <span>Prediction distribution: {Object.entries(lastTrialSummary.confusions || {}).map(([label, count]) => `${label} ${count}`).join(' · ') || 'Không có nhãn sai'}</span>
            {resultArtifactTarget === selectedLetter && videoBlobUrl && <a href={videoBlobUrl} download={videoDownloadName || 'pilot_trial_video.webm'}>Tải video trial chữ {selectedLetter}</a>}
            {resultArtifactTarget === selectedLetter && syncSidecarUrl && <a href={syncSidecarUrl} download={sidecarDownloadName || 'pilot_trial_sync.json'}>Tải sidecar đồng bộ chữ {selectedLetter}</a>}
          </div>}
          {(sequenceResultsByTarget[selectedLetter] || []).length > 0 && <div style={{ display: 'grid', gap: 4 }}><strong>Lịch sử trial chữ {selectedLetter}</strong>{(sequenceResultsByTarget[selectedLetter] || []).map(item => <div key={item.trialId} style={{ overflowWrap: 'anywhere' }}>{item.trialId}: {pilotTrialOutcome(item).duration === 'COMPLETED' ? '6s hoàn tất' : 'dừng sớm'} / {item.trialValid ? 'VALID' : `INVALID — ${item.reason || 'lý do validity trong diagnostics'}`}</div>)}</div>}
          {flushStatus && <div role="status">{flushStatus}</div>}
        </div>
      </details>
    </div>
  );
}
