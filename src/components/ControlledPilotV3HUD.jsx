import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  Play, Pause, RotateCcw, SkipForward, Square, Download,
  CheckCircle2, XCircle, AlertTriangle, RefreshCw, Eye, EyeOff,
  Activity, Video, ChevronRight, Layers, FileSpreadsheet, ShieldAlert
} from 'lucide-react';
import {
  createPilotV3Session,
  startPilotV3Session,
  pausePilotV3Session,
  resumePilotV3Session,
  stopPilotV3Session,
  getPilotV3Status,
  recoverPilotV3Session,
  startPilotV3Trial,
  startPilotV3Countdown,
  startPilotV3Record,
  sendPilotV3Frame,
  finishPilotV3Trial,
  retryPilotV3Trial,
  skipPilotV3Trial,
  getPilotV3Summary,
  exportPilotV3Session,
} from '../services/api';

export const VIETNAMESE_ALPHABET_29 = [
  'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 'H', 'I', 'K', 'L',
  'M', 'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y'
];

export default function ControlledPilotV3HUD({
  isActive = true,
  onToggleActive = null,
  cameraActive = false,
  latestLandmarks = null,
  videoElementRef = null,
  onFrameIntercept = null,
}) {
  // Session & Trial State (Synced with Backend Single Source of Truth)
  const [sessionState, setSessionState] = useState('CREATED'); // CREATED, READY, RUNNING, PAUSED, COMPLETED, STOPPED
  const [sessionId, setSessionId] = useState('');
  const [sessionMode, setSessionMode] = useState('FULL_29'); // FULL_29 | SINGLE_LETTER
  const [selectedSingleLetter, setSelectedSingleLetter] = useState('A');
  const [currentLetter, setCurrentLetter] = useState('A');
  const [currentTrialNumber, setCurrentTrialNumber] = useState(1);
  const [letterAccepted, setLetterAccepted] = useState(0);
  const [totalAccepted, setTotalAccepted] = useState(0);
  const [totalTarget, setTotalTarget] = useState(290);

  // Trial lifecycle
  const [trialState, setTrialState] = useState('PENDING'); // PENDING, READY, COUNTDOWN, RECORDING, SAVING, COMPLETED, INTERRUPTED, FAILED, SKIPPED
  const [trialId, setTrialId] = useState(null);
  const [attemptId, setAttemptId] = useState(null);
  const [countdownRemaining, setCountdownRemaining] = useState(3);
  const [recordingElapsed, setRecordingElapsed] = useState(0.0);

  // Diagnostic live feedback from recognition adapter
  const [latestDiag, setLatestDiag] = useState(null);
  const [lastTrialSummary, setLastTrialSummary] = useState(null);
  const [showSummaryTable, setShowSummaryTable] = useState(false);
  const [sessionSummary, setSessionSummary] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);
  const [recoverInput, setRecoverInput] = useState('');
  const [showRecoverBox, setShowRecoverBox] = useState(false);

  // Internal timer refs
  const countdownTimerRef = useRef(null);
  const syncIntervalRef = useRef(null);
  const recordingStartTsRef = useRef(null);

  // Fetch status from backend SSOT
  const refreshStatus = useCallback(async () => {
    try {
      const resp = await getPilotV3Status();
      if (resp && resp.success && resp.data) {
        const d = resp.data;
        setSessionState(d.state);
        setSessionId(d.session_id || '');
        setSessionMode(d.mode);
        if (d.current_letter) setCurrentLetter(d.current_letter);
        setCurrentTrialNumber(d.current_trial_number || 1);
        setLetterAccepted(d.letter_accepted_trials || 0);
        setTotalAccepted(d.total_accepted_trials || 0);
        setTotalTarget(d.total_target_accepted || (d.mode === 'SINGLE_LETTER' ? 10 : 290));
        setTrialState(d.trial_state);
        setTrialId(d.trial_id);
        setAttemptId(d.attempt_id);
        setRecordingElapsed(d.recording_elapsed_sec || 0.0);
      }
    } catch (err) {
      // Backend not yet initialized or network error
    }
  }, []);

  // Sync with backend every 1000ms while running
  useEffect(() => {
    refreshStatus();
    syncIntervalRef.current = setInterval(refreshStatus, 1000);
    return () => clearInterval(syncIntervalRef.current);
  }, [refreshStatus]);

  // Clean up countdown on unmount
  useEffect(() => {
    return () => {
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
    };
  }, []);

  // 1. Create & Start Session
  const handleCreateSession = async () => {
    setErrorMessage(null);
    try {
      await createPilotV3Session({
        mode: sessionMode,
        single_letter: sessionMode === 'SINGLE_LETTER' ? selectedSingleLetter : null,
      });
      await startPilotV3Session();
      await refreshStatus();
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // 2. Prepare Trial
  const handleStartTrial = async () => {
    setErrorMessage(null);
    setLastTrialSummary(null);
    try {
      await startPilotV3Trial();
      // Start 3-second countdown
      await startPilotV3Countdown();
      setCountdownRemaining(3);

      let count = 3;
      if (countdownTimerRef.current) clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = setInterval(async () => {
        count -= 1;
        setCountdownRemaining(count);
        if (count <= 0) {
          clearInterval(countdownTimerRef.current);
          countdownTimerRef.current = null;
          // Trigger recording on backend
          await startPilotV3Record();
          recordingStartTsRef.current = performance.now();
          await refreshStatus();
        }
      }, 1000);

      await refreshStatus();
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // 3. Pause Session
  const handlePauseSession = async () => {
    setErrorMessage(null);
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    try {
      await pausePilotV3Session();
      await refreshStatus();
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // 4. Resume Session
  const handleResumeSession = async () => {
    setErrorMessage(null);
    try {
      await resumePilotV3Session();
      await refreshStatus();
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // 5. Stop Session
  const handleStopSession = async () => {
    setErrorMessage(null);
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    try {
      await stopPilotV3Session('User clicked Stop');
      await refreshStatus();
      handleViewSummary();
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // 6. Retry Trial (creates new attempt)
  const handleRetryTrial = async () => {
    setErrorMessage(null);
    setLastTrialSummary(null);
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    try {
      await retryPilotV3Trial('User requested retry');
      await handleStartTrial();
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // 7. Skip Trial
  const handleSkipTrial = async () => {
    setErrorMessage(null);
    setLastTrialSummary(null);
    if (countdownTimerRef.current) {
      clearInterval(countdownTimerRef.current);
      countdownTimerRef.current = null;
    }
    try {
      await skipPilotV3Trial('User skipped');
      await refreshStatus();
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // 8. Accept Trial & Next
  const handleAcceptTrial = async () => {
    setErrorMessage(null);
    setLastTrialSummary(null);
    try {
      // Backend already accepted if 5.0s completed, or user confirms accept
      await refreshStatus();
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // 9. Recover Session from Disk
  const handleRecoverSession = async () => {
    if (!recoverInput.trim()) return;
    setErrorMessage(null);
    try {
      await recoverPilotV3Session(recoverInput.trim());
      setShowRecoverBox(false);
      await refreshStatus();
    } catch (err) {
      setErrorMessage(`Recovery failed: ${err.message}`);
    }
  };

  // 10. View & Export Summary
  const handleViewSummary = async () => {
    try {
      const resp = await getPilotV3Summary();
      if (resp && resp.data) {
        setSessionSummary(resp.data);
        setShowSummaryTable(true);
      }
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  const handleExportJSON = async () => {
    try {
      const resp = await exportPilotV3Session();
      const blob = new Blob([JSON.stringify(resp, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `controlled_pilot_v3_${sessionId || 'export'}.json`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      setErrorMessage(err.message);
    }
  };

  // Process incoming frame from parent camera recognizer
  const processFrame = useCallback(async (landmarks, trackingMeta = {}) => {
    if (sessionState !== 'RUNNING') return;
    try {
      const resp = await sendPilotV3Frame({
        landmarks,
        handedness: trackingMeta.handedness || 'Right',
        tracking_confidence: trackingMeta.tracking_confidence,
        handedness_score: trackingMeta.handedness_score,
        client_capture_ts: performance.now() / 1000.0,
      });

      if (resp && resp.success && resp.data) {
        setLatestDiag(resp.data);

        // If backend finished trial (elapsed >= 5.0s)
        if (resp.data.trial_state === 'COMPLETED' || resp.data.completion_status === 'COMPLETED') {
          setLastTrialSummary(resp.data);
          refreshStatus();
        }
      }
    } catch (err) {
      // Non-blocking frame transmission error
    }
  }, [sessionState, refreshStatus]);

  // Hook frame intercept if parent provided one
  useEffect(() => {
    if (onFrameIntercept) {
      onFrameIntercept(processFrame);
    }
  }, [onFrameIntercept, processFrame]);

  if (!isActive) return null;

  const isRecording = trialState === 'RECORDING';
  const isCountdown = trialState === 'COUNTDOWN';
  const isRunning = sessionState === 'RUNNING';
  const isPaused = sessionState === 'PAUSED';

  const predictedLetter = latestDiag?.final_prediction || '—';
  const confidencePct = Math.round((latestDiag?.final_confidence || 0) * 100);
  const handStatus = latestDiag?.hand_tracking?.tracking_status || 'NO_HAND';
  const isHandDetected = handStatus === 'HAND_DETECTED';

  // Compute progress percentage
  const progressPct = totalTarget > 0 ? Math.min(100, Math.round((totalAccepted / totalTarget) * 100)) : 0;
  const remainingTrials = Math.max(0, totalTarget - totalAccepted);

  return (
    <div style={{
      width: '100%',
      background: 'rgba(10, 15, 26, 0.95)',
      borderRadius: 'var(--radius-lg, 16px)',
      border: '1px solid rgba(0, 240, 255, 0.25)',
      boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5), inset 0 0 20px rgba(0, 240, 255, 0.05)',
      padding: '20px',
      color: '#e2e8f0',
      fontFamily: 'Inter, system-ui, sans-serif',
      marginBottom: '20px',
    }}>
      {/* HEADER SECTION */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
        paddingBottom: '14px',
        marginBottom: '16px',
        flexWrap: 'wrap',
        gap: '12px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            width: '36px',
            height: '36px',
            borderRadius: '10px',
            background: 'linear-gradient(135deg, #00f0ff 0%, #7000ff 100%)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            boxShadow: '0 0 15px rgba(0, 240, 255, 0.4)',
          }}>
            <Activity size={20} color="#fff" />
          </div>
          <div>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <h2 style={{ fontSize: '18px', fontWeight: '800', margin: 0, letterSpacing: '-0.01em' }}>
                Controlled Pilot <span style={{ color: '#00f0ff' }}>V3</span>
              </h2>
              <span style={{
                fontSize: '11px',
                fontWeight: '700',
                padding: '2px 8px',
                borderRadius: '6px',
                background: sessionState === 'RUNNING' ? 'rgba(0, 255, 128, 0.15)' : 'rgba(255, 255, 255, 0.08)',
                color: sessionState === 'RUNNING' ? '#00ff80' : '#94a3b8',
                border: `1px solid ${sessionState === 'RUNNING' ? '#00ff80' : 'rgba(255, 255, 255, 0.15)'}`,
              }}>
                SESSION: {sessionState}
              </span>
            </div>
            <p style={{ fontSize: '12px', color: '#94a3b8', margin: 0 }}>
              Công cụ kiểm thử 29 chữ cái tiếng Việt có kiểm soát (5.0s / trial, 10 trials / chữ)
            </p>
          </div>
        </div>

        {/* Global Action buttons */}
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            onClick={() => setShowRecoverBox(prev => !prev)}
            style={{
              padding: '6px 12px',
              borderRadius: '8px',
              fontSize: '12px',
              fontWeight: '600',
              background: 'rgba(255, 255, 255, 0.06)',
              color: '#94a3b8',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <RefreshCw size={14} /> Khôi phục phiên
          </button>

          <button
            onClick={handleViewSummary}
            style={{
              padding: '6px 12px',
              borderRadius: '8px',
              fontSize: '12px',
              fontWeight: '600',
              background: 'rgba(0, 240, 255, 0.12)',
              color: '#00f0ff',
              border: '1px solid rgba(0, 240, 255, 0.3)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <FileSpreadsheet size={14} /> Báo cáo tổng kết
          </button>

          <button
            onClick={handleExportJSON}
            style={{
              padding: '6px 12px',
              borderRadius: '8px',
              fontSize: '12px',
              fontWeight: '600',
              background: 'rgba(112, 0, 255, 0.15)',
              color: '#c084fc',
              border: '1px solid rgba(112, 0, 255, 0.35)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <Download size={14} /> Export JSON
          </button>

          {onToggleActive && (
            <button
              onClick={onToggleActive}
              style={{
                padding: '6px 10px',
                borderRadius: '8px',
                background: 'transparent',
                color: '#64748b',
                border: 'none',
                cursor: 'pointer'
              }}
              title="Đóng HUD"
            >
              ✕
            </button>
          )}
        </div>
      </div>

      {/* SESSION RECOVERY DRAWER */}
      {showRecoverBox && (
        <div style={{
          padding: '12px 16px',
          background: 'rgba(0, 0, 0, 0.35)',
          borderRadius: '8px',
          border: '1px solid rgba(255, 255, 255, 0.1)',
          marginBottom: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '12px'
        }}>
          <span style={{ fontSize: '13px', fontWeight: '600', color: '#94a3b8' }}>Session ID:</span>
          <input
            type="text"
            placeholder="pilot_v3_..."
            value={recoverInput}
            onChange={(e) => setRecoverInput(e.target.value)}
            style={{
              flex: 1,
              padding: '6px 10px',
              background: 'rgba(255, 255, 255, 0.05)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              borderRadius: '6px',
              color: '#fff',
              fontSize: '13px'
            }}
          />
          <button
            onClick={handleRecoverSession}
            style={{
              padding: '6px 14px',
              background: '#00f0ff',
              color: '#000',
              fontWeight: '700',
              borderRadius: '6px',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            Khôi phục
          </button>
        </div>
      )}

      {/* ERROR ALERT */}
      {errorMessage && (
        <div style={{
          padding: '10px 14px',
          background: 'rgba(239, 68, 68, 0.15)',
          borderRadius: '8px',
          border: '1px solid #ef4444',
          color: '#fca5a5',
          fontSize: '13px',
          marginBottom: '16px',
          display: 'flex',
          alignItems: 'center',
          gap: '8px'
        }}>
          <ShieldAlert size={16} />
          <span>{errorMessage}</span>
        </div>
      )}

      {/* SESSION CONFIGURATION & STATUS BAR */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
        gap: '12px',
        marginBottom: '16px',
        background: 'rgba(255, 255, 255, 0.02)',
        padding: '14px',
        borderRadius: '12px',
        border: '1px solid rgba(255, 255, 255, 0.06)'
      }}>
        {/* Mode & Target */}
        <div>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748b', fontWeight: '700' }}>Chế độ kiểm thử</div>
          {sessionState === 'CREATED' || sessionState === 'READY' ? (
            <div style={{ display: 'flex', gap: '8px', marginTop: '6px' }}>
              <select
                value={sessionMode}
                onChange={(e) => setSessionMode(e.target.value)}
                style={{
                  padding: '6px 10px',
                  background: 'rgba(0, 0, 0, 0.4)',
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  borderRadius: '6px',
                  color: '#fff',
                  fontSize: '13px'
                }}
              >
                <option value="FULL_29">Toàn bộ 29 Chữ Cái (290 lượt)</option>
                <option value="SINGLE_LETTER">Kiểm thử riêng 1 Chữ (10 lượt)</option>
              </select>

              {sessionMode === 'SINGLE_LETTER' && (
                <select
                  value={selectedSingleLetter}
                  onChange={(e) => setSelectedSingleLetter(e.target.value)}
                  style={{
                    padding: '6px 10px',
                    background: 'rgba(0, 0, 0, 0.4)',
                    border: '1px solid rgba(0, 240, 255, 0.3)',
                    borderRadius: '6px',
                    color: '#00f0ff',
                    fontWeight: '700',
                    fontSize: '14px'
                  }}
                >
                  {VIETNAMESE_ALPHABET_29.map(ch => (
                    <option key={ch} value={ch}>Chữ {ch}</option>
                  ))}
                </select>
              )}
            </div>
          ) : (
            <div style={{ fontSize: '14px', fontWeight: '700', color: '#00f0ff', marginTop: '4px' }}>
              {sessionMode === 'FULL_29' ? 'Toàn bộ 29 Chữ Cái' : `Kiểm thử Chữ ${selectedSingleLetter}`}
            </div>
          )}
        </div>

        {/* Current Target Letter */}
        <div>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748b', fontWeight: '700' }}>Chữ đang kiểm thử</div>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', marginTop: '2px' }}>
            <span style={{ fontSize: '28px', fontWeight: '900', color: '#00f0ff', lineHeight: 1 }}>
              {currentLetter}
            </span>
            <span style={{ fontSize: '13px', color: '#94a3b8' }}>
              (Đã đạt: <strong style={{ color: '#fff' }}>{letterAccepted}/10</strong> lượt)
            </span>
          </div>
        </div>

        {/* Full Session Progress */}
        <div>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748b', fontWeight: '700' }}>Tiến độ phiên</div>
          <div style={{ fontSize: '14px', fontWeight: '700', color: '#fff', marginTop: '4px' }}>
            {totalAccepted} / {totalTarget} lượt hoàn thành
          </div>
          <div style={{
            width: '100%',
            height: '6px',
            background: 'rgba(255, 255, 255, 0.08)',
            borderRadius: '3px',
            overflow: 'hidden',
            marginTop: '6px'
          }}>
            <div style={{
              width: `${progressPct}%`,
              height: '100%',
              background: 'linear-gradient(90deg, #00f0ff, #7000ff)',
              transition: 'width 0.3s ease'
            }} />
          </div>
          <div style={{ fontSize: '11px', color: '#64748b', marginTop: '4px' }}>
            Còn lại {remainingTrials} lượt
          </div>
        </div>

        {/* Hand & Recognition Status */}
        <div>
          <div style={{ fontSize: '11px', textTransform: 'uppercase', color: '#64748b', fontWeight: '700' }}>Bàn tay & Dự đoán AI</div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginTop: '6px' }}>
            <span style={{
              fontSize: '11px',
              padding: '3px 8px',
              borderRadius: '6px',
              background: isHandDetected ? 'rgba(0, 240, 255, 0.15)' : 'rgba(239, 68, 68, 0.15)',
              color: isHandDetected ? '#00f0ff' : '#ef4444',
              fontWeight: '700',
              border: `1px solid ${isHandDetected ? '#00f0ff' : '#ef4444'}`
            }}>
              {handStatus}
            </span>

            <span style={{ fontSize: '14px', fontWeight: '700', color: predictedLetter === currentLetter ? '#00ff80' : '#f59e0b' }}>
              AI: {predictedLetter} ({confidencePct}%)
            </span>
          </div>
        </div>
      </div>

      {/* ACTIVE TRIAL STATUS BANNER */}
      <div style={{
        padding: '16px',
        borderRadius: '12px',
        background: isRecording
          ? 'linear-gradient(135deg, rgba(239, 68, 68, 0.2) 0%, rgba(10, 15, 26, 0.9) 100%)'
          : isCountdown
          ? 'linear-gradient(135deg, rgba(245, 158, 11, 0.2) 0%, rgba(10, 15, 26, 0.9) 100%)'
          : 'rgba(255, 255, 255, 0.03)',
        border: `1px solid ${isRecording ? '#ef4444' : isCountdown ? '#f59e0b' : 'rgba(255, 255, 255, 0.1)'}`,
        marginBottom: '16px',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* BIG COUNTDOWN OVERLAY */}
        {isCountdown && (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            gap: '16px',
            padding: '10px 0'
          }}>
            <span style={{ fontSize: '16px', color: '#f59e0b', fontWeight: '700' }}>
              Chuẩn bị tư thế chữ <strong style={{ color: '#fff', fontSize: '20px' }}>{currentLetter}</strong>:
            </span>
            <span style={{
              fontSize: '44px',
              fontWeight: '900',
              color: '#f59e0b',
              lineHeight: 1,
              animation: 'pulse 1s infinite'
            }}>
              {countdownRemaining}
            </span>
            <span style={{ fontSize: '13px', color: '#94a3b8' }}>(Đếm ngược không tính vào 5.0s ghi hình)</span>
          </div>
        )}

        {/* RECORDING PROGRESS BAR */}
        {isRecording && (
          <div>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{
                  width: '10px',
                  height: '10px',
                  borderRadius: '50%',
                  background: '#ef4444',
                  boxShadow: '0 0 10px #ef4444'
                }} />
                <strong style={{ fontSize: '16px', color: '#fff' }}>
                  Letter {currentLetter} — Trial {currentTrialNumber}/10 — Recording {recordingElapsed.toFixed(1)} / 5.0s
                </strong>
              </div>
              <span style={{ fontSize: '13px', color: '#fca5a5', fontWeight: '700' }}>
                Còn lại: {Math.max(0, (5.0 - recordingElapsed)).toFixed(1)}s
              </span>
            </div>
            <div style={{ width: '100%', height: '8px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '4px', overflow: 'hidden' }}>
              <div style={{
                width: `${Math.min(100, (recordingElapsed / 5.0) * 100)}%`,
                height: '100%',
                background: '#ef4444',
                transition: 'width 0.1s linear'
              }} />
            </div>
          </div>
        )}

        {/* READY / PENDING / IDLE STATE PROMPT */}
        {!isRecording && !isCountdown && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
            <div>
              <strong style={{ fontSize: '15px', color: '#fff' }}>
                Letter {currentLetter} — Trial {currentTrialNumber}/10
              </strong>
              <div style={{ fontSize: '12px', color: '#94a3b8', marginTop: '2px' }}>
                Trạng thái: <span style={{ color: '#00f0ff', fontWeight: '700' }}>{trialState}</span>.
                Bấm "Start Trial" để bắt đầu 3s chuẩn bị + 5.0s ghi dữ liệu liên tục.
              </div>
            </div>

            <div style={{ fontSize: '12px', color: '#64748b' }}>
              Không cần giữ tay liên tục trước camera lúc nghỉ giữa các trial.
            </div>
          </div>
        )}
      </div>

      {/* CONTROL BUTTONS BAR */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: '10px',
        flexWrap: 'wrap',
        marginBottom: '16px'
      }}>
        {/* Session Start / Stop */}
        {sessionState === 'CREATED' || sessionState === 'READY' ? (
          <button
            onClick={handleCreateSession}
            style={{
              padding: '10px 20px',
              borderRadius: '8px',
              background: '#00f0ff',
              color: '#000',
              fontWeight: '800',
              fontSize: '14px',
              border: 'none',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '8px',
              boxShadow: '0 4px 14px rgba(0, 240, 255, 0.4)'
            }}
          >
            <Play size={16} /> Start Session
          </button>
        ) : (
          <button
            onClick={handleStopSession}
            style={{
              padding: '10px 16px',
              borderRadius: '8px',
              background: 'rgba(239, 68, 68, 0.15)',
              color: '#ef4444',
              fontWeight: '700',
              fontSize: '13px',
              border: '1px solid rgba(239, 68, 68, 0.4)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <Square size={16} /> Stop Session
          </button>
        )}

        {/* Start Trial */}
        <button
          onClick={handleStartTrial}
          disabled={!isRunning || isRecording || isCountdown}
          style={{
            padding: '10px 22px',
            borderRadius: '8px',
            background: isRunning && !isRecording && !isCountdown
              ? 'linear-gradient(135deg, #00ff80 0%, #00b4d8 100%)'
              : 'rgba(255, 255, 255, 0.08)',
            color: isRunning && !isRecording && !isCountdown ? '#000' : '#64748b',
            fontWeight: '800',
            fontSize: '14px',
            border: 'none',
            cursor: isRunning && !isRecording && !isCountdown ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            gap: '8px',
            boxShadow: isRunning && !isRecording && !isCountdown ? '0 4px 14px rgba(0, 255, 128, 0.3)' : 'none'
          }}
        >
          <Play size={16} /> Start Trial (5s)
        </button>

        {/* Pause / Resume */}
        {isRunning ? (
          <button
            onClick={handlePauseSession}
            style={{
              padding: '10px 16px',
              borderRadius: '8px',
              background: 'rgba(245, 158, 11, 0.15)',
              color: '#f59e0b',
              fontWeight: '700',
              fontSize: '13px',
              border: '1px solid rgba(245, 158, 11, 0.4)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <Pause size={16} /> Pause
          </button>
        ) : isPaused ? (
          <button
            onClick={handleResumeSession}
            style={{
              padding: '10px 16px',
              borderRadius: '8px',
              background: 'rgba(0, 255, 128, 0.15)',
              color: '#00ff80',
              fontWeight: '700',
              fontSize: '13px',
              border: '1px solid rgba(0, 255, 128, 0.4)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '6px'
            }}
          >
            <Play size={16} /> Resume
          </button>
        ) : null}

        {/* Retry Trial */}
        <button
          onClick={handleRetryTrial}
          disabled={!isRunning}
          style={{
            padding: '10px 16px',
            borderRadius: '8px',
            background: 'rgba(255, 255, 255, 0.06)',
            color: '#e2e8f0',
            fontWeight: '600',
            fontSize: '13px',
            border: '1px solid rgba(255, 255, 255, 0.15)',
            cursor: isRunning ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <RotateCcw size={16} /> Retry Trial
        </button>

        {/* Skip Trial */}
        <button
          onClick={handleSkipTrial}
          disabled={!isRunning}
          style={{
            padding: '10px 16px',
            borderRadius: '8px',
            background: 'rgba(255, 255, 255, 0.06)',
            color: '#94a3b8',
            fontWeight: '600',
            fontSize: '13px',
            border: '1px solid rgba(255, 255, 255, 0.12)',
            cursor: isRunning ? 'pointer' : 'not-allowed',
            display: 'flex',
            alignItems: 'center',
            gap: '6px'
          }}
        >
          <SkipForward size={16} /> Skip Trial
        </button>
      </div>

      {/* POST-TRIAL RESULT & QUALITY CARD */}
      {lastTrialSummary && (
        <div style={{
          padding: '16px',
          borderRadius: '12px',
          background: 'rgba(0, 240, 255, 0.04)',
          border: '1px solid rgba(0, 240, 255, 0.25)',
          marginBottom: '16px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '12px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <CheckCircle2 size={18} color="#00ff80" />
              <strong style={{ fontSize: '15px', color: '#fff' }}>
                Kết quả Trial {lastTrialSummary.trial_number} ({lastTrialSummary.expected_letter}) — Đã lưu thành công!
              </strong>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                onClick={handleAcceptTrial}
                style={{
                  padding: '6px 14px',
                  borderRadius: '6px',
                  background: '#00ff80',
                  color: '#000',
                  fontWeight: '700',
                  fontSize: '12px',
                  border: 'none',
                  cursor: 'pointer'
                }}
              >
                Chấp nhận & Tiếp tục
              </button>
              <button
                onClick={handleRetryTrial}
                style={{
                  padding: '6px 12px',
                  borderRadius: '6px',
                  background: 'rgba(255, 255, 255, 0.08)',
                  color: '#fff',
                  fontSize: '12px',
                  border: '1px solid rgba(255, 255, 255, 0.15)',
                  cursor: 'pointer'
                }}
              >
                Thử lại lượt này
              </button>
            </div>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(160px, 1fr))',
            gap: '10px',
            fontSize: '12px'
          }}>
            <div>
              <span style={{ color: '#64748b' }}>Tổng số frame: </span>
              <strong style={{ color: '#fff' }}>{lastTrialSummary.total_frames}</strong>
            </div>
            <div>
              <span style={{ color: '#64748b' }}>Khớp chính xác: </span>
              <strong style={{ color: '#00ff80' }}>
                {lastTrialSummary.recognition_match_rate !== null ? `${Math.round(lastTrialSummary.recognition_match_rate * 100)}%` : 'N/A'}
              </strong>
            </div>
            <div>
              <span style={{ color: '#64748b' }}>Nhầm lẫn (Mismatch): </span>
              <strong style={{ color: '#f59e0b' }}>
                {lastTrialSummary.recognition_mismatch_rate !== null ? `${Math.round(lastTrialSummary.recognition_mismatch_rate * 100)}%` : 'N/A'}
              </strong>
            </div>
            <div>
              <span style={{ color: '#64748b' }}>Landmarks hợp lệ: </span>
              <strong style={{ color: '#00f0ff' }}>
                {lastTrialSummary.valid_landmark_frames} / {lastTrialSummary.total_frames}
              </strong>
            </div>
            <div>
              <span style={{ color: '#64748b' }}>Chất lượng: </span>
              <span style={{ color: lastTrialSummary.quality_flags?.length ? '#f59e0b' : '#00ff80', fontWeight: '700' }}>
                {lastTrialSummary.quality_flags?.length ? lastTrialSummary.quality_flags.join(', ') : 'Tốt'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* FULL SESSION SUMMARY MODAL */}
      {showSummaryTable && sessionSummary && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(0, 0, 0, 0.8)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: '20px'
        }}>
          <div style={{
            width: '100%',
            maxWidth: '1000px',
            maxHeight: '90vh',
            background: '#0d1322',
            borderRadius: '16px',
            border: '1px solid rgba(0, 240, 255, 0.3)',
            boxShadow: '0 20px 60px rgba(0, 0, 0, 0.9)',
            display: 'flex',
            flexDirection: 'column',
            overflow: 'hidden'
          }}>
            {/* Modal Header */}
            <div style={{
              padding: '16px 20px',
              borderBottom: '1px solid rgba(255, 255, 255, 0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <div>
                <h3 style={{ fontSize: '18px', fontWeight: '800', margin: 0, color: '#fff' }}>
                  Bảng Tổng Hợp Controlled Pilot V3 (29 Chữ Cái)
                </h3>
                <span style={{ fontSize: '12px', color: '#94a3b8' }}>
                  Session: {sessionSummary.session_id} | Tổng lượt đạt: {sessionSummary.total_accepted_trials}
                </span>
              </div>
              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  onClick={handleExportJSON}
                  style={{
                    padding: '8px 14px',
                    borderRadius: '8px',
                    background: '#00f0ff',
                    color: '#000',
                    fontWeight: '700',
                    fontSize: '12px',
                    border: 'none',
                    cursor: 'pointer'
                  }}
                >
                  <Download size={14} style={{ display: 'inline', marginRight: '4px' }} /> Tải JSON
                </button>
                <button
                  onClick={() => setShowSummaryTable(false)}
                  style={{
                    padding: '8px 14px',
                    borderRadius: '8px',
                    background: 'rgba(255, 255, 255, 0.1)',
                    color: '#fff',
                    border: 'none',
                    cursor: 'pointer'
                  }}
                >
                  Đóng
                </button>
              </div>
            </div>

            {/* Modal Body: Table */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '16px 20px' }}>
              <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '13px', textAlign: 'left' }}>
                <thead>
                  <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.15)', color: '#64748b' }}>
                    <th style={{ padding: '8px' }}>Chữ</th>
                    <th style={{ padding: '8px' }}>Lượt đạt</th>
                    <th style={{ padding: '8px' }}>Frames</th>
                    <th style={{ padding: '8px' }}>Tỷ lệ Landmark</th>
                    <th style={{ padding: '8px' }}>Độ khớp AI (%)</th>
                    <th style={{ padding: '8px' }}>Nhầm lẫn nhiều nhất</th>
                    <th style={{ padding: '8px' }}>Chất lượng dữ liệu</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.values(sessionSummary.letter_reports || {}).map((rep) => (
                    <tr key={rep.letter} style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                      <td style={{ padding: '8px', fontWeight: '800', color: '#00f0ff', fontSize: '15px' }}>
                        {rep.letter}
                      </td>
                      <td style={{ padding: '8px' }}>{rep.accepted_trials} / 10</td>
                      <td style={{ padding: '8px' }}>{rep.total_frames}</td>
                      <td style={{ padding: '8px' }}>
                        {rep.valid_landmark_rate !== null ? `${Math.round(rep.valid_landmark_rate * 100)}%` : '—'}
                      </td>
                      <td style={{ padding: '8px', fontWeight: '700', color: (rep.recognition_match_rate || 0) >= 0.7 ? '#00ff80' : '#f59e0b' }}>
                        {rep.recognition_match_rate !== null ? `${Math.round(rep.recognition_match_rate * 100)}%` : '—'}
                      </td>
                      <td style={{ padding: '8px', color: '#94a3b8' }}>
                        {Object.entries(rep.most_confused_letters || {}).map(([c, cnt]) => `${c}(${cnt})`).join(', ') || '—'}
                      </td>
                      <td style={{ padding: '8px' }}>
                        <span style={{
                          fontSize: '11px',
                          fontWeight: '700',
                          padding: '2px 6px',
                          borderRadius: '4px',
                          background: rep.data_quality_status === 'EXCELLENT' ? 'rgba(0, 255, 128, 0.15)' : 'rgba(245, 158, 11, 0.15)',
                          color: rep.data_quality_status === 'EXCELLENT' ? '#00ff80' : '#f59e0b'
                        }}>
                          {rep.data_quality_status}
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
