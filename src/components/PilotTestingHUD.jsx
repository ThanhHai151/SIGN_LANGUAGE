import React, { useState, useEffect, useRef, useCallback } from 'react';
import {
  FlaskConical, Play, Square, RotateCcw, Save, AlertTriangle,
  CheckCircle2, XCircle, Download, BarChart3, SkipForward, Pause,
  FileJson, Sparkles, Check, ChevronRight, X, Clock, HelpCircle
} from 'lucide-react';
import {
  savePilotEvaluation,
  setPilotLogging,
  flushPilotLogs
} from '../services/api';
import {
  VSL_ALPHABET_29,
  PILOT_DURATION_SEC,
  PILOT_REPETITIONS,
  createPilotSessionId,
  createPilotTrialId,
  evaluateTrialFrames,
  buildEvaluationReport
} from './pilotProtocol';

export { VSL_ALPHABET_29 };

export default function PilotTestingHUD({
  isActive,
  onToggleActive,
  onMetadataUpdate,
  cameraActive,
  latestRecognition,
  videoElementRef
}) {
  // --- Session Configuration ---
  const [sessionId, setSessionId] = useState(() => createPilotSessionId());
  const [testMode, setTestMode] = useState('single_letter'); // 'single_letter' | 'all_29_letters'
  const [selectedLetter, setSelectedLetter] = useState('A');
  const [currentRep, setCurrentRep] = useState(1);
  const totalReps = PILOT_REPETITIONS; // 10 lượt
  const trialDurationSec = PILOT_DURATION_SEC; // 5.0s

  // --- Trial States ---
  // 'IDLE' | 'COUNTDOWN' | 'RECORDING' | 'TRIAL_REVIEW' | 'SESSION_COMPLETE'
  const [trialState, setTrialState] = useState('IDLE');
  const [countdownRemaining, setCountdownRemaining] = useState(3);
  const [recordingElapsedSec, setRecordingElapsedSec] = useState(0.0);
  const [autoAdvance, setAutoAdvance] = useState(true);
  const [saveFrameLogs, setSaveFrameLogs] = useState(true);

  // --- Realtime Frame & Prediction Buffer ---
  const [currentFrameStats, setCurrentFrameStats] = useState({
    totalFrames: 0,
    matchFrames: 0,
    mismatchFrames: 0,
    neutralFrames: 0,
    lastSign: 'NEUTRAL',
    lastConfidence: 0.0,
    isMatch: false
  });

  // --- Results Store ---
  // Structure: { [letter]: [ { rep: 1, matchPct: 88, status: 'PASS', confusions: {...}, frames: [...] }, ... ] }
  const [resultsByLetter, setResultsByLetter] = useState({});
  const [lastTrialSummary, setLastTrialSummary] = useState(null);
  const [showScoreboard, setShowScoreboard] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null); // { type: 'success'|'error', message: string }

  // --- Refs for Monotonic Loop & Clean State ---
  const trialStateRef = useRef(trialState);
  trialStateRef.current = trialState;

  const selectedLetterRef = useRef(selectedLetter);
  selectedLetterRef.current = selectedLetter;

  const currentRepRef = useRef(currentRep);
  currentRepRef.current = currentRep;

  const sessionIdRef = useRef(sessionId);
  sessionIdRef.current = sessionId;

  const trialFramesRef = useRef([]);
  const recordingStartTimeRef = useRef(0);
  const timerRafRef = useRef(null);
  const countdownIntervalRef = useRef(null);
  const autoAdvanceTimeoutRef = useRef(null);
  const frameCounterRef = useRef(0);

  // Gửi metadata sang CameraSignRecognizer
  useEffect(() => {
    if (!onMetadataUpdate) return;

    const getFrameMetadata = () => {
      if (trialStateRef.current !== 'RECORDING') return null;

      frameCounterRef.current += 1;
      const trialId = createPilotTrialId(sessionIdRef.current, selectedLetterRef.current, currentRepRef.current);

      return {
        session_id: sessionIdRef.current,
        trial_id: trialId,
        expected_sign: selectedLetterRef.current,
        frame_idx: frameCounterRef.current,
        client_capture_ts: performance.now() / 1000,
        signer_id: 'signer_01'
      };
    };

    const onFrameResult = (data, pilotMeta) => {
      if (trialStateRef.current !== 'RECORDING') return;

      const expected = selectedLetterRef.current;
      const predicted = data?.sign || 'NEUTRAL';
      const conf = Number(data?.confidence || 0);
      const isMatch = (predicted === expected);

      const frameRecord = {
        frame_idx: pilotMeta?.frame_idx || trialFramesRef.current.length + 1,
        expected_sign: expected,
        predicted_sign: predicted,
        confidence: conf,
        timestamp: performance.now() / 1000,
        is_match: isMatch,
        landmarks: data?.landmarks || null
      };

      trialFramesRef.current.push(frameRecord);

      setCurrentFrameStats(prev => {
        const isNeutral = (predicted === 'NEUTRAL' || predicted === 'unknown');
        return {
          totalFrames: prev.totalFrames + 1,
          matchFrames: prev.matchFrames + (isMatch ? 1 : 0),
          mismatchFrames: prev.mismatchFrames + (!isMatch && !isNeutral ? 1 : 0),
          neutralFrames: prev.neutralFrames + (isNeutral ? 1 : 0),
          lastSign: predicted,
          lastConfidence: conf,
          isMatch
        };
      });
    };

    onMetadataUpdate({ getFrameMetadata, onFrameResult });
  }, [onMetadataUpdate]);

  // Clean-up timers on unmount
  useEffect(() => {
    return () => {
      if (timerRafRef.current) cancelAnimationFrame(timerRafRef.current);
      if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
      if (autoAdvanceTimeoutRef.current) clearTimeout(autoAdvanceTimeoutRef.current);
    };
  }, []);

  // --- Lượt thử hoàn tất (End of 5.0s) ---
  const handleCompleteTrial = useCallback(() => {
    if (timerRafRef.current) cancelAnimationFrame(timerRafRef.current);

    const letter = selectedLetterRef.current;
    const rep = currentRepRef.current;
    const frames = [...trialFramesRef.current];

    // Đánh giá kết quả của lượt
    const summary = evaluateTrialFrames(frames, letter, trialDurationSec);
    summary.rep = rep;
    summary.letter = letter;
    summary.frames = frames;

    setLastTrialSummary(summary);
    setTrialState('TRIAL_REVIEW');

    // Cập nhật kết quả vào store
    setResultsByLetter(prev => {
      const existing = prev[letter] || [];
      // Thay thế lượt thử nếu đã tồn tại rep này (do retry), hoặc thêm mới
      const updated = [...existing.filter(t => t.rep !== rep), summary].sort((a, b) => a.rep - b.rep);
      return { ...prev, [letter]: updated };
    });

    // Auto advance nếu bật
    if (autoAdvance) {
      autoAdvanceTimeoutRef.current = setTimeout(() => {
        handleNextTrial(summary);
      }, 2500);
    }
  }, [trialDurationSec, autoAdvance]);

  // --- Bắt đầu 5.0s Recording ---
  const startRecordingPeriod = useCallback(() => {
    trialFramesRef.current = [];
    frameCounterRef.current = 0;
    setCurrentFrameStats({
      totalFrames: 0,
      matchFrames: 0,
      mismatchFrames: 0,
      neutralFrames: 0,
      lastSign: 'NEUTRAL',
      lastConfidence: 0.0,
      isMatch: false
    });
    setRecordingElapsedSec(0.0);
    setTrialState('RECORDING');

    recordingStartTimeRef.current = performance.now();

    const loop = () => {
      const elapsed = (performance.now() - recordingStartTimeRef.current) / 1000;
      setRecordingElapsedSec(Math.min(elapsed, trialDurationSec));

      if (elapsed >= trialDurationSec) {
        handleCompleteTrial();
      } else {
        timerRafRef.current = requestAnimationFrame(loop);
      }
    };

    timerRafRef.current = requestAnimationFrame(loop);
  }, [trialDurationSec, handleCompleteTrial]);

  // --- Bắt đầu Trial (Đếm ngược 3s chuẩn bị) ---
  const handleStartTrial = () => {
    if (!cameraActive) {
      alert('Vui lòng bật Camera trước khi bắt đầu kiểm thử!');
      return;
    }

    if (autoAdvanceTimeoutRef.current) clearTimeout(autoAdvanceTimeoutRef.current);
    setSaveStatus(null);
    setTrialState('COUNTDOWN');
    setCountdownRemaining(3);

    let count = 3;
    if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);

    countdownIntervalRef.current = setInterval(() => {
      count -= 1;
      if (count > 0) {
        setCountdownRemaining(count);
      } else {
        clearInterval(countdownIntervalRef.current);
        startRecordingPeriod();
      }
    }, 1000);
  };

  // --- Chuyển sang lượt thử tiếp theo ---
  const handleNextTrial = (prevSummary = null) => {
    if (autoAdvanceTimeoutRef.current) clearTimeout(autoAdvanceTimeoutRef.current);

    const letter = selectedLetterRef.current;
    const rep = currentRepRef.current;

    if (rep < totalReps) {
      // Tiếp tục lượt tiếp theo của chữ hiện tại
      setCurrentRep(rep + 1);
      setTrialState('IDLE');
    } else {
      // Đã xong 10 lượt của chữ này
      if (testMode === 'all_29_letters') {
        const currIdx = VSL_ALPHABET_29.indexOf(letter);
        if (currIdx < VSL_ALPHABET_29.length - 1) {
          // Sang chữ tiếp theo
          const nextLetter = VSL_ALPHABET_29[currIdx + 1];
          setSelectedLetter(nextLetter);
          setCurrentRep(1);
          setTrialState('IDLE');
        } else {
          // Hoàn thành cả 29 chữ!
          setTrialState('SESSION_COMPLETE');
          setShowScoreboard(true);
        }
      } else {
        // Hoàn thành 10 lượt của chữ đơn
        setTrialState('SESSION_COMPLETE');
        setShowScoreboard(true);
      }
    }
  };

  // --- Thử lại lượt này (Retry) ---
  const handleRetryTrial = () => {
    if (autoAdvanceTimeoutRef.current) clearTimeout(autoAdvanceTimeoutRef.current);
    setTrialState('IDLE');
    handleStartTrial();
  };

  // --- Bỏ qua lượt này (Skip) ---
  const handleSkipTrial = () => {
    if (autoAdvanceTimeoutRef.current) clearTimeout(autoAdvanceTimeoutRef.current);
    const letter = selectedLetterRef.current;
    const rep = currentRepRef.current;

    const skipSummary = {
      rep,
      letter,
      totalFrames: 0,
      activeFrames: 0,
      matchFrames: 0,
      mismatchFrames: 0,
      neutralFrames: 0,
      matchPct: 0.0,
      avgConfidence: 0.0,
      confusions: {},
      topConfusion: null,
      status: 'SKIPPED',
      isPassed: false,
      durationSec: 0,
      frames: []
    };

    setResultsByLetter(prev => {
      const existing = prev[letter] || [];
      const updated = [...existing.filter(t => t.rep !== rep), skipSummary].sort((a, b) => a.rep - b.rep);
      return { ...prev, [letter]: updated };
    });

    handleNextTrial();
  };

  // --- Dừng phiên thử nghiệm ---
  const handleStopSession = () => {
    if (timerRafRef.current) cancelAnimationFrame(timerRafRef.current);
    if (countdownIntervalRef.current) clearInterval(countdownIntervalRef.current);
    if (autoAdvanceTimeoutRef.current) clearTimeout(autoAdvanceTimeoutRef.current);
    setTrialState('IDLE');
  };

  // --- Tạo lại phiên mới ---
  const handleResetSession = () => {
    handleStopSession();
    const newSessId = createPilotSessionId();
    setSessionId(newSessId);
    setResultsByLetter({});
    setCurrentRep(1);
    setLastTrialSummary(null);
    setShowScoreboard(false);
    setSaveStatus(null);
  };

  // --- Lưu file JSON vào backend\data\diagnostics ---
  const handleSaveDiagnostics = async () => {
    setSaveStatus({ type: 'loading', message: 'Đang lưu báo cáo JSON vào backend/data/diagnostics...' });

    try {
      const report = buildEvaluationReport(sessionId, resultsByLetter, {
        trialsPerLetter: totalReps,
        trialDurationSec,
        mode: testMode,
        testedTarget: testMode === 'single_letter' ? selectedLetter : 'ALL_29'
      });

      // Lấy toàn bộ frame nếu bật
      let allFrames = [];
      if (saveFrameLogs) {
        Object.values(resultsByLetter).forEach(trials => {
          trials.forEach(t => {
            if (t.frames && t.frames.length) {
              allFrames = allFrames.concat(t.frames);
            }
          });
        });
      }

      const res = await savePilotEvaluation(sessionId, report, allFrames.length ? allFrames : null);

      if (res.success) {
        setSaveStatus({
          type: 'success',
          message: `Đã lưu thành công vào: backend/data/diagnostics/${res.filename}`,
          filepath: res.filepath
        });
      } else {
        setSaveStatus({
          type: 'error',
          message: res.error || 'Lỗi khi lưu vào backend'
        });
      }
    } catch (err) {
      setSaveStatus({
        type: 'error',
        message: err.message
      });
    }
  };

  // --- Tải file JSON về máy trực tiếp ---
  const handleDownloadJSON = () => {
    const report = buildEvaluationReport(sessionId, resultsByLetter, {
      trialsPerLetter: totalReps,
      trialDurationSec,
      mode: testMode,
      testedTarget: testMode === 'single_letter' ? selectedLetter : 'ALL_29'
    });

    const dataStr = 'data:text/json;charset=utf-8,' + encodeURIComponent(JSON.stringify(report, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute('href', dataStr);
    downloadAnchor.setAttribute('download', `pilot_eval_29_${sessionId}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  // Nếu HUD chưa được bật mở từ nút ngoài Camera
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
          title="Bật công cụ Controlled Pilot để kiểm thử 29 chữ cái và ghi file JSON chẩn đoán"
        >
          <FlaskConical size={14} style={{ color: '#38bdf8' }} />
          🧪 Controlled Pilot (Test 29 chữ)
        </button>
      </div>
    );
  }

  // Báo cáo tóm tắt hiện tại
  const currentReport = buildEvaluationReport(sessionId, resultsByLetter, {
    trialsPerLetter: totalReps,
    trialDurationSec,
    mode: testMode,
    testedTarget: selectedLetter
  });

  const letterTrials = resultsByLetter[selectedLetter] || [];
  const completedLetterTrials = letterTrials.filter(t => t.status !== 'SKIPPED');
  const letterMatchFrames = completedLetterTrials.reduce((sum, t) => sum + (t.matchFrames || 0), 0);
  const letterActiveFrames = completedLetterTrials.reduce((sum, t) => sum + (t.activeFrames || 0), 0);
  const letterAccuracyPct = letterActiveFrames > 0
    ? ((letterMatchFrames / letterActiveFrames) * 100).toFixed(1)
    : '0.0';

  return (
    <div style={{
      background: 'linear-gradient(145deg, #090e17, #0f172a)',
      border: '1px solid rgba(56, 189, 248, 0.35)',
      borderRadius: '12px',
      padding: '16px',
      display: 'grid',
      gap: '14px',
      color: '#e2e8f0',
      boxShadow: '0 12px 36px rgba(0,0,0,0.4)',
      width: '100%',
      boxSizing: 'border-box'
    }}>
      {/* Header */}
      <header style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '12px',
        borderBottom: '1px solid rgba(255,255,255,0.1)',
        paddingBottom: '12px'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
          <div style={{
            background: 'rgba(56, 189, 248, 0.15)',
            border: '1px solid rgba(56, 189, 248, 0.4)',
            padding: '8px',
            borderRadius: '8px'
          }}>
            <FlaskConical size={20} color="#38bdf8" />
          </div>
          <div>
            <h2 style={{ fontSize: '16px', fontWeight: 'bold', margin: 0, color: '#f8fafc' }}>
              Controlled Pilot — Kiểm thử 29 Chữ Cái VSL
            </h2>
            <div style={{ display: 'flex', gap: '8px', marginTop: '4px', fontSize: '11px', color: '#94a3b8' }}>
              <span>Tiêu chuẩn: <strong>10 lượt × 5.0 giây / chữ</strong></span>
              <span>•</span>
              <span>Session: <code style={{ color: '#38bdf8' }}>{sessionId}</code></span>
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button
            type="button"
            onClick={() => setShowScoreboard(true)}
            className="btn btn-secondary"
            style={{ fontSize: '12px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}
          >
            <BarChart3 size={14} />
            Bảng Chẩn đoán 29 Chữ
          </button>
          <button
            type="button"
            onClick={onToggleActive}
            className="btn btn-secondary"
            style={{ fontSize: '12px', padding: '6px 12px' }}
          >
            Đóng HUD
          </button>
        </div>
      </header>

      {/* Mode & Target Selector */}
      <div style={{
        display: 'grid',
        gridTemplateColumns: 'auto 1fr',
        gap: '16px',
        alignItems: 'center',
        background: 'rgba(15, 23, 42, 0.6)',
        padding: '10px 14px',
        borderRadius: '8px',
        border: '1px solid rgba(148, 163, 184, 0.15)'
      }}>
        {/* Chọn chế độ */}
        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            type="button"
            disabled={trialState === 'RECORDING' || trialState === 'COUNTDOWN'}
            onClick={() => setTestMode('single_letter')}
            style={{
              padding: '6px 12px',
              fontSize: '12px',
              borderRadius: '6px',
              border: testMode === 'single_letter' ? '1px solid #38bdf8' : '1px solid rgba(148, 163, 184, 0.2)',
              background: testMode === 'single_letter' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              color: testMode === 'single_letter' ? '#38bdf8' : '#94a3b8',
              fontWeight: testMode === 'single_letter' ? 'bold' : 'normal',
              cursor: 'pointer'
            }}
          >
            Kiểm thử 1 chữ cái
          </button>
          <button
            type="button"
            disabled={trialState === 'RECORDING' || trialState === 'COUNTDOWN'}
            onClick={() => setTestMode('all_29_letters')}
            style={{
              padding: '6px 12px',
              fontSize: '12px',
              borderRadius: '6px',
              border: testMode === 'all_29_letters' ? '1px solid #38bdf8' : '1px solid rgba(148, 163, 184, 0.2)',
              background: testMode === 'all_29_letters' ? 'rgba(56, 189, 248, 0.2)' : 'transparent',
              color: testMode === 'all_29_letters' ? '#38bdf8' : '#94a3b8',
              fontWeight: testMode === 'all_29_letters' ? 'bold' : 'normal',
              cursor: 'pointer'
            }}
          >
            Chạy toàn bộ 29 chữ
          </button>
        </div>

        {/* Thông tin tiến độ tổng thể */}
        <div style={{ display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '16px', fontSize: '12px' }}>
          <span>Chữ đang kiểm thử: <strong style={{ fontSize: '16px', color: '#38bdf8' }}>{selectedLetter}</strong></span>
          <span>Lượt: <strong style={{ color: '#f8fafc' }}>{currentRep} / {totalReps}</strong></span>
          <span>Độ chính xác chữ {selectedLetter}: <strong style={{ color: Number(letterAccuracyPct) >= 70 ? '#4ade80' : '#f87171' }}>{letterAccuracyPct}%</strong></span>
        </div>
      </div>

      {/* Grid 29 Chữ Cái Tiếng Việt */}
      <div style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '6px',
        padding: '6px 0',
        maxHeight: '110px',
        overflowY: 'auto'
      }}>
        {VSL_ALPHABET_29.map(letter => {
          const trials = resultsByLetter[letter] || [];
          const count = trials.length;
          const isSelected = letter === selectedLetter;
          const passedCount = trials.filter(t => t.status === 'PASS').length;
          const hasPassed = count > 0 && passedCount >= Math.ceil(count * 0.7);
          const needsFix = count > 0 && !hasPassed;

          let bg = 'rgba(30, 41, 59, 0.5)';
          let borderColor = 'rgba(148, 163, 184, 0.2)';
          let textColor = '#cbd5e1';

          if (isSelected) {
            bg = 'rgba(56, 189, 248, 0.25)';
            borderColor = '#38bdf8';
            textColor = '#38bdf8';
          } else if (hasPassed) {
            bg = 'rgba(34, 197, 94, 0.15)';
            borderColor = 'rgba(34, 197, 94, 0.4)';
            textColor = '#4ade80';
          } else if (needsFix) {
            bg = 'rgba(239, 68, 68, 0.15)';
            borderColor = 'rgba(239, 68, 68, 0.4)';
            textColor = '#f87171';
          }

          return (
            <button
              key={letter}
              type="button"
              disabled={trialState === 'RECORDING' || trialState === 'COUNTDOWN'}
              onClick={() => {
                setSelectedLetter(letter);
                setCurrentRep((resultsByLetter[letter]?.length || 0) + 1);
                setTrialState('IDLE');
              }}
              style={{
                width: '42px',
                height: '38px',
                borderRadius: '6px',
                border: `1px solid ${borderColor}`,
                background: bg,
                color: textColor,
                fontWeight: isSelected ? 'bold' : 'normal',
                fontSize: '14px',
                cursor: 'pointer',
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                justifyContent: 'center',
                position: 'relative'
              }}
              title={`Chữ ${letter} (${count}/${totalReps} lượt)`}
            >
              <span>{letter}</span>
              {count > 0 && (
                <span style={{ fontSize: '9px', opacity: 0.8 }}>{count}</span>
              )}
            </button>
          );
        })}
      </div>

      {/* Main Testing Stage */}
      <div style={{
        background: '#070b12',
        borderRadius: '10px',
        padding: '16px',
        border: '1px solid rgba(148, 163, 184, 0.15)',
        display: 'grid',
        gap: '12px'
      }}>
        {/* Countdown Overlay */}
        {trialState === 'COUNTDOWN' && (
          <div style={{
            textAlign: 'center',
            padding: '24px 0',
            background: 'rgba(56, 189, 248, 0.08)',
            borderRadius: '8px',
            border: '1px dashed #38bdf8'
          }}>
            <p style={{ margin: 0, fontSize: '13px', color: '#94a3b8' }}>Chuẩn bị tạo tư thế cho chữ cái</p>
            <h1 style={{ fontSize: '48px', margin: '4px 0', color: '#38bdf8' }}>{selectedLetter}</h1>
            <div style={{ fontSize: '28px', fontWeight: 'bold', color: '#fbbf24' }}>
              Bắt đầu ghi sau: {countdownRemaining}s
            </div>
          </div>
        )}

        {/* Recording Active Display */}
        {trialState === 'RECORDING' && (
          <div style={{ display: 'grid', gap: '10px' }}>
            {/* Thanh tiến độ 5.0s */}
            <div>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', marginBottom: '4px' }}>
                <span style={{ color: '#ef4444', fontWeight: 'bold', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: 8, height: 8, borderRadius: '50%', background: '#ef4444', animation: 'pulse 1s infinite' }} />
                  ĐANG THU THẬP DỮ LIỆU WEBCAM
                </span>
                <span style={{ fontVariantNumeric: 'tabular-nums', fontWeight: 'bold' }}>
                  {recordingElapsedSec.toFixed(1)}s / {trialDurationSec.toFixed(1)}s
                </span>
              </div>
              <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
                <div style={{
                  height: '100%',
                  width: `${(recordingElapsedSec / trialDurationSec) * 100}%`,
                  background: 'linear-gradient(90deg, #38bdf8, #22c55e)',
                  transition: 'width 0.05s linear'
                }} />
              </div>
            </div>

            {/* Thông số nhận diện trực tiếp */}
            <div style={{
              display: 'grid',
              gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))',
              gap: '10px',
              marginTop: '4px'
            }}>
              <div style={{ background: 'rgba(15,23,42,0.8)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }}>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Mục tiêu (Expected)</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#38bdf8' }}>{selectedLetter}</div>
              </div>

              <div style={{
                background: currentFrameStats.isMatch ? 'rgba(34,197,94,0.15)' : 'rgba(239,68,68,0.15)',
                padding: '10px',
                borderRadius: '8px',
                border: `1px solid ${currentFrameStats.isMatch ? '#22c55e' : '#ef4444'}`
              }}>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>AI Dự đoán (Predicted)</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: currentFrameStats.isMatch ? '#4ade80' : '#f87171' }}>
                  {currentFrameStats.lastSign}
                </div>
              </div>

              <div style={{ background: 'rgba(15,23,42,0.8)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }}>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Độ tin cậy</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#cbd5e1' }}>
                  {(currentFrameStats.lastConfidence * 100).toFixed(0)}%
                </div>
              </div>

              <div style={{ background: 'rgba(15,23,42,0.8)', padding: '10px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }}>
                <div style={{ fontSize: '11px', color: '#94a3b8' }}>Khớp lượt này</div>
                <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#facc15' }}>
                  {currentFrameStats.matchFrames} / {currentFrameStats.totalFrames}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Post-Trial Review */}
        {trialState === 'TRIAL_REVIEW' && lastTrialSummary && (
          <div style={{
            background: lastTrialSummary.isPassed ? 'rgba(34,197,94,0.12)' : 'rgba(239,68,68,0.12)',
            border: `1px solid ${lastTrialSummary.isPassed ? 'rgba(34,197,94,0.4)' : 'rgba(239,68,68,0.4)'}`,
            borderRadius: '8px',
            padding: '14px',
            display: 'grid',
            gap: '12px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                {lastTrialSummary.isPassed ? (
                  <CheckCircle2 size={24} color="#4ade80" />
                ) : (
                  <XCircle size={24} color="#f87171" />
                )}
                <div>
                  <h3 style={{ margin: 0, fontSize: '15px', color: lastTrialSummary.isPassed ? '#4ade80' : '#f87171' }}>
                    {lastTrialSummary.isPassed
                      ? `LƯỢT ${lastTrialSummary.rep} ĐẠT — Khớp ${lastTrialSummary.matchPct}% chữ ${selectedLetter}`
                      : `LƯỢT ${lastTrialSummary.rep} CẦN FIX — Chỉ khớp ${lastTrialSummary.matchPct}% chữ ${selectedLetter}`}
                  </h3>
                  <div style={{ fontSize: '11px', color: '#94a3b8', marginTop: '2px' }}>
                    {lastTrialSummary.topConfusion
                      ? `⚠️ Thường bị nhận diện nhầm sang chữ: "${lastTrialSummary.topConfusion}" (${lastTrialSummary.confusions[lastTrialSummary.topConfusion]} frames)`
                      : 'Không có nhầm lẫn đáng kể.'}
                  </div>
                </div>
              </div>

              <div style={{ textAlign: 'right', fontSize: '11px', color: '#cbd5e1' }}>
                <div>Thu thập: <strong>{lastTrialSummary.totalFrames} frames / 5.0s</strong></div>
                <div>Độ tin cậy TB: <strong>{(lastTrialSummary.avgConfidence * 100).toFixed(0)}%</strong></div>
              </div>
            </div>

            {/* Nút hành động sau lượt */}
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'flex-end', flexWrap: 'wrap' }}>
              <button
                type="button"
                onClick={handleRetryTrial}
                className="btn btn-secondary"
                style={{ fontSize: '12px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                <RotateCcw size={13} />
                Thử lại lượt này
              </button>
              <button
                type="button"
                onClick={handleSkipTrial}
                className="btn btn-secondary"
                style={{ fontSize: '12px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                <SkipForward size={13} />
                Bỏ qua
              </button>
              <button
                type="button"
                onClick={() => handleNextTrial(lastTrialSummary)}
                className="btn btn-primary"
                style={{ fontSize: '12px', padding: '6px 14px', display: 'flex', alignItems: 'center', gap: '4px' }}
              >
                <Check size={14} />
                Chấp nhận & Tiếp tục
              </button>
            </div>
          </div>
        )}

        {/* Session Complete Banner */}
        {trialState === 'SESSION_COMPLETE' && (
          <div style={{
            background: 'rgba(56, 189, 248, 0.15)',
            border: '1px solid #38bdf8',
            borderRadius: '8px',
            padding: '16px',
            textAlign: 'center'
          }}>
            <Sparkles size={28} color="#38bdf8" style={{ margin: '0 auto 8px' }} />
            <h3 style={{ margin: 0, fontSize: '16px', color: '#f8fafc' }}>
              Hoàn thành phiên kiểm thử Controlled Pilot!
            </h3>
            <p style={{ fontSize: '12px', color: '#94a3b8', margin: '4px 0 12px' }}>
              Đã thu thập dữ liệu webcam cho các chữ cái được chỉ định. Vui lòng lưu file JSON vào backend/data/diagnostics.
            </p>
            <div style={{ display: 'flex', gap: '8px', justifyContent: 'center' }}>
              <button
                type="button"
                onClick={() => setShowScoreboard(true)}
                className="btn btn-primary"
                style={{ fontSize: '12px', padding: '6px 14px' }}
              >
                Xem chi tiết các chữ cần fix
              </button>
              <button
                type="button"
                onClick={handleSaveDiagnostics}
                className="btn btn-secondary"
                style={{ fontSize: '12px', padding: '6px 14px', background: '#1e3a8a', color: '#93c5fd' }}
              >
                <Save size={13} style={{ marginRight: 4 }} />
                Lưu vào backend/data/diagnostics
              </button>
            </div>
          </div>
        )}

        {/* Controls Bar (Khi Idle) */}
        {trialState === 'IDLE' && (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
              <button
                type="button"
                onClick={handleStartTrial}
                className="btn btn-primary"
                style={{
                  fontSize: '13px',
                  padding: '8px 18px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: 'linear-gradient(135deg, #0284c7, #2563eb)',
                  boxShadow: '0 4px 12px rgba(37,99,235,0.3)'
                }}
              >
                <Play size={16} fill="#fff" />
                Bắt đầu Lượt thử {currentRep} / {totalReps} (Chữ {selectedLetter})
              </button>

              <label style={{ fontSize: '12px', display: 'flex', alignItems: 'center', gap: '6px', cursor: 'pointer', color: '#94a3b8' }}>
                <input
                  type="checkbox"
                  checked={autoAdvance}
                  onChange={e => setAutoAdvance(e.target.checked)}
                />
                Tự động chuyển lượt sau khi ghi
              </label>
            </div>

            <div style={{ display: 'flex', gap: '8px' }}>
              <button
                type="button"
                onClick={handleResetSession}
                className="btn btn-secondary"
                style={{ fontSize: '12px', padding: '6px 10px', color: '#94a3b8' }}
                title="Tạo phiên mới và xóa dữ liệu tạm hiện tại"
              >
                <RotateCcw size={13} style={{ marginRight: 4 }} />
                Reset Phiên
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Thông báo trạng thái lưu file */}
      {saveStatus && (
        <div style={{
          padding: '8px 12px',
          borderRadius: '6px',
          fontSize: '12px',
          background: saveStatus.type === 'success' ? 'rgba(34, 197, 94, 0.15)' : 'rgba(239, 68, 68, 0.15)',
          border: `1px solid ${saveStatus.type === 'success' ? '#22c55e' : '#ef4444'}`,
          color: saveStatus.type === 'success' ? '#4ade80' : '#f87171',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between'
        }}>
          <span>{saveStatus.message}</span>
          <button
            type="button"
            onClick={() => setSaveStatus(null)}
            style={{ background: 'none', border: 'none', color: 'inherit', cursor: 'pointer' }}
          >
            <X size={14} />
          </button>
        </div>
      )}

      {/* Footer Actions: Lưu JSON & Xem Scoreboard */}
      <footer style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        flexWrap: 'wrap',
        gap: '10px',
        paddingTop: '6px',
        borderTop: '1px solid rgba(255,255,255,0.06)'
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px', color: '#94a3b8' }}>
          <label style={{ display: 'flex', alignItems: 'center', gap: '4px', cursor: 'pointer' }}>
            <input
              type="checkbox"
              checked={saveFrameLogs}
              onChange={e => setSaveFrameLogs(e.target.checked)}
            />
            Kèm file frame chi tiết (.jsonl) khi lưu
          </label>
        </div>

        <div style={{ display: 'flex', gap: '8px' }}>
          <button
            type="button"
            onClick={handleDownloadJSON}
            className="btn btn-secondary"
            style={{ fontSize: '11px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '4px' }}
          >
            <Download size={13} />
            Tải JSON
          </button>
          <button
            type="button"
            onClick={handleSaveDiagnostics}
            className="btn btn-secondary"
            style={{
              fontSize: '11px',
              padding: '6px 14px',
              display: 'flex',
              alignItems: 'center',
              gap: '4px',
              background: '#1e3a8a',
              borderColor: '#3b82f6',
              color: '#bfdbfe'
            }}
          >
            <Save size={13} />
            Lưu vào backend\data\diagnostics
          </button>
        </div>
      </footer>

      {/* --- SCOREBOARD / BÁO CÁO CHẨN ĐOÁN 29 CHỮ MODAL --- */}
      {showScoreboard && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.75)',
          backdropFilter: 'blur(4px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: '16px'
        }}>
          <div style={{
            background: '#090e17',
            border: '1px solid rgba(56, 189, 248, 0.4)',
            borderRadius: '12px',
            width: '100%',
            maxWidth: '850px',
            maxHeight: '85vh',
            display: 'flex',
            flexDirection: 'column',
            boxShadow: '0 20px 50px rgba(0,0,0,0.6)',
            color: '#f8fafc'
          }}>
            {/* Modal Header */}
            <div style={{
              padding: '16px 20px',
              borderBottom: '1px solid rgba(255,255,255,0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between'
            }}>
              <div>
                <h3 style={{ margin: 0, fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <BarChart3 size={18} color="#38bdf8" />
                  Báo cáo Đánh giá Chẩn đoán 29 Chữ cái VSL
                </h3>
                <span style={{ fontSize: '11px', color: '#94a3b8' }}>Session: {sessionId}</span>
              </div>
              <button
                type="button"
                onClick={() => setShowScoreboard(false)}
                style={{ background: 'none', border: 'none', color: '#94a3b8', cursor: 'pointer', padding: 4 }}
              >
                <X size={20} />
              </button>
            </div>

            {/* Modal Body */}
            <div style={{ padding: '20px', overflowY: 'auto', display: 'grid', gap: '16px' }}>
              {/* Thẻ tóm tắt tổng quan */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                gap: '10px'
              }}>
                <div style={{ background: 'rgba(15,23,42,0.8)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8' }}>Chữ đã kiểm thử</span>
                  <div style={{ fontSize: '22px', fontWeight: 'bold', color: '#38bdf8' }}>
                    {currentReport.summary.evaluated_letters_count} / 29
                  </div>
                </div>

                <div style={{ background: 'rgba(34,197,94,0.12)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(34,197,94,0.3)' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8' }}>Chữ ĐẠT</span>
                  <div style={{ fontSize: '22px', fontWeight: 'bold', color: '#4ade80' }}>
                    {currentReport.summary.letters_passed.length}
                  </div>
                </div>

                <div style={{ background: 'rgba(239,68,68,0.12)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(239,68,68,0.3)' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8' }}>Chữ CẦN FIX</span>
                  <div style={{ fontSize: '22px', fontWeight: 'bold', color: '#f87171' }}>
                    {currentReport.summary.letters_needing_fix.length}
                  </div>
                </div>

                <div style={{ background: 'rgba(15,23,42,0.8)', padding: '12px', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.08)' }}>
                  <span style={{ fontSize: '11px', color: '#94a3b8' }}>Độ chính xác TB</span>
                  <div style={{ fontSize: '22px', fontWeight: 'bold', color: '#facc15' }}>
                    {currentReport.summary.overall_accuracy_pct}%
                  </div>
                </div>
              </div>

              {/* Danh sách chữ cần FIX */}
              {currentReport.summary.letters_needing_fix.length > 0 && (
                <div style={{
                  background: 'rgba(239, 68, 68, 0.08)',
                  border: '1px solid rgba(239, 68, 68, 0.3)',
                  borderRadius: '8px',
                  padding: '12px'
                }}>
                  <div style={{ fontSize: '13px', fontWeight: 'bold', color: '#f87171', marginBottom: '8px' }}>
                    ⚠️ CÁC CHỮ CẦN SỬA LOGIC NHẬN DIỆN ({currentReport.summary.letters_needing_fix.length} chữ):
                  </div>
                  <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
                    {currentReport.summary.letters_needing_fix.map(letter => {
                      const letterData = currentReport.letters[letter];
                      const confList = Object.entries(letterData.topConfusions || {})
                        .sort((a, b) => b[1] - a[1])
                        .slice(0, 2)
                        .map(([c, n]) => `${c}(${n})`)
                        .join(', ');

                      return (
                        <div
                          key={letter}
                          style={{
                            background: 'rgba(15, 23, 42, 0.8)',
                            border: '1px solid rgba(239, 68, 68, 0.4)',
                            borderRadius: '6px',
                            padding: '6px 10px',
                            fontSize: '12px'
                          }}
                        >
                          <strong style={{ color: '#f87171', fontSize: '14px' }}>{letter}</strong>
                          <span style={{ color: '#94a3b8', marginLeft: '6px' }}>({letterData.accuracyPct}%)</span>
                          {confList && (
                            <span style={{ color: '#fca5a5', display: 'block', fontSize: '10px' }}>
                              Nhầm sang: {confList}
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}

              {/* Bảng chi tiết 29 chữ cái */}
              <div>
                <h4 style={{ margin: '0 0 8px', fontSize: '13px', color: '#cbd5e1' }}>Chi tiết từng chữ cái (A — Y):</h4>
                <div style={{ border: '1px solid rgba(255,255,255,0.08)', borderRadius: '8px', overflow: 'hidden' }}>
                  <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: '12px' }}>
                    <thead>
                      <tr style={{ background: 'rgba(255,255,255,0.04)', color: '#94a3b8', textAlign: 'left' }}>
                        <th style={{ padding: '8px 12px' }}>Chữ</th>
                        <th style={{ padding: '8px 12px' }}>Lượt đạt</th>
                        <th style={{ padding: '8px 12px' }}>Độ khớp %</th>
                        <th style={{ padding: '8px 12px' }}>Trạng thái</th>
                        <th style={{ padding: '8px 12px' }}>Nhầm lẫn nhiều nhất</th>
                      </tr>
                    </thead>
                    <tbody>
                      {VSL_ALPHABET_29.map(letter => {
                        const lData = currentReport.letters[letter];
                        if (!lData || lData.status === 'UNTESTED') {
                          return (
                            <tr key={letter} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)', opacity: 0.5 }}>
                              <td style={{ padding: '8px 12px', fontWeight: 'bold' }}>{letter}</td>
                              <td style={{ padding: '8px 12px' }}>0 / 10</td>
                              <td style={{ padding: '8px 12px' }}>—</td>
                              <td style={{ padding: '8px 12px' }}>
                                <span style={{ padding: '2px 6px', borderRadius: '4px', background: 'rgba(148,163,184,0.1)', color: '#94a3b8', fontSize: '10px' }}>
                                  Chưa test
                                </span>
                              </td>
                              <td style={{ padding: '8px 12px' }}>—</td>
                            </tr>
                          );
                        }

                        const confs = Object.entries(lData.topConfusions || {})
                          .sort((a, b) => b[1] - a[1])
                          .slice(0, 3)
                          .map(([c, n]) => `${c} (${n})`)
                          .join(', ');

                        return (
                          <tr key={letter} style={{ borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
                            <td style={{ padding: '8px 12px', fontWeight: 'bold', fontSize: '13px', color: '#38bdf8' }}>{letter}</td>
                            <td style={{ padding: '8px 12px' }}>{lData.passedTrials} / {lData.totalTrials}</td>
                            <td style={{ padding: '8px 12px', fontWeight: 'bold', color: lData.accuracyPct >= 70 ? '#4ade80' : '#f87171' }}>
                              {lData.accuracyPct}%
                            </td>
                            <td style={{ padding: '8px 12px' }}>
                              <span style={{
                                padding: '2px 6px',
                                borderRadius: '4px',
                                fontSize: '10px',
                                fontWeight: 'bold',
                                background: lData.status === 'PASS' ? 'rgba(34,197,94,0.2)' : 'rgba(239,68,68,0.2)',
                                color: lData.status === 'PASS' ? '#4ade80' : '#f87171'
                              }}>
                                {lData.status}
                              </span>
                            </td>
                            <td style={{ padding: '8px 12px', color: '#fca5a5' }}>
                              {confs || 'Không có'}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Modal Footer */}
            <div style={{
              padding: '14px 20px',
              borderTop: '1px solid rgba(255,255,255,0.1)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              background: 'rgba(0,0,0,0.2)'
            }}>
              <button
                type="button"
                onClick={() => setShowScoreboard(false)}
                className="btn btn-secondary"
                style={{ fontSize: '12px', padding: '6px 14px' }}
              >
                Đóng
              </button>

              <div style={{ display: 'flex', gap: '8px' }}>
                <button
                  type="button"
                  onClick={handleDownloadJSON}
                  className="btn btn-secondary"
                  style={{ fontSize: '12px', padding: '6px 14px', display: 'flex', alignItems: 'center', gap: '4px' }}
                >
                  <Download size={14} />
                  Tải file JSON
                </button>
                <button
                  type="button"
                  onClick={handleSaveDiagnostics}
                  className="btn btn-primary"
                  style={{ fontSize: '12px', padding: '6px 16px', display: 'flex', alignItems: 'center', gap: '4px' }}
                >
                  <Save size={14} />
                  Lưu vào backend\data\diagnostics
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
