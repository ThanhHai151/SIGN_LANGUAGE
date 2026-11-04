import React, { useState, useEffect, useRef } from 'react';
import {
  FlaskConical, Play, Square, CheckCircle, RotateCcw,
  Save, AlertTriangle, ShieldCheck, Database, Timer, Activity
} from 'lucide-react';
import {
  setPilotLogging,
  flushPilotLogs,
  fetchPilotStats
} from '../services/api';

// 22 chữ cái tĩnh tiếng Việt VSL chuẩn hóa (VSL Alphabet V2)
export const STATIC_LETTERS_22 = [
  'A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L',
  'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y'
];

export default function PilotTestingHUD({
  isActive,
  onToggleActive,
  onMetadataUpdate,
  cameraActive
}) {
  // Session Configuration
  const [sessionId, setSessionId] = useState(() => {
    const d = new Date();
    const pad = (n) => String(n).padStart(2, '0');
    return `pilot_${d.getFullYear()}${pad(d.getMonth() + 1)}${pad(d.getDate())}_${pad(d.getHours())}${pad(d.getMinutes())}`;
  });

  const [selectedLetter, setSelectedLetter] = useState('A');
  const [currentRep, setCurrentRep] = useState(1);
  const [totalReps, setTotalReps] = useState(10);

  // Trial State: 'idle' | 'countdown' | 'recording' | 'completed'
  const [trialState, setTrialState] = useState('idle');
  const [countdownSeconds, setCountdownSeconds] = useState(5.0);
  const [trialFrameCount, setTrialFrameCount] = useState(0);

  // Backend Logging Telemetry
  const [loggingEnabled, setLoggingEnabled] = useState(false);
  const [backendStats, setBackendStats] = useState({
    queued_count: 0,
    logged_count: 0,
    dropped_queue_full: 0,
    dropped_io_error: 0,
    total_dropped: 0
  });
  const [flushStatus, setFlushStatus] = useState(null);

  // Refs for real-time frame metadata (non-reactive for 60 FPS loop)
  const trialStateRef = useRef(trialState);
  trialStateRef.current = trialState;

  const frameIdxRef = useRef(0);
  const timerIntervalRef = useRef(null);

  // Current Unique Trial ID
  const trialId = `${sessionId}_${selectedLetter}_rep${String(currentRep).padStart(2, '0')}`;

  // Đồng bộ metadata ra ngoài CameraSignRecognizer qua callback
  useEffect(() => {
    if (trialState === 'recording') {
      onMetadataUpdate({
        getFrameMetadata: () => {
          // BẢO VỆ RÀNG BUỘC VI GIÂY (Task 7F.5D): Kiểm tra ref đồng bộ ngay bên trong callback
          if (trialStateRef.current !== 'recording') {
            return null;
          }
          frameIdxRef.current += 1;
          setTrialFrameCount(frameIdxRef.current);
          return {
            expected_sign: selectedLetter,
            session_id: sessionId,
            trial_id: trialId,
            frame_idx: frameIdxRef.current
          };
        }
      });
    } else {
      // Khi không trong trial, metadata là null tuyệt đối
      onMetadataUpdate({
        getFrameMetadata: () => null
      });
    }
  }, [trialState, selectedLetter, sessionId, trialId]);

  // Quản lý Bật/Tắt chế độ Pilot Logging trên Backend
  const handleTogglePilotLogging = async (enable) => {
    const res = await setPilotLogging(enable, null, sessionId);
    if (res && res.success) {
      setLoggingEnabled(enable);
      if (enable && res.stats) {
        setBackendStats(res.stats);
      }
    }
  };

  // Định kỳ cập nhật stats từ Backend mỗi 2 giây khi mở HUD
  useEffect(() => {
    if (!isActive) return;
    const interval = setInterval(async () => {
      const res = await fetchPilotStats();
      if (res && res.success && res.stats) {
        setBackendStats(res.stats);
      }
    }, 2000);
    return () => clearInterval(interval);
  }, [isActive]);

  // Bắt đầu một Trial ghi nhận 5 giây
  const handleStartTrial = async () => {
    if (!cameraActive) {
      alert('Vui lòng bật Camera trước khi bắt đầu Trial thử nghiệm!');
      return;
    }

    // Tự động bật pilot logging nếu chưa bật
    if (!loggingEnabled) {
      await handleTogglePilotLogging(true);
    }

    frameIdxRef.current = 0;
    setTrialFrameCount(0);
    setFlushStatus(null);
    setTrialState('recording');
    setCountdownSeconds(5.0);

    const startTime = Date.now();
    const durationMs = 5000;

    clearInterval(timerIntervalRef.current);
    timerIntervalRef.current = setInterval(() => {
      const elapsed = Date.now() - startTime;
      const remaining = Math.max(0, (durationMs - elapsed) / 1000);
      setCountdownSeconds(remaining);

      if (elapsed >= durationMs) {
        clearInterval(timerIntervalRef.current);
        handleEndTrial();
      }
    }, 50);
  };

  // Kết thúc một Trial (tự động sau 5s hoặc bấm Dừng thủ công)
  const handleEndTrial = async () => {
    trialStateRef.current = 'completed'; // Vô hiệu hóa metadata ngay lập tức ở cấp vi giây (Task 7F.5D)
    clearInterval(timerIntervalRef.current);
    setTrialState('completed');

    // Flush ngay lập tức để đẩy dữ liệu trial xuống file JSONL
    const flushRes = await flushPilotLogs();
    if (flushRes && flushRes.success && flushRes.stats) {
      setBackendStats(flushRes.stats);
      setFlushStatus(`Đã lưu ${frameIdxRef.current} frames vào JSONL`);
    }

    // Tự động tăng repetition cho lần thử tiếp theo nếu chưa đạt tối đa
    if (currentRep < totalReps) {
      setCurrentRep(prev => prev + 1);
    }
  };

  // Hủy trial đang chạy
  const handleCancelTrial = () => {
    trialStateRef.current = 'idle'; // Vô hiệu hóa metadata ngay lập tức (Task 7F.5D)
    clearInterval(timerIntervalRef.current);
    setTrialState('idle');
    frameIdxRef.current = 0;
    setTrialFrameCount(0);
  };

  // Kết thúc toàn bộ Session và Flush dữ liệu
  const handleFinishSession = async () => {
    handleCancelTrial();
    const res = await flushPilotLogs();
    await handleTogglePilotLogging(false);
    if (res && res.success && res.stats) {
      setBackendStats(res.stats);
      alert(`Đã hoàn tất Session "${sessionId}"!\nTổng số record đã lưu: ${res.stats.logged_count}\nThất thoát: ${res.stats.total_dropped || 0}`);
    }
    setFlushStatus('Session hoàn tất');
  };

  if (!isActive) {
    return (
      <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '-8px' }}>
        <button
          onClick={onToggleActive}
          className="btn btn-secondary"
          style={{
            fontSize: '11px',
            padding: '5px 10px',
            display: 'flex',
            alignItems: 'center',
            gap: '6px',
            background: 'rgba(30, 41, 59, 0.7)',
            borderColor: 'var(--border-subtle)'
          }}
          title="Bật giao diện kiểm thử Controlled Webcam Pilot (Task 7F.5B)"
        >
          <FlaskConical size={14} style={{ color: '#38bdf8' }} />
          Mở Pilot Testing HUD
        </button>
      </div>
    );
  }

  const progressPercent = Math.min(100, Math.max(0, ((5.0 - countdownSeconds) / 5.0) * 100));

  return (
    <div
      style={{
        background: 'linear-gradient(135deg, rgba(15, 23, 42, 0.95), rgba(30, 41, 59, 0.95))',
        border: '1px solid rgba(56, 189, 248, 0.35)',
        borderRadius: 'var(--radius-md)',
        padding: '14px',
        display: 'flex',
        flexDirection: 'column',
        gap: '12px',
        boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
        animation: 'fadeIn 0.2s ease-in-out'
      }}
    >
      {/* HUD Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.08)', paddingBottom: '8px' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <FlaskConical size={18} style={{ color: '#38bdf8' }} />
          <div>
            <h4 style={{ fontSize: '13px', margin: 0, fontWeight: '700', color: '#f8fafc', display: 'flex', alignItems: 'center', gap: '6px' }}>
              Controlled Webcam Pilot HUD
              <span style={{
                fontSize: '10px',
                padding: '2px 6px',
                borderRadius: '4px',
                background: loggingEnabled ? 'rgba(34, 197, 94, 0.2)' : 'rgba(148, 163, 184, 0.2)',
                color: loggingEnabled ? '#4ade80' : '#94a3b8',
                border: `1px solid ${loggingEnabled ? 'rgba(34, 197, 94, 0.4)' : 'rgba(148, 163, 184, 0.3)'}`
              }}>
                {loggingEnabled ? 'LOGGING ACTIVE' : 'LOGGING IDLE'}
              </span>
            </h4>
            <p style={{ fontSize: '11px', margin: 0, color: 'var(--text-muted)' }}>
              Thu thập dữ liệu định lượng 22 chữ cái tĩnh kèm Ground Truth có kiểm soát
            </p>
          </div>
        </div>

        <button
          onClick={onToggleActive}
          className="btn btn-secondary"
          style={{ fontSize: '11px', padding: '4px 8px' }}
        >
          Đóng HUD
        </button>
      </div>

      {/* Control Grid: Session, Target Letter, Repetition */}
      <div style={{ display: 'grid', gridTemplateColumns: '1.2fr 1fr 1fr', gap: '10px', alignItems: 'center' }}>
        {/* Session ID */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <label style={{ fontSize: '10px', color: '#94a3b8', fontWeight: '600', textTransform: 'uppercase' }}>
            Session ID:
          </label>
          <input
            type="text"
            value={sessionId}
            onChange={(e) => setSessionId(e.target.value.replace(/[^a-zA-Z0-9_-]/g, ''))}
            disabled={trialState === 'recording'}
            style={{
              fontSize: '11px',
              padding: '6px 8px',
              background: 'rgba(0,0,0,0.3)',
              border: '1px solid var(--border-subtle)',
              borderRadius: '4px',
              color: '#f8fafc'
            }}
          />
        </div>

        {/* Target Letter (22 chữ tĩnh) */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <label style={{ fontSize: '10px', color: '#94a3b8', fontWeight: '600', textTransform: 'uppercase' }}>
            Chữ mục tiêu (Ground Truth):
          </label>
          <select
            value={selectedLetter}
            onChange={(e) => {
              setSelectedLetter(e.target.value);
              setCurrentRep(1);
            }}
            disabled={trialState === 'recording'}
            style={{
              fontSize: '12px',
              fontWeight: '700',
              padding: '5px 8px',
              background: 'rgba(15, 23, 42, 0.8)',
              border: '1px solid #38bdf8',
              borderRadius: '4px',
              color: '#38bdf8'
            }}
          >
            {STATIC_LETTERS_22.map((letCode) => (
              <option key={letCode} value={letCode}>
                Chữ cái {letCode}
              </option>
            ))}
          </select>
        </div>

        {/* Repetition Selector */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
          <label style={{ fontSize: '10px', color: '#94a3b8', fontWeight: '600', textTransform: 'uppercase' }}>
            Lần lặp (Repetition):
          </label>
          <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
            <select
              value={currentRep}
              onChange={(e) => setCurrentRep(Number(e.target.value))}
              disabled={trialState === 'recording'}
              style={{
                fontSize: '11px',
                padding: '5px 8px',
                background: 'rgba(0,0,0,0.3)',
                border: '1px solid var(--border-subtle)',
                borderRadius: '4px',
                color: '#f8fafc',
                flex: 1
              }}
            >
              {Array.from({ length: totalReps }, (_, i) => i + 1).map((n) => (
                <option key={n} value={n}>
                  Lần {n} / {totalReps}
                </option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Trial Action & Countdown Banner */}
      <div
        style={{
          background: trialState === 'recording' ? 'rgba(56, 189, 248, 0.12)' : 'rgba(0, 0, 0, 0.25)',
          border: `1px solid ${trialState === 'recording' ? 'rgba(56, 189, 248, 0.5)' : 'var(--border-subtle)'}`,
          borderRadius: 'var(--radius-md)',
          padding: '12px',
          display: 'flex',
          flexDirection: 'column',
          gap: '8px'
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '8px' }}>
          <div>
            <span style={{ fontSize: '10px', color: '#94a3b8', fontWeight: '600' }}>TRIAL HIỆN TẠI:</span>
            <div style={{ fontSize: '12px', fontWeight: '700', color: '#f8fafc', fontFamily: 'monospace' }}>
              {trialId}
            </div>
          </div>

          <div style={{ display: 'flex', gap: '8px' }}>
            {trialState === 'recording' ? (
              <button
                onClick={handleCancelTrial}
                className="btn btn-danger"
                style={{ fontSize: '11px', padding: '6px 12px', display: 'flex', alignItems: 'center', gap: '6px' }}
              >
                <Square size={13} /> Dừng Trial
              </button>
            ) : (
              <button
                onClick={handleStartTrial}
                className="btn btn-primary"
                style={{
                  fontSize: '12px',
                  fontWeight: '700',
                  padding: '7px 16px',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '6px',
                  background: '#0284c7',
                  borderColor: '#38bdf8'
                }}
              >
                <Play size={14} /> Bắt đầu Trial (5s Hold)
              </button>
            )}

            <button
              onClick={handleFinishSession}
              className="btn btn-secondary"
              style={{ fontSize: '11px', padding: '6px 10px', display: 'flex', alignItems: 'center', gap: '5px' }}
              title="Flush toàn bộ dữ liệu và đóng phiên"
            >
              <Save size={13} /> Flush & Xong
            </button>
          </div>
        </div>

        {/* Progress bar countdown when recording */}
        {trialState === 'recording' && (
          <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', fontWeight: '600', color: '#38bdf8' }}>
              <span>GIỮ YÊN TƯ THẾ CHỮ {selectedLetter}...</span>
              <span>{countdownSeconds.toFixed(1)}s ({trialFrameCount} frames)</span>
            </div>
            <div style={{ width: '100%', height: '8px', background: 'rgba(255,255,255,0.1)', borderRadius: '4px', overflow: 'hidden' }}>
              <div
                style={{
                  width: `${progressPercent}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, #38bdf8, #22c55e)',
                  transition: 'width 0.05s linear'
                }}
              />
            </div>
          </div>
        )}

        {flushStatus && trialState !== 'recording' && (
          <div style={{ fontSize: '11px', color: '#4ade80', display: 'flex', alignItems: 'center', gap: '5px' }}>
            <CheckCircle size={13} /> {flushStatus}
          </div>
        )}
      </div>

      {/* Real-time Telemetry Stats Footer */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        fontSize: '10px',
        color: '#94a3b8',
        paddingTop: '6px',
        borderTop: '1px solid rgba(255,255,255,0.06)'
      }}>
        <div style={{ display: 'flex', gap: '14px' }}>
          <span>Queued: <strong style={{ color: '#f8fafc' }}>{backendStats.queued_count}</strong></span>
          <span>Logged: <strong style={{ color: '#4ade80' }}>{backendStats.logged_count}</strong></span>
          <span>Thất thoát: <strong style={{ color: backendStats.total_dropped > 0 ? '#ef4444' : '#f8fafc' }}>{backendStats.total_dropped || 0}</strong></span>
        </div>
        <div>
          <span>Chữ tĩnh: <strong style={{ color: '#38bdf8' }}>{selectedLetter}</strong> ({currentRep}/{totalReps})</span>
        </div>
      </div>
    </div>
  );
}

