import React, { useState, useEffect } from 'react';
import {
  fetchWordTargets,
  fetchWordDebugStatus,
  startWordDebugSession,
  stopWordDebugSession,
  setWordDebugActive
} from '../services/wordDebugApi';

/**
 * Panel debug cho Word Mode: chọn target + trial, Start/Stop để backend ghi log có ground truth.
 * Chỉ được render khi mode === 'words' và Word Debug được bật (xem CameraSignRecognizer).
 */
export default function WordDebugPanel() {
  const [targets, setTargets] = useState([]);
  const [targetClass, setTargetClass] = useState('');
  const [trial, setTrial] = useState(1);
  const [status, setStatus] = useState({ active: false });
  const [lastSummary, setLastSummary] = useState(null);
  const [error, setError] = useState('');
  const [busy, setBusy] = useState(false);

  // Nạp 63 từ (từ backend) + đồng bộ trạng thái phiên (phòng khi reload trang giữa chừng)
  useEffect(() => {
    let cancelled = false;
    fetchWordTargets()
      .then((list) => { if (!cancelled) setTargets(list); })
      .catch((e) => { if (!cancelled) setError(`Không tải được danh sách từ: ${e.message}`); });
    fetchWordDebugStatus()
      .then((st) => {
        if (cancelled) return;
        setStatus(st);
        setWordDebugActive(st.active);
        if (st.active) {
          setTargetClass(String(st.target_class));
          setTrial(st.trial || 1);
        }
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // Khi đang ghi: cập nhật số lượt dự đoán mỗi giây (chỉ đọc status)
  useEffect(() => {
    if (!status.active) return undefined;
    const timer = setInterval(() => {
      fetchWordDebugStatus().then(setStatus).catch(() => {});
    }, 1000);
    return () => clearInterval(timer);
  }, [status.active]);

  const handleStart = async () => {
    const selected = targets.find((t) => String(t.class) === String(targetClass));
    if (!selected) {
      setError('Hãy chọn Target word trước khi Start.');
      return;
    }
    setBusy(true);
    setError('');
    setLastSummary(null);
    try {
      const st = await startWordDebugSession(selected.class, selected.word, Number(trial) || 1, 'A01_1_T01');
      setStatus(st);
      setWordDebugActive(true);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const handleStop = async () => {
    setBusy(true);
    setError('');
    try {
      const res = await stopWordDebugSession();
      setWordDebugActive(false);
      setStatus({ active: false });
      if (res && res.summary) {
        setLastSummary(res.summary);
      }
      setTrial((n) => (Number(n) || 1) + 1); // gợi ý trial kế tiếp (vẫn sửa tay được)
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  };

  const recording = Boolean(status.active);

  return (
    <div
      style={{
        border: '1px dashed var(--accent-amber, #f59e0b)',
        borderRadius: 'var(--radius-sm, 8px)',
        padding: '10px 12px',
        marginTop: '10px',
        background: 'rgba(245, 158, 11, 0.06)',
        fontSize: '12px'
      }}
    >
      <div style={{ fontWeight: 700, marginBottom: '8px', display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span>🧪 WORD DEBUG TEST (ĐỘC LẬP TỪNG TRIAL)</span>
        <span style={{ fontSize: '11px', background: 'rgba(59, 130, 246, 0.15)', color: '#2563eb', padding: '2px 6px', borderRadius: '4px', fontWeight: 600 }}>
          🏷️ Đợt test: A01_1_T01
        </span>
      </div>

      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
        <label style={{ minWidth: '78px' }}>Target word</label>
        <select
          value={targetClass}
          onChange={(e) => setTargetClass(e.target.value)}
          disabled={recording || busy}
          style={{ flex: 1, padding: '4px 6px' }}
        >
          <option value="">-- chọn từ --</option>
          {targets.map((t) => (
            <option key={t.class} value={t.class}>{t.word} (class {t.class})</option>
          ))}
        </select>
      </div>

      <div style={{ display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '8px' }}>
        <label style={{ minWidth: '78px' }}>Trial</label>
        <input
          type="number"
          min="1"
          value={trial}
          onChange={(e) => setTrial(e.target.value)}
          disabled={recording || busy}
          style={{ width: '70px', padding: '4px 6px' }}
        />
      </div>

      <div style={{ display: 'flex', gap: '8px', marginBottom: '8px' }}>
        <button
          className="btn btn-primary"
          onClick={handleStart}
          disabled={recording || busy || !targetClass}
          style={{ flex: 1, padding: '6px', fontSize: '12px', justifyContent: 'center' }}
        >
          Start Test
        </button>
        <button
          className="btn btn-secondary"
          onClick={handleStop}
          disabled={!recording || busy}
          style={{ flex: 1, padding: '6px', fontSize: '12px', justifyContent: 'center' }}
        >
          Stop Test
        </button>
      </div>

      <div style={{ marginBottom: lastSummary ? '8px' : '0' }}>
        Status:{' '}
        <b style={{ color: recording ? '#ef4444' : 'inherit' }}>{recording ? '● Recording' : 'Idle'}</b>
        {recording && (
          <span>
            {' '}— {status.target_word} (class {status.target_class}), trial {status.trial},{' '}
            {status.seq_count || 0} cửa sổ (windows)
          </span>
        )}
      </div>

      {lastSummary && (
        <div
          style={{
            marginTop: '8px',
            padding: '8px 10px',
            borderRadius: '6px',
            background: lastSummary.verdict === 'CORRECT'
              ? 'rgba(16, 185, 129, 0.15)'
              : lastSummary.verdict === 'PARTIAL'
              ? 'rgba(245, 158, 11, 0.15)'
              : 'rgba(239, 68, 68, 0.15)',
            border: `1px solid ${
              lastSummary.verdict === 'CORRECT' ? '#10b981' : lastSummary.verdict === 'PARTIAL' ? '#f59e0b' : '#ef4444'
            }`
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '4px' }}>
            <span style={{ fontWeight: 700 }}>📊 KẾT QUẢ TRIAL #{lastSummary.trial}: {lastSummary.target_word}</span>
            <span
              style={{
                fontWeight: 700,
                padding: '2px 6px',
                borderRadius: '4px',
                fontSize: '11px',
                background: lastSummary.verdict === 'CORRECT' ? '#10b981' : lastSummary.verdict === 'PARTIAL' ? '#f59e0b' : '#ef4444',
                color: '#fff'
              }}
            >
              {lastSummary.verdict}
            </span>
          </div>
          <div style={{ fontSize: '11px', lineHeight: '1.4' }}>
            <div>• Cửa sổ đúng: <b>{lastSummary.correct_windows}/{lastSummary.total_windows}</b> ({Math.round(lastSummary.accuracy * 100)}%)</div>
            <div>• Từ dự đoán chủ đạo: <b>{lastSummary.dominant_predicted_label}</b> | Conf max: <b>{lastSummary.target_max_confidence}</b></div>
            <div>• Nguồn mốc: <b>{lastSummary.landmarks_source}</b> {lastSummary.is_static_shoulders ? '(Vai tĩnh ⚠️)' : '(Vai động ✓)'}</div>
            <div>• Reset buffer: <b>{lastSummary.total_resets}</b> lần | FPS TB: <b>{lastSummary.avg_client_fps || 'N/A'}</b> | Model infer: <b>{lastSummary.avg_inference_ms || 'N/A'} ms</b></div>
            <div style={{ fontSize: '10px', color: '#6b7280', marginTop: '4px' }}>
              Đã lưu CSV & logs: <code>{lastSummary.session_id}</code>
            </div>
          </div>
        </div>
      )}

      {error && <div style={{ color: '#ef4444', marginTop: '6px' }}>{error}</div>}
    </div>
  );
}
