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
    try {
      const st = await startWordDebugSession(selected.class, selected.word, Number(trial) || 1);
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
      await stopWordDebugSession();
      setWordDebugActive(false);
      setStatus({ active: false });
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
      <div style={{ fontWeight: 700, marginBottom: '8px' }}>🧪 WORD DEBUG TEST</div>

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

      <div>
        Status:{' '}
        <b style={{ color: recording ? '#ef4444' : 'inherit' }}>{recording ? '● Recording' : 'Idle'}</b>
        {recording && (
          <span>
            {' '}— {status.target_word} (class {status.target_class}), trial {status.trial},{' '}
            {status.seq_count || 0} lượt dự đoán
          </span>
        )}
      </div>
      {error && <div style={{ color: '#ef4444', marginTop: '6px' }}>{error}</div>}
    </div>
  );
}
