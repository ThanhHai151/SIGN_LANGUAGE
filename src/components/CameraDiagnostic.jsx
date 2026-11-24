import { useRef, useState } from 'react';
import { DIAGNOSTIC_RECORD_MS } from './useCameraDiagnostic';
import './CameraDiagnostic.css';

const RECORD_SEC = Math.round(DIAGNOSTIC_RECORD_MS / 1000);
const MODES = { words: 'Từ vựng', letters: 'Chữ cái', letters_numbers: 'Chữ cái và số', digits: 'Chữ số' };
const SOURCES = {
  qipedc_template: 'Mẫu QIPEDC',
  qipedc_practice: 'Luyện QIPEDC',
  qipedc_gate: 'Cổng lọc QIPEDC',
  sstcn: 'SSTCN',
  landmarks: 'Nhận diện tay',
};
const FRONTEND_ACTIONS = {
  accepted_qipedc: 'Đã chốt QIPEDC',
  accepted_sstcn: 'Đã chốt SSTCN',
  confirmed_backend: 'Đã chốt',
  confirmed_caption: 'Đã ghi phụ đề',
  waiting_qipedc_2nd_vote: 'Chờ phiếu QIPEDC 2/2',
  waiting_sstcn_vote: 'Chờ phiếu SSTCN 2/2',
  ignored_template_locked: 'Đang khóa sau mẫu trước',
  ignored_already_has_token: 'Đã có từ trong lượt này',
  discarded_stale_window: 'Bỏ qua (cửa sổ cũ)',
  neutral: 'Chưa chốt (NEUTRAL)',
  unconfirmed_smoothing: 'Chờ ổn định / cooldown',
  backend_error: 'Lỗi backend',
  network_exception: 'Lỗi mạng',
};
const REASONS = {
  matched: 'Khớp mẫu “mấy giờ?”',
  word_matched: 'Khớp chuyển động với mẫu QIPEDC',
  ambiguous_word: 'Chuyển động gần nhiều mẫu QIPEDC',
  practice_matched: 'Khớp mẫu đã chọn',
  practice_mismatch: 'Chuỗi chuyển động còn khác mẫu đã chọn',
  low_similarity: 'Thấy cả hai pha, nhưng chuỗi chuyển động còn khác mẫu',
  missing_both_hands: 'Thấy hai pha nhưng thiếu khung hình hai tay để xác nhận mẫu',
  lost_hand_during_interaction: 'Camera thấy hai tay đưa lại gần, sau đó mất dấu một tay',
  tracking_gaps: 'Mất dấu tay hoặc vai trong chuỗi',
  lift_not_seen: 'Thấy pha chỉ cổ tay, chưa thấy tay chỉ đưa lên sau đó',
  wrist_not_seen: 'Chưa thấy ngón trỏ đến gần cổ tay bên kia',
  no_motion: 'Chuyển động tay chưa đủ ngưỡng (0.16) trong cửa sổ này',
  insufficient_landmarks: 'Chưa thu đủ khung hình có vai và tay',
  topic_excluded: 'Chủ đề đang chọn loại trừ mẫu “mấy giờ?”',
  reference_unavailable: 'Mẫu QIPEDC chưa sẵn sàng',
  holistic_pose_missing: 'Mất dấu khung thân trên',
  hands_fallback_pose_blocked: 'Chế độ này cần khung thân trên thật',
  inference_error: 'Yêu cầu nhận diện bị lỗi',
};
const PHASE_ORDER = ['matched', 'low_similarity', 'missing_both_hands', 'tracking_gaps', 'lift_not_seen',
  'wrist_not_seen', 'reference_unavailable', 'topic_excluded', 'no_motion', 'insufficient_landmarks'];

const seconds = (ms) => `${(ms / 1000).toFixed(2).replace('.', ',')} s`;
const percent = (value) => `${Math.round(value * 100)}%`;
const wordKey = (value) => (value || '').normalize('NFC').toLowerCase().replace(/[!?.,]+$/u, '').trim();

function buildSummary(review) {
  const frames = review?.frames || [];
  const events = review ? [...review.events].sort((a, b) => a.ms - b.ms) : [];
  const pipelineLog = review?.pipelineLog || [];
  const poseFrames = frames.filter((frame) => frame.shoulders !== null);
  const leftHands = frames.filter((frame) => frame.left).length;
  const rightHands = frames.filter((frame) => frame.right).length;
  const bothHands = frames.filter((frame) => frame.left && frame.right).length;
  const noHands = frames.filter((frame) => !frame.left && !frame.right).length;
  const bridgedFrames = frames.filter((frame) => frame.leftBridged || frame.rightBridged).length;
  const rtts = events.map((e) => e.rttMs).filter((v) => typeof v === 'number');
  const serverTimes = events.map((e) => e.serverMs).filter((v) => typeof v === 'number');
  const avgFps = frames.length && review.durationMs > 0
    ? Number(((frames.length * 1000) / review.durationMs).toFixed(1))
    : 0;
  const avgRttMs = rtts.length ? Math.round(rtts.reduce((a, b) => a + b, 0) / rtts.length) : null;
  const avgServerMs = serverTimes.length ? Math.round(serverTimes.reduce((a, b) => a + b, 0) / serverTimes.length) : null;
  const captions = events.filter((e) => e.caption).map((e) => e.caption);
  const stalePendingSkips = pipelineLog.filter((item) => item.kind === 'request_pending_kept_stale').length;
  const bufferResets = pipelineLog.filter((item) => item.kind === 'buffer_reset').length;
  const forcedBilateralCount = events.filter((e) => e.backendDiagnostic?.qipedc?.bilateral_forced_by_2frame_rule).length;

  return {
    totalFrames: frames.length,
    avgFps,
    leftHandRatio: frames.length ? Number((leftHands / frames.length).toFixed(3)) : 0,
    rightHandRatio: frames.length ? Number((rightHands / frames.length).toFixed(3)) : 0,
    bothHandsRatio: frames.length ? Number((bothHands / frames.length).toFixed(3)) : 0,
    noHandsRatio: frames.length ? Number((noHands / frames.length).toFixed(3)) : 0,
    shouldersRatio: poseFrames.length ? Number((poseFrames.filter((f) => f.shoulders).length / poseFrames.length).toFixed(3)) : null,
    bridgedFrames,
    totalPredictions: events.length,
    avgRttMs,
    avgServerMs,
    captions,
    stalePendingSkips,
    bufferResets,
    forcedBilateralCount,
  };
}

function formatTextReport(review, summary) {
  const events = [...review.events].sort((a, b) => a.ms - b.ms);
  const lines = [
    `=== VSL CHẨN ĐOÁN CAMERA (${(review.durationMs / 1000).toFixed(2)}s) ===`,
    `Chế độ: ${review.settings.mode} (${review.settings.captureMode}) | Engine: ${review.settings.trackerEngine || 'unknown'} | Độ phân giải: ${review.settings.resolution || 'n/a'} | Chủ đề: ${review.settings.topic || 'Tất cả'}`,
    `Mục tiêu: ${review.settings.expected || '(không nhập)'}`,
    `Tracking: ${summary.totalFrames} frames (~${summary.avgFps} FPS) | Trái ${percent(summary.leftHandRatio)} | Phải ${percent(summary.rightHandRatio)} | 2 tay ${percent(summary.bothHandsRatio)} | Mất tay ${percent(summary.noHandsRatio)}${summary.shouldersRatio !== null ? ` | 2 vai ${percent(summary.shouldersRatio)}` : ''}`,
    `Hiệu năng: ${summary.totalPredictions} lượt dự đoán | RTT TB: ${summary.avgRttMs ?? 'n/a'} ms | Server TB: ${summary.avgServerMs ?? 'n/a'} ms | Reset buffer: ${summary.bufferResets} | Bỏ cửa sổ pending mới: ${summary.stalePendingSkips} | Ép 2 tay (2-frame): ${summary.forcedBilateralCount}`,
    `Phụ đề đã ghi: ${summary.captions.join(' · ') || '(chưa có)'}`,
    '',
    '--- CHI TIẾT TỪNG LƯỢT PHÂN TÍCH ---',
  ];
  for (const ev of events) {
    const mg = ev.backendDiagnostic?.motion_gate;
    const qp = ev.backendDiagnostic?.qipedc;
    const mgText = mg ? `extent(L=${mg.left_extent ?? '-'},R=${mg.right_extent ?? '-'},thr=${mg.motion_threshold})` : '';
    const qpText = qp?.top_candidates?.length
      ? `QIPEDC[${qp.bucket_hands}tay/${qp.bucket_total_words},gap=${qp.gap_top1_top2 ?? '-'}]: ` +
        qp.top_candidates.slice(0, 3).map((c) => `${c.word}(${c.distance})`).join(', ')
      : '';
    const sstcnTop = ev.backendDiagnostic?.sstcn?.top_k?.length
      ? `SSTCN: ${ev.backendDiagnostic.sstcn.top_k.slice(0, 3).map((t) => `${t.term}(${Math.round(t.confidence * 100)}%)`).join(', ')}`
      : '';
    lines.push(
      `[#${ev.reqId ?? '-'} @ ${(ev.ms / 1000).toFixed(2)}s | ${ev.windowFrames ?? '-'}f/${ev.windowDurationMs ?? '-'}ms | RTT=${ev.rttMs ?? '-'}ms srv=${ev.serverMs ?? '-'}ms${ev.isFlush ? ' FLUSH' : ''}] ` +
      `Kết quả=${ev.label || ev.displayLabel || 'NEUTRAL'}${ev.confidence !== null ? ` (${percent(ev.confidence)})` : ''} | Nguồn=${ev.source} | Frontend=${ev.frontendAction || '-'}${ev.caption ? ` | GHI="${ev.caption}"` : ''}`
    );
    if (ev.explanation) lines.push(`  Lý do: ${ev.explanation}`);
    if (mgText || qpText || sstcnTop) lines.push(`  Số liệu: ${[mgText, qpText, sstcnTop].filter(Boolean).join(' | ')}`);
  }
  return lines.join('\n');
}

function exportReport(review, summary) {
  const payload = {
    schema_version: 'vsl-camera-diagnostic-v2',
    settings: review.settings,
    durationMs: review.durationMs,
    summary,
    frames: review.frames,
    events: review.events,
    pipelineLog: review.pipelineLog || [],
  };
  const data = JSON.stringify(payload, null, 2);
  const reportUrl = URL.createObjectURL(new Blob([data], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = reportUrl;
  link.download = 'vsl-chan-doan-full-log.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(reportUrl), 1000);
}

export default function CameraDiagnostic({
  capture,
  cameraActive,
  stream,
  mode,
  captureMode,
  timed,
  topic,
  expectedSign,
  trackerEngine,
  detailedTracking,
  getResolution,
}) {
  const [expected, setExpected] = useState(expectedSign || '');
  const [showAllRows, setShowAllRows] = useState(true);
  const [copiedLog, setCopiedLog] = useState(false);
  const playerRef = useRef(null);
  const { status, remainingSec, review, error } = capture;
  const busy = status !== 'idle';
  const seek = (ms) => {
    const player = playerRef.current;
    if (!player) return;
    player.pause();
    player.currentTime = ms / 1000;
  };

  const events = review ? [...review.events].sort((a, b) => a.ms - b.ms) : [];
  const rows = showAllRows
    ? events
    : events.filter((event, i) => !i || event.caption || event.label !== events[i - 1].label ||
      event.status !== events[i - 1].status || event.frontendAction !== events[i - 1].frontendAction ||
      Math.floor(event.ms / 1000) !== Math.floor(events[i - 1].ms / 1000));
  const summary = review ? buildSummary(review) : null;
  const expectedKey = wordKey(review?.settings.expected);
  const bestPhase = expectedKey === 'mấy giờ' || (!expectedKey && events.some((event) => event.liftMs !== null))
    ? PHASE_ORDER.map((key) => events.find((event) => event.status === key)).find(Boolean)
    : null;
  const frames = review?.frames || [];
  const modeMismatch = expectedKey === 'mấy giờ' && review.settings.captureMode !== 'live_words';
  const trackingWindows = new Map();
  for (const frame of frames) {
    const second = Math.floor(frame.ms / 1000);
    if (!trackingWindows.has(second)) trackingWindows.set(second, []);
    trackingWindows.get(second).push(frame);
  }
  const alternative = expectedKey && events.reduce((best, event) =>
    event.source === 'sstcn' && event.label && wordKey(event.label) !== expectedKey &&
      (best === null || event.confidence > best.confidence) ? event : best, null);

  const handleCopySummary = () => {
    if (!review || !summary) return;
    navigator.clipboard?.writeText(formatTextReport(review, summary)).then(() => {
      setCopiedLog(true);
      setTimeout(() => setCopiedLog(false), 2000);
    }).catch(() => {});
  };

  return (
    <section className="glass-panel camera-diagnostic" aria-labelledby="camera-diagnostic-title">
      <div className="camera-diagnostic-actions">
        <h3 id="camera-diagnostic-title">Chẩn đoán video {RECORD_SEC} giây (Full Log)</h3>
        <button type="button" className="btn btn-primary"
          disabled={status !== 'recording' && (!cameraActive || timed || busy)}
          onClick={() => status === 'recording'
            ? capture.stop()
            : capture.start(stream(), {
              mode,
              captureMode,
              topic,
              expected: expected.trim(),
              trackerEngine,
              detailedTracking,
              resolution: getResolution ? getResolution() : null,
            })}>
          {status === 'recording'
            ? `Dừng ghi (còn ${remainingSec}s)`
            : status === 'starting'
              ? 'Đang bật bộ ghi…'
              : status === 'finishing'
                ? 'Đang tổng hợp full log…'
                : `Ghi chẩn đoán ${RECORD_SEC} giây`}
        </button>
      </div>
      <label className="camera-diagnostic-field">
        Các ký hiệu bạn dự định thử (tùy chọn)
        <input className="input" value={expected} onChange={(event) => setExpected(event.target.value)}
          disabled={busy} placeholder="Ví dụ: bực mình, xin lỗi, mấy giờ?" maxLength={120} />
      </label>
      <p className="camera-diagnostic-note" role="status">
        {error || (status === 'recording'
          ? `Đang ghi (${remainingSec}s còn lại). Hãy thực hiện các từ bạn muốn kiểm tra; toàn bộ quy trình camera → backend → ra chữ đang được ghi lại.`
          : 'Ghi hình tối đa 10 giây kèm toàn bộ log từng bước (tracking, biên độ chuyển động, Top ứng viên QIPEDC/SSTCN, khoảng cách DTW, độ trễ và lý do chốt chữ).')}
      </p>
      {review && summary && (
        <div className="camera-diagnostic-review">
          <video ref={playerRef} src={review.url} controls playsInline preload="metadata" aria-label="Video chẩn đoán đã ghi" />
          <div className="camera-diagnostic-actions">
            <span>{seconds(review.durationMs)} · {MODES[review.settings.mode]}{review.settings.expected && ` · Mục tiêu “${review.settings.expected}”`}</span>
            <a className="btn btn-secondary" href={review.url} download={`vsl-chan-doan.${review.extension}`}>Tải video</a>
            <button type="button" className="btn btn-primary" onClick={() => exportReport(review, summary)}>Tải full log (JSON)</button>
            <button type="button" className="btn btn-secondary" onClick={handleCopySummary}>
              {copiedLog ? 'Đã chép log' : 'Chép tóm tắt log'}
            </button>
            <button type="button" className="btn btn-secondary" onClick={capture.clear}>Xóa clip</button>
          </div>
          {frames.length ? (
            <div className="camera-diagnostic-summary">
              <span><strong>{summary.totalFrames}</strong> frame (~{summary.avgFps} FPS)</span>
              <span>Trái <strong>{percent(summary.leftHandRatio)}</strong> · Phải <strong>{percent(summary.rightHandRatio)}</strong> · 2 tay <strong>{percent(summary.bothHandsRatio)}</strong> · Mất tay <strong>{percent(summary.noHandsRatio)}</strong></span>
              {summary.shouldersRatio !== null && <span>Thấy 2 vai <strong>{percent(summary.shouldersRatio)}</strong></span>}
              <span><strong>{summary.totalPredictions}</strong> lượt gửi · RTT TB <strong>{summary.avgRttMs ?? '—'} ms</strong> · Server TB <strong>{summary.avgServerMs ?? '—'} ms</strong></span>
              {(summary.forcedBilateralCount > 0 || summary.stalePendingSkips > 0) && (
                <span className="camera-diagnostic-warn">
                  {summary.forcedBilateralCount > 0 && `Ép sang 2 tay (do tay thừa >=2f): ${summary.forcedBilateralCount} lượt`}
                  {summary.forcedBilateralCount > 0 && summary.stalePendingSkips > 0 && ' · '}
                  {summary.stalePendingSkips > 0 && `Bỏ qua cửa sổ mới vì pending cũ: ${summary.stalePendingSkips} lần`}
                </span>
              )}
            </div>
          ) : <p>Camera ghi được video, bộ theo dõi chưa trả về khung hình trong lượt này.</p>}
          <p><strong>Phụ đề ghi trong lượt này:</strong> {summary.captions.join(' · ') || 'Chưa có từ nào được ghi.'}</p>
          {modeMismatch && <p>Chế độ Từ → Tự do có thể chẩn đoán từng pha của mẫu “mấy giờ?”.</p>}
          {bestPhase && <p><strong>Đối chiếu “mấy giờ?”</strong> · {REASONS[bestPhase.status]}</p>}
          {alternative && <p>SSTCN từng dự đoán “{alternative.label}”{alternative.confidence !== null && ` (${percent(alternative.confidence)})`} trong tập từ của mô hình.</p>}
          {trackingWindows.size > 0 && (
            <div className="camera-diagnostic-actions" aria-label="Theo dõi tay và vai từng giây">
              {Array.from(trackingWindows, ([second, items]) => {
                const poses = items.filter((frame) => frame.shoulders !== null);
                const leftPct = percent(items.filter((frame) => frame.left).length / items.length);
                const rightPct = percent(items.filter((frame) => frame.right).length / items.length);
                return <button type="button" className="btn btn-secondary" key={second} onClick={() => seek(second * 1000)}>
                  {second}s · T {leftPct} / P {rightPct}
                  {poses.length > 0 && ` · vai ${percent(poses.filter((frame) => frame.shoulders).length / poses.length)}`}
                </button>;
              })}
            </div>
          )}
          <div className="camera-diagnostic-actions">
            <strong>Quy trình phân tích ({rows.length}/{events.length} lượt)</strong>
            <label className="camera-diagnostic-toggle">
              <input type="checkbox" checked={showAllRows} onChange={(e) => setShowAllRows(e.target.checked)} />
              Hiện đầy đủ tất cả lượt gửi
            </label>
          </div>
          {rows.length ? (
            <div className="camera-diagnostic-table">
              <table>
                <thead>
                  <tr>
                    <th>Thời điểm &amp; Cửa sổ</th>
                    <th>Kết quả &amp; Chốt chữ</th>
                    <th>Tại sao hệ thống quyết định vậy (Chi tiết quy trình)</th>
                  </tr>
                </thead>
                <tbody>
                  {rows.map((event, i) => {
                    const mg = event.backendDiagnostic?.motion_gate;
                    const qp = event.backendDiagnostic?.qipedc;
                    const sstcn = event.backendDiagnostic?.sstcn;
                    return (
                      <tr key={`${event.ms}-${event.reqId ?? i}`}>
                        <td>
                          <button type="button" onClick={() => seek(event.ms)}>
                            {event.reqId ? `#${event.reqId} · ` : ''}{seconds(event.ms)}
                          </button>
                          <small>
                            {event.windowFrames !== null ? `${event.windowFrames}f` : '—'}
                            {event.windowDurationMs !== null ? ` · ${event.windowDurationMs}ms` : ''}
                            {event.isFlush && ' · Flush'}
                          </small>
                          <small>
                            RTT {event.rttMs ?? '—'}ms · Srv {event.serverMs ?? '—'}ms
                            {event.payloadKB ? ` · ${event.payloadKB}KB` : ''}
                          </small>
                        </td>
                        <td>
                          <strong>{event.label || event.displayLabel || 'Đang chờ'}</strong>
                          {event.confidence !== null && ` (${percent(event.confidence)})`}
                          <small>{SOURCES[event.source] || event.source}</small>
                          {event.frontendAction && (
                            <small className={event.caption ? 'diag-tag-ok' : 'diag-tag-muted'}>
                              {FRONTEND_ACTIONS[event.frontendAction] || event.frontendAction}
                              {event.caption && ` → “${event.caption}”`}
                            </small>
                          )}
                        </td>
                        <td>
                          <div className="diag-explain">
                            {event.explanation || (event.matchMethod === 'phases'
                              ? 'Khớp pha chỉ cổ tay và đưa tay lên rồi mở ngón'
                              : event.matchMethod === 'hand_shape'
                                ? 'Khớp tư thế hai tay ở nhiều đoạn của động tác'
                                : REASONS[event.status] || '—')}
                          </div>
                          {mg && (
                            <small>
                              Chuyển động: trái={mg.left_extent ?? 0}, phải={mg.right_extent ?? 0} (ngưỡng {mg.motion_threshold}) ·
                              tay hợp lệ T={mg.filled_left_frames ?? 0}/P={mg.filled_right_frames ?? 0}/{mg.visible_frames ?? 0}f
                            </small>
                          )}
                          {qp && (
                            <small>
                              QIPEDC: nhóm {qp.bucket_hands ?? '?'} tay ({qp.bucket_total_words ?? 0} từ)
                              {qp.active_hands ? ` · active=${qp.active_hands}` : ''}
                              {qp.gap_top1_top2 !== null && qp.gap_top1_top2 !== undefined ? ` · gap=${qp.gap_top1_top2} (cần >=${qp.min_separation})` : ''}
                              {qp.bilateral_forced_by_2frame_rule && ' · ⚠ bị ép 2 tay (tay kia >=2f)'}
                            </small>
                          )}
                          {qp?.top_candidates?.length > 0 && (
                            <small className="diag-topk">
                              Top QIPEDC: {qp.top_candidates.slice(0, 5).map((c, idx) =>
                                `${idx + 1}. ${c.word} (dist=${c.distance}${c.common === false ? ', hiếm' : ''})`
                              ).join(' · ')}
                            </small>
                          )}
                          {sstcn?.called && sstcn.top_k?.length > 0 && (
                            <small className="diag-topk">
                              Top SSTCN: {sstcn.top_k.slice(0, 3).map((t, idx) =>
                                `${idx + 1}. ${t.term} (${percent(t.confidence)})`
                              ).join(' · ')}
                            </small>
                          )}
                          {event.wristMs !== null && <button type="button" onClick={() => seek(event.wristMs)}>Chỉ cổ tay · {seconds(event.wristMs)}</button>}
                          {event.liftMs !== null && <button type="button" onClick={() => seek(event.liftMs)}>
                            {event.matchMethod === 'phases' ? 'Mở ngón' : 'Đưa tay lên'} · {seconds(event.liftMs)}
                          </button>}
                          {(event.backendDiagnostic || event.frontendDecision || event.classifierDetails) && (
                            <details className="diag-raw-details">
                              <summary>Xem log chi tiết lượt này</summary>
                              <pre>{JSON.stringify({
                                frontendDecision: event.frontendDecision,
                                pipeline_steps: event.backendDiagnostic?.pipeline_steps,
                                timings_ms: event.backendDiagnostic?.timings_ms,
                                motion_gate: event.backendDiagnostic?.motion_gate,
                                qipedc: event.backendDiagnostic?.qipedc,
                                sstcn: event.backendDiagnostic?.sstcn,
                                classifier_details: event.classifierDetails,
                              }, null, 2)}</pre>
                            </details>
                          )}
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          ) : <p>Chưa có kết quả nhận diện trong clip. Kiểm tra xem tay có nằm trong khung lúc ghi.</p>}
          {review.pipelineLog?.length > 0 && (
            <details className="diag-raw-details">
              <summary>Xem full log dòng thời gian ({review.pipelineLog.length} sự kiện từ camera → chữ)</summary>
              <pre>{review.pipelineLog.map((item) =>
                `[${(item.ms / 1000).toFixed(2)}s] ${item.kind}: ${JSON.stringify(item.detail)}`
              ).join('\n')}</pre>
            </details>
          )}
        </div>
      )}
    </section>
  );
}
