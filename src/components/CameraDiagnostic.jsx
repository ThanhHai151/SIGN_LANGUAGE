import { useRef, useState } from 'react';
import './CameraDiagnostic.css';

const MODES = { words: 'Từ', letters: 'Chữ cái', letters_numbers: 'Chữ cái và số', digits: 'Chữ số', sentence: 'Ghép câu' };
const SOURCES = { qipedc_template: 'Mẫu QIPEDC', qipedc_practice: 'Luyện QIPEDC', sstcn: 'SSTCN', landmarks: 'Nhận diện tay' };
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
  no_motion: 'Chuyển động tay chưa đủ rõ trong cửa sổ này',
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

function exportReport({ settings, durationMs, frames, events, faceSummary = null }) {
  const data = JSON.stringify({ settings, durationMs, frames, events, faceSummary }, null, 2);
  const reportUrl = URL.createObjectURL(new Blob([data], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = reportUrl;
  link.download = 'vsl-chan-doan.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(reportUrl), 1000);
}

function exportFaceReport(report) {
  const payload = {
    type: 'vsl_face_tracking_metrics_only',
    settings: report.settings,
    durationMs: report.durationMs,
    summary: report.summary,
    frames: report.frames,
  };
  const reportUrl = URL.createObjectURL(new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' }));
  const link = document.createElement('a');
  link.href = reportUrl;
  link.download = 'vsl-face-diagnostics.json';
  link.click();
  setTimeout(() => URL.revokeObjectURL(reportUrl), 1000);
}

export default function CameraDiagnostic({ capture, cameraActive, stream, mode, captureMode, timed, topic, expectedSign }) {
  const [expected, setExpected] = useState(expectedSign || '');
  const playerRef = useRef(null);
  const { status, review, error, faceStatus = 'idle', faceReport = null } = capture;
  const busy = status !== 'idle';
  const faceBusy = faceStatus === 'running';
  const seek = (ms) => {
    const player = playerRef.current;
    if (!player) return;
    player.pause();
    player.currentTime = ms / 1000;
  };

  const events = review ? [...review.events].sort((a, b) => a.ms - b.ms) : [];
  const rows = events.filter((event, i) => !i || event.caption || event.label !== events[i - 1].label ||
    event.status !== events[i - 1].status || Math.floor(event.ms / 1000) !== Math.floor(events[i - 1].ms / 1000));
  const expectedKey = wordKey(review?.settings.expected);
  const bestPhase = expectedKey === 'mấy giờ' || (!expectedKey && events.some((event) => event.liftMs !== null))
    ? PHASE_ORDER.map((key) => events.find((event) => event.status === key)).find(Boolean)
    : null;
  const frames = review?.frames || [];
  const poseFrames = frames.filter((frame) => frame.shoulders !== null);
  const bothHands = frames.filter((frame) => frame.left && frame.right).length;
  const noHands = frames.filter((frame) => !frame.left && !frame.right).length;
  const captions = events.filter((event) => event.caption).map((event) => event.caption);
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
  const faceSummary = faceReport?.summary || null;

  return (
    <section className="glass-panel camera-diagnostic" aria-labelledby="camera-diagnostic-title">
      <div className="camera-diagnostic-actions">
        <h3 id="camera-diagnostic-title">Chẩn đoán video 5 giây</h3>
        <button type="button" className="btn btn-primary"
          disabled={status !== 'recording' && (!cameraActive || timed || busy || faceBusy)}
          onClick={() => status === 'recording' ? capture.stop() : capture.start(stream(), { mode, captureMode, topic, expected: expected.trim() })}>
          {status === 'recording' ? 'Dừng ghi' : status === 'starting' ? 'Đang bật bộ ghi…' : status === 'finishing' ? 'Đang chuẩn bị clip…' : 'Ghi chẩn đoán 5 giây'}
        </button>
      </div>
      <label className="camera-diagnostic-field">
        Ký hiệu bạn định làm (tùy chọn)
        <input className="input" value={expected} onChange={(event) => setExpected(event.target.value)}
          disabled={busy || faceBusy} placeholder="Ví dụ: mấy giờ?" maxLength={80} />
      </label>
      <p className="camera-diagnostic-note" role="status">
        {error || (status === 'recording'
          ? 'Đang ghi. Thực hiện trọn động tác một lần.'
          : 'Video không có âm thanh và được giữ trong trình duyệt.')}
      </p>
      {review && (
        <div className="camera-diagnostic-review">
          <video ref={playerRef} src={review.url} controls playsInline preload="metadata" aria-label="Video chẩn đoán đã ghi" />
          <div className="camera-diagnostic-actions">
            <span>{seconds(review.durationMs)} · {MODES[review.settings.mode]}{review.settings.expected && ` · Mục tiêu “${review.settings.expected}”`}</span>
            <a className="btn btn-secondary" href={review.url} download={`vsl-chan-doan.${review.extension}`}>Tải video</a>
            <button type="button" className="btn btn-secondary" onClick={() => exportReport(review)}>Tải báo cáo</button>
            <button type="button" className="btn btn-secondary" onClick={capture.clear}>Xóa clip</button>
          </div>
          {frames.length ? (
            <p className="camera-diagnostic-note">
              {frames.length} khung hình phân tích · Thấy hai tay {percent(bothHands / frames.length)} · Mất dấu cả hai tay {percent(noHands / frames.length)}
              {poseFrames.length > 0 && ` · Thấy hai vai ${percent(poseFrames.filter((frame) => frame.shoulders).length / poseFrames.length)}`}
            </p>
          ) : <p>Camera ghi được video, bộ theo dõi chưa trả về khung hình trong lượt này.</p>}
          {review.faceSummary && review.faceSummary.totalFrames > 0 && (
            <p className="camera-diagnostic-note">
              Khuôn mặt (Holistic): phát hiện {percent(review.faceSummary.faceDetectionRate || 0)} ({review.faceSummary.faceDetectedFrames}/{review.faceSummary.totalFrames}) · TB {review.faceSummary.faceLandmarkCountAvg ?? 0} điểm · Render TB {review.faceSummary.avgRenderMs ?? '—'} ms · Video {review.faceSummary.videoResolution || '—'} / Xử lý {review.faceSummary.processingResolution || '—'}
            </p>
          )}
          <p>Phụ đề ghi trong lượt này: {captions.join(' · ') || 'Chưa có từ nào được ghi.'}</p>
          {modeMismatch && <p>Chế độ Từ → Tự do có thể chẩn đoán từng pha của mẫu “mấy giờ?”.</p>}
          {bestPhase && <p><strong>Đối chiếu “mấy giờ?”</strong> · {REASONS[bestPhase.status]}</p>}
          {alternative && <p>SSTCN từng dự đoán “{alternative.label}”{alternative.confidence !== null && ` (${percent(alternative.confidence)})`} trong tập từ của mô hình.</p>}
          {trackingWindows.size > 0 && (
            <div className="camera-diagnostic-actions" aria-label="Theo dõi tay và vai từng giây">
              {Array.from(trackingWindows, ([second, items]) => {
                const poses = items.filter((frame) => frame.shoulders !== null);
                return <button type="button" className="btn btn-secondary" key={second} onClick={() => seek(second * 1000)}>
                  {second} s · 2 tay {percent(items.filter((frame) => frame.left && frame.right).length / items.length)}
                  {poses.length > 0 && ` · vai ${percent(poses.filter((frame) => frame.shoulders).length / poses.length)}`}
                </button>;
              })}
            </div>
          )}
          {rows.length ? (
            <div className="camera-diagnostic-table">
              <table>
                <thead><tr><th>Đến giây</th><th>Dự đoán</th><th>{review.settings.captureMode === 'live_words' ? 'So khớp QIPEDC' : 'Dấu hiệu trong chuỗi'}</th></tr></thead>
                <tbody>
                  {rows.map((event, i) => (
                    <tr key={`${event.ms}-${i}`}>
                      <td><button type="button" onClick={() => seek(event.ms)}>{seconds(event.ms)}</button></td>
                      <td>{event.label || 'Đang chờ'}{event.confidence !== null && ` (${percent(event.confidence)})`}
                        <small>{SOURCES[event.source] || event.source}{event.caption && ' · Đã ghi phụ đề'}</small>
                      </td>
                      <td>{event.matchMethod === 'phases' ? 'Khớp pha chỉ cổ tay và đưa tay lên rồi mở ngón'
                        : event.matchMethod === 'hand_shape' ? 'Khớp tư thế hai tay ở nhiều đoạn của động tác'
                          : REASONS[event.status] || '—'}
                        {event.wristMs !== null && <button type="button" onClick={() => seek(event.wristMs)}>Chỉ cổ tay · {seconds(event.wristMs)}</button>}
                        {event.liftMs !== null && <button type="button" onClick={() => seek(event.liftMs)}>
                          {event.matchMethod === 'phases' ? 'Mở ngón' : 'Đưa tay lên'} · {seconds(event.liftMs)}
                        </button>}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : <p>Chưa có kết quả nhận diện trong clip. Kiểm tra xem tay có nằm trong khung lúc ghi.</p>}
        </div>
      )}

      <hr style={{ borderColor: 'var(--border-subtle)', opacity: 0.4, margin: '4px 0' }} />

      <div className="camera-diagnostic-actions">
        <h3>Chẩn đoán khuôn mặt (chỉ số — không lưu video)</h3>
        <button
          type="button"
          className="btn btn-secondary"
          disabled={!faceBusy && (!cameraActive || timed || busy)}
          onClick={() => (
            faceBusy
              ? capture.stopFaceMetrics()
              : capture.startFaceMetrics({ mode, captureMode })
          )}
        >
          {faceBusy ? 'Dừng đo' : 'Đo chỉ số khuôn mặt (10 giây)'}
        </button>
      </div>
      <p className="camera-diagnostic-note" role="status">
        {faceBusy
          ? 'Đang đo chỉ số khuôn mặt cục bộ (không lưu ảnh hoặc video)...'
          : mode !== 'words'
            ? 'Lưu ý: Face Contours chỉ chạy ở chế độ Từ (Holistic); các chế độ khác dùng Hands (0 điểm mặt). Chỉ số tính cục bộ, không lưu ảnh/video.'
            : 'Chỉ ghi nhận chỉ số số học trong trình duyệt, không lưu ảnh hoặc video khuôn mặt.'}
      </p>
      {faceSummary && (
        <div className="camera-diagnostic-review">
          {faceSummary.totalFrames > 0 ? (
            <>
              <p className="camera-diagnostic-note">
                Phát hiện khuôn mặt: {faceSummary.faceDetectedFrames}/{faceSummary.totalFrames} frame ({percent(faceSummary.faceDetectionRate || 0)})
                {' · '}Mất khuôn mặt: {faceSummary.faceMissingFrames} frame (tối đa {faceSummary.maxConsecutiveMissingFrames} liên tiếp)
              </p>
              <p className="camera-diagnostic-note">
                Số landmark (faceLandmarkCount): {faceSummary.faceLandmarkCountAvg !== null
                  ? `${faceSummary.faceLandmarkCountMin}–${faceSummary.faceLandmarkCountMax} (TB ${faceSummary.faceLandmarkCountAvg})`
                  : '0'}
                {' · '}FPS xử lý: {faceSummary.processingFps ?? '—'}
                {' · '}Xử lý Holistic TB: {faceSummary.avgProcessMs ?? '—'} ms (P95: {faceSummary.p95ProcessMs ?? '—'} ms)
                {' · '}Render TB: {faceSummary.avgRenderMs ?? '—'} ms (P95: {faceSummary.p95RenderMs ?? '—'} ms)
              </p>
              <p className="camera-diagnostic-note">
                Video: {faceSummary.videoResolution || '—'}
                {' · '}Canvas xử lý Holistic: {faceSummary.processingResolution || '—'}
                {' · '}Canvas hiển thị: {faceSummary.canvasResolution || '—'}
              </p>
              <p className="camera-diagnostic-note">
                {faceSummary.stabilitySufficientData
                  ? `Ổn định anchor (${faceSummary.consecutiveDetectedPairs} cặp liên tiếp): dịch chuyển TB ${faceSummary.meanFrameToFrameDisplacementPx !== null ? `${faceSummary.meanFrameToFrameDisplacementPx} px` : faceSummary.meanFrameToFrameDisplacementNorm}/frame · độ lệch chuẩn ${faceSummary.anchorStdDevPx !== null ? `${faceSummary.anchorStdDevPx} px` : faceSummary.anchorStdDevNorm}`
                  : (faceSummary.stabilityNote || 'Chưa đủ dữ liệu ổn định.')}
              </p>
            </>
          ) : (
            <p className="camera-diagnostic-note">
              Chưa nhận được frame Holistic nào trong lượt đo (kiểm tra camera đang bật ở chế độ Từ / Holistic).
            </p>
          )}
          <div className="camera-diagnostic-actions">
            <button type="button" className="btn btn-secondary" onClick={() => exportFaceReport(faceReport)}>
              Tải báo cáo Face JSON
            </button>
            <button type="button" className="btn btn-secondary" onClick={capture.clearFaceMetrics}>
              Xóa kết quả
            </button>
          </div>
        </div>
      )}
    </section>
  );
}
