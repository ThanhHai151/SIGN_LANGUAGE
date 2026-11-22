import { useEffect, useRef, useState } from 'react';

const RECORD_MS = 5000;

const inFrame = (point) => point && point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1;

export function useCameraDiagnostic() {
  const sessionRef = useRef(null);
  const [status, setStatus] = useState('idle');
  const [review, setReview] = useState(null);
  const [error, setError] = useState('');

  const discard = () => {
    const session = sessionRef.current;
    sessionRef.current = null;
    if (!session) return;
    clearTimeout(session.timer);
    if (session.recorder.state !== 'inactive') session.recorder.stop();
    if (session.url) URL.revokeObjectURL(session.url);
  };

  useEffect(() => discard, []);

  const stop = () => {
    const session = sessionRef.current;
    if (!session || session.recorder.state === 'inactive') return;
    clearTimeout(session.timer);
    session.end = performance.now();
    setStatus('finishing');
    session.recorder.stop();
  };

  const start = (stream, settings) => {
    if (sessionRef.current?.recorder.state === 'recording') return;
    const track = stream?.getVideoTracks().find((item) => item.readyState === 'live');
    if (!track || typeof MediaRecorder === 'undefined') {
      setError(track ? 'Trình duyệt này chưa hỗ trợ ghi video.' : 'Hãy bật camera trước khi ghi.');
      return;
    }
    let recorder;
    try {
      const mimeType = ['video/webm;codecs=vp8', 'video/webm', 'video/mp4']
        .find((type) => MediaRecorder.isTypeSupported(type));
      const videoOnly = new MediaStream([track]);
      recorder = mimeType ? new MediaRecorder(videoOnly, { mimeType }) : new MediaRecorder(videoOnly);
    } catch {
      setError('Không khởi tạo được bộ ghi video trên trình duyệt này.');
      return;
    }

    discard();
    setStatus('starting');
    setReview(null);
    setError('');
    const session = { recorder, settings, start: null, end: null, frames: [], events: [], chunks: [] };
    sessionRef.current = session;
    recorder.onstart = () => {
      if (sessionRef.current !== session || session.end !== null) return;
      session.start = performance.now();
      session.timer = setTimeout(stop, RECORD_MS);
      setStatus('recording');
    };
    recorder.ondataavailable = ({ data }) => {
      if (sessionRef.current === session && data.size) session.chunks.push(data);
    };
    recorder.onstop = () => {
      if (sessionRef.current !== session) return;
      clearTimeout(session.timer);
      session.end ??= performance.now();
      setStatus('idle');
      if (!session.chunks.length || session.start === null) {
        setError('Lượt ghi chưa thu được video. Hãy thử lại.');
        return;
      }
      const blob = new Blob(session.chunks, { type: recorder.mimeType || session.chunks[0].type });
      session.chunks = [];
      session.url = URL.createObjectURL(blob);
      setReview({ url: session.url, extension: blob.type.includes('mp4') ? 'mp4' : 'webm',
        settings, durationMs: Math.round(session.end - session.start), frames: session.frames.slice(), events: session.events.slice() });
    };
    recorder.onerror = () => {
      if (sessionRef.current !== session) return;
      discard();
      setStatus('idle');
      setError('Quá trình ghi video bị gián đoạn.');
    };
    try {
      recorder.start();
    } catch {
      discard();
      setStatus('idle');
      setError('Không bắt đầu được lượt ghi video.');
    }
  };

  const includesTime = (time) => {
    const session = sessionRef.current;
    return session && session.start !== null && time >= session.start && (session.end === null || time <= session.end);
  };

  const trackFrame = (left, right, pose = null) => {
    const now = performance.now();
    if (!includesTime(now)) return;
    const session = sessionRef.current;
    const shoulders = pose === null ? null : [pose[11], pose[12]].every((point) =>
      inFrame(point) && (point.visibility ?? 1) >= 0.5);
    session.frames.push({ ms: Math.round(now - session.start), shoulders, left: Boolean(inFrame(left?.[0])), right: Boolean(inFrame(right?.[0])) });
  };

  const addPrediction = (time, data, times, source, caption, failure) => {
    if (!includesTime(time)) return;
    const session = sessionRef.current;
    const relativeTime = (frame) => {
      const timestamp = times?.[frame];
      return Number.isFinite(timestamp) && includesTime(timestamp) ? Math.round(timestamp - session.start) : null;
    };
    const reason = data?.diagnostic;
    session.events.push({
      ms: Math.round(time - session.start), source: data?.source || source, caption: caption || null,
      label: data?.sign === 'NEUTRAL' ? null : data?.label || data?.sign || null,
      confidence: data?.sign === 'NEUTRAL' ? null : data?.confidence ?? null,
      status: failure ? 'inference_error' : reason?.status || data?.blocked_reason || null,
      similarity: reason?.similarity ?? null,
      matchMethod: reason?.match_method ?? null,
      wristMs: relativeTime(reason?.tap_frame), liftMs: relativeTime(reason?.lift_frame),
    });
    if (session.url) setReview((current) => current?.url === session.url ? { ...current, events: session.events.slice() } : current);
  };

  const clear = () => {
    discard();
    setReview(null);
    setError('');
    setStatus('idle');
  };

  return { status, review, error, start, stop, clear, trackFrame, addPrediction, includesTime };
}
