import { useEffect, useRef, useState } from 'react';

export const DIAGNOSTIC_RECORD_MS = 10000;

const inFrame = (point) => point && point.x >= 0 && point.x <= 1 && point.y >= 0 && point.y <= 1;

function roundNum(value, digits = 4) {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  const factor = 10 ** digits;
  return Math.round(value * factor) / factor;
}

export function useCameraDiagnostic() {
  const sessionRef = useRef(null);
  const [status, setStatus] = useState('idle');
  const [remainingSec, setRemainingSec] = useState(DIAGNOSTIC_RECORD_MS / 1000);
  const [review, setReview] = useState(null);
  const [error, setError] = useState('');

  const discard = () => {
    const session = sessionRef.current;
    sessionRef.current = null;
    if (!session) return;
    clearTimeout(session.timer);
    clearInterval(session.ticker);
    if (session.recorder.state !== 'inactive') session.recorder.stop();
    if (session.url) URL.revokeObjectURL(session.url);
  };

  useEffect(() => discard, []);

  const stop = () => {
    const session = sessionRef.current;
    if (!session || session.recorder.state === 'inactive') return;
    clearTimeout(session.timer);
    clearInterval(session.ticker);
    session.end = performance.now();
    session.pipelineLog.push({
      ms: session.start !== null ? Math.round(session.end - session.start) : 0,
      kind: 'session_stop',
      detail: { totalFrames: session.frames.length, totalEvents: session.events.length },
    });
    setStatus('finishing');
    setRemainingSec(0);
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
    setRemainingSec(DIAGNOSTIC_RECORD_MS / 1000);
    setReview(null);
    setError('');
    const session = {
      recorder,
      settings: { ...settings, recordDurationMs: DIAGNOSTIC_RECORD_MS, startedAtIso: new Date().toISOString() },
      start: null,
      end: null,
      frames: [],
      events: [],
      pipelineLog: [],
      reqSeq: 0,
      chunks: [],
    };
    sessionRef.current = session;
    recorder.onstart = () => {
      if (sessionRef.current !== session || session.end !== null) return;
      session.start = performance.now();
      session.pipelineLog.push({ ms: 0, kind: 'session_start', detail: session.settings });
      session.timer = setTimeout(stop, DIAGNOSTIC_RECORD_MS);
      session.ticker = setInterval(() => {
        if (sessionRef.current !== session || session.start === null) return;
        const left = Math.max(0, Math.ceil((DIAGNOSTIC_RECORD_MS - (performance.now() - session.start)) / 1000));
        setRemainingSec(left);
      }, 200);
      setStatus('recording');
    };
    recorder.ondataavailable = ({ data }) => {
      if (sessionRef.current === session && data.size) session.chunks.push(data);
    };
    recorder.onstop = () => {
      if (sessionRef.current !== session) return;
      clearTimeout(session.timer);
      clearInterval(session.ticker);
      session.end ??= performance.now();
      setStatus('idle');
      setRemainingSec(DIAGNOSTIC_RECORD_MS / 1000);
      if (!session.chunks.length || session.start === null) {
        setError('Lượt ghi chưa thu được video. Hãy thử lại.');
        return;
      }
      const blob = new Blob(session.chunks, { type: recorder.mimeType || session.chunks[0].type });
      session.chunks = [];
      session.url = URL.createObjectURL(blob);
      setReview({
        url: session.url,
        extension: blob.type.includes('mp4') ? 'mp4' : 'webm',
        settings: session.settings,
        durationMs: Math.round(session.end - session.start),
        frames: session.frames.slice(),
        events: session.events.slice(),
        pipelineLog: session.pipelineLog.slice(),
      });
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
    return Boolean(session && session.start !== null && time >= session.start && (session.end === null || time <= session.end));
  };

  const nextRequestId = () => {
    const session = sessionRef.current;
    if (!session) return null;
    session.reqSeq += 1;
    return session.reqSeq;
  };

  const addPipelineEvent = (time, kind, detail = null) => {
    if (!includesTime(time)) return;
    const session = sessionRef.current;
    session.pipelineLog.push({
      ms: Math.round(time - session.start),
      kind,
      detail,
    });
    if (session.url) {
      setReview((current) => current?.url === session.url
        ? { ...current, pipelineLog: session.pipelineLog.slice() } : current);
    }
  };

  const trackFrame = (left, right, pose = null, meta = null) => {
    const now = performance.now();
    if (!includesTime(now)) return;
    const session = sessionRef.current;
    const shoulders = pose === null ? null : [pose[11], pose[12]].every((point) =>
      inFrame(point) && (point.visibility ?? 1) >= 0.5);
    const frameEntry = {
      ms: Math.round(now - session.start),
      shoulders,
      left: Boolean(inFrame(left?.[0])),
      right: Boolean(inFrame(right?.[0])),
    };
    if (meta) {
      if (meta.fps !== undefined) frameEntry.fps = meta.fps;
      if (meta.leftBridged !== undefined) frameEntry.leftBridged = Boolean(meta.leftBridged);
      if (meta.rightBridged !== undefined) frameEntry.rightBridged = Boolean(meta.rightBridged);
      if (meta.maxWristSpeed !== undefined) frameEntry.wristSpeed = roundNum(meta.maxWristSpeed, 4);
      if (meta.handsClasped !== undefined) frameEntry.handsClasped = Boolean(meta.handsClasped);
      if (meta.bufLen !== undefined) frameEntry.bufLen = meta.bufLen;
      if (meta.bufDurationMs !== undefined) frameEntry.bufDurationMs = meta.bufDurationMs;
    }
    session.frames.push(frameEntry);
  };

  const addPrediction = (time, data, times, source, caption, failure, meta = null) => {
    if (!includesTime(time)) return;
    const session = sessionRef.current;
    const relativeTime = (frame) => {
      const timestamp = times?.[frame];
      return Number.isFinite(timestamp) && includesTime(timestamp) ? Math.round(timestamp - session.start) : null;
    };
    const diag = data?.diagnostic || null;
    const backendExplanation = diag?.decision?.explanation || null;
    const frontendAction = meta?.frontendDecision?.action || null;
    const frontendReason = meta?.frontendDecision?.reason || null;
    const explanation = [backendExplanation, frontendReason].filter(Boolean).join(' → ') || null;

    const topK = (data?.top_k && data.top_k.length)
      ? data.top_k
      : (diag?.qipedc?.top_candidates || []).map((item) => ({
        term: item.word || item.entry_id,
        entry_id: item.entry_id,
        distance: item.distance,
        confidence: item.similarity,
        common: item.common,
      }));

    const eventRecord = {
      reqId: meta?.reqId ?? null,
      ms: Math.round(time - session.start),
      receivedMs: meta?.receivedAt ? Math.round(meta.receivedAt - session.start) : null,
      rttMs: meta?.rttMs ?? null,
      serverMs: diag?.timings_ms?.total_ms ?? data?.inference_time_ms ?? null,
      windowFrames: meta?.windowFrames ?? diag?.raw_frames ?? (Array.isArray(times) ? times.length : null),
      windowDurationMs: meta?.windowDurationMs ?? (Array.isArray(times) && times.length >= 2
        ? Math.round(times[times.length - 1] - times[0]) : null),
      payloadKB: meta?.payloadKB ?? null,
      isFlush: Boolean(meta?.isFlush),
      source: data?.source || source,
      caption: caption || null,
      replacedCaption: meta?.replaced || null,
      sign: data?.sign || null,
      label: data?.sign === 'NEUTRAL' ? null : data?.label || data?.sign || null,
      displayLabel: data?.label || null,
      confidence: data?.sign === 'NEUTRAL' ? null : data?.confidence ?? null,
      status: failure ? 'inference_error' : diag?.status || data?.blocked_reason || null,
      similarity: diag?.similarity ?? null,
      matchMethod: diag?.match_method || data?.match_method || null,
      wristMs: relativeTime(diag?.tap_frame),
      liftMs: relativeTime(diag?.lift_frame),
      topK,
      frontendAction,
      frontendDecision: meta?.frontendDecision || null,
      classifierDetails: data?.classifier_details || null,
      backendDiagnostic: diag,
      explanation,
    };
    session.events.push(eventRecord);
    session.pipelineLog.push({
      ms: eventRecord.receivedMs ?? eventRecord.ms,
      kind: 'prediction_decision',
      detail: {
        reqId: eventRecord.reqId,
        sentAtMs: eventRecord.ms,
        rttMs: eventRecord.rttMs,
        serverMs: eventRecord.serverMs,
        sign: eventRecord.sign,
        displayLabel: eventRecord.displayLabel,
        source: eventRecord.source,
        status: eventRecord.status,
        frontendAction: eventRecord.frontendAction,
        caption: eventRecord.caption,
        explanation: eventRecord.explanation,
      },
    });
    if (session.url) {
      setReview((current) => current?.url === session.url
        ? { ...current, events: session.events.slice(), pipelineLog: session.pipelineLog.slice() } : current);
    }
  };

  const clear = () => {
    discard();
    setReview(null);
    setError('');
    setStatus('idle');
    setRemainingSec(DIAGNOSTIC_RECORD_MS / 1000);
  };

  return {
    status,
    remainingSec,
    review,
    error,
    start,
    stop,
    clear,
    trackFrame,
    addPrediction,
    addPipelineEvent,
    nextRequestId,
    includesTime,
  };
}
