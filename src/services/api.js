/**
 * VSL Connect API Service
 * Giao tiếp thời gian thực giữa Frontend React và Python FastAPI Backend
 */

import { API_BASE_URL } from './apiBase';

export async function fetchLetters() {
  const response = await fetch(`${API_BASE_URL}/api/letters`);
  return response.ok ? response.json() : null;
}

export async function fetchLetter(char) {
  const response = await fetch(`${API_BASE_URL}/api/letter/${encodeURIComponent(char)}`);
  const data = response.ok ? await response.json() : null;
  return data?.sequence?.[0] ?? null;
}

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, { method: 'GET' });
    if (!res.ok) throw new Error('Backend offline');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Backend health check failed:', err.message);
    return { status: 'offline', error: err.message };
  }
}

export async function fetchDictionary() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/dictionary`);
    if (!res.ok) throw new Error('Cannot load dictionary');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Using fallback dictionary due to error:', err);
    return null;
  }
}

export async function fetchVSL102Classes() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/vsl_102_classes`);
    if (!res.ok) throw new Error('Cannot load VSL 102 classes');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Cannot fetch 102 classes:', err);
    return null;
  }
}

/**
 * Bộ từ vựng VSL thống nhất (QIPEDC + giáo trình 102 + robot 3D) — dùng chung cho cả 2 chiều dịch
 */
let lexiconCache = null;
export async function fetchLexicon() {
  if (lexiconCache) return lexiconCache;
  try {
    const res = await fetch(`${API_BASE_URL}/api/lexicon`);
    if (!res.ok) throw new Error('Cannot load lexicon');
    lexiconCache = await res.json();
    return lexiconCache;
  } catch (err) {
    console.warn('[VSL API] Cannot fetch lexicon:', err);
    return null;
  }
}

/**
 * Hoạt ảnh chụp chuyển động (mocap) của robot cho 1 mục từ — trích từ video người thật QIPEDC
 */
const motionCache = new Map();
// ---------- Video/ảnh mẫu QIPEDC phát qua bộ đệm của backend (nhanh, mượt, tua được) ----------
const QIPEDC_ID_RE = /qipedc\.moet\.gov\.vn\/(videos|thumbs)\/([A-Za-z0-9_-]+)\.(mp4|png)/;

/** Link qua backend cho 1 URL QIPEDC (giữ nguyên nếu không phải QIPEDC). */
export function mediaUrl(url) {
  const m = QIPEDC_ID_RE.exec(url || '');
  return m ? `${API_BASE_URL}/api/media/${m[1]}/${m[2]}.${m[3]}` : url;
}

export function mediaId(url) {
  const m = QIPEDC_ID_RE.exec(url || '');
  return m ? m[2] : null;
}

const warmed = new Set();
/** Nhờ backend tải trước (ở nền) các video sắp xem. */
export function prefetchVideos(urls) {
  const ids = urls.map(mediaId).filter((id) => id && !warmed.has(id));
  if (!ids.length) return;
  ids.forEach((id) => warmed.add(id));
  fetch(`${API_BASE_URL}/api/media/prefetch`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ids, kind: 'videos' }),
  }).catch(() => ids.forEach((id) => warmed.delete(id)));
}

const browserWarmed = new Set();
/** Tải sẵn cả file vào bộ nhớ đệm của trình duyệt (khi di chuột tới ô) → bấm xem là phát ngay. */
export function warmVideo(url) {
  const u = mediaUrl(url);
  if (!u || browserWarmed.has(u)) return;
  browserWarmed.add(u);
  fetch(u).then((r) => r.blob()).catch(() => browserWarmed.delete(u));
}

const entryCache = new Map();
export function fetchLexiconEntry(id) {
  if (!id) return Promise.resolve(null);
  if (!entryCache.has(id)) {
    entryCache.set(id, fetch(`${API_BASE_URL}/api/lexicon/${encodeURIComponent(id)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => (j && j.entry) || null)
      .catch(() => { entryCache.delete(id); return null; }));
  }
  return entryCache.get(id);
}

export function fetchMotion(id) {
  if (!id) return Promise.resolve(null);
  if (!motionCache.has(id)) {
    motionCache.set(id, fetch(`${API_BASE_URL}/api/motion/${encodeURIComponent(id)}`)
      .then((r) => (r.ok ? r.json() : null))
      .then((j) => (j && j.motion) || null)
      .catch(() => { motionCache.delete(id); return null; }));
  }
  return motionCache.get(id);
}

export async function sendSequenceForRecognition(sequence, threshold = 0.65, mode = 'words', debugMeta = null, options = null) {
  try {
    const payload = { sequence, threshold, mode };
    if (debugMeta) {
      payload.debug_meta = debugMeta;
    }
    if (options && options.topic_id) {
      payload.topic_id = options.topic_id;
    }
    if (options && Array.isArray(options.timestamps) && options.timestamps.length === sequence.length) {
      payload.timestamps = options.timestamps;
    }
    if (options?.live_words) {
      payload.live_words = true;
      payload.aspect_ratio = options.aspect_ratio;
      if (options.vocab_scope) payload.vocab_scope = options.vocab_scope;
      if (options.qipedc_entries?.length) payload.qipedc_entries = options.qipedc_entries;
    }
    if (options?.include_diagnostic) payload.include_diagnostic = true;
    if (typeof options?.allow_short_sequences === 'boolean') {
      payload.allow_short_sequences = options.allow_short_sequences;
    }
    const res = await fetch(`${API_BASE_URL}/api/recognize_sequence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) throw new Error('Sequence recognition failed');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Sequence recognition error:', err);
    return {
      success: false,
      error: err.message
    };
  }
}

export async function requestQipedcPractice(entryId, videoId, recording) {
  const action = recording ? 'match' : 'prepare';
  const res = await fetch(`${API_BASE_URL}/api/qipedc/practice/${encodeURIComponent(entryId)}/${action}?video_id=${encodeURIComponent(videoId)}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: recording ? JSON.stringify(recording) : undefined,
  });
  const data = await res.json();
  if (!res.ok) throw new Error(typeof data.detail === 'string' ? data.detail : 'Không chấm được lần ghi này.');
  return data;
}

export async function prepareQipedcLive(entryId) {
  const query = entryId ? `?entry_id=${encodeURIComponent(entryId)}` : '';
  const response = await fetch(`${API_BASE_URL}/api/qipedc/live/prepare${query}`, { method: 'POST' });
  const data = await response.json();
  if (!response.ok) throw new Error(data.detail || 'Chưa chuẩn bị được mẫu QIPEDC.');
  return data;
}

export async function fetchSSTCNTopics() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/sstcn/topics`);
    if (!res.ok) throw new Error('Cannot load SSTCN topics');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Cannot fetch SSTCN topics:', err);
    return null;
  }
}

export async function saveCalibrationSample({ label, entry_id = null, sequence, timestamps = null, signer_id = 'personal' }) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/calibration/save_sample`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ label, entry_id, sequence, timestamps, signer_id })
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || 'Failed to save calibration sample');
    }
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Save calibration error:', err);
    return { success: false, error: err.message };
  }
}

export async function fetchCalibrationSummary() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/calibration/summary`);
    if (!res.ok) throw new Error('Cannot load calibration summary');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Cannot fetch calibration summary:', err);
    return null;
  }
}

export async function sendLandmarksForRecognition(
  landmarks,
  multi_hands = null,
  handedness = 'Right',
  mode = null,
  metadata = null
) {
  try {
    const headers = {
      'Content-Type': 'application/json',
      ...(mode ? { 'X-Conversion-Mode': mode } : {})
    };
    const resolvedTimestamp =
      metadata?.timestamp !== undefined && metadata?.timestamp !== null
        ? Number(metadata.timestamp)
        : (metadata?.landmark_extract_ts !== undefined && metadata?.landmark_extract_ts !== null
            ? Number(metadata.landmark_extract_ts)
            : null);

    const bodyPayload = {
      landmarks,
      multi_hands,
      handedness,
      mode,
      ...(resolvedTimestamp !== null && Number.isFinite(resolvedTimestamp) ? { timestamp: resolvedTimestamp } : {}),
      ...(metadata?.expected_sign ? { expected_sign: metadata.expected_sign } : {}),
      ...(metadata?.session_id ? { session_id: metadata.session_id } : {}),
      ...(metadata?.trial_id ? { trial_id: metadata.trial_id } : {}),
      ...(metadata?.frame_idx !== undefined && metadata?.frame_idx !== null ? { frame_idx: Number(metadata.frame_idx) } : {}),
      ...(metadata?.signer_id ? { signer_id: metadata.signer_id } : {}),
      ...(metadata?.client_capture_ts !== undefined && metadata?.client_capture_ts !== null ? { client_capture_ts: Number(metadata.client_capture_ts) } : {}),
      ...(metadata?.landmark_extract_ts !== undefined && metadata?.landmark_extract_ts !== null ? { landmark_extract_ts: Number(metadata.landmark_extract_ts) } : {}),
      ...(metadata?.request_send_ts !== undefined && metadata?.request_send_ts !== null ? { request_send_ts: Number(metadata.request_send_ts) } : {}),
      ...(metadata?.video_pts_sec !== undefined && metadata?.video_pts_sec !== null ? { video_pts_sec: metadata.video_pts_sec } : {}),
      ...(metadata?.video_frame_idx !== undefined && metadata?.video_frame_idx !== null ? { video_frame_idx: metadata.video_frame_idx } : {}),
      ...(metadata?.phase ? { phase: metadata.phase } : {}),
      ...(metadata?.image_width ? { image_width: Number(metadata.image_width) } : {}),
      ...(metadata?.image_height ? { image_height: Number(metadata.image_height) } : {}),
      ...(metadata?.handedness_score !== undefined && metadata?.handedness_score !== null ? { handedness_score: Number(metadata.handedness_score) } : {}),
      ...(metadata?.tracking_confidence !== undefined && metadata?.tracking_confidence !== null ? { tracking_confidence: Number(metadata.tracking_confidence) } : {}),
      ...(metadata?.mediapipe_model_complexity !== undefined && metadata?.mediapipe_model_complexity !== null ? { mediapipe_model_complexity: Number(metadata.mediapipe_model_complexity) } : {}),
      ...(metadata?.processing_resolution ? { processing_resolution: String(metadata.processing_resolution) } : {}),
      ...(metadata?.camera_fps !== undefined && metadata?.camera_fps !== null ? { camera_fps: Number(metadata.camera_fps) } : {})
    };
    const res = await fetch(`${API_BASE_URL}/api/recognize_landmarks`, {
      method: 'POST',
      headers,
      body: JSON.stringify(bodyPayload)
    });
    if (!res.ok) throw new Error('Recognition failed');
    return await res.json();
  } catch (err) {
    // Fallback: Local Client-side heuristic if backend is temporarily disconnected
    return {
      success: true,
      backend_error: true,
      error: err.message,
      data: clientSideLocalRecognize(landmarks)
    };
  }
}

/**
 * Điều khiển bật/tắt ghi log chẩn đoán Controlled Webcam Pilot (Task 7F.5B / 7F.5D)
 * Tuyệt đối không truyền filepath qua HTTP API để bảo vệ an toàn đường dẫn.
 */
export async function setPilotLogging(enabled, filepath_or_session_id = null, maybe_session_id = null) {
  const session_id = maybe_session_id || (typeof filepath_or_session_id === 'string' && !filepath_or_session_id.includes('/') && !filepath_or_session_id.includes('\\') ? filepath_or_session_id : null);
  try {
    const payload = { enabled };
    if (session_id) {
      payload.session_id = session_id;
    }
    const res = await fetch(`${API_BASE_URL}/api/pilot/control`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || 'Failed to toggle pilot logging');
    }
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Pilot control error:', err);
    return { success: false, error: err.message };
  }
}

/**
 * Xả toàn bộ bộ đệm log chẩn đoán pilot xuống đĩa JSONL
 */
export async function flushPilotLogs(sessionId = null) {
  try {
    const request = { method: 'POST' };
    if (sessionId) {
      request.headers = { 'Content-Type': 'application/json' };
      request.body = JSON.stringify({ session_id: sessionId });
    }
    const res = await fetch(`${API_BASE_URL}/api/pilot/flush`, request);
    let body = null;
    try { body = await res.json(); } catch { /* return explicit unknown/error below */ }
    if (!res.ok) {
      return {
        ...(body || {}),
        success: false,
        flush_success: false,
        http_status: res.status,
        error: body?.detail || body?.error || `Failed to flush pilot logs (HTTP ${res.status})`
      };
    }
    return body || { success: false, flush_success: false, error_code: 'invalid_response', error: 'Flush endpoint returned no JSON body.' };
  } catch (err) {
    console.warn('[VSL API] Pilot flush error:', err);
    return { success: false, flush_success: false, http_status: null, error_code: 'network_error', error: err.message };
  }
}

/**
 * Lấy số liệu thống kê ghi log pilot (queued, logged, dropped)
 */
export async function fetchPilotStats() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/pilot/stats`);
    if (!res.ok) throw new Error('Failed to fetch pilot stats');
    return await res.json();
  } catch (err) {
    return { success: false, error: err.message };
  }
}

export async function openPilotTrial(sessionId, trialId) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/pilot/trial/open`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, trial_id: trialId })
    });
    const body = await res.json().catch(() => ({}));
    return res.ok ? body : { ...body, success: false, http_status: res.status };
  } catch (err) {
    return { success: false, error_code: 'network_error', error: err.message };
  }
}

export async function flushPilotTrial(sessionId, trialId) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/pilot/trial/flush`, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, trial_id: trialId })
    });
    const body = await res.json().catch(() => ({}));
    return res.ok ? body : { ...body, success: false, flush_success: false, http_status: res.status };
  } catch (err) {
    return { success: false, flush_success: false, error_code: 'network_error', error: err.message };
  }
}

export async function fetchPilotTrialAccounting(sessionId, trialId) {
  try {
    const params = new URLSearchParams({ session_id: sessionId, trial_id: trialId });
    const res = await fetch(`${API_BASE_URL}/api/pilot/trial/accounting?${params}`);
    const body = await res.json().catch(() => ({}));
    return res.ok ? body : { ...body, success: false, http_status: res.status };
  } catch (err) {
    return { success: false, error_code: 'network_error', error: err.message };
  }
}

export async function fetchPilotRuntimeConfig() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/pilot/config`);
    if (!res.ok) throw new Error('Failed to verify pilot runtime configuration');
    return await res.json();
  } catch (err) {
    return { success: false, error: err.message };
  }
}

/**
 * Lưu session manifest kèm SHA256 cho phiên thử nghiệm pilot (Task 3.10 Section 8)
 */
export async function savePilotSessionManifest(sessionId, manifest) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/pilot/manifest`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, manifest })
    });
    if (!res.ok) throw new Error('Failed to save session manifest');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Pilot manifest save error:', err);
    return { success: false, error: err.message };
  }
}

/**
 * Lưu sidecar synchronization file ánh xạ video và JSONL (Task 3.10 Implementation C)
 */
export async function savePilotSidecarSync(sessionId, syncData) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/pilot/sidecar_sync`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ session_id: sessionId, sync_data: syncData })
    });
    if (!res.ok) throw new Error('Failed to save sidecar sync');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Pilot sidecar save error:', err);
    return { success: false, error: err.message };
  }
}

/**
 * Lưu kết quả chẩn đoán Controlled Pilot 29 Chữ cái vào backend/data/diagnostics/
 */
export async function savePilotEvaluation(sessionId, report, frames = null) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/pilot/save_evaluation`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        report,
        frames
      })
    });
    if (!res.ok) {
      const errJson = await res.json().catch(() => ({}));
      throw new Error(errJson.detail || `HTTP ${res.status}`);
    }
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Save pilot evaluation error:', err);
    return { success: false, error: err.message };
  }
}

/**
 * Lấy danh sách các file báo cáo chẩn đoán Controlled Pilot
 */
export async function fetchPilotEvaluations() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/pilot/evaluations`);
    if (!res.ok) throw new Error('Cannot fetch pilot evaluations');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Fetch pilot evaluations error:', err);
    return { success: false, evaluations: [] };
  }
}

export async function convertTextToVSL(text) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/text_to_sign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error('Text to sign failed');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Fallback to client-side text tokenizer:', err);
    return clientSideTextToSign(text);
  }
}

export async function clearCurrentSentence() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/clear_sentence`, { method: 'POST' });
    return await res.json();
  } catch (err) {
    return { success: true };
  }
}

let cachedVietnameseVoice = null;
let speechVoicesInitialized = false;
let activeUtteranceCount = 0;
let lastTtsTelemetry = {
  text: null,
  callEpochMs: null,
  onstartLatencyMs: null,
  voiceName: null,
  localService: null,
};

function pickBestVietnameseVoice(voices) {
  if (!Array.isArray(voices) || voices.length === 0) return null;
  const viVoices = voices.filter(
    (v) =>
      v &&
      typeof v.lang === 'string' &&
      (/^vi([-_]|$)/i.test(v.lang) || v.lang.toUpperCase().includes('VN'))
  );
  if (viVoices.length === 0) return null;
  // Ưu tiên giọng nội bộ trên máy (localService) để triệt tiêu độ trễ mạng trước khi phát âm
  return viVoices.find((v) => v.localService) || viVoices[0];
}

export function initSpeechSynthesisVoices() {
  if (typeof window === 'undefined' || !window.speechSynthesis) return null;
  const synth = window.speechSynthesis;
  const refreshVoices = () => {
    try {
      const voices = synth.getVoices();
      const selected = pickBestVietnameseVoice(voices);
      if (selected) {
        cachedVietnameseVoice = selected;
      }
    } catch {
      // Bỏ qua lỗi truy vấn danh sách giọng đọc của trình duyệt
    }
  };

  refreshVoices();
  if (!speechVoicesInitialized) {
    speechVoicesInitialized = true;
    if (typeof synth.addEventListener === 'function') {
      synth.addEventListener('voiceschanged', refreshVoices);
    } else if ('onvoiceschanged' in synth) {
      synth.onvoiceschanged = refreshVoices;
    }
  }
  return cachedVietnameseVoice;
}

if (typeof window !== 'undefined' && window.speechSynthesis) {
  initSpeechSynthesisVoices();
}

export function cancelVietnameseSpeech() {
  activeUtteranceCount = 0;
  if (typeof window === 'undefined' || !window.speechSynthesis) return;
  try {
    if (window.speechSynthesis.speaking || window.speechSynthesis.pending) {
      window.speechSynthesis.cancel();
    }
  } catch {
    // Bỏ qua lỗi hủy speechSynthesis
  }
}

export function getLastTtsTelemetry() {
  return { ...lastTtsTelemetry };
}

/**
 * Phát âm tiếng Việt (Text-to-Speech) sử dụng Web Speech API.
 * - Không gọi cancel() vô điều kiện khi hàng đợi rỗng để tránh độ trễ khởi động lại audio pipeline.
 * - Tự động dùng giọng tiếng Việt đã nạp sẵn qua voiceschanged.
 */
export function speakVietnamese(text, options = {}) {
  if (typeof window === 'undefined' || !window.speechSynthesis || !text) return null;

  const synth = window.speechSynthesis;
  const { interrupt = false, onStart = null } = options || {};

  if (interrupt) {
    if (synth.speaking || synth.pending) {
      synth.cancel();
    }
    activeUtteranceCount = 0;
  } else {
    if (synth.paused) {
      try { synth.resume(); } catch { /* ignore */ }
    }
    // Chỉ dọn hàng đợi nếu bị ứ đọng >= 2 câu chờ chưa đọc xong; không hủy khi đang rảnh hoặc chỉ có 1 âm tiết ngắn
    if ((synth.speaking || synth.pending) && activeUtteranceCount >= 2) {
      synth.cancel();
      activeUtteranceCount = 0;
    }
  }

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'vi-VN';
  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  const viVoice = cachedVietnameseVoice || initSpeechSynthesisVoices();
  if (viVoice) {
    utterance.voice = viVoice;
  }

  const callPerfNow = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const callEpochMs = Date.now();
  lastTtsTelemetry = {
    text,
    callEpochMs,
    onstartLatencyMs: null,
    voiceName: viVoice?.name || null,
    localService: typeof viVoice?.localService === 'boolean' ? viVoice.localService : null,
  };

  activeUtteranceCount += 1;
  let settled = false;
  const markSettled = () => {
    if (!settled) {
      settled = true;
      activeUtteranceCount = Math.max(0, activeUtteranceCount - 1);
    }
  };

  utterance.onstart = () => {
    const startPerfNow = typeof performance !== 'undefined' ? performance.now() : Date.now();
    const latencyMs = Math.max(0, Math.round((startPerfNow - callPerfNow) * 10) / 10);
    lastTtsTelemetry = {
      ...lastTtsTelemetry,
      onstartLatencyMs: latencyMs,
    };
    if (typeof onStart === 'function') {
      try { onStart(latencyMs); } catch { /* ignore */ }
    }
  };
  utterance.onend = markSettled;
  utterance.onerror = markSettled;

  synth.speak(utterance);
  return utterance;
}

/**
 * Fallback bộ nhận diện tại Client (Edge) khi mất kết nối Backend
 */
function clientSideLocalRecognize(landmarks) {
  if (!landmarks || landmarks.length < 21) {
    return { sign: 'NEUTRAL', label: 'Chờ cử chỉ...', confidence: 0.0, confirmed: false };
  }
  const p = landmarks;
  const thumbExt = p[4].x < p[3].x;
  const indexExt = p[8].y < p[6].y;
  const middleExt = p[12].y < p[10].y;
  const ringExt = p[16].y < p[14].y;
  const pinkyExt = p[20].y < p[18].y;

  if (indexExt && middleExt && !ringExt && !pinkyExt) {
    return { sign: 'V', label: 'Chữ V / Hòa bình', confidence: 0.94, confirmed: true };
  }
  if (indexExt && thumbExt && !middleExt && !ringExt && !pinkyExt) {
    return { sign: 'L', label: 'Chữ L', confidence: 0.95, confirmed: true };
  }
  if (indexExt && thumbExt && pinkyExt && !middleExt && !ringExt) {
    return { sign: 'YEU_THUONG', label: 'Yêu thương', confidence: 0.96, confirmed: true };
  }
  if (indexExt && middleExt && ringExt && pinkyExt) {
    return { sign: 'XIN_CHAO', label: 'Xin chào', confidence: 0.92, confirmed: true };
  }
  return { sign: 'GESTURE', label: 'Cử chỉ VSL', confidence: 0.85, confirmed: true };
}

/**
 * Fallback Text-to-Sign Client-side
 */
function clientSideTextToSign(text) {
  // Mất kết nối backend → không có bộ từ vựng để ghép cụm từ / tra chuyển động.
  // Không giả vờ diễn ký hiệu: robot ĐÁNH VẦN chữ cái ngón tay từng từ (đúng chữ, dù chậm hơn).
  const toSpell = (w) => w
    .replace(/đ/g, 'd').replace(/Đ/g, 'D')
    .normalize('NFD').replace(/[\u0300-\u036f]/g, '')
    .toUpperCase().split('').filter(c => /[A-Z]/.test(c)).slice(0, 12);
  const words = text.trim().split(/\s+/).map(w => w.replace(/[?!.,;:"“”]+/g, '')).filter(Boolean);
  const sequence = words.map(w => ({
    token: w,
    gloss: w.toUpperCase(),
    label: w.charAt(0).toUpperCase() + w.slice(1),
    type: 'vocabulary',
    duration_ms: 1500,
    description: `Chưa kết nối máy chủ — robot đánh vần chữ cái ngón tay cho từ "${w}"`,
    pose_key: null,
    motion_id: null,
    has_pose: false,
    spell: toSpell(w),
    offline: true,
  }));
  return {
    success: true,
    offline: true,
    input_text: text,
    sequence,
    total_tokens: sequence.length,
    total_duration_ms: sequence.length * 1500
  };
}
