/**
 * Robot tự HỌC ký hiệu mới ngay trong trình duyệt
 * ================================================
 * Từ chưa có chuyển động robot → chạy MediaPipe Holistic trên video QIPEDC của từ đó (phát qua bộ đệm
 * backend để đọc được khung hình), lấy điểm khớp vai/khuỷu/cổ tay + 21 khớp mỗi bàn tay (8 khung/giây),
 * gửi lên backend (/api/motion/learn) để chuyển sang góc khớp robot và lưu lại cho các lần sau.
 */
import { mediaUrl } from './api';

const API_BASE_URL = 'http://127.0.0.1:8000';
const MP_VER = '0.10.14';
const MODEL = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task';
const FPS = 8;

let holisticPromise = null;
let ts = 1;

function getHolistic() {
  if (!holisticPromise) {
    holisticPromise = (async () => {
      const url = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/vision_bundle.mjs`;
      const { FilesetResolver, HolisticLandmarker } = await import(/* @vite-ignore */ url);
      const fs = await FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm`);
      try {
        return await HolisticLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' }, runningMode: 'VIDEO' });
      } catch {
        return HolisticLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: MODEL, delegate: 'CPU' }, runningMode: 'VIDEO' });
      }
    })().catch((e) => { holisticPromise = null; throw e; });
  }
  return holisticPromise;
}

const r3 = (x) => Math.round(x * 1000);
const pack = (lms, idx) => (lms ? (idx ? idx.map((i) => lms[i]) : lms).flatMap((p) => [r3(p.x), r3(p.y), r3(p.z)]) : null);

function loadVideo(src) {
  return new Promise((resolve, reject) => {
    const v = document.createElement('video');
    v.crossOrigin = 'anonymous'; // backend gửi CORS → MediaPipe đọc được khung hình
    v.muted = true;
    v.playsInline = true;
    v.preload = 'auto';
    const t = setTimeout(() => reject(new Error('Quá thời gian tải video')), 25000);
    v.onloadeddata = () => { clearTimeout(t); resolve(v); };
    v.onerror = () => { clearTimeout(t); reject(new Error('Không tải được video')); };
    v.src = src;
  });
}

async function extract(videoId, onProgress) {
  const holistic = await getHolistic();
  const v = await loadVideo(mediaUrl(`https://qipedc.moet.gov.vn/videos/${videoId}.mp4`));
  const frames = [];
  const dur = v.duration;
  const n = Math.max(1, Math.ceil(dur * FPS));
  for (let k = 0; k < n; k++) {
    const t = k / FPS;
    await new Promise((res) => { v.onseeked = res; v.currentTime = Math.min(t, dur - 0.01); });
    ts += 40;
    const out = holistic.detectForVideo(v, ts);
    frames.push({
      p: pack(out.poseWorldLandmarks?.[0], [0, 11, 12, 13, 14, 15, 16, 23, 24]),
      l: pack(out.leftHandWorldLandmarks?.[0]),
      r: pack(out.rightHandWorldLandmarks?.[0]),
    });
    onProgress?.((k + 1) / n);
    if (k % 4 === 3) await new Promise((res) => setTimeout(res, 0)); // nhường luồng cho giao diện/robot
  }
  v.removeAttribute('src');
  v.load();
  return { id: videoId, fps: FPS, dur: Math.round(dur * 1000) / 1000, frames };
}

// Hàng đợi: học lần lượt từng từ (MediaPipe dùng chung 1 mô hình)
const results = new Map();
let chain = Promise.resolve();

/**
 * Học chuyển động cho 1 video QIPEDC. Trả về {success, motion_id, clip, sign_description} hoặc {success:false}.
 * Gọi lại cùng mã → dùng kết quả cũ.
 */
export function learnMotion(videoId, onProgress) {
  if (!videoId) return Promise.resolve({ success: false });
  if (results.has(videoId)) return results.get(videoId);
  const p = (chain = chain.then(async () => {
    try {
      const raw = await extract(videoId, onProgress);
      const res = await fetch(`${API_BASE_URL}/api/motion/learn`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(raw),
      });
      const j = await res.json();
      if (!j.success) results.delete(videoId);
      return j;
    } catch (e) {
      console.warn('[VSL] Không học được ký hiệu', videoId, e);
      results.delete(videoId);
      return { success: false, reason: String(e?.message || e) };
    }
  }));
  results.set(videoId, p);
  return p;
}
