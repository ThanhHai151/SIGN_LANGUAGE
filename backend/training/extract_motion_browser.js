/*
 * TRÍCH CHUYỂN ĐỘNG KÝ HIỆU TỪ VIDEO QIPEDC (chạy trong trình duyệt)
 * ===================================================================
 * Robot "học" ký hiệu bằng cách bắt chước người Điếc trong video Danh mục NNKH QIPEDC (Bộ GD&ĐT).
 *
 * Cách dùng:
 *   1. Mở https://qipedc.moet.gov.vn/dictionary trong Chrome/Edge.
 *   2. F12 → Console → dán toàn bộ file này → Enter.
 *   3. Gọi:  await vslExtract(['D0002', 'W00489', ...])     // mã video (trường "_id" của QIPEDC)
 *      hoặc: await vslExtractAll(800)                         // 800 video đầu tiên của từ điển
 *   4. Khi xong, trình duyệt tải về file  qipedc_motion_raw.json.gz
 *   5. Giải nén vào thư mục raw_motion/ (mỗi video 1 file) rồi chạy:
 *        python -m backend.training.batch_retarget raw_motion
 *
 * Mỗi khung hình (8 khung/giây) lưu:
 *   p = 9 điểm pose world (mũi, vai T/P, khuỷu T/P, cổ tay T/P, hông T/P), đơn vị mm
 *   l, r = 21 điểm bàn tay trái / phải (hand world landmarks, gốc tại cổ tay pose), mm — null nếu không thấy tay
 * Kết quả lưu trong IndexedDB 'vslmotion' nên có thể dừng/chạy tiếp mà không mất dữ liệu.
 */
(() => {
  const MP_VER = '0.10.14';
  const MODEL = 'https://storage.googleapis.com/mediapipe-models/holistic_landmarker/holistic_landmarker/float16/latest/holistic_landmarker.task';
  let holistic = null;
  let ts = 1;

  async function init() {
    if (holistic) return holistic;
    const { FilesetResolver, HolisticLandmarker } = await import(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/vision_bundle.mjs`);
    const fs = await FilesetResolver.forVisionTasks(`https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VER}/wasm`);
    try {
      holistic = await HolisticLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: MODEL, delegate: 'GPU' }, runningMode: 'VIDEO' });
    } catch (e) {
      holistic = await HolisticLandmarker.createFromOptions(fs, { baseOptions: { modelAssetPath: MODEL, delegate: 'CPU' }, runningMode: 'VIDEO' });
    }
    return holistic;
  }

  const db = new Promise((res, rej) => {
    const r = indexedDB.open('vslmotion', 1);
    r.onupgradeneeded = () => r.result.createObjectStore('m', { keyPath: 'id' });
    r.onsuccess = () => res(r.result);
    r.onerror = () => rej(r.error);
  });
  const dbGet = async (id) => new Promise((res) => { const t = (/** @type {IDBDatabase} */ (dbCache)).transaction('m').objectStore('m').get(id); t.onsuccess = () => res(t.result); t.onerror = () => res(null); });
  const dbPut = async (o) => new Promise((res) => { const t = dbCache.transaction('m', 'readwrite').objectStore('m').put(o); t.onsuccess = () => res(true); t.onerror = () => res(false); });
  const dbAll = async () => new Promise((res) => { const t = dbCache.transaction('m').objectStore('m').getAll(); t.onsuccess = () => res(t.result); });
  let dbCache = null;

  async function extractOne(id, fps = 8) {
    const v = document.createElement('video');
    v.muted = true; v.preload = 'auto'; v.src = `/videos/${id}.mp4`;
    await new Promise((res, rej) => { v.onloadeddata = res; v.onerror = () => rej(new Error('Không tải được ' + id)); setTimeout(() => rej(new Error('Quá thời gian ' + id)), 20000); });
    const r3 = (x) => Math.round(x * 1000);
    const pk = (lms, idx) => (lms ? (idx ? idx.map((i) => lms[i]) : lms).flatMap((p) => [r3(p.x), r3(p.y), r3(p.z)]) : null);
    const frames = [];
    for (let t = 0; t < v.duration; t += 1 / fps) {
      await new Promise((res) => { v.onseeked = res; v.currentTime = Math.min(t, v.duration - 0.01); });
      ts += 40;
      const out = holistic.detectForVideo(v, ts);
      frames.push({
        p: pk(out.poseWorldLandmarks?.[0], [0, 11, 12, 13, 14, 15, 16, 23, 24]),
        l: pk(out.leftHandWorldLandmarks?.[0]),
        r: pk(out.rightHandWorldLandmarks?.[0]),
      });
    }
    const dur = Math.round(v.duration * 1000) / 1000;
    v.removeAttribute('src'); v.load();
    return { id, fps, dur, frames };
  }

  async function download() {
    const all = await dbAll();
    const gz = new Blob([JSON.stringify(all)]).stream().pipeThrough(new CompressionStream('gzip'));
    const url = URL.createObjectURL(await new Response(gz).blob());
    const a = Object.assign(document.createElement('a'), { href: url, download: 'qipedc_motion_raw.json.gz' });
    document.body.appendChild(a); a.click(); a.remove();
    console.log(`[VSL] Đã tải ${all.length} video về máy.`);
  }

  window.vslExtract = async (ids, { fps = 8, autoDownload = true } = {}) => {
    dbCache = await db;
    await init();
    let done = 0;
    for (const id of ids) {
      if (await dbGet(id)) { done++; continue; }
      const t0 = performance.now();
      try { await dbPut(await extractOne(id, fps)); } catch (e) { console.warn('[VSL]', e.message); }
      done++;
      console.log(`[VSL] ${done}/${ids.length} ${id} (${Math.round(performance.now() - t0)} ms)`);
    }
    if (autoDownload) await download();
  };

  window.vslExtractAll = async (limit = 800) => {
    const j = await fetch('/dictionary/getAll', { method: 'POST', headers: { 'Content-Type': 'application/x-www-form-urlencoded' }, body: 'group=100000&text=' }).then((r) => r.json());
    return window.vslExtract(j.data.slice(0, limit).map((x) => x._id));
  };
  window.vslDownload = async () => { dbCache = await db; return download(); };
  console.log('[VSL] Sẵn sàng: await vslExtract([...mã video]) hoặc await vslExtractAll(800)');
})();
