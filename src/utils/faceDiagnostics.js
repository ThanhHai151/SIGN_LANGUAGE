/**
 * Pure telemetry and stability summarization helpers for controlled Face Tracking diagnostics.
 * Stores numeric metrics only — zero image frames or video blobs.
 */

export const FACE_STABILITY_ANCHORS = [1, 33, 263, 152, 13];
const MIN_STABILITY_PAIRS = 5;

function toPositiveInt(value) {
  return Number.isFinite(value) && value > 0 ? Math.round(value) : null;
}

function roundTo(value, decimals) {
  const factor = 10 ** decimals;
  return Math.round(value * factor) / factor;
}

function computeSeriesStats(values, decimals) {
  if (!values.length) {
    return { avg: null, p95: null, max: null };
  }
  let sum = 0;
  let max = -Infinity;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    sum += v;
    if (v > max) max = v;
  }
  const sorted = [...values].sort((a, b) => a - b);
  const p95Idx = Math.min(sorted.length - 1, Math.max(0, Math.ceil(0.95 * sorted.length) - 1));
  return {
    avg: roundTo(sum / values.length, decimals),
    p95: roundTo(sorted[p95Idx], decimals),
    max: roundTo(max, decimals),
  };
}

/**
 * Extract canonical 2D anchor landmarks ([nose, eye_R, eye_L, chin, lip]) for stability checks.
 * Returns null when fewer than 3 anchors are valid (never fabricates points).
 */
export function extractFaceAnchorPoints(faceLandmarks) {
  if (!Array.isArray(faceLandmarks) || faceLandmarks.length === 0) {
    return null;
  }
  const anchors = [];
  for (let i = 0; i < FACE_STABILITY_ANCHORS.length; i++) {
    const idx = FACE_STABILITY_ANCHORS[i];
    const lm = faceLandmarks[idx];
    if (lm && Number.isFinite(lm.x) && Number.isFinite(lm.y)) {
      anchors.push({ idx, x: lm.x, y: lm.y });
    }
  }
  return anchors.length >= 3 ? anchors : null;
}

/**
 * Build one numeric-only diagnostic sample for a processed Holistic frame.
 */
export function createFaceFrameSample({
  ms,
  faceLandmarks,
  renderMs,
  holisticProcessMs,
  videoWidth,
  videoHeight,
  processingWidth,
  processingHeight,
  canvasWidth,
  canvasHeight,
}) {
  const detected = Array.isArray(faceLandmarks) && faceLandmarks.length > 0;
  return {
    ms: Number.isFinite(ms) ? Math.round(ms) : 0,
    faceDetected: detected,
    faceLandmarkCount: detected ? faceLandmarks.length : 0,
    renderMs: Number.isFinite(renderMs) && renderMs >= 0 ? roundTo(renderMs, 3) : null,
    holisticProcessMs:
      Number.isFinite(holisticProcessMs) && holisticProcessMs >= 0 ? roundTo(holisticProcessMs, 2) : null,
    videoWidth: toPositiveInt(videoWidth),
    videoHeight: toPositiveInt(videoHeight),
    processingWidth: toPositiveInt(processingWidth),
    processingHeight: toPositiveInt(processingHeight),
    canvasWidth: toPositiveInt(canvasWidth),
    canvasHeight: toPositiveInt(canvasHeight),
    anchors: detected ? extractFaceAnchorPoints(faceLandmarks) : null,
  };
}

function findLatestResolution(samples, widthKey, heightKey) {
  for (let i = samples.length - 1; i >= 0; i--) {
    const w = samples[i]?.[widthKey];
    const h = samples[i]?.[heightKey];
    if (Number.isFinite(w) && w > 0 && Number.isFinite(h) && h > 0) {
      return { width: w, height: h, text: `${w}x${h}` };
    }
  }
  return null;
}

/**
 * Summarize numeric Face Tracking metrics and anchor stability over collected samples.
 */
export function summarizeFaceDiagnostics(samples, meta = {}) {
  const list = Array.isArray(samples) ? samples : [];
  const totalFrames = list.length;

  let faceDetectedFrames = 0;
  let faceMissingFrames = 0;
  let currentMissingStreak = 0;
  let maxConsecutiveMissingFrames = 0;

  let lmMin = Infinity;
  let lmMax = -Infinity;
  let lmSum = 0;
  let lmCounted = 0;

  const renderTimes = [];
  const processTimes = [];

  for (let i = 0; i < totalFrames; i++) {
    const s = list[i];
    if (s.faceDetected) {
      faceDetectedFrames += 1;
      currentMissingStreak = 0;
      if (Number.isFinite(s.faceLandmarkCount) && s.faceLandmarkCount > 0) {
        if (s.faceLandmarkCount < lmMin) lmMin = s.faceLandmarkCount;
        if (s.faceLandmarkCount > lmMax) lmMax = s.faceLandmarkCount;
        lmSum += s.faceLandmarkCount;
        lmCounted += 1;
      }
    } else {
      faceMissingFrames += 1;
      currentMissingStreak += 1;
      if (currentMissingStreak > maxConsecutiveMissingFrames) {
        maxConsecutiveMissingFrames = currentMissingStreak;
      }
    }
    if (Number.isFinite(s.renderMs) && s.renderMs >= 0) renderTimes.push(s.renderMs);
    if (Number.isFinite(s.holisticProcessMs) && s.holisticProcessMs >= 0) processTimes.push(s.holisticProcessMs);
  }

  const spanMs = totalFrames >= 2 ? Math.max(0, list[totalFrames - 1].ms - list[0].ms) : 0;
  const durationMs = Number.isFinite(meta.durationMs) && meta.durationMs >= 0
    ? Math.round(meta.durationMs)
    : spanMs;

  const processingFps = totalFrames >= 2 && spanMs > 0
    ? roundTo((totalFrames - 1) / (spanMs / 1000), 1)
    : null;

  const renderStats = computeSeriesStats(renderTimes, 3);
  const processStats = computeSeriesStats(processTimes, 2);

  const videoRes = findLatestResolution(list, 'videoWidth', 'videoHeight');
  const procRes = findLatestResolution(list, 'processingWidth', 'processingHeight');
  const canvasRes = findLatestResolution(list, 'canvasWidth', 'canvasHeight');

  // Stability calculations across consecutive detected anchor frames
  let consecutiveDetectedPairs = 0;
  for (let i = 1; i < totalFrames; i++) {
    if (Array.isArray(list[i - 1].anchors) && Array.isArray(list[i].anchors)) {
      consecutiveDetectedPairs += 1;
    }
  }

  const stabilitySufficientData = consecutiveDetectedPairs >= MIN_STABILITY_PAIRS;
  let meanFrameToFrameDisplacementNorm = null;
  let meanFrameToFrameDisplacementPx = null;
  let anchorStdDevNorm = null;
  let anchorStdDevPx = null;
  let stabilityNote = null;

  if (stabilitySufficientData) {
    const pxW = videoRes?.width || canvasRes?.width || null;
    const pxH = videoRes?.height || canvasRes?.height || null;
    const hasPxScale = Number.isFinite(pxW) && pxW > 0 && Number.isFinite(pxH) && pxH > 0;

    let stepNormSum = 0;
    let stepPxSum = 0;
    let stepCount = 0;

    for (let i = 1; i < totalFrames; i++) {
      const prevA = list[i - 1].anchors;
      const currA = list[i].anchors;
      if (!Array.isArray(prevA) || !Array.isArray(currA)) continue;

      const prevByIdx = new Map();
      for (let p = 0; p < prevA.length; p++) prevByIdx.set(prevA[p].idx, prevA[p]);

      for (let c = 0; c < currA.length; c++) {
        const curPt = currA[c];
        const prevPt = prevByIdx.get(curPt.idx);
        if (!prevPt) continue;
        const dx = curPt.x - prevPt.x;
        const dy = curPt.y - prevPt.y;
        stepNormSum += Math.hypot(dx, dy);
        if (hasPxScale) {
          stepPxSum += Math.hypot(dx * pxW, dy * pxH);
        }
        stepCount += 1;
      }
    }

    if (stepCount > 0) {
      meanFrameToFrameDisplacementNorm = roundTo(stepNormSum / stepCount, 5);
      meanFrameToFrameDisplacementPx = hasPxScale ? roundTo(stepPxSum / stepCount, 2) : null;
    }

    // Per-anchor positional scatter across all frames with valid anchors
    const byAnchor = new Map();
    for (let i = 0; i < totalFrames; i++) {
      const anchors = list[i].anchors;
      if (!Array.isArray(anchors)) continue;
      for (let a = 0; a < anchors.length; a++) {
        const pt = anchors[a];
        if (!byAnchor.has(pt.idx)) byAnchor.set(pt.idx, []);
        byAnchor.get(pt.idx).push(pt);
      }
    }

    let stdNormSum = 0;
    let stdPxSum = 0;
    let stdCount = 0;
    for (const pts of byAnchor.values()) {
      if (pts.length < MIN_STABILITY_PAIRS) continue;
      let mx = 0;
      let my = 0;
      for (let k = 0; k < pts.length; k++) {
        mx += pts[k].x;
        my += pts[k].y;
      }
      mx /= pts.length;
      my /= pts.length;

      let varNorm = 0;
      let varPx = 0;
      for (let k = 0; k < pts.length; k++) {
        const dx = pts[k].x - mx;
        const dy = pts[k].y - my;
        varNorm += dx * dx + dy * dy;
        if (hasPxScale) {
          varPx += (dx * pxW) ** 2 + (dy * pxH) ** 2;
        }
      }
      stdNormSum += Math.sqrt(varNorm / pts.length);
      if (hasPxScale) {
        stdPxSum += Math.sqrt(varPx / pts.length);
      }
      stdCount += 1;
    }

    if (stdCount > 0) {
      anchorStdDevNorm = roundTo(stdNormSum / stdCount, 5);
      anchorStdDevPx = hasPxScale ? roundTo(stdPxSum / stdCount, 2) : null;
    }
  } else {
    stabilityNote =
      'Chưa đủ frame liên tiếp có khuôn mặt (cần >= 5 cặp frame) để kết luận độ ổn định.';
  }

  return {
    totalFrames,
    faceDetectedFrames,
    faceMissingFrames,
    faceDetectionRate: totalFrames > 0 ? roundTo(faceDetectedFrames / totalFrames, 4) : null,
    maxConsecutiveMissingFrames,
    faceLandmarkCountMin: lmCounted > 0 ? lmMin : null,
    faceLandmarkCountMax: lmCounted > 0 ? lmMax : null,
    faceLandmarkCountAvg: lmCounted > 0 ? roundTo(lmSum / lmCounted, 1) : null,
    durationMs,
    processingFps,
    avgRenderMs: renderStats.avg,
    p95RenderMs: renderStats.p95,
    maxRenderMs: renderStats.max,
    avgProcessMs: processStats.avg,
    p95ProcessMs: processStats.p95,
    videoResolution: videoRes?.text || null,
    processingResolution: procRes?.text || null,
    canvasResolution: canvasRes?.text || null,
    consecutiveDetectedPairs,
    stabilitySufficientData,
    meanFrameToFrameDisplacementNorm,
    meanFrameToFrameDisplacementPx,
    anchorStdDevNorm,
    anchorStdDevPx,
    stabilityNote,
  };
}

/**
 * Remove internal anchor coordinates before exporting JSON reports.
 */
export function stripInternalAnchors(samples) {
  if (!Array.isArray(samples)) return [];
  return samples.map((s) => ({
    ms: s.ms,
    faceDetected: s.faceDetected,
    faceLandmarkCount: s.faceLandmarkCount,
    renderMs: s.renderMs,
    holisticProcessMs: s.holisticProcessMs,
    videoWidth: s.videoWidth,
    videoHeight: s.videoHeight,
    processingWidth: s.processingWidth,
    processingHeight: s.processingHeight,
    canvasWidth: s.canvasWidth,
    canvasHeight: s.canvasHeight,
  }));
}
