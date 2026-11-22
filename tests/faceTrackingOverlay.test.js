import test from 'node:test';
import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

import {
  extractValidFaceLandmarks,
  resolveFaceContourGroups,
  filterBodyPoseConnections,
  drawFaceContours,
} from '../src/utils/faceContoursRenderer.js';
import {
  FACE_STABILITY_ANCHORS,
  extractFaceAnchorPoints,
  createFaceFrameSample,
  summarizeFaceDiagnostics,
  stripInternalAnchors,
} from '../src/utils/faceDiagnostics.js';
import { HandTrackerStabilizer } from '../src/utils/handTrackerStabilizer.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, '..');

function makeSyntheticFaceLandmarks(count = 468, offset = 0) {
  return Array.from({ length: count }, (_, idx) => ({
    x: 0.4 + ((idx % 20) / 100) + offset,
    y: 0.25 + (Math.floor(idx / 20) / 100) + offset * 0.5,
    z: -0.02,
  }));
}

function makeMockCanvasContext() {
  const calls = [];
  const ctx = {
    lineCap: 'butt',
    lineJoin: 'miter',
    strokeStyle: '#000',
    lineWidth: 1,
    save() { calls.push({ op: 'save' }); },
    restore() { calls.push({ op: 'restore' }); },
    beginPath() {
      calls.push({ op: 'beginPath', strokeStyle: this.strokeStyle, lineWidth: this.lineWidth });
    },
    moveTo(x, y) { calls.push({ op: 'moveTo', x, y }); },
    lineTo(x, y) { calls.push({ op: 'lineTo', x, y }); },
    stroke() {
      calls.push({ op: 'stroke', strokeStyle: this.strokeStyle, lineWidth: this.lineWidth });
    },
  };
  return { ctx, calls };
}

test('extractValidFaceLandmarks accepts 468 landmarks and rejects empty/invalid without synthesizing points', () => {
  const valid = makeSyntheticFaceLandmarks(468);
  assert.equal(extractValidFaceLandmarks(valid), valid);
  assert.equal(extractValidFaceLandmarks(null), null);
  assert.equal(extractValidFaceLandmarks(undefined), null);
  assert.equal(extractValidFaceLandmarks([]), null);
  assert.equal(extractValidFaceLandmarks([{ x: NaN, y: 0.5 }]), null);
});

test('resolveFaceContourGroups includes contours, ignores FACEMESH_TESSELATION, adds nose only from API', () => {
  const withoutNose = resolveFaceContourGroups({
    FACEMESH_TESSELATION: [[0, 1], [1, 2]],
  });
  assert.deepEqual(
    withoutNose.map((g) => g.name),
    ['faceOval', 'leftEyebrow', 'rightEyebrow', 'leftEye', 'rightEye', 'lips']
  );
  assert.equal(withoutNose.reduce((sum, g) => sum + g.edges.length, 0), 124);

  const apiNose = [[1, 2], [2, 98], [98, 327]];
  const withApiGlobals = resolveFaceContourGroups({
    FACEMESH_TESSELATION: [[0, 1]],
    FACEMESH_FACE_OVAL: [[10, 338]],
    FACEMESH_NOSE: apiNose,
  });
  assert.deepEqual(
    withApiGlobals.map((g) => g.name),
    ['faceOval', 'leftEyebrow', 'rightEyebrow', 'leftEye', 'rightEye', 'lips', 'nose']
  );
  assert.equal(withApiGlobals[0].source, 'api');
  assert.equal(withApiGlobals[6].source, 'api');
  assert.equal(withApiGlobals[6].edges.length, 3);
});

test('drawFaceContours batches lightweight contours on Canvas2D without mutating landmarks', () => {
  const landmarks = makeSyntheticFaceLandmarks(468);
  const snapshot = JSON.stringify(landmarks);
  const { ctx, calls } = makeMockCanvasContext();

  const res = drawFaceContours(ctx, landmarks, 640, 480, {});
  assert.equal(res.drawn, true);
  assert.equal(res.landmarkCount, 468);
  assert.equal(res.segmentsDrawn, 124);
  assert.deepEqual(res.groupsDrawn, ['faceOval', 'leftEyebrow', 'rightEyebrow', 'leftEye', 'rightEye', 'lips']);

  const strokeCalls = calls.filter((c) => c.op === 'stroke');
  const moveCalls = calls.filter((c) => c.op === 'moveTo');
  const lineCalls = calls.filter((c) => c.op === 'lineTo');
  assert.equal(strokeCalls.length, 6);
  assert.equal(moveCalls.length, 124);
  assert.equal(lineCalls.length, 124);
  assert.equal(JSON.stringify(landmarks), snapshot);

  // Also test with API FACEMESH_NOSE present
  const { ctx: ctxNose } = makeMockCanvasContext();
  const resNose = drawFaceContours(ctxNose, landmarks, 640, 480, {
    FACEMESH_NOSE: [[1, 2], [2, 5]],
  });
  assert.equal(resNose.drawn, true);
  assert.equal(resNose.segmentsDrawn, 126);
  assert.ok(resNose.groupsDrawn.includes('nose'));
});

test('drawFaceContours gracefully does nothing when face landmarks are missing or canvas invalid', () => {
  const { ctx, calls } = makeMockCanvasContext();
  assert.deepEqual(drawFaceContours(ctx, null, 640, 480, {}), {
    drawn: false,
    segmentsDrawn: 0,
    groupsDrawn: [],
    landmarkCount: 0,
  });
  assert.deepEqual(drawFaceContours(ctx, [], 640, 480, {}), {
    drawn: false,
    segmentsDrawn: 0,
    groupsDrawn: [],
    landmarkCount: 0,
  });
  assert.deepEqual(drawFaceContours(ctx, makeSyntheticFaceLandmarks(468), 0, 480, {}), {
    drawn: false,
    segmentsDrawn: 0,
    groupsDrawn: [],
    landmarkCount: 0,
  });
  assert.equal(calls.length, 0);
});

test('filterBodyPoseConnections strips head/face pose indices 0..10 without mutating source connections', () => {
  const samplePoseConnections = [
    [0, 1], [1, 2], [9, 10], [0, 4], // pose face indices -> must drop when Face Contours active
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
    { start: 0, end: 7 },
    { start: 15, end: 17 },
  ];
  const frozenBefore = JSON.stringify(samplePoseConnections);
  const filtered = filterBodyPoseConnections(samplePoseConnections);

  assert.equal(JSON.stringify(samplePoseConnections), frozenBefore);
  assert.equal(filtered.length, 8);
  assert.deepEqual(filtered.slice(0, 7), [
    [11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
  ]);
  assert.deepEqual(filtered[7], { start: 15, end: 17 });
});

test('faceDiagnostics computes detection, counts, FPS, render latency, resolutions, missing streaks, and stability', () => {
  const samples = [];
  // 6 detected frames (5 consecutive pairs) at ~30fps, 1 missing frame, then 2 detected
  for (let i = 0; i < 9; i++) {
    const isMissing = i === 6 || i === 7;
    const faceLandmarks = isMissing ? null : makeSyntheticFaceLandmarks(468, i * 0.001);
    samples.push(createFaceFrameSample({
      ms: i * 33.33,
      faceLandmarks,
      renderMs: isMissing ? 0.05 : 0.25,
      holisticProcessMs: 18.5,
      videoWidth: 640,
      videoHeight: 480,
      processingWidth: 480,
      processingHeight: 360,
      canvasWidth: 640,
      canvasHeight: 480,
    }));
  }

  const summary = summarizeFaceDiagnostics(samples);
  assert.equal(summary.totalFrames, 9);
  assert.equal(summary.faceDetectedFrames, 7);
  assert.equal(summary.faceMissingFrames, 2);
  assert.equal(summary.maxConsecutiveMissingFrames, 2);
  assert.equal(summary.faceDetectionRate, 0.7778);
  assert.equal(summary.faceLandmarkCountMin, 468);
  assert.equal(summary.faceLandmarkCountMax, 468);
  assert.equal(summary.faceLandmarkCountAvg, 468);
  assert.equal(summary.videoResolution, '640x480');
  assert.equal(summary.processingResolution, '480x360');
  assert.equal(summary.canvasResolution, '640x480');
  assert.ok(summary.processingFps >= 29 && summary.processingFps <= 31);
  assert.ok(summary.avgRenderMs > 0.15 && summary.avgRenderMs < 0.3);
  assert.equal(summary.avgProcessMs, 18.5);
  assert.equal(summary.consecutiveDetectedPairs, 5);
  assert.equal(summary.stabilitySufficientData, true);
  assert.ok(summary.meanFrameToFrameDisplacementNorm > 0);
  assert.ok(summary.meanFrameToFrameDisplacementPx > 0);
  assert.ok(summary.anchorStdDevNorm > 0);
  assert.ok(summary.anchorStdDevPx > 0);

  const exportedFrames = stripInternalAnchors(samples);
  assert.equal(exportedFrames.length, 9);
  assert.equal('anchors' in exportedFrames[0], false);
  assert.equal('image' in exportedFrames[0], false);
  assert.equal('video' in exportedFrames[0], false);

  // Verify insufficient data returns null stability metrics
  const shortSummary = summarizeFaceDiagnostics(samples.slice(0, 4));
  assert.equal(shortSummary.stabilitySufficientData, false);
  assert.equal(shortSummary.meanFrameToFrameDisplacementNorm, null);
  assert.equal(shortSummary.meanFrameToFrameDisplacementPx, null);
  assert.equal(shortSummary.anchorStdDevNorm, null);
  assert.equal(shortSummary.anchorStdDevPx, null);
  assert.ok(shortSummary.stabilityNote.includes('Chưa đủ frame liên tiếp'));
});

test('extractFaceAnchorPoints requires >= 3 valid anchor landmarks and never synthesizes missing ones', () => {
  const valid = makeSyntheticFaceLandmarks(468);
  const anchors = extractFaceAnchorPoints(valid);
  assert.equal(anchors.length, FACE_STABILITY_ANCHORS.length);

  const partial = [];
  partial[1] = { x: 0.5, y: 0.3 };
  partial[33] = { x: 0.4, y: 0.2 };
  assert.equal(extractFaceAnchorPoints(partial), null);
});

test('HandTrackerStabilizer & 75-joint Holistic contract remain unchanged alongside face contour rendering', () => {
  const stabA = new HandTrackerStabilizer();
  const stabB = new HandTrackerStabilizer();
  const pose = Array.from({ length: 33 }, (_, i) => ({ x: 0.5 + i * 0.001, y: 0.4, z: 0, visibility: 0.9 }));
  const left = Array.from({ length: 21 }, (_, i) => ({ x: 0.6 + i * 0.005, y: 0.5 + i * 0.002, z: -0.01 }));
  const right = Array.from({ length: 21 }, (_, i) => ({ x: 0.3 + i * 0.005, y: 0.5 + i * 0.002, z: -0.01 }));
  const face = makeSyntheticFaceLandmarks(468);
  const { ctx } = makeMockCanvasContext();

  const outA = stabA.process(pose, left, right, 1000);
  drawFaceContours(ctx, face, 640, 480, {});
  const outB = stabB.process(pose, left, right, 1000);

  assert.deepEqual(outA.modelLeft, outB.modelLeft);
  assert.deepEqual(outA.modelRight, outB.modelRight);
  assert.deepEqual(outA.drawLeft, outB.drawLeft);
  assert.deepEqual(outA.drawRight, outB.drawRight);

  // Verify CameraSignRecognizer.jsx preserves guardrails
  const csrSrc = fs.readFileSync(path.join(ROOT, 'src/components/CameraSignRecognizer.jsx'), 'utf8');
  assert.equal(csrSrc.includes('refineFaceLandmarks: true'), false);
  assert.equal(csrSrc.includes('hCanvas.width = 480;'), true);
  assert.equal(csrSrc.includes('hCanvas.height = 360;'), true);
  assert.equal(csrSrc.includes('pCanvas.width = 320;'), true);
  assert.equal(csrSrc.includes('pCanvas.height = 240;'), true);
  assert.equal(
    csrSrc.includes("conversionModeRef.current === 'words'"),
    true
  );
  assert.equal(
    csrSrc.includes("onHandsResultsFast"),
    true
  );
  assert.equal(
    csrSrc.includes("renderSkeletonsOverlay(ctx, drawLeft, drawRight, null, canvas);"),
    true
  );
});
