/**
 * Lightweight Canvas2D renderer for MediaPipe Holistic 468-point Face Landmarks.
 * Draws clean face contours (face oval, eyebrows, eyes, lips, and optional nose if
 * exposed by MediaPipe globals) without dense tessellation or synthetic points.
 * Intended strictly for UI visualization in Holistic ('words') mode.
 */

// Canonical MediaPipe 468-landmark contour connections used only when
// the CDN bundle does not expose the corresponding FACEMESH_* global.
const CANONICAL_FACE_OVAL = [
  [10, 338], [338, 297], [297, 332], [332, 284], [284, 251], [251, 389],
  [389, 356], [356, 454], [454, 323], [323, 361], [361, 288], [288, 397],
  [397, 365], [365, 379], [379, 378], [378, 400], [400, 377], [377, 152],
  [152, 148], [148, 176], [176, 149], [149, 150], [150, 136], [136, 172],
  [172, 58], [58, 132], [132, 93], [93, 234], [234, 127], [127, 162],
  [162, 21], [21, 54], [54, 103], [103, 67], [67, 109], [109, 10],
];

const CANONICAL_LEFT_EYE = [
  [263, 249], [249, 390], [390, 373], [373, 374], [374, 380], [380, 381], [381, 382], [382, 362],
  [263, 466], [466, 388], [388, 387], [387, 386], [386, 385], [385, 384], [384, 398], [398, 362],
];

const CANONICAL_RIGHT_EYE = [
  [33, 7], [7, 163], [163, 144], [144, 145], [145, 153], [153, 154], [154, 155], [155, 133],
  [33, 246], [246, 161], [161, 160], [160, 159], [159, 158], [158, 157], [157, 173], [173, 133],
];

const CANONICAL_LEFT_EYEBROW = [
  [276, 283], [283, 282], [282, 295], [295, 285],
  [300, 293], [293, 334], [334, 296], [296, 336],
];

const CANONICAL_RIGHT_EYEBROW = [
  [46, 53], [53, 52], [52, 65], [65, 55],
  [70, 63], [63, 105], [105, 66], [66, 107],
];

const CANONICAL_LIPS = [
  [61, 146], [146, 91], [91, 181], [181, 84], [84, 17], [17, 314], [314, 405], [405, 321], [321, 375], [375, 291],
  [61, 185], [185, 40], [40, 39], [39, 37], [37, 0], [0, 267], [267, 269], [269, 270], [270, 409], [409, 291],
  [78, 95], [95, 88], [88, 178], [178, 87], [87, 14], [14, 317], [317, 402], [402, 318], [318, 324], [324, 308],
  [78, 191], [191, 80], [80, 81], [81, 82], [82, 13], [13, 312], [312, 311], [311, 310], [310, 415], [415, 308],
];

const CONTOUR_SPECS = [
  {
    name: 'faceOval',
    globalKey: 'FACEMESH_FACE_OVAL',
    fallback: CANONICAL_FACE_OVAL,
    color: 'rgba(148, 163, 184, 0.58)',
    lineWidth: 1.5,
  },
  {
    name: 'leftEyebrow',
    globalKey: 'FACEMESH_LEFT_EYEBROW',
    fallback: CANONICAL_LEFT_EYEBROW,
    color: 'rgba(125, 211, 252, 0.72)',
    lineWidth: 1.5,
  },
  {
    name: 'rightEyebrow',
    globalKey: 'FACEMESH_RIGHT_EYEBROW',
    fallback: CANONICAL_RIGHT_EYEBROW,
    color: 'rgba(125, 211, 252, 0.72)',
    lineWidth: 1.5,
  },
  {
    name: 'leftEye',
    globalKey: 'FACEMESH_LEFT_EYE',
    fallback: CANONICAL_LEFT_EYE,
    color: 'rgba(56, 189, 248, 0.88)',
    lineWidth: 1.6,
  },
  {
    name: 'rightEye',
    globalKey: 'FACEMESH_RIGHT_EYE',
    fallback: CANONICAL_RIGHT_EYE,
    color: 'rgba(56, 189, 248, 0.88)',
    lineWidth: 1.6,
  },
  {
    name: 'lips',
    globalKey: 'FACEMESH_LIPS',
    fallback: CANONICAL_LIPS,
    color: 'rgba(244, 114, 182, 0.82)',
    lineWidth: 1.6,
  },
];

const NOSE_SPEC = {
  name: 'nose',
  globalKey: 'FACEMESH_NOSE',
  color: 'rgba(148, 163, 184, 0.65)',
  lineWidth: 1.4,
};

function parseEdgePair(edge) {
  if (Array.isArray(edge) && edge.length >= 2) {
    const a = edge[0];
    const b = edge[1];
    return Number.isInteger(a) && Number.isInteger(b) ? [a, b] : null;
  }
  if (edge && typeof edge === 'object') {
    const a = edge.start;
    const b = edge.end;
    return Number.isInteger(a) && Number.isInteger(b) ? [a, b] : null;
  }
  return null;
}

/**
 * Validate MediaPipe Holistic `results.faceLandmarks` without inventing missing points.
 */
export function extractValidFaceLandmarks(rawFaceLandmarks) {
  if (!Array.isArray(rawFaceLandmarks) || rawFaceLandmarks.length === 0) {
    return null;
  }
  for (let i = 0; i < rawFaceLandmarks.length; i++) {
    const lm = rawFaceLandmarks[i];
    if (lm && Number.isFinite(lm.x) && Number.isFinite(lm.y)) {
      return rawFaceLandmarks;
    }
  }
  return null;
}

/**
 * Resolve contour groups from MediaPipe globals when available, falling back
 * to canonical 468-mesh contour tables (never using dense FACEMESH_TESSELATION).
 */
export function resolveFaceContourGroups(apiGlobals = globalThis) {
  const groups = CONTOUR_SPECS.map((spec) => {
    const apiEdges = apiGlobals?.[spec.globalKey];
    const hasApiEdges = Array.isArray(apiEdges) && apiEdges.length > 0;
    return {
      name: spec.name,
      edges: hasApiEdges ? apiEdges : spec.fallback,
      color: spec.color,
      lineWidth: spec.lineWidth,
      source: hasApiEdges ? 'api' : 'canonical',
    };
  });

  const apiNose = apiGlobals?.[NOSE_SPEC.globalKey];
  if (Array.isArray(apiNose) && apiNose.length > 0) {
    groups.push({
      name: NOSE_SPEC.name,
      edges: apiNose,
      color: NOSE_SPEC.color,
      lineWidth: NOSE_SPEC.lineWidth,
      source: 'api',
    });
  }
  return groups;
}

/**
 * Return a filtered copy of POSE_CONNECTIONS keeping only upper-body/torso/arm links
 * (indices >= 11) so coarse Pose face segments (0..10) do not overlap Face Contours.
 */
export function filterBodyPoseConnections(poseConnections) {
  if (!Array.isArray(poseConnections)) {
    return [];
  }
  const filtered = [];
  for (let k = 0; k < poseConnections.length; k++) {
    const pair = parseEdgePair(poseConnections[k]);
    if (pair && pair[0] >= 11 && pair[1] >= 11) {
      filtered.push(poseConnections[k]);
    }
  }
  return filtered;
}

/**
 * Render batched Canvas2D face contours onto the overlay canvas.
 */
export function drawFaceContours(ctx, faceLandmarks, width, height, apiGlobals = globalThis) {
  if (
    !ctx ||
    !Array.isArray(faceLandmarks) ||
    faceLandmarks.length === 0 ||
    !Number.isFinite(width) ||
    !Number.isFinite(height) ||
    width <= 0 ||
    height <= 0
  ) {
    return { drawn: false, segmentsDrawn: 0, groupsDrawn: [], landmarkCount: 0 };
  }

  const groups = resolveFaceContourGroups(apiGlobals);
  let totalSegments = 0;
  const groupsDrawn = [];

  ctx.save();
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';

  for (let g = 0; g < groups.length; g++) {
    const group = groups[g];
    const edges = group.edges;
    let groupSegments = 0;

    ctx.beginPath();
    ctx.strokeStyle = group.color;
    ctx.lineWidth = group.lineWidth;

    for (let k = 0; k < edges.length; k++) {
      const pair = parseEdgePair(edges[k]);
      if (!pair) continue;
      const p1 = faceLandmarks[pair[0]];
      const p2 = faceLandmarks[pair[1]];
      if (
        !p1 ||
        !p2 ||
        !Number.isFinite(p1.x) ||
        !Number.isFinite(p1.y) ||
        !Number.isFinite(p2.x) ||
        !Number.isFinite(p2.y)
      ) {
        continue;
      }
      ctx.moveTo(p1.x * width, p1.y * height);
      ctx.lineTo(p2.x * width, p2.y * height);
      groupSegments += 1;
    }

    if (groupSegments > 0) {
      ctx.stroke();
      totalSegments += groupSegments;
      groupsDrawn.push(group.name);
    }
  }

  ctx.restore();
  return {
    drawn: totalSegments > 0,
    segmentsDrawn: totalSegments,
    groupsDrawn,
    landmarkCount: faceLandmarks.length,
  };
}
