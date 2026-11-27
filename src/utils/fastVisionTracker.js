/**
 * High-Accuracy WebGL GPU Dual-Model Vision Tracker for VSL Connect AI.
 *
 * Replaces single-threaded CPU Holistic's "Pose-wrist-lagged hand crop" bottleneck with:
 *   1. Dedicated `@mediapipe/tasks-vision` GPU `HandLandmarker` (SSD palm detector + 21-joint
 *      hand landmarker running EVERY frame directly on WebGL textures — catches rapid hand swings
 *      anywhere in the frame with zero Pose wrist lag).
 *   2. Lightweight GPU `PoseLandmarker` (`pose_landmarker_lite`, 33 body joints WITHOUT 468-point
 *      FaceMesh overhead), supporting decoupled Eco cadence (1/1, 1/2, 1/3) on low-end machines
 *      with live hand-wrist kinematic pose anchoring on intermediate frames.
 *   3. Automatic fallback to legacy `window.Holistic` (`smoothLandmarks: false` + `window.Hands`
 *      fast-motion palm rescue) if WebGL / Tasks Vision CDN is unavailable.
 */

const MP_TASKS_VER = '0.10.14';
const TASKS_BUNDLE_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_TASKS_VER}/vision_bundle.mjs`;
const TASKS_WASM_URL = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_TASKS_VER}/wasm`;
const HAND_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task';
const POSE_LITE_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task';
const POSE_FULL_MODEL_URL =
  'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task';

let tasksRuntimePromise = null;

function loadTasksRuntime() {
  if (!tasksRuntimePromise) {
    tasksRuntimePromise = (async () => {
      const mod = await import(/* @vite-ignore */ TASKS_BUNDLE_URL);
      const fileset = await mod.FilesetResolver.forVisionTasks(TASKS_WASM_URL);
      return { ...mod, fileset };
    })().catch((err) => {
      tasksRuntimePromise = null;
      throw err;
    });
  }
  return tasksRuntimePromise;
}

function dist2D(a, b) {
  if (!a || !b) return Infinity;
  return Math.hypot((a.x || 0) - (b.x || 0), (a.y || 0) - (b.y || 0));
}

function validJoint(lm, minVis = 0.25) {
  return Boolean(
    lm &&
      Number.isFinite(lm.x) &&
      Number.isFinite(lm.y) &&
      !(lm.x === 0 && lm.y === 0) &&
      (lm.visibility === undefined || lm.visibility >= minVis)
  );
}

/**
 * Assign 1 or 2 independently detected 21-joint hands to anatomical Left (pose[15]) and Right (pose[16]),
 * matching MediaPipe Holistic's `leftHandLandmarks` / `rightHandLandmarks` convention on raw video frames.
 */
export function assignHandsToAnatomicalSides(handLandmarksList, handednessesList, poseLandmarks, prevLeft, prevRight) {
  if (!handLandmarksList || handLandmarksList.length === 0) {
    return { leftHandLandmarks: null, rightHandLandmarks: null };
  }

  const lPose = validJoint(poseLandmarks?.[15], 0.22) ? poseLandmarks[15] : null;
  const rPose = validJoint(poseLandmarks?.[16], 0.22) ? poseLandmarks[16] : null;
  const shoulderSpan = dist2D(poseLandmarks?.[11], poseLandmarks?.[12]);
  const closeWrist = Number.isFinite(shoulderSpan) ? Math.min(0.11, Math.max(0.055, 0.40 * shoulderSpan)) : 0.075;

  const costForSide = (lms, rawCat, side) => {
    const wrist = lms[0];
    const poseTarget = side === 'left' ? lPose : rPose;
    const prevTarget = side === 'left' ? prevLeft?.[0] : prevRight?.[0];
    // A missing observation carries uncertainty, never a perfect distance match.
    let cost = lPose || rPose ? (poseTarget ? dist2D(wrist, poseTarget) : 1.55 * closeWrist) : 0;
    if (prevLeft || prevRight) cost += 0.50 * (prevTarget ? dist2D(wrist, prevTarget) : closeWrist);
    const hasAnchor = Boolean(lPose || rPose || prevLeft || prevRight);
    // MediaPipe Tasks HandLandmarker assumes a mirrored selfie image, so on an unmirrored <video>
    // frame, anatomical Right hand is reported as 'Left' and anatomical Left hand as 'Right'.
    const expectedSelfieCat = side === 'left' ? 'Right' : 'Left';
    if (rawCat) {
      cost += rawCat === expectedSelfieCat ? 0 : (hasAnchor ? 0.035 : 0.35);
    } else if (!hasAnchor) {
      // Fallback spatial prior in unflipped camera frame: Left arm ~ x>0.5, Right arm ~ x<0.5
      cost += side === 'left' ? Math.max(0, 0.5 - wrist.x) : Math.max(0, wrist.x - 0.5);
    }
    return cost;
  };

  const h0 = handLandmarksList[0];
  const h1 = handLandmarksList[1];
  const cat0 = handednessesList?.[0]?.[0]?.categoryName || null;
  const cat1 = handednessesList?.[1]?.[0]?.categoryName || null;

  // Avoid duplicate BlazePalm boxes locked onto the exact same physical hand
  if (!h1 || sameDetectedHand(h0, h1)) {
    const closeLeft = dist2D(h0[0], lPose) <= closeWrist;
    const closeRight = dist2D(h0[0], rPose) <= closeWrist;
    const followsLeft = dist2D(h0[0], prevLeft?.[0]) <= 0.60 * closeWrist &&
      dist2D(h0[0], prevRight?.[0]) >= 1.7 * closeWrist;
    const followsRight = dist2D(h0[0], prevRight?.[0]) <= 0.60 * closeWrist &&
      dist2D(h0[0], prevLeft?.[0]) >= 1.7 * closeWrist;
    const useLeft = closeLeft !== closeRight ? closeLeft : followsLeft || followsRight ? followsLeft
      : costForSide(h0, cat0, 'left') <= costForSide(h0, cat0, 'right');
    return useLeft
      ? { leftHandLandmarks: h0, rightHandLandmarks: null }
      : { leftHandLandmarks: null, rightHandLandmarks: h0 };
  }

  const assign01 = costForSide(h0, cat0, 'left') + costForSide(h1, cat1, 'right');
  const assign10 = costForSide(h1, cat1, 'left') + costForSide(h0, cat0, 'right');
  return assign01 <= assign10
    ? { leftHandLandmarks: h0, rightHandLandmarks: h1 }
    : { leftHandLandmarks: h1, rightHandLandmarks: h0 };
}

function sameDetectedHand(left, right) {
  if (!left || !right || dist2D(left[0], right[0]) >= 0.040) return false;
  // Overlapping detector boxes can disagree on the tips of curled fingers.
  // Use the four palm knuckles to distinguish a duplicate from two hands in contact.
  const palmGap = [5, 9, 13, 17].reduce((sum, index) => sum + dist2D(left[index], right[index]), 0) / 4;
  return palmGap < 0.025;
}

/**
 * Update cached Pose wrist/palm/elbow joints from live high-FPS hand landmarks on Eco pose-skip frames.
 */
function syncPoseArmsWithLiveHands(cachedPose, leftHand, rightHand) {
  if (!cachedPose || cachedPose.length < 33) return cachedPose;
  const next = cachedPose.slice();
  const updateArm = (hand, shIdx, elIdx, wrIdx, pkIdx, ixIdx, thIdx) => {
    if (!hand || hand.length < 21) return;
    const w = hand[0];
    const prevW = cachedPose[wrIdx];
    next[wrIdx] = { x: w.x, y: w.y, z: w.z || 0, visibility: 0.95 };
    next[pkIdx] = { x: hand[17].x, y: hand[17].y, z: hand[17].z || 0, visibility: 0.9 };
    next[ixIdx] = { x: hand[5].x, y: hand[5].y, z: hand[5].z || 0, visibility: 0.9 };
    next[thIdx] = { x: hand[2].x, y: hand[2].y, z: hand[2].z || 0, visibility: 0.9 };
    if (validJoint(prevW, 0.2) && validJoint(cachedPose[elIdx], 0.2) && validJoint(cachedPose[shIdx], 0.2)) {
      const dx = w.x - prevW.x;
      const dy = w.y - prevW.y;
      next[elIdx] = {
        ...cachedPose[elIdx],
        x: cachedPose[elIdx].x + 0.52 * dx,
        y: cachedPose[elIdx].y + 0.52 * dy,
      };
    }
  };
  updateArm(leftHand, 11, 13, 15, 17, 19, 21);
  updateArm(rightHand, 12, 14, 16, 18, 20, 22);
  return next;
}

async function createWithDelegateFallback(factory) {
  try {
    return { instance: await factory('GPU'), delegate: 'GPU' };
  } catch {
    return { instance: await factory('CPU'), delegate: 'CPU' };
  }
}

/**
 * Create an accelerated tracker with the same `.send({ image, runPose })` and `.close()` contract.
 */
export async function createAcceleratedTracker({
  engine,
  detailed = false,
  onHandsResults,
  onHolisticResults,
}) {
  // 1. Try WebGL GPU Tasks Vision (`HandLandmarker` + `PoseLandmarker` without 468-pt FaceMesh)
  try {
    const { HandLandmarker, PoseLandmarker, fileset } = await loadTasksRuntime();
    let lastTsMs = 0;
    const nextMonotonicTs = () => {
      const now = Math.round(performance.now());
      lastTsMs = now > lastTsMs ? now : lastTsMs + 1;
      return lastTsMs;
    };

    if (engine === 'hands') {
      const { instance: handLandmarker, delegate } = await createWithDelegateFallback((del) =>
        HandLandmarker.createFromOptions(fileset, {
          baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: del },
          runningMode: 'VIDEO',
          numHands: 2,
          minHandDetectionConfidence: 0.32,
          minHandPresenceConfidence: 0.28,
          minTrackingConfidence: 0.24,
        })
      );
      return {
        _engineType: 'hands',
        _backend: `tasks_${delegate.toLowerCase()}`,
        _delegate: delegate,
        send: async ({ image }) => {
          const out = handLandmarker.detectForVideo(image, nextMonotonicTs());
          const multiHandLandmarks = out.landmarks || [];
          const multiHandedness = (out.handednesses || []).map((cats) => ({
            label: cats?.[0]?.categoryName || 'Right',
            score: cats?.[0]?.score ?? 0.9,
          }));
          onHandsResults({ multiHandLandmarks, multiHandedness });
        },
        close: () => {
          try {
            handLandmarker.close();
          } catch {
            /* ignore */
          }
        },
      };
    }

    if (engine === 'holistic') {
      const poseUrl = detailed ? POSE_FULL_MODEL_URL : POSE_LITE_MODEL_URL;
      const [{ instance: handLandmarker, delegate: hDel }, { instance: poseLandmarker, delegate: pDel }] =
        await Promise.all([
          createWithDelegateFallback((del) =>
            HandLandmarker.createFromOptions(fileset, {
              baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: del },
              runningMode: 'VIDEO',
              numHands: 2,
              minHandDetectionConfidence: 0.30,
              minHandPresenceConfidence: 0.26,
              minTrackingConfidence: 0.22,
            })
          ),
          createWithDelegateFallback((del) =>
            PoseLandmarker.createFromOptions(fileset, {
              baseOptions: { modelAssetPath: poseUrl, delegate: del },
              runningMode: 'VIDEO',
              numPoses: 1,
              minPoseDetectionConfidence: 0.35,
              minPosePresenceConfidence: 0.30,
              minTrackingConfidence: 0.28,
            })
          ),
        ]);

      let cachedPose = null;
      let prevLeft = null;
      let prevRight = null;
      let lastLeftSeen = -Infinity;
      let lastRightSeen = -Infinity;

      return {
        _engineType: 'holistic',
        _backend: `tasks_dual_${hDel.toLowerCase()}`,
        _delegate: hDel === 'GPU' && pDel === 'GPU' ? 'GPU' : 'CPU',
        send: async ({ image, runPose = true }) => {
          const ts1 = nextMonotonicTs();
          // Always run dedicated palm+finger detector every frame so rapid swings never drop!
          const handOut = handLandmarker.detectForVideo(image, ts1);

          let poseLandmarks = cachedPose;
          if (runPose || !cachedPose) {
            const ts2 = nextMonotonicTs();
            const poseOut = poseLandmarker.detectForVideo(image, ts2);
            poseLandmarks = poseOut.landmarks?.[0] || null;
            cachedPose = poseLandmarks;
          }

          const assigned = assignHandsToAnatomicalSides(
            handOut.landmarks || [],
            handOut.handednesses || [],
            poseLandmarks,
            ts1 - lastLeftSeen <= 180 ? prevLeft : null,
            ts1 - lastRightSeen <= 180 ? prevRight : null
          );
          if (assigned.leftHandLandmarks) {
            prevLeft = assigned.leftHandLandmarks;
            lastLeftSeen = ts1;
          }
          if (assigned.rightHandLandmarks) {
            prevRight = assigned.rightHandLandmarks;
            lastRightSeen = ts1;
          }

          if (!runPose && poseLandmarks) {
            poseLandmarks = syncPoseArmsWithLiveHands(
              poseLandmarks,
              assigned.leftHandLandmarks,
              assigned.rightHandLandmarks
            );
          }

          onHolisticResults({
            poseLandmarks,
            leftHandLandmarks: assigned.leftHandLandmarks,
            rightHandLandmarks: assigned.rightHandLandmarks,
          });
        },
        close: () => {
          try {
            handLandmarker.close();
          } catch {
            /* ignore */
          }
          try {
            poseLandmarker.close();
          } catch {
            /* ignore */
          }
        },
      };
    }
  } catch (tasksErr) {
    console.warn('[Tracker] Tasks Vision GPU unavailable, falling back to WASM engine:', tasksErr?.message || tasksErr);
  }

  // 2. Fallback to legacy CDN MediaPipe with zero-lag Pose + dedicated Hands fast-swing rescue
  if (engine === 'hands') {
    if (!window.Hands) {
      throw new Error('Thư viện MediaPipe Hands chưa sẵn sàng. Vui lòng kiểm tra kết nối mạng CDN.');
    }
    const tracker = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
    });
    tracker.setOptions({
      maxNumHands: 2,
      modelComplexity: 0,
      minDetectionConfidence: 0.35,
      minTrackingConfidence: 0.25,
    });
    tracker.onResults(onHandsResults);
    tracker._engineType = 'hands';
    tracker._backend = 'legacy_wasm';
    tracker._delegate = 'CPU';
    return tracker;
  }

  if (engine === 'holistic') {
    if (!window.Holistic) {
      throw new Error('Thư viện MediaPipe Holistic chưa sẵn sàng. Chế độ Từ vựng yêu cầu Holistic 75 khớp thật.');
    }
    const holistic = new window.Holistic({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`,
    });
    holistic.setOptions({
      modelComplexity: detailed ? 1 : 0,
      // Disable internal Pose smoothing so Pose wrists do not lag behind fast swings!
      smoothLandmarks: false,
      minDetectionConfidence: 0.35,
      minTrackingConfidence: 0.22,
    });

    // Optional standalone Hands palm detector rescue when Holistic loses a fast-swinging hand
    let rescueHands = null;
    let lastRescueOut = null;
    if (window.Hands) {
      try {
        rescueHands = new window.Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`,
        });
        rescueHands.setOptions({
          maxNumHands: 2,
          modelComplexity: 0,
          minDetectionConfidence: 0.32,
          minTrackingConfidence: 0.22,
        });
        rescueHands.onResults((res) => {
          lastRescueOut = res;
        });
      } catch {
        rescueHands = null;
      }
    }

    let lastImage = null;
    let prevLeft = null;
    let prevRight = null;
    let holisticOut = null;
    holistic.onResults((res) => {
      holisticOut = res;
    });

    return {
      _engineType: 'holistic',
      _backend: rescueHands ? 'legacy_holistic_plus_hands' : 'legacy_holistic',
      _delegate: 'CPU',
      send: async ({ image }) => {
        lastImage = image;
        holisticOut = null;
        await holistic.send({ image });
        if (!holisticOut) return;
        let { poseLandmarks, leftHandLandmarks, rightHandLandmarks } = holisticOut;

        if (rescueHands && lastImage && (!leftHandLandmarks || !rightHandLandmarks)) {
          lastRescueOut = null;
          try {
            await rescueHands.send({ image: lastImage });
            if (lastRescueOut?.multiHandLandmarks?.length) {
              const fakeCats = (lastRescueOut.multiHandedness || []).map((h) => [{ categoryName: h.label }]);
              const assigned = assignHandsToAnatomicalSides(
                lastRescueOut.multiHandLandmarks,
                fakeCats,
                poseLandmarks,
                prevLeft,
                prevRight
              );
              if (!leftHandLandmarks && !sameDetectedHand(assigned.leftHandLandmarks, rightHandLandmarks)) {
                leftHandLandmarks = assigned.leftHandLandmarks;
              }
              if (!rightHandLandmarks && !sameDetectedHand(assigned.rightHandLandmarks, leftHandLandmarks)) {
                rightHandLandmarks = assigned.rightHandLandmarks;
              }
            }
          } catch {
            /* ignore rescue error */
          }
        }
        prevLeft = leftHandLandmarks;
        prevRight = rightHandLandmarks;
        onHolisticResults({ poseLandmarks, leftHandLandmarks, rightHandLandmarks });
      },
      close: () => {
        try {
          holistic.close?.();
        } catch {
          /* ignore */
        }
        try {
          rescueHands?.close?.();
        } catch {
          /* ignore */
        }
      },
    };
  }

  throw new Error(`Engine MediaPipe không hợp lệ: ${engine}`);
}
