import React, { useEffect, useRef } from 'react';
import { createPortal } from 'react-dom';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { createRestPose, findMixamoBone, getMixamoBoneName, getMixamoToneSymbol, resolveVietnameseSign } from '../utils/MixamoPoseAdapter';

function normalizeSignToken(value) {
  if (typeof value !== 'string') return '';
  const token = value.trim().normalize('NFC').toUpperCase().replace(/^ALPHA_/, '');
  return getMixamoToneSymbol(token);
}

function interpolateTargets(fromTargets, toTargets, progress) {
  const fromByBone = new Map(fromTargets.map((target) => [target.bone, target]));
  return toTargets.map((to) => {
    const from = fromByBone.get(to.bone) || to;
    const fromEuler = new THREE.Euler(...from.rot, 'XYZ');
    const toEuler = new THREE.Euler(...to.rot, 'XYZ');
    const fromQuat = new THREE.Quaternion().setFromEuler(fromEuler);
    const toQuat = new THREE.Quaternion().setFromEuler(toEuler);
    fromQuat.slerp(toQuat, progress);
    const result = {
      ...to,
      rot: new THREE.Euler().setFromQuaternion(fromQuat, 'XYZ').toArray().slice(0, 3)
    };
    if (from.pos || to.pos) {
      const a = from.pos || [0, 0, 0];
      const b = to.pos || [0, 0, 0];
      result.pos = a.map((value, axis) => THREE.MathUtils.lerp(value, b[axis], progress));
    }
    return result;
  });
}

function sampleKeyframes(keyframes, progress) {
  if (!keyframes?.length) return [];
  if (progress <= keyframes[0].t) return keyframes[0].targets;
  for (let index = 0; index < keyframes.length - 1; index += 1) {
    const from = keyframes[index];
    const to = keyframes[index + 1];
    if (progress > to.t) continue;
    const fraction = (progress - from.t) / Math.max(0.001, to.t - from.t);
    return interpolateTargets(from.targets, to.targets, fraction);
  }
  return keyframes[keyframes.length - 1].targets;
}

function sampleAction(action, progress) {
  if (action.type === 'keyframes') return sampleKeyframes(action.keyframes, progress);
  if (action.type !== 'motion') return action.data || [];
  const base = action.pose || [];
  const frameTargets = sampleKeyframes(action.trajectory, progress);
  const merged = new Map(base.map((target) => [target.bone, target]));
  for (const target of frameTargets) {
    const previous = merged.get(target.bone) || { bone: target.bone, rot: [0, 0, 0] };
    merged.set(target.bone, { ...previous, ...target });
  }
  return [...merged.values()];
}

function sampleActionQueue(queue, elapsedMs) {
  let elapsed = elapsedMs;
  for (const action of queue) {
    const durationMs = Math.max(1, (action.duration || 0.4) * 1000);
    if (elapsed <= durationMs) return sampleAction(action, THREE.MathUtils.clamp(elapsed / durationMs, 0, 1));
    elapsed -= durationMs;
  }
  return queue.length ? sampleAction(queue[queue.length - 1], 1) : createRestPose();
}

export default function MixamoAvatarSigner({ activeLetter, currentStep, currentSign, isPlaying = false, sequence = [], currentIndex = 0, controlsTarget }) {
  const mountRef = useRef(null);
  const [modelLoaded, setModelLoaded] = React.useState(false);
  const [modelError, setModelError] = React.useState(false);
  const [activeView, setActiveView] = React.useState('half');
  const viewPresetsRef = useRef(null);
  const cameraTargetRef = useRef(null);
  const lookTargetRef = useRef(null);
  const viewTransitionRef = useRef(false);
  const poseTargetsRef = useRef(createRestPose());
  const poseAnimationRef = useRef(null);
  const gloss = activeLetter || currentSign || currentStep?.avatar_symbol || currentStep?.gloss || currentStep?.token;
  const signToken = normalizeSignToken(gloss);

  const hasNextStep = currentIndex < sequence.length - 1;


  useEffect(() => {
    const actions = resolveVietnameseSign(signToken, { hasNextStep });

    if (actions.length) {
      poseAnimationRef.current = { actions, startedAt: performance.now(), token: signToken };
      poseTargetsRef.current = sampleActionQueue(actions, 0);
    } else {
      poseAnimationRef.current = null;
      poseTargetsRef.current = createRestPose();
    }
  }, [activeLetter, currentSign, currentStep, currentIndex, hasNextStep, isPlaying, modelLoaded, signToken]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return undefined;
    const bonesMap = new Map();
    const restPositions = new Map();

    const scene = new THREE.Scene();
    scene.background = new THREE.Color('#0b1120');

    const viewPresets = {
      half: {
        position: new THREE.Vector3(0, 1.35, 2.4),
        target: new THREE.Vector3(0, 1.05, 0)
      },
      full: {
        position: new THREE.Vector3(0, 1.3, 4.5),
        target: new THREE.Vector3(0, 0.35, 0)
      },
    };
    viewPresetsRef.current = viewPresets;
    cameraTargetRef.current = viewPresets.half.position.clone();
    lookTargetRef.current = viewPresets.half.target.clone();

    const camera = new THREE.PerspectiveCamera(35, 1, 0.1, 100);
    camera.position.copy(viewPresets.half.position);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    container.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 1.5));
    const keyLight = new THREE.DirectionalLight(0xffffff, 2.2);
    keyLight.position.set(2, 4, 3);
    scene.add(keyLight);
    const fillLight = new THREE.DirectionalLight(0xffffff, 0.8);
    fillLight.position.set(-3, 2, -2);
    scene.add(fillLight);

    const avatarRoot = new THREE.Group();
    avatarRoot.position.set(0, -1, 0);
    scene.add(avatarRoot);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.enableZoom = true;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    controls.mouseButtons.MIDDLE = null;
    controls.target.copy(viewPresets.half.target);
    controls.minDistance = 0.35;
    controls.maxDistance = 8;
    controls.update();

    // Touchpads report two-finger scrolling as wheel input. Intercept scrolling
    // before OrbitControls, while leaving ctrl-wheel pinch zoom to its handler.
    const handleWheelPan = (event) => {
      if (event.ctrlKey || !Number.isFinite(event.deltaY) || event.deltaY === 0) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      viewTransitionRef.current = false;
      const unit = event.deltaMode === 1 ? 16 : event.deltaMode === 2 ? container.clientHeight : 1;
      const nextY = THREE.MathUtils.clamp(
        controls.target.y - event.deltaY * unit * 0.002,
        viewPresets.full.target.y,
        1.7
      );
      const deltaY = nextY - controls.target.y;
      controls.target.y = nextY;
      camera.position.y += deltaY;
      controls.update();
    };
    renderer.domElement.addEventListener('wheel', handleWheelPan, { capture: true, passive: false });

    // Safari also exposes pinch through gesture events with a cumulative scale.
    let gestureScale = 1;
    const handleGestureStart = (event) => {
      event.preventDefault();
      gestureScale = 1;
    };
    const handleGestureChange = (event) => {
      if (!Number.isFinite(event.scale) || event.scale <= 0) return;
      event.preventDefault();
      const scale = event.scale / gestureScale;
      gestureScale = event.scale;
      if (scale === 1) return;

      viewTransitionRef.current = false;
      if (scale > 1) controls.dollyIn(1 / scale);
      else controls.dollyOut(scale);
    };
    renderer.domElement.addEventListener('gesturestart', handleGestureStart, { passive: false });
    renderer.domElement.addEventListener('gesturechange', handleGestureChange, { passive: false });

    let activePanPointerId = null;
    let lastPanY = 0;
    const activeTouchPointers = new Map();
    const handlePointerDown = (event) => {
      if (event.pointerType === 'mouse' && event.button === 2) {
        activePanPointerId = event.pointerId;
        lastPanY = event.clientY;
        viewTransitionRef.current = false;
        // Two-finger touchpad clicks are reported as secondary mouse input.
        // Keep this drag out of OrbitControls so only vertical panning applies.
        event.preventDefault();
        event.stopImmediatePropagation();
        renderer.domElement.setPointerCapture(event.pointerId);
      } else if (event.pointerType === 'touch') {
        activeTouchPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
      }
    };
    const handlePointerMove = (event) => {
      if (event.pointerType === 'touch' && activeTouchPointers.size > 1) {
        activeTouchPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });
        return;
      }
      const isMousePan = event.pointerType === 'mouse'
        && event.pointerId === activePanPointerId
        && (event.buttons & 2);
      const isSingleTouch = event.pointerType === 'touch' && activeTouchPointers.size === 1;
      if (!isMousePan && !isSingleTouch) return;

      const previous = isSingleTouch ? activeTouchPointers.get(event.pointerId) : null;
      if (isSingleTouch && !previous) return;
      event.preventDefault();
      event.stopImmediatePropagation();
      const deltaX = isSingleTouch ? event.clientX - previous.x : 0;
      const deltaY = isSingleTouch ? event.clientY - previous.y : event.clientY - lastPanY;
      if (isMousePan) lastPanY = event.clientY;
      viewTransitionRef.current = false;
      if (isSingleTouch) activeTouchPointers.set(event.pointerId, { x: event.clientX, y: event.clientY });

      if (isSingleTouch && deltaX) {
        const offset = camera.position.clone().sub(controls.target);
        offset.applyAxisAngle(new THREE.Vector3(0, 1, 0), -deltaX / Math.max(1, container.clientHeight) * Math.PI * 2);
        camera.position.copy(controls.target).add(offset);
      }

      const nextTargetY = THREE.MathUtils.clamp(controls.target.y - deltaY * 0.002, 0.6, 1.7);
      const verticalDelta = nextTargetY - controls.target.y;
      controls.target.y = nextTargetY;
      camera.position.y += verticalDelta;
      controls.update();
    };
    const handlePointerUp = (event) => {
      if (event.pointerId === activePanPointerId) {
        activePanPointerId = null;
        if (renderer.domElement.hasPointerCapture(event.pointerId)) {
          renderer.domElement.releasePointerCapture(event.pointerId);
        }
      }
      activeTouchPointers.delete(event.pointerId);
    };
    const preventContextMenu = (event) => event.preventDefault();
    renderer.domElement.addEventListener('pointerdown', handlePointerDown, true);
    renderer.domElement.addEventListener('pointermove', handlePointerMove, { capture: true, passive: false });
    renderer.domElement.addEventListener('pointerup', handlePointerUp, true);
    renderer.domElement.addEventListener('pointercancel', handlePointerUp, true);
    renderer.domElement.addEventListener('lostpointercapture', handlePointerUp, true);
    renderer.domElement.addEventListener('contextmenu', preventContextMenu);

    const resize = () => {
      const width = Math.max(1, container.clientWidth);
      const height = Math.max(1, container.clientHeight);
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    };
    const resizeObserver = new ResizeObserver(resize);
    resizeObserver.observe(container);
    resize();

    let disposed = false;
    const bindShoulders = new Map();
    const loader = new GLTFLoader();
    loader.load('/models/avatar.glb', (gltf) => {
      if (disposed) return;

      const model = gltf.scene;
      const bounds = new THREE.Box3().setFromObject(model);
      const size = bounds.getSize(new THREE.Vector3());
      const center = bounds.getCenter(new THREE.Vector3());
      const targetHeight = 2.7;
      const scale = targetHeight / Math.max(size.y, 0.001);
      model.scale.setScalar(scale);
      model.position.set(-center.x * scale, 0, -center.z * scale);
      avatarRoot.add(model);
      controls.update();

      const skeletonBones = new Set();
      model.traverse((object) => {
        if (object.isSkinnedMesh) object.skeleton.bones.forEach((bone) => skeletonBones.add(bone));
      });
      for (const bone of skeletonBones) {
        const name = getMixamoBoneName(bone.name);
        bonesMap.set(name, bone);
        restPositions.set(name, bone.position.clone());
        if (name === 'RightShoulder' || name === 'LeftShoulder') {
          bindShoulders.set(bone, bone.quaternion.clone());
        }
      }
      setModelLoaded(true);
    }, undefined, (error) => {
      if (!disposed) {
        setModelError(true);
        console.error('Could not load Mixamo avatar:', error);
      }
    });

    let frameId;
    let lastFrameTime = performance.now();
    const render = () => {
      frameId = requestAnimationFrame(render);
      const now = performance.now();
      const deltaSeconds = Math.min(0.05, Math.max(0.001, (now - lastFrameTime) / 1000));
      lastFrameTime = now;
      if (viewTransitionRef.current) {
        camera.position.lerp(cameraTargetRef.current, 0.08);
        controls.target.lerp(lookTargetRef.current, 0.08);
        if (camera.position.distanceTo(cameraTargetRef.current) < 0.01
          && controls.target.distanceTo(lookTargetRef.current) < 0.01) {
          camera.position.copy(cameraTargetRef.current);
          controls.target.copy(lookTargetRef.current);
          viewTransitionRef.current = false;
        }
      }
      // Action queue: pose / tone trajectory steps run in order. Replacing this
      // ref on a new selection cancels the old sequence without leaving timers.
      const poseAnimation = poseAnimationRef.current;
      if (poseAnimation) {
        poseTargetsRef.current = sampleActionQueue(
          poseAnimation.actions,
          now - poseAnimation.startedAt
        );
      }
      const easing = 1 - Math.exp(-deltaSeconds * 16);
      const posedShoulders = new Set();
      for (const target of poseTargetsRef.current) {
        const bone = findMixamoBone(bonesMap, target.bone);
        if (!bone?.rotation) continue;
        if (bindShoulders.has(bone)) posedShoulders.add(bone);
        const targetQuaternion = new THREE.Quaternion().setFromEuler(
          new THREE.Euler(target.rot[0], target.rot[1], target.rot[2], bone.rotation.order)
        );
        bone.quaternion.slerp(targetQuaternion, easing);
        if (target.bone === 'RightHand') {
          const restPosition = restPositions.get('RightHand');
          if (restPosition) {
            const offset = target.pos || [0, 0, 0];
            const targetPosition = restPosition.clone().add(new THREE.Vector3(...offset));
            bone.position.lerp(targetPosition, easing);
          }
        }
      }
      // Legacy poses omit clavicles. Return those joints to their imported bind rotations.
      for (const [bone, rotation] of bindShoulders) {
        if (!posedShoulders.has(bone)) bone.quaternion.slerp(rotation, easing);
      }
      controls.update();
      renderer.render(scene, camera);
    };
    render();

    return () => {
      disposed = true;
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener('wheel', handleWheelPan, true);
      renderer.domElement.removeEventListener('gesturestart', handleGestureStart);
      renderer.domElement.removeEventListener('gesturechange', handleGestureChange);
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown, true);
      renderer.domElement.removeEventListener('pointermove', handlePointerMove, true);
      renderer.domElement.removeEventListener('pointerup', handlePointerUp, true);
      renderer.domElement.removeEventListener('pointercancel', handlePointerUp, true);
      renderer.domElement.removeEventListener('lostpointercapture', handlePointerUp, true);
      renderer.domElement.removeEventListener('contextmenu', preventContextMenu);
      scene.traverse((object) => {
        object.geometry?.dispose();
        if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
        else object.material?.dispose();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [tunerHostRef]);

  const resetToRestPose = () => {
    poseAnimationRef.current = null;
    poseTargetsRef.current = createRestPose();
  };
  const selectView = (viewName) => {
    const preset = viewPresetsRef.current?.[viewName];
    if (!preset || !cameraTargetRef.current || !lookTargetRef.current) return;
    cameraTargetRef.current.copy(preset.position);
    lookTargetRef.current.copy(preset.target);
    viewTransitionRef.current = true;
    setActiveView(viewName);
  };

  const viewButtons = [
    { id: 'half', label: 'Bán thân' },
    { id: 'full', label: 'Toàn thân' },
  ];

  const viewControls = (
    <div className={controlsTarget ? 'control-grid model-view-controls' : undefined}
      style={controlsTarget ? undefined : {
        position: 'absolute', top: 12, right: 12, zIndex: 5, display: 'grid', gap: 4, padding: 5,
        borderRadius: 9, background: 'rgba(8, 15, 28, 0.88)', border: '1px solid var(--border-subtle)'
      }}>
      {viewButtons.map((view) => (
        <button key={view.id} type="button" className={`btn ${activeView === view.id ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => selectView(view.id)} aria-pressed={activeView === view.id}>{view.label}</button>
      ))}
      <button type="button" className="btn btn-secondary model-reset" disabled={!modelLoaded} onClick={resetToRestPose}>
        Về tư thế nghỉ
      </button>
    </div>
  );

  return (
    <>
      <div ref={mountRef} style={{ position: 'absolute', inset: 0, cursor: 'grab' }} title="Chụm, mở hai ngón hoặc cuộn bánh xe để zoom. Kéo chuột trái để xoay, kéo chuột phải để dịch góc nhìn." />
      {!modelLoaded && (
        <p className="avatar-load-status" role="status">
          {modelError ? 'Lỗi tải Avatar. Chọn Robot để tiếp tục.' : 'Đang tải Avatar…'}
        </p>
      )}
      {controlsTarget === undefined ? viewControls : controlsTarget && createPortal(viewControls, controlsTarget)}
    </>
  );
}
