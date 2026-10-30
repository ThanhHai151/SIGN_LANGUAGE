import React, { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { createRestPose, getPoseForLetter, MIXAMO_LETTER_ANIMATIONS } from '../utils/MixamoPoseAdapter';
import { setupHandGUI } from '../utils/handPoseTuner';

function interpolatePoseTargets(keyframes, progress) {
  if (progress <= keyframes[0].t) return keyframes[0].targets;
  for (let index = 0; index < keyframes.length - 1; index += 1) {
    const from = keyframes[index];
    const to = keyframes[index + 1];
    if (progress > to.t) continue;
    const fraction = (progress - from.t) / Math.max(0.001, to.t - from.t);
    const toByBone = new Map(to.targets.map((target) => [target.bone, target]));
    return from.targets.map((target) => {
      const next = toByBone.get(target.bone) || target;
      return {
        bone: target.bone,
        rot: target.rot.map((angle, axis) => THREE.MathUtils.lerp(angle, next.rot[axis], fraction))
      };
    });
  }
  return keyframes[keyframes.length - 1].targets;
}

export default function MixamoAvatarSigner({ activeLetter, currentStep, currentSign, isPlaying = false }) {
  const mountRef = useRef(null);
  const bonesMapRef = useRef(new Map());
  const handGUIRef = useRef(null);
  const [modelLoaded, setModelLoaded] = React.useState(false);
  const [activeView, setActiveView] = React.useState('half');
  const viewPresetsRef = useRef(null);
  const cameraTargetRef = useRef(null);
  const lookTargetRef = useRef(null);
  const viewTransitionRef = useRef(false);
  const poseTargetsRef = useRef(createRestPose());
  const poseAnimationRef = useRef(null);
  const gloss = activeLetter || currentSign || currentStep?.gloss || currentStep?.token;
  const letter = typeof gloss === 'string' && /^[a-z]$/i.test(gloss.trim()) ? gloss.trim() : '';

  useEffect(() => {
    const hasActiveLetter = letter && (activeLetter || !currentStep || currentStep.type === 'fingerspelling' || isPlaying);
    poseTargetsRef.current = hasActiveLetter ? getPoseForLetter(letter) : createRestPose();
    const animation = hasActiveLetter ? MIXAMO_LETTER_ANIMATIONS[letter.toUpperCase()] : null;
    poseAnimationRef.current = animation
      ? { ...animation, startedAt: performance.now() }
      : null;
    if (modelLoaded && hasActiveLetter) {
      handGUIRef.current?.setLetter(letter);
    } else if (modelLoaded) {
      for (const target of poseTargetsRef.current) {
        if (!/^RightHand(?:Thumb|Index|Middle|Ring|Pinky)\d+$/.test(target.bone)) continue;
        const bone = bonesMapRef.current.get(target.bone)
          || bonesMapRef.current.get(`mixamorig${target.bone}`);
        bone?.rotation?.set(target.rot[0], target.rot[1], target.rot[2]);
      }
    }
  }, [activeLetter, currentSign, currentStep, isPlaying, modelLoaded]);

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return undefined;

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
      rHand: {
        position: new THREE.Vector3(-0.18, 1.35, 1.5),
        target: new THREE.Vector3(-0.16, 1.30, 0.0)
      },
      lHand: {
        position: new THREE.Vector3(0.18, 1.35, 1.5),
        target: new THREE.Vector3(0.16, 1.30, 0.0)
      }
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
    controls.enableZoom = false;
    controls.enablePan = true;
    controls.screenSpacePanning = true;
    controls.mouseButtons.RIGHT = THREE.MOUSE.PAN;
    controls.target.copy(viewPresets.half.target);
    controls.minDistance = 0.35;
    controls.maxDistance = 8;
    controls.update();

    const handleWheel = (event) => {
      event.preventDefault();
      const nextTargetY = THREE.MathUtils.clamp(
        controls.target.y - event.deltaY * 0.002,
        0.6,
        1.7
      );
      const verticalDelta = nextTargetY - controls.target.y;
      controls.target.y = nextTargetY;
      camera.position.y += verticalDelta;
      controls.update();
    };
    renderer.domElement.addEventListener('wheel', handleWheel, { passive: false });

    let activePanPointerId = null;
    const activeTouchPointers = new Map();
    const handlePointerDown = (event) => {
      if (event.pointerType === 'mouse' && event.button === 2) {
        activePanPointerId = event.pointerId;
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
      const deltaY = isSingleTouch ? event.clientY - previous.y : event.movementY;
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
      if (event.pointerId === activePanPointerId) activePanPointerId = null;
      activeTouchPointers.delete(event.pointerId);
    };
    const preventContextMenu = (event) => event.preventDefault();
    renderer.domElement.addEventListener('pointerdown', handlePointerDown, true);
    renderer.domElement.addEventListener('pointermove', handlePointerMove, { capture: true, passive: false });
    renderer.domElement.addEventListener('pointerup', handlePointerUp, true);
    renderer.domElement.addEventListener('pointercancel', handlePointerUp, true);
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

      const boneNames = [];
      model.traverse((object) => {
        if (object.isBone) {
          bonesMapRef.current.set(object.name, object);
          if (/arm|forearm|hand|shoulder|thumb|index|middle|ring|pinky/i.test(object.name)) {
            boneNames.push(object.name);
          }
        }
      });
      handGUIRef.current?.destroy();
      handGUIRef.current = setupHandGUI(bonesMapRef.current, {
        letter: /^[A-Z]$/i.test(letter) ? letter : 'A',
        onPoseChange: (pose) => {
          poseTargetsRef.current = pose;
        }
      });
      poseTargetsRef.current = createRestPose();
      setModelLoaded(true);
      console.info('Mixamo arm and finger bones:', boneNames);
    }, undefined, (error) => {
      if (!disposed) console.error('Could not load Mixamo avatar:', error);
    });

    let frameId;
    const render = () => {
      frameId = requestAnimationFrame(render);
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
      // Ease each tracked joint toward the latest UI-selected pose.
      const poseAnimation = poseAnimationRef.current;
      if (poseAnimation) {
        const progress = THREE.MathUtils.clamp(
          (performance.now() - poseAnimation.startedAt) / poseAnimation.durationMs,
          0,
          1
        );
        poseTargetsRef.current = interpolatePoseTargets(poseAnimation.keyframes, progress);
      }
      for (const target of poseTargetsRef.current) {
        if (/^RightHand(?:Thumb|Index|Middle|Ring|Pinky)\d+$/.test(target.bone)) continue;
        const bone = bonesMapRef.current.get(target.bone) || bonesMapRef.current.get(`mixamorig${target.bone}`);
        if (!bone?.rotation) continue;
        bone.rotation.x = THREE.MathUtils.lerp(bone.rotation.x, target.rot[0], 0.12);
        bone.rotation.y = THREE.MathUtils.lerp(bone.rotation.y, target.rot[1], 0.12);
        bone.rotation.z = THREE.MathUtils.lerp(bone.rotation.z, target.rot[2], 0.12);
      }
      controls.update();
      renderer.render(scene, camera);
    };
    render();

    return () => {
      disposed = true;
      handGUIRef.current?.destroy();
      handGUIRef.current = null;
      cancelAnimationFrame(frameId);
      resizeObserver.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener('wheel', handleWheel);
      renderer.domElement.removeEventListener('pointerdown', handlePointerDown, true);
      renderer.domElement.removeEventListener('pointermove', handlePointerMove, true);
      renderer.domElement.removeEventListener('pointerup', handlePointerUp, true);
      renderer.domElement.removeEventListener('pointercancel', handlePointerUp, true);
      renderer.domElement.removeEventListener('contextmenu', preventContextMenu);
      scene.traverse((object) => {
        object.geometry?.dispose();
        if (Array.isArray(object.material)) object.material.forEach((material) => material.dispose());
        else object.material?.dispose();
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, []);

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
    { id: 'half', label: '👤 Bán thân' },
    { id: 'full', label: '🧍 Toàn thân' },
    { id: 'rHand', label: '✋ Cận tay phải' },
    { id: 'lHand', label: '🤚 Cận tay trái' }
  ];

  return (
    <>
      <div ref={mountRef} style={{ position: 'absolute', inset: 0, cursor: 'grab' }} title="Chuột phải hoặc vuốt một ngón dọc để dịch góc nhìn; vuốt ngang để xoay" />
      <div style={{ position: 'absolute', top: '12px', right: '12px', zIndex: 5, display: 'flex', flexDirection: 'column', gap: '4px', padding: '5px', borderRadius: '9px', background: 'rgba(8, 15, 28, 0.88)', border: '1px solid rgba(148, 163, 184, 0.25)' }}>
        {viewButtons.map((view) => (
          <button key={view.id} type="button" className={`btn ${activeView === view.id ? 'btn-primary' : 'btn-secondary'}`} onClick={() => selectView(view.id)} style={{ justifyContent: 'flex-start', padding: '5px 8px', fontSize: '11px' }} aria-pressed={activeView === view.id}>
            {view.label}
          </button>
        ))}
      </div>
      <div style={{ position: 'absolute', right: '12px', bottom: '12px', zIndex: 5, display: 'flex', gap: '5px', padding: '5px', borderRadius: '9px', background: 'rgba(8, 15, 28, 0.88)', border: '1px solid rgba(148, 163, 184, 0.25)' }}>
        <button type="button" className="btn btn-secondary" disabled={!modelLoaded} onClick={resetToRestPose} style={{ padding: '5px 8px', fontSize: '11px' }}>
          Về Tư Thế Nghỉ
        </button>
      </div>
    </>
  );
}
