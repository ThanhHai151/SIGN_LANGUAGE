import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { RoundedBoxGeometry } from 'three/examples/jsm/geometries/RoundedBoxGeometry.js';
import { getHandPoseByGloss, NATURAL_FINGERS_REST } from '../utils/handPoses';
import { fetchMotion } from '../services/api';
import { Eye, ZoomIn, RotateCcw, Crosshair } from 'lucide-react';

/**
 * RobotSigner3D — Robot diễn hoạt Ngôn ngữ Ký hiệu VSL
 *
 * Thay thế Avatar3D (nhân vật da người) bằng một robot dễ quan sát hơn:
 * - Mỗi ngón tay một màu riêng (Cái / Trỏ / Giữa / Áp út / Út) → nhìn rõ ngón nào duỗi, ngón nào gập.
 * - Mặt lòng bàn tay màu xanh phát sáng, mu bàn tay màu trắng → biết ngay bàn tay đang hướng về đâu.
 * - Khớp ngón màu tối tương phản, bàn tay phóng to hơn.
 * - Màn hình LED trên ngực hiển thị từ đang diễn, mắt LED đổi màu khi đang diễn hoạt.
 * - Lớp phụ đề: câu đầy đủ với từ hiện tại được tô sáng, kèm chú thích màu ngón tay.
 *
 * Hệ khung xương (vị trí pivot + trục xoay khớp) giữ NGUYÊN như Avatar3D cũ để
 * dùng lại toàn bộ dữ liệu tư thế trong utils/handPoses.js.
 */

export const FINGER_COLORS = {
  thumb: { hex: 0xf97316, css: '#f97316', name: 'Cái' },
  index: { hex: 0x3b82f6, css: '#3b82f6', name: 'Trỏ' },
  middle: { hex: 0x22c55e, css: '#22c55e', name: 'Giữa' },
  ring: { hex: 0xeab308, css: '#eab308', name: 'Áp út' },
  pinky: { hex: 0xec4899, css: '#ec4899', name: 'Út' }
};

// Tỉ lệ cơ thể lấy theo người thật trong video QIPEDC (vai 0.40 : cánh tay trên 0.30 : cẳng tay 0.29)
// để chuyển động học từ video đặt tay đúng chỗ (chạm ngực, chạm trán, hai tay chạm nhau)
const SHOULDER_HALF = 0.20;
const UPPER_LEN = 0.30;
const FORE_LEN = 0.29;
const HAND_SCALE = 1.3;

// Vẽ nội dung màn hình LED trên ngực robot
function drawChestScreen(ctx, canvas, text, sub, alpha = 1, offsetY = 0) {
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = '#04121c';
  ctx.fillRect(0, 0, w, h);

  // Đường quét LED
  ctx.fillStyle = 'rgba(34, 211, 238, 0.06)';
  for (let y = 0; y < h; y += 6) ctx.fillRect(0, y, w, 2);

  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.globalAlpha = alpha;

  ctx.fillStyle = '#67e8f9';
  ctx.font = '600 34px "Plus Jakarta Sans", "Segoe UI", Arial, sans-serif';
  ctx.fillText(sub || 'VSL ROBOT', w / 2, 44);

  let size = 112;
  const label = text || '•••';
  ctx.font = `800 ${size}px "Outfit", "Segoe UI", Arial, sans-serif`;
  while (ctx.measureText(label).width > w - 50 && size > 36) {
    size -= 6;
    ctx.font = `800 ${size}px "Outfit", "Segoe UI", Arial, sans-serif`;
  }
  ctx.shadowColor = '#22d3ee';
  ctx.shadowBlur = 24;
  ctx.fillStyle = '#ecfeff';
  ctx.fillText(label, w / 2, h / 2 + 22 + offsetY);
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;
}

// ---------- Hoạt ảnh chụp chuyển động (mocap) từ video QIPEDC ----------
// Nội suy tuyến tính đệ quy giữa 2 keyframe (số, mảng, object lồng nhau)
function lerpDeep(a, b, u) {
  if (typeof a === 'number' && typeof b === 'number') return a + (b - a) * u;
  if (Array.isArray(a) && Array.isArray(b)) return a.map((v, i) => lerpDeep(v, b[i], u));
  if (a && b && typeof a === 'object') {
    const o = {};
    for (const k of Object.keys(a)) o[k] = k in b ? lerpDeep(a[k], b[k], u) : a[k];
    return o;
  }
  return u < 0.5 ? a : b;
}

// Lấy mẫu hoạt ảnh tại thời điểm tSec (giây)
export const SPELL_LETTER_MS = 550;

/** Ký hiệu chưa có chuyển động học từ video, chưa có tư thế vẽ tay → robot đánh vần tạm. */
export function isTempSpell(step) {
  return !!(step && step.type !== 'fingerspelling' && !step.motion_id && !step.pose_key && step.spell?.length);
}

export function sampleMotion(motion, tSec) {
  const kfs = motion.keyframes;
  if (!kfs || !kfs.length) return null;
  if (tSec <= kfs[0].t) return kfs[0];
  for (let i = 0; i < kfs.length - 1; i++) {
    const a = kfs[i];
    const b = kfs[i + 1];
    if (tSec <= b.t) return lerpDeep(a, b, (tSec - a.t) / Math.max(1e-3, b.t - a.t));
  }
  return kfs[kfs.length - 1];
}

function applyKeyframe(cur, kf) {
  if (!kf) return;
  cur.body = kf.body || cur.body;
  ['right', 'left'].forEach((s) => {
    if (kf[s + 'Arm']) cur[s + 'Arm'] = kf[s + 'Arm'];
    if (kf[s + 'Forearm']) cur[s + 'Forearm'] = kf[s + 'Forearm'];
    if (kf[s + 'Wrist']) cur[s + 'Wrist'] = kf[s + 'Wrist'];
    if (kf[s + 'Fingers']) cur[s + 'Fingers'] = kf[s + 'Fingers'];
    cur[s + 'Twist'] = typeof kf[s + 'Twist'] === 'number' ? kf[s + 'Twist'] : undefined;
  });
}

export default function RobotSigner3D({ currentStep, isPlaying, speed = 1.0, sequence = [], currentIndex = 0, freezeAt, videoElRef }) {
  const mountRef = useRef(null);
  const rigRef = useRef({});
  const animFrameRef = useRef(null);
  const isPlayingRef = useRef(isPlaying);
  const speedRef = useRef(speed);
  const screenRef = useRef(null);
  const captionRef = useRef(null);

  const [cameraMode, setCameraMode] = useState('full');
  const cameraModeRef = useRef('full');
  const targetCamPosRef = useRef(new THREE.Vector3(0, 1.12, 2.15));
  const targetCamLookRef = useRef(new THREE.Vector3(0, 0.98, 0));

  useEffect(() => { isPlayingRef.current = isPlaying; }, [isPlaying]);
  useEffect(() => { speedRef.current = speed; }, [speed]);
  const freezeRef = useRef(freezeAt);
  useEffect(() => { freezeRef.current = freezeAt; }, [freezeAt]);

  // ---------- Bàn tay robot (giữ nguyên toạ độ khớp của Avatar3D) ----------
  const createRobotHand = (mats, isLeft = false) => {
    const handGroup = new THREE.Group();
    const flip = isLeft ? -1 : 1;

    // Mu bàn tay (mặt -Z): giáp trắng
    const backGeo = new RoundedBoxGeometry(0.062, 0.072, 0.014, 2, 0.006);
    backGeo.translate(0, -0.035, -0.003);
    handGroup.add(new THREE.Mesh(backGeo, mats.armor));

    // Lòng bàn tay (mặt +Z): tấm xanh phát sáng để nhận biết hướng lòng bàn tay
    const palmGeo = new RoundedBoxGeometry(0.056, 0.064, 0.006, 2, 0.0025);
    palmGeo.translate(0, -0.035, 0.0065);
    handGroup.add(new THREE.Mesh(palmGeo, mats.palm));

    // Thanh khớp đốt ngón tay (knuckle bar)
    const knuckleGeo = new THREE.CapsuleGeometry(0.006, 0.05, 4, 10);
    knuckleGeo.rotateZ(Math.PI / 2);
    const knuckle = new THREE.Mesh(knuckleGeo, mats.joint);
    knuckle.position.set(0, -0.068, -0.001);
    handGroup.add(knuckle);

    // Một đốt ngón: viên nang màu của ngón
    const segment = (radius, length, mat) => {
      const geo = new THREE.CapsuleGeometry(radius, Math.max(0.001, length - radius * 2), 4, 12);
      geo.translate(0, -length / 2, 0);
      return new THREE.Mesh(geo, mat);
    };
    const jointBall = (r) => new THREE.Mesh(new THREE.SphereGeometry(r, 14, 14), mats.joint);

    const createFinger = (key, lengthScale, isThumb = false) => {
      const fingerRoot = new THREE.Group();
      const segMat = mats.fingers[key];
      const tipMat = mats.tips[key];

      if (isThumb) {
        const cmcJoint = new THREE.Group();
        cmcJoint.add(jointBall(0.0118));
        cmcJoint.add(segment(0.0112, 0.024 * lengthScale, segMat));

        const mcpJoint = new THREE.Group();
        mcpJoint.position.set(0, -0.024 * lengthScale, 0);
        mcpJoint.add(jointBall(0.0108));
        mcpJoint.add(segment(0.0098, 0.022 * lengthScale, segMat));

        const ipJoint = new THREE.Group();
        ipJoint.position.set(0, -0.022 * lengthScale, 0);
        ipJoint.add(jointBall(0.0096));
        ipJoint.add(segment(0.0088, 0.019 * lengthScale, tipMat));

        mcpJoint.add(ipJoint);
        cmcJoint.add(mcpJoint);
        fingerRoot.add(cmcJoint);
        return { root: fingerRoot, cmc: cmcJoint, mcp: mcpJoint, ip: ipJoint, isThumb: true };
      }

      const mcpJoint = new THREE.Group();
      mcpJoint.add(jointBall(0.0102));
      mcpJoint.add(segment(0.0090, 0.038 * lengthScale, segMat));

      const pipJoint = new THREE.Group();
      pipJoint.position.set(0, -0.038 * lengthScale, 0);
      pipJoint.add(jointBall(0.0092));
      pipJoint.add(segment(0.0082, 0.026 * lengthScale, segMat));

      const dipJoint = new THREE.Group();
      dipJoint.position.set(0, -0.026 * lengthScale, 0);
      dipJoint.add(jointBall(0.0084));
      dipJoint.add(segment(0.0076, 0.021 * lengthScale, tipMat));

      pipJoint.add(dipJoint);
      mcpJoint.add(pipJoint);
      fingerRoot.add(mcpJoint);
      return { root: fingerRoot, mcp: mcpJoint, pip: pipJoint, dip: dipJoint, isThumb: false };
    };

    const indexRig = createFinger('index', 1.0);
    indexRig.root.position.set(-0.020 * flip, -0.070, 0.0);
    handGroup.add(indexRig.root);

    const middleRig = createFinger('middle', 1.08);
    middleRig.root.position.set(-0.006 * flip, -0.072, 0.0);
    handGroup.add(middleRig.root);

    const ringRig = createFinger('ring', 0.98);
    ringRig.root.position.set(0.007 * flip, -0.070, 0.0);
    handGroup.add(ringRig.root);

    const pinkyRig = createFinger('pinky', 0.82);
    pinkyRig.root.position.set(0.020 * flip, -0.067, 0.0);
    handGroup.add(pinkyRig.root);

    const thumbRig = createFinger('thumb', 1.0, true);
    thumbRig.root.position.set(-0.031 * flip, -0.026, 0.006);
    thumbRig.root.rotation.set(0.25, -0.35 * flip, -0.40 * flip);
    handGroup.add(thumbRig.root);

    return {
      group: handGroup,
      fingers: { thumb: thumbRig, index: indexRig, middle: middleRig, ring: ringRig, pinky: pinkyRig }
    };
  };

  // ---------- Dựng cảnh ----------
  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 500;
    const height = container.clientHeight || 480;

    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.05, 100);
    camera.position.copy(targetCamPosRef.current);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.05;
    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // Ánh sáng trung tính, sáng rõ để màu ngón tay không bị ám
    scene.add(new THREE.HemisphereLight(0xffffff, 0x1e293b, 1.6));
    const key = new THREE.DirectionalLight(0xffffff, 2.4);
    key.position.set(1.5, 3, 3);
    scene.add(key);
    const fill = new THREE.DirectionalLight(0xdbeafe, 1.0);
    fill.position.set(-2.5, 1.5, 2);
    scene.add(fill);
    const rim = new THREE.DirectionalLight(0x22d3ee, 1.6);
    rim.position.set(0, 2.5, -3);
    scene.add(rim);

    // Vật liệu
    const mats = {
      armor: new THREE.MeshStandardMaterial({ color: 0xeef2f7, roughness: 0.32, metalness: 0.25 }),
      dark: new THREE.MeshStandardMaterial({ color: 0x334155, roughness: 0.45, metalness: 0.6 }),
      joint: new THREE.MeshStandardMaterial({ color: 0x1f2937, roughness: 0.35, metalness: 0.7 }),
      palm: new THREE.MeshStandardMaterial({ color: 0x0891b2, emissive: 0x06b6d4, emissiveIntensity: 0.45, roughness: 0.4 }),
      glow: new THREE.MeshStandardMaterial({ color: 0x22d3ee, emissive: 0x22d3ee, emissiveIntensity: 1.4 }),
      visor: new THREE.MeshStandardMaterial({ color: 0x050b14, roughness: 0.12, metalness: 0.8 }),
      eye: new THREE.MeshBasicMaterial({ color: 0x67e8f9 }),
      fingers: {},
      tips: {}
    };
    Object.entries(FINGER_COLORS).forEach(([k, c]) => {
      mats.fingers[k] = new THREE.MeshStandardMaterial({ color: c.hex, roughness: 0.38, metalness: 0.15 });
      mats.tips[k] = new THREE.MeshStandardMaterial({ color: c.hex, emissive: c.hex, emissiveIntensity: 0.55, roughness: 0.3 });
    });

    // Bệ đỡ phát sáng
    const avatarGroup = new THREE.Group();
    avatarGroup.position.set(0, -0.2, 0);

    // Vòng hào quang "lơ lửng" bên dưới robot
    const baseRing = new THREE.Mesh(new THREE.TorusGeometry(0.2, 0.007, 8, 64), mats.glow);
    baseRing.rotation.x = Math.PI / 2;
    baseRing.position.set(0, 0.46, 0);
    avatarGroup.add(baseRing);
    const baseRing2 = new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.005, 8, 48), mats.glow);
    baseRing2.rotation.x = Math.PI / 2;
    baseRing2.position.set(0, 0.42, 0);
    avatarGroup.add(baseRing2);

    // Thân robot
    const torsoGroup = new THREE.Group();
    torsoGroup.position.set(0, 0.7, 0);

    // Hông thuôn nhọn + động cơ đẩy phát sáng (robot bay lơ lửng)
    const waist = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.07, 0.2, 32), mats.dark);
    waist.position.set(0, 0.02, 0);
    torsoGroup.add(waist);
    const thruster = new THREE.Mesh(new THREE.CylinderGeometry(0.06, 0.045, 0.03, 24), mats.glow);
    thruster.position.set(0, -0.09, 0);
    torsoGroup.add(thruster);

    const torsoMesh = new THREE.Mesh(new RoundedBoxGeometry(0.44, 0.52, 0.24, 4, 0.07), mats.armor);
    torsoMesh.position.set(0, 0.36, 0);
    torsoGroup.add(torsoMesh);

    // Màn hình LED trên ngực (hiển thị từ đang diễn)
    const canvas = document.createElement('canvas');
    canvas.width = 512;
    canvas.height = 256;
    const ctx = canvas.getContext('2d');
    drawChestScreen(ctx, canvas, '', 'VSL ROBOT');
    const screenTex = new THREE.CanvasTexture(canvas);
    screenTex.colorSpace = THREE.SRGBColorSpace;
    screenTex.anisotropy = 4;
    const screenFrame = new THREE.Mesh(new RoundedBoxGeometry(0.31, 0.17, 0.012, 2, 0.01), mats.dark);
    screenFrame.position.set(0, 0.42, 0.122);
    torsoGroup.add(screenFrame);
    const screen = new THREE.Mesh(
      new THREE.PlaneGeometry(0.29, 0.145),
      new THREE.MeshBasicMaterial({ map: screenTex, toneMapped: false })
    );
    screen.position.set(0, 0.42, 0.1295);
    torsoGroup.add(screen);
    screenRef.current = { canvas, ctx, tex: screenTex, shown: { text: '', sub: 'VSL ROBOT' }, anim: null };

    // Đèn trạng thái dưới màn hình
    const statusLed = new THREE.Mesh(new THREE.CapsuleGeometry(0.008, 0.08, 4, 8), mats.glow);
    statusLed.rotation.z = Math.PI / 2;
    statusLed.position.set(0, 0.27, 0.122);
    torsoGroup.add(statusLed);

    // Đầu robot
    const headGroup = new THREE.Group();
    headGroup.position.set(0, 0.63, 0);

    const neck = new THREE.Mesh(new THREE.CylinderGeometry(0.045, 0.055, 0.08, 20), mats.dark);
    neck.position.set(0, 0.02, 0);
    headGroup.add(neck);

    const headMesh = new THREE.Mesh(new RoundedBoxGeometry(0.30, 0.24, 0.24, 4, 0.08), mats.armor);
    headMesh.position.set(0, 0.17, 0);
    headGroup.add(headMesh);

    const visor = new THREE.Mesh(new RoundedBoxGeometry(0.25, 0.12, 0.03, 3, 0.03), mats.visor);
    visor.position.set(0, 0.18, 0.112);
    headGroup.add(visor);

    const eyeGeo = new THREE.CapsuleGeometry(0.013, 0.02, 4, 10);
    const leftEye = new THREE.Mesh(eyeGeo, mats.eye);
    leftEye.position.set(-0.055, 0.185, 0.129);
    const rightEye = new THREE.Mesh(eyeGeo, mats.eye);
    rightEye.position.set(0.055, 0.185, 0.129);
    headGroup.add(leftEye, rightEye);

    const mouth = new THREE.Mesh(new THREE.CapsuleGeometry(0.004, 0.05, 4, 8), mats.eye);
    mouth.rotation.z = Math.PI / 2;
    mouth.position.set(0, 0.145, 0.129);
    headGroup.add(mouth);

    [-1, 1].forEach((s) => {
      const ear = new THREE.Mesh(new THREE.CylinderGeometry(0.035, 0.035, 0.03, 20), mats.dark);
      ear.rotation.z = Math.PI / 2;
      ear.position.set(0.16 * s, 0.17, 0);
      headGroup.add(ear);
      const earGlow = new THREE.Mesh(new THREE.TorusGeometry(0.026, 0.004, 6, 24), mats.glow);
      earGlow.rotation.y = Math.PI / 2;
      earGlow.position.set(0.177 * s, 0.17, 0);
      headGroup.add(earGlow);
    });

    const antenna = new THREE.Mesh(new THREE.CylinderGeometry(0.005, 0.005, 0.07, 8), mats.dark);
    antenna.position.set(0, 0.32, 0);
    headGroup.add(antenna);
    const antennaTip = new THREE.Mesh(new THREE.SphereGeometry(0.016, 16, 16), mats.glow);
    antennaTip.position.set(0, 0.36, 0);
    headGroup.add(antennaTip);

    torsoGroup.add(headGroup);

    // Cánh tay (giữ nguyên vị trí vai/khuỷu/cổ tay của Avatar3D)
    const buildArm = (side) => {
      const isLeft = side === 'left';
      const shoulder = new THREE.Group();
      shoulder.position.set(isLeft ? SHOULDER_HALF : -SHOULDER_HALF, 0.60, 0);

      const pad = new THREE.Mesh(new THREE.SphereGeometry(0.068, 24, 18), mats.armor);
      pad.scale.set(1, 0.85, 1);
      shoulder.add(pad);
      const padRing = new THREE.Mesh(new THREE.TorusGeometry(0.06, 0.006, 8, 32), mats.glow);
      padRing.rotation.y = Math.PI / 2;
      padRing.position.x = isLeft ? 0.03 : -0.03;
      shoulder.add(padRing);

      const upperGeo = new THREE.CapsuleGeometry(0.036, UPPER_LEN - 0.10, 6, 16);
      upperGeo.translate(0, -UPPER_LEN / 2, 0);
      shoulder.add(new THREE.Mesh(upperGeo, mats.dark));

      const elbow = new THREE.Group();
      elbow.position.set(0, -UPPER_LEN, 0);
      elbow.add(new THREE.Mesh(new THREE.SphereGeometry(0.044, 18, 18), mats.joint));
      const elbowRing = new THREE.Mesh(new THREE.TorusGeometry(0.045, 0.005, 8, 28), mats.glow);
      elbowRing.rotation.y = Math.PI / 2;
      elbow.add(elbowRing);

      const foreGeo = new THREE.CapsuleGeometry(0.035, FORE_LEN - 0.09, 6, 16);
      foreGeo.translate(0, -FORE_LEN / 2, 0);
      elbow.add(new THREE.Mesh(foreGeo, mats.armor));

      // Khớp xoay cẳng tay (sấp/ngửa): tự quay lòng bàn tay ra phía người xem khi giơ tay
      const twist = new THREE.Group();
      twist.position.set(0, -FORE_LEN, 0);
      elbow.add(twist);

      const wrist = new THREE.Group();
      wrist.add(new THREE.Mesh(new THREE.SphereGeometry(0.026, 16, 16), mats.joint));
      const cuff = new THREE.Mesh(new THREE.TorusGeometry(0.03, 0.005, 8, 28), mats.glow);
      cuff.rotation.x = Math.PI / 2;
      cuff.position.y = 0.012;
      wrist.add(cuff);

      const hand = createRobotHand(mats, isLeft);
      hand.group.scale.setScalar(HAND_SCALE);
      hand.group.position.set(0, -0.012, 0);
      wrist.add(hand.group);

      twist.add(wrist);
      shoulder.add(elbow);
      torsoGroup.add(shoulder);
      return { shoulder, elbow, twist, wrist, hand };
    };

    const right = buildArm('right');
    const left = buildArm('left');

    avatarGroup.add(torsoGroup);
    scene.add(avatarGroup);

    rigRef.current = {
      avatarGroup,
      torsoGroup,
      headGroup,
      rightShoulder: right.shoulder,
      rightElbow: right.elbow,
      rightTwist: right.twist,
      rightWrist: right.wrist,
      leftShoulder: left.shoulder,
      leftElbow: left.elbow,
      leftTwist: left.twist,
      leftWrist: left.wrist,
      rightHand: right.hand,
      leftHand: left.hand,
      currentPose: {
        body: { torso: [0, 0, 0], head: [0, 0, 0] },
        rightArm: [-0.20, 0.05, 0.05], rightForearm: [-0.35, 0.05, 0.05], rightWrist: [0, 0, 0],
        rightFingers: NATURAL_FINGERS_REST,
        leftArm: [-0.20, -0.05, -0.05], leftForearm: [-0.35, -0.05, -0.05], leftWrist: [0, 0, 0],
        leftFingers: NATURAL_FINGERS_REST
      }
    };

    // Kéo chuột / chạm để xoay robot
    let isDragging = false;
    let prevX = 0;
    const down = (e) => { isDragging = true; prevX = e.clientX ?? e.touches?.[0]?.clientX ?? 0; };
    const move = (e) => {
      if (!isDragging) return;
      const x = e.clientX ?? e.touches?.[0]?.clientX ?? prevX;
      avatarGroup.rotation.y += (x - prevX) * 0.01;
      prevX = x;
    };
    const up = () => { isDragging = false; };
    container.addEventListener('mousedown', down);
    container.addEventListener('touchstart', down, { passive: true });
    window.addEventListener('mousemove', move);
    window.addEventListener('touchmove', move, { passive: true });
    window.addEventListener('mouseup', up);
    window.addEventListener('touchend', up);

    // Tự co giãn theo kích thước khung
    const ro = new ResizeObserver(() => {
      const w = container.clientWidth;
      const h = container.clientHeight;
      if (!w || !h) return;
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
      renderer.setSize(w, h);
    });
    ro.observe(container);

    // Lò xo giảm chấn tới hạn (critically-damped spring / SmoothDamp):
    // khớp tăng tốc rồi giảm tốc mềm mại, vận tốc liên tục khi đổi ký hiệu giữa chừng,
    // và không phụ thuộc FPS (dùng dt thực).
    let DT = 1 / 60;
    let ST = 0.16;
    const smoothDamp = (current, target, state, key) => {
      const omega = 2 / ST;
      const x = omega * DT;
      const exp = 1 / (1 + x + 0.48 * x * x + 0.235 * x * x * x);
      const change = current - target;
      const vel = state[key] || 0;
      const temp = (vel + omega * change) * DT;
      state[key] = (vel - omega * temp) * exp;
      return target + (change + temp) * exp;
    };
    const springAxis = (obj, axis, target) => {
      const st = obj.userData.vel || (obj.userData.vel = {});
      obj.rotation[axis] = smoothDamp(obj.rotation[axis], target, st, axis);
    };
    const lerpRot = (obj, target) => {
      springAxis(obj, 'x', target[0]);
      springAxis(obj, 'y', target[1]);
      springAxis(obj, 'z', target[2]);
    };

    const applyFingers = (fingers, target, mirror) => {
      const m = mirror ? -1 : 1;
      const th = target?.thumb || {};
      const cmcY = th.cmc_y ?? 0.0;
      const cmcZ = th.cmc_z ?? 0.0;
      const mcpY = th.mcp_y ?? (th.cmc_y ? th.cmc_y * 0.45 : 0.0);
      const mcpZ = th.mcp_z ?? (th.cmc_z ? th.cmc_z * 0.25 : 0.0);

      springAxis(fingers.thumb.cmc, 'x', th.cmc_x ?? 0.0);
      springAxis(fingers.thumb.cmc, 'y', m * cmcY);
      springAxis(fingers.thumb.cmc, 'z', m * cmcZ);
      springAxis(fingers.thumb.mcp, 'x', -(th.mcp_x ?? 0.1));
      springAxis(fingers.thumb.mcp, 'y', m * mcpY);
      springAxis(fingers.thumb.mcp, 'z', m * mcpZ);
      springAxis(fingers.thumb.ip, 'x', -(th.ip_x ?? 0.1));
      springAxis(fingers.thumb.ip, 'z', m * (th.ip_z ?? 0.0));

      ['index', 'middle', 'ring', 'pinky'].forEach((name) => {
        const f = fingers[name];
        const tg = target?.[name];
        if (!f || !tg) return;
        springAxis(f.mcp, 'x', -tg.mcp_x);
        springAxis(f.mcp, 'z', m * tg.spread_z);
        springAxis(f.pip, 'x', -tg.pip_x);
        springAxis(f.dip, 'x', -tg.dip_x);
      });
    };

    const handR = new THREE.Vector3();
    const handL = new THREE.Vector3();
    const camOffNear = new THREE.Vector3(0, 0.04, 0.52);
    const camOffBoth = new THREE.Vector3(0, 0.05, 0);
    const tmpQuat = new THREE.Quaternion();
    const tmpVec = new THREE.Vector3();
    const camLook = targetCamLookRef.current.clone();
    const clock = new THREE.Clock();
    const idleEye = new THREE.Color(0x67e8f9);
    const activeEye = new THREE.Color(0x4ade80);
    let nextBlink = 2.5;

    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate);
      DT = Math.min(0.05, clock.getDelta());
      const time = clock.elapsedTime;
      const playing = isPlayingRef.current;

      // Camera cận cảnh tự bám theo tâm bàn tay
      const mode = cameraModeRef.current;
      if (mode !== 'full' && rigRef.current.rightHand) {
        const r = rigRef.current;
        r.rightHand.group.localToWorld(handR.set(0, -0.05, 0.01));
        r.leftHand.group.localToWorld(handL.set(0, -0.05, 0.01));
        if (mode === 'hand_right') {
          targetCamLookRef.current.copy(handR);
          targetCamPosRef.current.copy(handR).add(camOffNear);
        } else if (mode === 'hand_left') {
          targetCamLookRef.current.copy(handL);
          targetCamPosRef.current.copy(handL).add(camOffNear);
        } else {
          targetCamLookRef.current.copy(handR).add(handL).multiplyScalar(0.5);
          const spread = handR.distanceTo(handL);
          targetCamPosRef.current.copy(targetCamLookRef.current).add(camOffBoth).setZ(targetCamLookRef.current.z + 0.55 + spread * 1.1);
        }
      }
      const camK = 1 - Math.exp(-DT * 4.5);
      camera.position.lerp(targetCamPosRef.current, camK);
      camLook.lerp(targetCamLookRef.current, camK);
      camera.lookAt(camLook);

      const rig = rigRef.current;
      const cur = rig.currentPose;

      // Hoạt ảnh mocap: lấy mẫu theo thời gian thực (lặp lại khi đang tạm dừng để xem kỹ)
      const mo = rig.motion;
      if (mo) {
        const frozen = typeof freezeRef.current === 'number';
        const durS = mo.data.duration_ms / 1000;
        // Chỉ diễn đoạn có động tác (bỏ khoảng đứng yên đầu/cuối video)
        const a0 = mo.data.active ? mo.data.active[0] : 0;
        const a1 = mo.data.active ? Math.min(durS, mo.data.active[1]) : durS;
        let el;
        // Đồng bộ với video mẫu: nếu video của đúng ký hiệu này đang phát → lấy thời điểm của video
        const vid = videoElRef?.current;
        const vidSynced = vid && !vid.paused && vid.readyState >= 2 && mo.data.id && (vid.currentSrc || vid.src || '').includes('/' + mo.data.id + '.mp4');
        if (frozen) {
          el = freezeRef.current;
        } else if (vidSynced) {
          el = vid.currentTime;
          mo.start = time - el / Math.max(0.25, speedRef.current);
        } else {
          if (mo.start === undefined) mo.start = time;
          el = a0 + (time - mo.start) * speedRef.current;
          if (el > a1 + 0.7 && !isPlayingRef.current) { mo.start = time; el = a0; }
        }
        applyKeyframe(cur, sampleMotion(mo.data, Math.min(Math.max(el, a0), a1)));
        ST = frozen ? 0.01 : Math.max(0.02, 0.035 / Math.max(0.5, speedRef.current)); // bám sát chuyển động học từ video
      } else {
        // Thời gian chuyển tiếp: nhanh hơn khi tăng tốc độ phát, ngắn hơn với keyframe động
        ST = Math.max(0.05, (cur.smoothTime || 0.12) / Math.max(0.25, speedRef.current));
      }

      lerpRot(rig.torsoGroup, cur.body.torso);
      lerpRot(rig.headGroup, cur.body.head);

      lerpRot(rig.rightShoulder, cur.rightArm);
      lerpRot(rig.rightElbow, cur.rightForearm);
      lerpRot(rig.rightWrist, cur.rightWrist);
      applyFingers(rig.rightHand.fingers, cur.rightFingers, false);

      lerpRot(rig.leftShoulder, cur.leftArm);
      lerpRot(rig.leftElbow, cur.leftForearm);
      lerpRot(rig.leftWrist, cur.leftWrist);
      applyFingers(rig.leftHand.fingers, cur.leftFingers, true);

      // Xoay cẳng tay: cẳng tay càng hướng lên → lòng bàn tay càng quay ra trước
      // (dữ liệu tư thế có thể ghi đè bằng palmTwist: số radian, hoặc 0 để tắt)
      [['right', rig.rightElbow, rig.rightTwist, 1], ['left', rig.leftElbow, rig.leftTwist, -1]].forEach(([side, elbow, twist, sign]) => {
        let target;
        if (typeof cur[side + 'Twist'] === 'number') {
          target = cur[side + 'Twist'];            // góc xoay cẳng tay đo từ video (mocap)
        } else if (typeof cur.palmTwist === 'number') {
          target = cur.palmTwist * sign;
        } else {
          elbow.getWorldQuaternion(tmpQuat);
          tmpVec.set(0, -1, 0).applyQuaternion(tmpQuat);
          const k = THREE.MathUtils.smoothstep(tmpVec.y, -0.1, 0.35);
          target = Math.PI * k * sign;
        }
        springAxis(twist, 'y', target);
      });

      // Hiệu ứng sống động: mắt chớp, đèn nhịp thở, robot lơ lửng nhẹ
      mats.eye.color.lerp(playing ? activeEye : idleEye, 1 - Math.exp(-DT * 6));
      if (time > nextBlink) {
        const k = (time - nextBlink) / 0.14;
        const s = k < 1 ? Math.max(0.12, Math.abs(1 - 2 * k)) : 1;
        leftEye.scale.y = s;
        rightEye.scale.y = s;
        if (k >= 1) nextBlink = time + 2.5 + Math.random() * 2.5;
      }
      const pulse = 1.0 + Math.sin(time * (playing ? 6 : 2)) * 0.45;
      mats.glow.emissiveIntensity = pulse;
      torsoGroup.position.y = 0.7 + Math.sin(time * 1.4) * 0.006;

      // Màn hình ngực: chuyển chữ mờ dần (cross-fade) thay vì đổi đột ngột
      const sa = screenRef.current?.anim;
      if (sa) {
        const p = Math.min(1, (time - (sa.start ?? (sa.start = time))) / 0.32);
        const sc = screenRef.current;
        if (p < 0.5) {
          const k = p / 0.5;
          drawChestScreen(sc.ctx, sc.canvas, sa.from.text, sa.from.sub, 1 - k, -10 * k);
        } else {
          const k = (p - 0.5) / 0.5;
          drawChestScreen(sc.ctx, sc.canvas, sa.to.text, sa.to.sub, k, 10 * (1 - k));
        }
        sc.tex.needsUpdate = true;
        if (p >= 1) { sc.shown = sa.to; sc.anim = null; }
      }

      renderer.render(scene, camera);
    };
    animate();

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      ro.disconnect();
      container.removeEventListener('mousedown', down);
      container.removeEventListener('touchstart', down);
      window.removeEventListener('mousemove', move);
      window.removeEventListener('touchmove', move);
      window.removeEventListener('mouseup', up);
      window.removeEventListener('touchend', up);
      scene.traverse((o) => {
        if (o.geometry) o.geometry.dispose();
      });
      screenTex.dispose();
      renderer.dispose();
    };
  }, []);

  // ---------- Cập nhật tư thế theo ký hiệu hiện tại ----------
  useEffect(() => {
    const rig = rigRef.current;
    if (!rig || !rig.currentPose) return;

    // Cập nhật màn hình ngực
    const scr = screenRef.current;
    if (scr) {
      const isSpell = currentStep?.type === 'fingerspelling';
      const tempSpell = isTempSpell(currentStep);
      const text = currentStep ? (isSpell ? (currentStep.token || currentStep.label) : currentStep.label) : '';
      const next = { text, sub: currentStep ? (isSpell || tempSpell ? 'ĐÁNH VẦN' : 'KÝ HIỆU') : 'VSL ROBOT' };
      const from = scr.anim ? scr.anim.to : scr.shown;
      if (from.text !== next.text || from.sub !== next.sub) {
        scr.anim = { from, to: next, start: undefined };
      }
    }

    // 1) Có hoạt ảnh chụp chuyển động từ video người thật → phát hoạt ảnh
    rig.motion = null;
    rig.currentPose.rightTwist = undefined;
    rig.currentPose.leftTwist = undefined;
    let cancelled = false;
    const motionId = currentStep?.motion_id;
    // (trường "motion" cũ của API là chuỗi 'static'/'dynamic' — chỉ nhận khi là dữ liệu chuyển động thật)
    if (currentStep?.motion && typeof currentStep.motion === 'object' && currentStep.motion.keyframes?.length) {
      rig.motion = { data: currentStep.motion, start: undefined };
      return undefined;
    }
    if (motionId) {
      fetchMotion(motionId).then((data) => {
        // bỏ qua kết quả về muộn của ký hiệu trước (đã chuyển sang ký hiệu khác)
        if (!cancelled && data && data.keyframes?.length) rig.motion = { data, start: undefined };
      });
    }

    const applyStatic = (pd) => {
      rig.currentPose.body = pd.body || { torso: [0.02, 0, 0], head: [0.02, 0, 0] };
      rig.currentPose.rightArm = pd.rightArm || [-1.15, 0.15, -0.08];
      rig.currentPose.rightForearm = pd.rightForearm || [-1.10, 0.20, 0.08];
      rig.currentPose.rightWrist = pd.rightWrist || [0.15, 0.10, 0];
      rig.currentPose.rightFingers = { ...(pd.rightFingers || pd) };
      rig.currentPose.leftArm = pd.leftArm || [-0.22, -0.08, 0.05];
      rig.currentPose.leftForearm = pd.leftForearm || [-0.35, -0.08, -0.05];
      rig.currentPose.leftWrist = pd.leftWrist || [0, 0, 0];
      rig.currentPose.leftFingers = pd.leftFingers ? { ...pd.leftFingers } : NATURAL_FINGERS_REST;
      rig.currentPose.palmTwist = pd.palmTwist;
    };

    // 1b) Từ chưa có chuyển động (đang học từ video) → đánh vần tạm bằng bảng chữ cái ngón tay
    if (isTempSpell(currentStep)) {
      const letters = currentStep.spell;
      let i = 0;
      rig.currentPose.smoothTime = 0.1;
      const run = () => { applyStatic(getHandPoseByGloss(letters[i % letters.length])); i += 1; };
      run();
      const iv = setInterval(() => { if (!rig.motion) run(); }, SPELL_LETTER_MS / Math.max(0.25, speed));
      return () => { cancelled = true; clearInterval(iv); };
    }

    // 2) Tư thế vẽ tay (handPoses.js); từ chưa có tư thế robot → tư thế nghỉ (xem video mẫu)
    const gloss = currentStep?.has_pose === false
      ? 'REST'
      : (currentStep?.pose_key || currentStep?.gloss || currentStep?.token);
    const poseData = getHandPoseByGloss(gloss);

    if (poseData.isDynamic && poseData.keyframes && poseData.keyframes.length > 0) {
      let idx = 0;
      const run = () => {
        const kf = poseData.keyframes[idx];
        if (kf) {
          rig.currentPose.body = kf.body || { torso: [0.05, 0, 0], head: [0.05, 0, 0] };
          rig.currentPose.rightArm = kf.rightArm || [-1.0, 0.2, 0.1];
          rig.currentPose.rightForearm = kf.rightForearm || [-0.6, 0.3, 0.2];
          rig.currentPose.rightWrist = kf.rightWrist || [0, 0, 0];
          rig.currentPose.rightFingers = { ...kf.rightFingers };
          rig.currentPose.leftArm = kf.leftArm || [-0.4, -0.2, -0.1];
          rig.currentPose.leftForearm = kf.leftForearm || [-0.6, -0.1, -0.1];
          rig.currentPose.leftWrist = kf.leftWrist || [0, 0, 0];
          rig.currentPose.leftFingers = { ...kf.leftFingers };
          rig.currentPose.palmTwist = kf.palmTwist ?? poseData.palmTwist;
        }
        idx = (idx + 1) % poseData.keyframes.length;
      };
      const kfMs = (currentStep?.duration_ms || 1800) / poseData.keyframes.length;
      // Keyframe nối tiếp nhau: lò xo ngắn hơn để kịp tới từng tư thế nhưng vẫn mượt
      rig.currentPose.smoothTime = Math.min(0.17, (kfMs / 1000) * 0.35);
      run();
      const interval = setInterval(() => { if (!rig.motion) run(); }, kfMs / speed);
      return () => { cancelled = true; clearInterval(interval); };
    }

    rig.currentPose.smoothTime = 0.12;
    applyStatic(poseData);
    return () => { cancelled = true; };
  }, [currentStep, speed]);

  // Tự cuộn phụ đề tới từ đang diễn (chỉ cuộn trong khung phụ đề, không cuộn cả trang)
  useEffect(() => {
    const box = captionRef.current;
    const el = box?.querySelector('[data-active="1"]');
    if (!box || !el) return;
    const top = el.offsetTop - box.offsetTop;
    if (top < box.scrollTop || top + el.offsetHeight > box.scrollTop + box.clientHeight) {
      box.scrollTo({ top: Math.max(0, top - box.clientHeight / 2 + el.offsetHeight / 2), behavior: 'smooth' });
    }
  }, [currentIndex, sequence]);

  // ---------- Góc nhìn camera ----------
  const switchCameraMode = (mode) => {
    setCameraMode(mode);
    cameraModeRef.current = mode;
    if (mode === 'full') {
      targetCamPosRef.current.set(0, 1.12, 2.15);
      targetCamLookRef.current.set(0, 0.98, 0);
    }
  };

  const resetRotation = () => {
    rigRef.current?.avatarGroup?.rotation.set(0, 0, 0);
  };

  const cameraButtons = [
    { id: 'full', icon: <Eye size={13} />, label: 'Toàn thân', title: 'Toàn cảnh robot' },
    { id: 'hand_both', icon: <Crosshair size={13} />, label: '2 tay', title: 'Cận cảnh cả 2 bàn tay' },
    { id: 'hand_right', icon: <ZoomIn size={13} />, label: 'Tay phải', title: 'Phóng to tay phải' },
    { id: 'hand_left', icon: <ZoomIn size={13} />, label: 'Tay trái', title: 'Phóng to tay trái' }
  ];

  return (
    <div style={{ display: 'flex', flexDirection: 'column', width: '100%', height: '100%' }}>
      <style>{`
        @keyframes rsSwapIn {
          from { opacity: 0; transform: translateY(8px); filter: blur(2px); }
          to   { opacity: 1; transform: translateY(0);   filter: blur(0); }
        }
        .rs-swap { animation: rsSwapIn 0.32s cubic-bezier(0.22, 1, 0.36, 1) both; }
        @media (prefers-reduced-motion: reduce) { .rs-swap { animation: none; } }
      `}</style>
      {/* ===== Khung 3D ===== */}
      <div style={{ position: 'relative', flex: 1, minHeight: 0 }}>
        <div
          ref={mountRef}
          style={{ position: 'absolute', inset: 0, cursor: 'grab' }}
          title="Kéo chuột để xoay robot 360°"
        />

        {/* Thanh góc nhìn (dọc, góc phải) */}
        <div style={{
          position: 'absolute', top: '12px', right: '12px', display: 'flex', flexDirection: 'column',
          gap: '4px', background: 'rgba(8, 15, 28, 0.85)', padding: '4px', borderRadius: '10px',
          border: '1px solid var(--border-subtle)', backdropFilter: 'blur(10px)'
        }}>
          {cameraButtons.map((b) => (
            <button
              key={b.id}
              onClick={() => switchCameraMode(b.id)}
              className={`btn ${cameraMode === b.id ? 'btn-primary' : 'btn-secondary'}`}
              style={{ padding: '5px 9px', fontSize: '11px', justifyContent: 'flex-start' }}
              title={b.title}
            >
              {b.icon} {b.label}
            </button>
          ))}
          <button onClick={resetRotation} className="btn btn-secondary" style={{ padding: '5px 9px', fontSize: '11px', justifyContent: 'flex-start' }} title="Khôi phục góc nhìn thẳng">
            <RotateCcw size={13} /> Xoay lại
          </button>
        </div>
      </div>

      {/* ===== Dải thông tin dưới khung 3D (không che robot) ===== */}
      <div style={{
        borderTop: '1px solid var(--border-subtle)', background: 'rgba(8, 15, 28, 0.92)',
        padding: '8px 12px 10px', display: 'flex', flexDirection: 'column', gap: '8px'
      }}>
        {/* Chú thích màu ngón tay */}
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '6px 14px', justifyContent: 'center', fontSize: '11.5px', color: '#e2e8f0' }}>
          {Object.entries(FINGER_COLORS).map(([k, c]) => (
            <span key={k} style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
              <span style={{ width: 10, height: 10, borderRadius: 3, background: c.css }} />
              {c.name}
            </span>
          ))}
          <span style={{ display: 'inline-flex', alignItems: 'center', gap: '5px' }}>
            <span style={{ width: 10, height: 10, borderRadius: 3, background: '#06b6d4', boxShadow: '0 0 6px #06b6d4' }} />
            Lòng bàn tay
          </span>
        </div>

        {/* Phụ đề: câu đầy đủ, tô sáng từ đang diễn */}
        {sequence.length > 0 && (
          <div
            ref={captionRef}
            style={{
              display: 'flex', flexWrap: 'wrap', gap: '8px 9px', alignItems: 'center', justifyContent: 'center',
              maxHeight: '78px', overflowY: 'auto', padding: '4px 6px'
            }}
          >
            {sequence.map((s, i) => {
              const active = i === currentIndex;
              const done = i < currentIndex;
              const spell = s.type === 'fingerspelling';
              return (
                <span
                  key={i}
                  data-active={active ? '1' : undefined}
                  style={{
                    padding: '3px 9px',
                    borderRadius: '8px',
                    fontSize: '14px',
                    fontWeight: 700,
                    color: active ? '#04121c' : done ? '#64748b' : '#e2e8f0',
                    background: active ? (spell ? '#fbbf24' : '#22d3ee') : 'rgba(255,255,255,0.06)',
                    border: `1px solid ${active ? 'transparent' : spell ? 'rgba(251,191,36,0.35)' : 'rgba(255,255,255,0.08)'}`,
                    boxShadow: active ? `0 0 12px ${spell ? 'rgba(251,191,36,0.55)' : 'rgba(34,211,238,0.55)'}` : '0 0 0 rgba(0,0,0,0)',
                    transform: active ? 'scale(1.12)' : 'scale(1)',
                    transition: 'background-color 0.35s ease, color 0.35s ease, border-color 0.35s ease, box-shadow 0.35s ease, transform 0.35s cubic-bezier(0.34, 1.4, 0.64, 1)',
                    willChange: 'transform'
                  }}
                >
                  {spell ? (s.token || s.label) : s.label}
                </span>
              );
            })}
          </div>
        )}
      </div>
    </div>
  );
}
