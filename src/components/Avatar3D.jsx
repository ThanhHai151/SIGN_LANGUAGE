import React, { useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { getHandPoseByGloss, NATURAL_FINGERS_REST } from '../utils/handPoses';
import { Eye, ZoomIn, RotateCcw, Crosshair, Sparkles } from 'lucide-react';

/**
 * Component Avatar 3D Toàn Thân & Hai Tay Chuẩn Giải Phẫu (Two-Handed & Whole-Body 3D Avatar)
 * Cấu trúc:
 * 1. Toàn bộ thân trên: Thân (Torso), Cổ & Đầu (Head nod/tilt), Hai vai (Shoulders).
 * 2. Cả 2 cánh tay & cẳng tay: Cử động đồng bộ và hỗ trợ biểu đạt hai tay.
 * 3. Cả 2 bàn tay: Mỗi bàn tay có 15 khớp đốt (MCP, PIP, DIP), móng tay định hướng, lòng bàn tay giải phẫu rõ nét.
 */
export default function Avatar3D({ currentStep, isPlaying, speed = 1.0 }) {
  const mountRef = useRef(null);
  const sceneRef = useRef(null);
  const cameraRef = useRef(null);
  const rendererRef = useRef(null);
  const rigRef = useRef({});
  const animFrameRef = useRef(null);

  const [cameraMode, setCameraMode] = useState('full');
  const targetCamPosRef = useRef(new THREE.Vector3(0, 1.30, 2.35));
  const targetCamLookRef = useRef(new THREE.Vector3(0, 1.05, 0));

  // Hàm tạo một bàn tay giải phẫu tự nhiên & rõ từng khớp ngón (không cục tròn thừa)
  const createArticulatedHand = (skinMat, creaseMat, nailMat, palmarMat, isLeft = false) => {
    const handGroup = new THREE.Group();
    const flip = isLeft ? -1 : 1;

    // 1. Khối lòng bàn tay giải phẫu thon gọn tự nhiên (Palm)
    const palmGeo = new THREE.BoxGeometry(0.060, 0.070, 0.018);
    palmGeo.translate(0, -0.035, 0);
    const palmMesh = new THREE.Mesh(palmGeo, skinMat);
    handGroup.add(palmMesh);

    // Tấm đệm lòng bàn tay mềm mại (Palmar Surface Pad - mặt trước +Z)
    const palmarGeo = new THREE.BoxGeometry(0.054, 0.064, 0.003);
    palmarGeo.translate(0, -0.035, 0.0095);
    const palmarMesh = new THREE.Mesh(palmarGeo, palmarMat);
    handGroup.add(palmarMesh);

    // Hàm tạo 1 ngón tay giải phẫu rõ khớp
    const createFinger = (lengthScale, isThumb = false) => {
      const fingerRoot = new THREE.Group();

      if (isThumb) {
        // --- NGÓN CÁI GIẢI PHẪU SINH HỌC CHUẨN XÁC: 3 ĐỐT, 2 KHỚP PHÂN BIỆT RÕ RÀNG ---
        // 1. Khớp cổ bàn ngón cái (CMC Joint)
        const cmcJoint = new THREE.Group();

        // Đốt bàn 1 (1st Metacarpal) - đoạn thân đốt bàn nối từ mép ngoài bàn tay
        const metaGeo = new THREE.CylinderGeometry(0.011, 0.0125, 0.024 * lengthScale, 14);
        metaGeo.translate(0, -0.012 * lengthScale, 0);
        cmcJoint.add(new THREE.Mesh(metaGeo, skinMat));

        // 2. Khớp bàn - ngón cái (MCP Joint - Khớp gốc ngón cái rõ ràng)
        const mcpJoint = new THREE.Group();
        mcpJoint.position.set(0, -0.024 * lengthScale, 0);

        // Chỏm khớp MCP (Khớp đốt ngón tròn mịn màu da, không bị lộ cục tròn lạc màu)
        const mcpPivot = new THREE.Mesh(new THREE.SphereGeometry(0.0115, 14, 14), skinMat);
        mcpJoint.add(mcpPivot);

        // Nếp gấp khớp bàn ngón cái (MCP Flexion Crease) - đường viền khớp rõ nét phân biệt đốt bàn và đốt gần
        const mcpCrease = new THREE.Mesh(new THREE.TorusGeometry(0.0116, 0.0016, 8, 16), creaseMat);
        mcpCrease.rotation.x = Math.PI / 2;
        mcpJoint.add(mcpCrease);

        // Đốt gần ngón cái (Proximal Phalanx) - thon gọn, có eo sinh học
        const proxGeo = new THREE.CylinderGeometry(0.010, 0.0112, 0.022 * lengthScale, 14);
        proxGeo.translate(0, -0.011 * lengthScale, 0);
        mcpJoint.add(new THREE.Mesh(proxGeo, skinMat));

        // Đệm thịt phẳng nhẹ ở đốt gần (không lồi thành cục)
        const proxPadGeo = new THREE.BoxGeometry(0.014, 0.018 * lengthScale, 0.003);
        proxPadGeo.translate(0, -0.011 * lengthScale, 0.007);
        mcpJoint.add(new THREE.Mesh(proxPadGeo, palmarMat));

        // 3. Khớp liên đốt ngón cái (IP Joint - Khớp giữa ngón cái rõ nét)
        const ipJoint = new THREE.Group();
        ipJoint.position.set(0, -0.022 * lengthScale, 0);

        // Chỏm khớp liên đốt
        const ipPivot = new THREE.Mesh(new THREE.SphereGeometry(0.0102, 14, 14), skinMat);
        ipJoint.add(ipPivot);

        // Nếp gấp khớp liên đốt (IP Flexion Crease) - giúp nhìn rõ mồn một khớp ngón cái gập duỗi
        const ipCrease = new THREE.Mesh(new THREE.TorusGeometry(0.0103, 0.0015, 8, 16), creaseMat);
        ipCrease.rotation.x = Math.PI / 2;
        ipJoint.add(ipCrease);

        // 4. Đốt xa ngón cái (Distal Phalanx & Fingertip)
        const distGeo = new THREE.CylinderGeometry(0.0085, 0.010, 0.018 * lengthScale, 14);
        distGeo.translate(0, -0.009 * lengthScale, 0);
        const distMesh = new THREE.Mesh(distGeo, skinMat);

        // Búp thịt đầu ngón cái (Thumb Pulp Pad) - mặt lòng ngón (+Z)
        const pulpGeo = new THREE.SphereGeometry(0.0105, 14, 14);
        pulpGeo.scale(1.15, 1.25, 0.85);
        pulpGeo.translate(0, -0.009 * lengthScale, 0.0055);
        distMesh.add(new THREE.Mesh(pulpGeo, palmarMat));

        // Móng ngón cái hình vòm siêu nét phản quang cao cấp (Thumb Specular Nail) - mặt mu ngón (-Z)
        const nailGeo = new THREE.BoxGeometry(0.0125, 0.012 * lengthScale, 0.003);
        const nailMesh = new THREE.Mesh(nailGeo, nailMat);
        nailMesh.position.set(0, -0.010 * lengthScale, -0.0075);
        distMesh.add(nailMesh);

        // Viền chân móng (Cuticle Rim)
        const cuticleGeo = new THREE.BoxGeometry(0.013, 0.0025, 0.0035);
        const cuticleMesh = new THREE.Mesh(cuticleGeo, creaseMat);
        cuticleMesh.position.set(0, -0.0035 * lengthScale, -0.0072);
        distMesh.add(cuticleMesh);

        ipJoint.add(distMesh);
        mcpJoint.add(ipJoint);
        cmcJoint.add(mcpJoint);
        fingerRoot.add(cmcJoint);

        return { root: fingerRoot, cmc: cmcJoint, mcp: mcpJoint, ip: ipJoint, isThumb: true };
      } else {
        // --- 4 NGÓN TAY (MCP -> PIP -> DIP) ĐỒNG BỘ MÀU DA VÀ RÕ NẾP KHỚP ---
        const mcpJoint = new THREE.Group();
        const mcpPivot = new THREE.Mesh(new THREE.SphereGeometry(0.0105, 14, 14), skinMat);
        mcpJoint.add(mcpPivot);
        const mcpCrease = new THREE.Mesh(new THREE.TorusGeometry(0.0105, 0.0015, 8, 16), creaseMat);
        mcpCrease.rotation.x = Math.PI / 2;
        mcpJoint.add(mcpCrease);

        const proxGeo = new THREE.CylinderGeometry(0.009, 0.0102, 0.038 * lengthScale, 14);
        proxGeo.translate(0, -0.019 * lengthScale, 0);
        mcpJoint.add(new THREE.Mesh(proxGeo, skinMat));

        const pipJoint = new THREE.Group();
        pipJoint.position.set(0, -0.038 * lengthScale, 0);
        const pipPivot = new THREE.Mesh(new THREE.SphereGeometry(0.0092, 14, 14), skinMat);
        pipJoint.add(pipPivot);
        const pipCrease = new THREE.Mesh(new THREE.TorusGeometry(0.0092, 0.0014, 8, 16), creaseMat);
        pipCrease.rotation.x = Math.PI / 2;
        pipJoint.add(pipCrease);

        const midGeo = new THREE.CylinderGeometry(0.008, 0.009, 0.026 * lengthScale, 14);
        midGeo.translate(0, -0.013 * lengthScale, 0);
        pipJoint.add(new THREE.Mesh(midGeo, skinMat));

        const dipJoint = new THREE.Group();
        dipJoint.position.set(0, -0.026 * lengthScale, 0);
        const dipPivot = new THREE.Mesh(new THREE.SphereGeometry(0.0082, 14, 14), skinMat);
        dipJoint.add(dipPivot);
        const dipCrease = new THREE.Mesh(new THREE.TorusGeometry(0.0082, 0.0012, 8, 16), creaseMat);
        dipCrease.rotation.x = Math.PI / 2;
        dipJoint.add(dipCrease);

        const distGeo = new THREE.CylinderGeometry(0.007, 0.008, 0.020 * lengthScale, 14);
        distGeo.translate(0, -0.010 * lengthScale, 0);
        const distMesh = new THREE.Mesh(distGeo, skinMat);

        // Móng tay (Nail Plate)
        const nailGeo = new THREE.BoxGeometry(0.0075, 0.009, 0.0028);
        const nailMesh = new THREE.Mesh(nailGeo, nailMat);
        nailMesh.position.set(0, -0.012 * lengthScale, -0.0068);
        distMesh.add(nailMesh);
        dipJoint.add(distMesh);

        pipJoint.add(dipJoint);
        mcpJoint.add(pipJoint);
        fingerRoot.add(mcpJoint);

        return { root: fingerRoot, mcp: mcpJoint, pip: pipJoint, dip: dipJoint, isThumb: false };
      }
    };

    // 2. Gắn 5 ngón tay vào vị trí giải phẫu chuẩn
    const indexRig = createFinger(1.0, false);
    indexRig.root.position.set(-0.020 * flip, -0.070, 0.0);
    handGroup.add(indexRig.root);

    const middleRig = createFinger(1.08, false);
    middleRig.root.position.set(-0.006 * flip, -0.072, 0.0);
    handGroup.add(middleRig.root);

    const ringRig = createFinger(0.98, false);
    ringRig.root.position.set(0.007 * flip, -0.070, 0.0);
    handGroup.add(ringRig.root);

    const pinkyRig = createFinger(0.82, false);
    pinkyRig.root.position.set(0.020 * flip, -0.067, 0.0);
    handGroup.add(pinkyRig.root);

    // Ngón cái gắn ở mép ngoài bàn tay (Radial margin)
    const thumbRig = createFinger(1.0, true);
    thumbRig.root.position.set(-0.031 * flip, -0.026, 0.006);
    thumbRig.root.rotation.set(0.25, -0.35 * flip, -0.40 * flip);
    handGroup.add(thumbRig.root);

    return {
      group: handGroup,
      fingers: { thumb: thumbRig, index: indexRig, middle: middleRig, ring: ringRig, pinky: pinkyRig }
    };
  };

  useEffect(() => {
    const container = mountRef.current;
    if (!container) return;

    const width = container.clientWidth || 450;
    const height = container.clientHeight || 450;

    // 1. Scene, Camera, Renderer
    const scene = new THREE.Scene();
    sceneRef.current = scene;

    const camera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
    camera.position.set(0, 1.30, 2.35);
    cameraRef.current = camera;

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(width, height);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.shadowMap.enabled = true;
    renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    rendererRef.current = renderer;

    container.innerHTML = '';
    container.appendChild(renderer.domElement);

    // 2. Chiếu sáng Studio 4 điểm (Key, Fill, Rim, Hand Accent Light)
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0x38bdf8, 2.2);
    keyLight.position.set(2.5, 3.5, 3);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x818cf8, 1.5);
    fillLight.position.set(-2.5, 2.0, 2);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0x34d399, 1.8);
    rimLight.position.set(0, 3.5, -2.5);
    scene.add(rimLight);

    // Đèn rọi bàn tay làm rõ nét khớp ngón tay
    const handSpot = new THREE.PointLight(0xffffff, 1.2, 3);
    handSpot.position.set(0, 1.2, 1.5);
    scene.add(handSpot);

    // 3. Vật liệu giải phẫu cao cấp
    const skinMat = new THREE.MeshStandardMaterial({
      color: 0xf2c19e,
      roughness: 0.42,
      metalness: 0.02
    });

    const palmarMat = new THREE.MeshStandardMaterial({
      color: 0xf8d8c2,
      roughness: 0.50,
      metalness: 0.01
    });

    const creaseMat = new THREE.MeshStandardMaterial({
      color: 0xc88764,
      roughness: 0.65,
      metalness: 0.02
    });

    const clothMat = new THREE.MeshStandardMaterial({
      color: 0x1e293b,
      roughness: 0.7,
      metalness: 0.15
    });

    const nailMat = new THREE.MeshStandardMaterial({
      color: 0xfff6f0,
      roughness: 0.08,
      metalness: 0.25
    });

    // 4. CẤU TRÚC TOÀN BỘ CƠ THỂ AVATAR
    const avatarGroup = new THREE.Group();
    avatarGroup.position.set(0, -0.2, 0);

    // Khối thân (Torso Group) - có thể cúi, xoay, nghiêng
    const torsoGroup = new THREE.Group();
    torsoGroup.position.set(0, 0.7, 0);

    const torsoGeo = new THREE.CylinderGeometry(0.24, 0.19, 0.65, 24);
    torsoGeo.translate(0, 0.325, 0);
    const torsoMesh = new THREE.Mesh(torsoGeo, clothMat);
    torsoGroup.add(torsoMesh);

    // Đầu & Cổ (Head Group) - có thể gật, nghiêng, xoay
    const headGroup = new THREE.Group();
    headGroup.position.set(0, 0.75, 0); // Đỉnh thân

    const headMesh = new THREE.Mesh(new THREE.SphereGeometry(0.16, 24, 24), skinMat);
    headGroup.add(headMesh);

    // Mắt & Lông mày biểu cảm
    const eyeMat = new THREE.MeshBasicMaterial({ color: 0x0f172a });
    const leftEye = new THREE.Mesh(new THREE.SphereGeometry(0.018, 12, 12), eyeMat);
    leftEye.position.set(-0.06, 0.02, 0.145);
    const rightEye = new THREE.Mesh(new THREE.SphereGeometry(0.018, 12, 12), eyeMat);
    rightEye.position.set(0.06, 0.02, 0.145);
    headGroup.add(leftEye);
    headGroup.add(rightEye);

    // Miệng mỉm cười thân thiện
    const smileGeo = new THREE.TorusGeometry(0.03, 0.005, 8, 16, Math.PI);
    const smileMesh = new THREE.Mesh(smileGeo, eyeMat);
    smileMesh.position.set(0, -0.05, 0.145);
    smileMesh.rotation.x = Math.PI;
    headGroup.add(smileMesh);

    torsoGroup.add(headGroup);

    // --- CÁNH TAY PHẢI (RIGHT ARM) - NẰM BÊN PHẢI CƠ THỂ AVATAR (-X) ---
    const rightShoulder = new THREE.Group();
    rightShoulder.position.set(-0.25, 0.60, 0);

    const rightUpperArmGeo = new THREE.CylinderGeometry(0.048, 0.040, 0.28, 16);
    rightUpperArmGeo.translate(0, -0.14, 0);
    rightShoulder.add(new THREE.Mesh(rightUpperArmGeo, clothMat));

    const rightElbow = new THREE.Group();
    rightElbow.position.set(0, -0.28, 0);
    rightElbow.add(new THREE.Mesh(new THREE.SphereGeometry(0.042, 16, 16), clothMat));

    const rightForearmGeo = new THREE.CylinderGeometry(0.040, 0.032, 0.26, 16);
    rightForearmGeo.translate(0, -0.13, 0);
    rightElbow.add(new THREE.Mesh(rightForearmGeo, skinMat));

    const rightWrist = new THREE.Group();
    rightWrist.position.set(0, -0.26, 0);
    const rightWristJoint = new THREE.Mesh(new THREE.SphereGeometry(0.028, 16, 16), skinMat);
    rightWrist.add(rightWristJoint);
    const rightWristRing = new THREE.Mesh(new THREE.TorusGeometry(0.0285, 0.002, 8, 24), creaseMat);
    rightWristRing.rotation.x = Math.PI / 2;
    rightWrist.add(rightWristRing);

    // Bàn tay phải giải phẫu (Tỉ lệ chuẩn tự nhiên 1.12x)
    const rightHandRig = createArticulatedHand(skinMat, creaseMat, nailMat, palmarMat, false);
    rightHandRig.group.scale.set(1.12, 1.12, 1.12);
    rightHandRig.group.position.set(0, -0.010, 0);
    rightWrist.add(rightHandRig.group);
    rightElbow.add(rightWrist);
    rightShoulder.add(rightElbow);
    torsoGroup.add(rightShoulder);

    // --- CÁNH TAY TRÁI (LEFT ARM) - NẰM BÊN TRÁI CƠ THỂ AVATAR (+X) ---
    const leftShoulder = new THREE.Group();
    leftShoulder.position.set(0.25, 0.60, 0);

    const leftUpperArmGeo = new THREE.CylinderGeometry(0.048, 0.040, 0.28, 16);
    leftUpperArmGeo.translate(0, -0.14, 0);
    leftShoulder.add(new THREE.Mesh(leftUpperArmGeo, clothMat));

    const leftElbow = new THREE.Group();
    leftElbow.position.set(0, -0.28, 0);
    leftElbow.add(new THREE.Mesh(new THREE.SphereGeometry(0.042, 16, 16), clothMat));

    const leftForearmGeo = new THREE.CylinderGeometry(0.040, 0.032, 0.26, 16);
    leftForearmGeo.translate(0, -0.13, 0);
    leftElbow.add(new THREE.Mesh(leftForearmGeo, skinMat));

    const leftWrist = new THREE.Group();
    leftWrist.position.set(0, -0.26, 0);
    const leftWristJoint = new THREE.Mesh(new THREE.SphereGeometry(0.028, 16, 16), skinMat);
    leftWrist.add(leftWristJoint);
    const leftWristRing = new THREE.Mesh(new THREE.TorusGeometry(0.0285, 0.002, 8, 24), creaseMat);
    leftWristRing.rotation.x = Math.PI / 2;
    leftWrist.add(leftWristRing);

    // Bàn tay trái giải phẫu (Tỉ lệ chuẩn tự nhiên 1.12x)
    const leftHandRig = createArticulatedHand(skinMat, creaseMat, nailMat, palmarMat, true);
    leftHandRig.group.scale.set(1.12, 1.12, 1.12);
    leftHandRig.group.position.set(0, -0.010, 0);
    leftWrist.add(leftHandRig.group);
    leftElbow.add(leftWrist);
    leftShoulder.add(leftElbow);
    torsoGroup.add(leftShoulder);

    avatarGroup.add(torsoGroup);
    scene.add(avatarGroup);

    // Lưu trữ tham chiếu hệ xương để điều khiển FK
    rigRef.current = {
      avatarGroup,
      torsoGroup,
      headGroup,
      rightShoulder,
      rightElbow,
      rightWrist,
      leftShoulder,
      leftElbow,
      leftWrist,
      rightHand: rightHandRig,
      leftHand: leftHandRig,
      currentPose: {
        body: { torso: [0, 0, 0], head: [0, 0, 0] },
        rightArm: [-0.20, 0.05, 0.05], rightForearm: [-0.35, 0.05, 0.05], rightWrist: [0, 0, 0],
        rightFingers: NATURAL_FINGERS_REST,
        leftArm: [-0.20, -0.05, -0.05], leftForearm: [-0.35, -0.05, -0.05], leftWrist: [0, 0, 0],
        leftFingers: NATURAL_FINGERS_REST
      }
    };

    // 5. Chuột xoay Orbit 360 độ
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;

    const handleMouseDown = (e) => {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    };

    const handleMouseMove = (e) => {
      if (!isDragging) return;
      const deltaX = e.clientX - prevMouseX;
      avatarGroup.rotation.y += deltaX * 0.01;
      prevMouseX = e.clientX;
    };

    const handleMouseUp = () => { isDragging = false; };

    container.addEventListener('mousedown', handleMouseDown);
    window.addEventListener('mousemove', handleMouseMove);
    window.addEventListener('mouseup', handleMouseUp);

    // 6. Animation Loop chính với nội suy mượt mà
    let clock = new THREE.Clock();

    const animate = () => {
      animFrameRef.current = requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Camera lerp mượt mà
      camera.position.lerp(targetCamPosRef.current, 0.08);
      camera.lookAt(targetCamLookRef.current);

      const rig = rigRef.current;
      if (!rig || !rig.rightShoulder) return;

      const lerpSpeed = 0.14 * speed;
      const cur = rig.currentPose;

      // 1. Thân người (Torso)
      rig.torsoGroup.rotation.x = THREE.MathUtils.lerp(rig.torsoGroup.rotation.x, cur.body.torso[0], lerpSpeed);
      rig.torsoGroup.rotation.y = THREE.MathUtils.lerp(rig.torsoGroup.rotation.y, cur.body.torso[1], lerpSpeed);
      rig.torsoGroup.rotation.z = THREE.MathUtils.lerp(rig.torsoGroup.rotation.z, cur.body.torso[2], lerpSpeed);

      // 2. Đầu & Cổ (Head)
      rig.headGroup.rotation.x = THREE.MathUtils.lerp(rig.headGroup.rotation.x, cur.body.head[0], lerpSpeed);
      rig.headGroup.rotation.y = THREE.MathUtils.lerp(rig.headGroup.rotation.y, cur.body.head[1], lerpSpeed);
      rig.headGroup.rotation.z = THREE.MathUtils.lerp(rig.headGroup.rotation.z, cur.body.head[2], lerpSpeed);

      // 3. Cánh tay & Bàn tay phải
      rig.rightShoulder.rotation.x = THREE.MathUtils.lerp(rig.rightShoulder.rotation.x, cur.rightArm[0], lerpSpeed);
      rig.rightShoulder.rotation.y = THREE.MathUtils.lerp(rig.rightShoulder.rotation.y, cur.rightArm[1], lerpSpeed);
      rig.rightShoulder.rotation.z = THREE.MathUtils.lerp(rig.rightShoulder.rotation.z, cur.rightArm[2], lerpSpeed);

      rig.rightElbow.rotation.x = THREE.MathUtils.lerp(rig.rightElbow.rotation.x, cur.rightForearm[0], lerpSpeed);
      rig.rightElbow.rotation.y = THREE.MathUtils.lerp(rig.rightElbow.rotation.y, cur.rightForearm[1], lerpSpeed);
      rig.rightElbow.rotation.z = THREE.MathUtils.lerp(rig.rightElbow.rotation.z, cur.rightForearm[2], lerpSpeed);

      rig.rightWrist.rotation.x = THREE.MathUtils.lerp(rig.rightWrist.rotation.x, cur.rightWrist[0], lerpSpeed);
      rig.rightWrist.rotation.y = THREE.MathUtils.lerp(rig.rightWrist.rotation.y, cur.rightWrist[1], lerpSpeed);
      rig.rightWrist.rotation.z = THREE.MathUtils.lerp(rig.rightWrist.rotation.z, cur.rightWrist[2], lerpSpeed);

      // Ngón tay phải - Xoay 3 trục sinh học đối chiếu (Multi-Axis Opposition)
      const rf = rig.rightHand.fingers;
      const rt = cur.rightFingers?.thumb || {};
      const rt_cmc_x = rt.cmc_x ?? 0.0;
      const rt_cmc_y = rt.cmc_y ?? 0.0;
      const rt_cmc_z = rt.cmc_z ?? 0.0;
      const rt_mcp_x = rt.mcp_x ?? 0.1;
      const rt_mcp_y = rt.mcp_y ?? (rt.cmc_y ? rt.cmc_y * 0.45 : 0.0);
      const rt_mcp_z = rt.mcp_z ?? (rt.cmc_z ? rt.cmc_z * 0.25 : 0.0);
      const rt_ip_x = rt.ip_x ?? 0.1;
      const rt_ip_z = rt.ip_z ?? 0.0;

      rf.thumb.cmc.rotation.x = THREE.MathUtils.lerp(rf.thumb.cmc.rotation.x, rt_cmc_x, lerpSpeed);
      rf.thumb.cmc.rotation.y = THREE.MathUtils.lerp(rf.thumb.cmc.rotation.y, rt_cmc_y, lerpSpeed);
      rf.thumb.cmc.rotation.z = THREE.MathUtils.lerp(rf.thumb.cmc.rotation.z, rt_cmc_z, lerpSpeed);

      rf.thumb.mcp.rotation.x = THREE.MathUtils.lerp(rf.thumb.mcp.rotation.x, -rt_mcp_x, lerpSpeed);
      rf.thumb.mcp.rotation.y = THREE.MathUtils.lerp(rf.thumb.mcp.rotation.y, rt_mcp_y, lerpSpeed);
      rf.thumb.mcp.rotation.z = THREE.MathUtils.lerp(rf.thumb.mcp.rotation.z, rt_mcp_z, lerpSpeed);

      rf.thumb.ip.rotation.x = THREE.MathUtils.lerp(rf.thumb.ip.rotation.x, -rt_ip_x, lerpSpeed);
      rf.thumb.ip.rotation.z = THREE.MathUtils.lerp(rf.thumb.ip.rotation.z, rt_ip_z, lerpSpeed);

      ['index', 'middle', 'ring', 'pinky'].forEach((name) => {
        const finger = rf[name];
        const target = cur.rightFingers[name];
        if (finger && target) {
          finger.mcp.rotation.x = THREE.MathUtils.lerp(finger.mcp.rotation.x, -target.mcp_x, lerpSpeed);
          finger.mcp.rotation.z = THREE.MathUtils.lerp(finger.mcp.rotation.z, target.spread_z, lerpSpeed);
          finger.pip.rotation.x = THREE.MathUtils.lerp(finger.pip.rotation.x, -target.pip_x, lerpSpeed);
          finger.dip.rotation.x = THREE.MathUtils.lerp(finger.dip.rotation.x, -target.dip_x, lerpSpeed);
        }
      });

      // 4. Cánh tay & Bàn tay trái
      rig.leftShoulder.rotation.x = THREE.MathUtils.lerp(rig.leftShoulder.rotation.x, cur.leftArm[0], lerpSpeed);
      rig.leftShoulder.rotation.y = THREE.MathUtils.lerp(rig.leftShoulder.rotation.y, cur.leftArm[1], lerpSpeed);
      rig.leftShoulder.rotation.z = THREE.MathUtils.lerp(rig.leftShoulder.rotation.z, cur.leftArm[2], lerpSpeed);

      rig.leftElbow.rotation.x = THREE.MathUtils.lerp(rig.leftElbow.rotation.x, cur.leftForearm[0], lerpSpeed);
      rig.leftElbow.rotation.y = THREE.MathUtils.lerp(rig.leftElbow.rotation.y, cur.leftForearm[1], lerpSpeed);
      rig.leftElbow.rotation.z = THREE.MathUtils.lerp(rig.leftElbow.rotation.z, cur.leftForearm[2], lerpSpeed);

      rig.leftWrist.rotation.x = THREE.MathUtils.lerp(rig.leftWrist.rotation.x, cur.leftWrist[0], lerpSpeed);
      rig.leftWrist.rotation.y = THREE.MathUtils.lerp(rig.leftWrist.rotation.y, cur.leftWrist[1], lerpSpeed);
      rig.leftWrist.rotation.z = THREE.MathUtils.lerp(rig.leftWrist.rotation.z, cur.leftWrist[2], lerpSpeed);

      // Ngón tay trái - Xoay 3 trục phản chiếu đối xứng hoàn hảo
      const lf = rig.leftHand.fingers;
      const lt = cur.leftFingers?.thumb || {};
      const lt_cmc_x = lt.cmc_x ?? 0.0;
      const lt_cmc_y = -(lt.cmc_y ?? 0.0);
      const lt_cmc_z = -(lt.cmc_z ?? 0.0);
      const lt_mcp_x = lt.mcp_x ?? 0.1;
      const lt_mcp_y = -(lt.mcp_y ?? (lt.cmc_y ? lt.cmc_y * 0.45 : 0.0));
      const lt_mcp_z = -(lt.mcp_z ?? (lt.cmc_z ? lt.cmc_z * 0.25 : 0.0));
      const lt_ip_x = lt.ip_x ?? 0.1;
      const lt_ip_z = -(lt.ip_z ?? 0.0);

      lf.thumb.cmc.rotation.x = THREE.MathUtils.lerp(lf.thumb.cmc.rotation.x, lt_cmc_x, lerpSpeed);
      lf.thumb.cmc.rotation.y = THREE.MathUtils.lerp(lf.thumb.cmc.rotation.y, lt_cmc_y, lerpSpeed);
      lf.thumb.cmc.rotation.z = THREE.MathUtils.lerp(lf.thumb.cmc.rotation.z, lt_cmc_z, lerpSpeed);

      lf.thumb.mcp.rotation.x = THREE.MathUtils.lerp(lf.thumb.mcp.rotation.x, -lt_mcp_x, lerpSpeed);
      lf.thumb.mcp.rotation.y = THREE.MathUtils.lerp(lf.thumb.mcp.rotation.y, lt_mcp_y, lerpSpeed);
      lf.thumb.mcp.rotation.z = THREE.MathUtils.lerp(lf.thumb.mcp.rotation.z, lt_mcp_z, lerpSpeed);

      lf.thumb.ip.rotation.x = THREE.MathUtils.lerp(lf.thumb.ip.rotation.x, -lt_ip_x, lerpSpeed);
      lf.thumb.ip.rotation.z = THREE.MathUtils.lerp(lf.thumb.ip.rotation.z, lt_ip_z, lerpSpeed);

      ['index', 'middle', 'ring', 'pinky'].forEach((name) => {
        const finger = lf[name];
        const target = cur.leftFingers[name];
        if (finger && target) {
          finger.mcp.rotation.x = THREE.MathUtils.lerp(finger.mcp.rotation.x, -target.mcp_x, lerpSpeed);
          finger.mcp.rotation.z = THREE.MathUtils.lerp(finger.mcp.rotation.z, -target.spread_z, lerpSpeed);
          finger.pip.rotation.x = THREE.MathUtils.lerp(finger.pip.rotation.x, -target.pip_x, lerpSpeed);
          finger.dip.rotation.x = THREE.MathUtils.lerp(finger.dip.rotation.x, -target.dip_x, lerpSpeed);
        }
      });

      // Nhịp thở tự nhiên
      if (!isPlaying) {
        torsoMesh.scale.x = 1 + Math.sin(elapsedTime * 1.5) * 0.012;
        headGroup.position.y = 0.75 + Math.sin(elapsedTime * 1.5) * 0.004;
      }

      renderer.render(scene, camera);
    };

    animate();

    return () => {
      cancelAnimationFrame(animFrameRef.current);
      container.removeEventListener('mousedown', handleMouseDown);
      window.removeEventListener('mousemove', handleMouseMove);
      window.removeEventListener('mouseup', handleMouseUp);
      renderer.dispose();
    };
  }, []);

  // Cập nhật mục tiêu chuyển động (Keyframe hoặc Static pose)
  useEffect(() => {
    const rig = rigRef.current;
    if (!rig || !rig.currentPose) return;

    const gloss = currentStep?.gloss || currentStep?.token;
    const poseData = getHandPoseByGloss(gloss);

    if (poseData.isDynamic && poseData.keyframes && poseData.keyframes.length > 0) {
      let keyframeIdx = 0;
      const runKeyframes = () => {
        if (!poseData.keyframes) return;
        const kf = poseData.keyframes[keyframeIdx];
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
        }
        keyframeIdx = (keyframeIdx + 1) % poseData.keyframes.length;
      };

      runKeyframes();
      const interval = setInterval(runKeyframes, (currentStep.duration_ms || 1800) / poseData.keyframes.length / speed);
      return () => clearInterval(interval);
    } else {
      // Cử chỉ tĩnh (Bảng chữ cái A-Z hoặc từ đơn)
      rig.currentPose.body = poseData.body || { torso: [0.02, 0, 0], head: [0.02, 0, 0] };
      rig.currentPose.rightArm = poseData.rightArm || [-1.15, 0.15, -0.08];
      rig.currentPose.rightForearm = poseData.rightForearm || [-1.10, 0.20, 0.08];
      rig.currentPose.rightWrist = poseData.rightWrist || [0.15, 0.10, 0];
      rig.currentPose.rightFingers = { ...(poseData.rightFingers || poseData) };

      rig.currentPose.leftArm = poseData.leftArm || [-0.22, -0.08, 0.05];
      rig.currentPose.leftForearm = poseData.leftForearm || [-0.35, -0.08, -0.05];
      rig.currentPose.leftWrist = poseData.leftWrist || [0, 0, 0];
      rig.currentPose.leftFingers = poseData.leftFingers ? { ...poseData.leftFingers } : NATURAL_FINGERS_REST;
    }
  }, [currentStep, speed]);

  // Các chế độ góc nhìn Camera
  const switchCameraMode = (mode) => {
    setCameraMode(mode);
    if (mode === 'hand_right') {
      targetCamPosRef.current.set(-0.35, 1.15, 1.25);
      targetCamLookRef.current.set(-0.20, 1.15, 0.35);
    } else if (mode === 'hand_left') {
      targetCamPosRef.current.set(0.35, 1.15, 1.25);
      targetCamLookRef.current.set(0.20, 1.15, 0.35);
    } else if (mode === 'hand_both') {
      targetCamPosRef.current.set(0.0, 1.20, 1.50);
      targetCamLookRef.current.set(0.0, 1.15, 0.25);
    } else {
      // Toàn cảnh 2 tay và cơ thể
      targetCamPosRef.current.set(0, 1.30, 2.35);
      targetCamLookRef.current.set(0, 1.05, 0);
    }
  };

  const resetRotation = () => {
    if (rigRef.current && rigRef.current.avatarGroup) {
      rigRef.current.avatarGroup.rotation.set(0, 0, 0);
    }
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '100%', minHeight: '430px' }}>
      <div 
        ref={mountRef} 
        style={{ width: '100%', height: '100%', cursor: 'grab' }}
        title="Nhấp và kéo chuột để xoay nhân vật 360 độ"
      />

      {/* Thanh công cụ Chuyển đổi Góc nhìn Camera */}
      <div style={{
        position: 'absolute',
        top: '12px',
        right: '12px',
        display: 'flex',
        gap: '6px',
        background: 'rgba(15, 23, 42, 0.88)',
        padding: '4px',
        borderRadius: 'var(--radius-md)',
        backdropFilter: 'blur(10px)',
        border: '1px solid var(--border-subtle)'
      }}>
        <button
          onClick={() => switchCameraMode('full')}
          className={`btn ${cameraMode === 'full' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '6px 10px', fontSize: '11px' }}
          title="Toàn cảnh 2 tay và cơ thể"
        >
          <Eye size={13} /> Toàn thân 2 tay
        </button>

        <button
          onClick={() => switchCameraMode('hand_both')}
          className={`btn ${cameraMode === 'hand_both' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '6px 10px', fontSize: '11px' }}
          title="Cận cảnh trực diện cả 2 bàn tay"
        >
          <Crosshair size={13} /> Cận cảnh 2 tay
        </button>

        <button
          onClick={() => switchCameraMode('hand_right')}
          className={`btn ${cameraMode === 'hand_right' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '6px 10px', fontSize: '11px' }}
          title="Phóng to soi từng đốt ngón tay phải"
        >
          <ZoomIn size={13} /> Soi tay phải
        </button>

        <button
          onClick={() => switchCameraMode('hand_left')}
          className={`btn ${cameraMode === 'hand_left' ? 'btn-primary' : 'btn-secondary'}`}
          style={{ padding: '6px 10px', fontSize: '11px' }}
          title="Phóng to soi từng đốt ngón tay trái"
        >
          <ZoomIn size={13} /> Soi tay trái
        </button>

        <button
          onClick={resetRotation}
          className="btn btn-secondary"
          style={{ padding: '6px 10px', fontSize: '11px' }}
          title="Khôi phục góc nhìn thẳng ban đầu"
        >
          <RotateCcw size={13} />
        </button>
      </div>

      {/* Gợi ý tương tác */}
      <div style={{
        position: 'absolute',
        bottom: '12px',
        left: '50%',
        transform: 'translateX(-50%)',
        fontSize: '11px',
        color: 'var(--text-muted)',
        background: 'rgba(15, 23, 42, 0.75)',
        padding: '3px 12px',
        borderRadius: '20px',
        pointerEvents: 'none',
        backdropFilter: 'blur(6px)',
        border: '1px solid var(--border-subtle)',
        whiteSpace: 'nowrap'
      }}>
        👐 Diễn hoạt 2 tay & Toàn thân (Head/Torso/FK 30 khớp ngón) • Kéo chuột xoay 360°
      </div>
    </div>
  );
}
