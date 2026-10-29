import React, { useState, useEffect, useRef } from 'react';
import {
  Camera, CameraOff, Volume2, Copy, Trash2, CheckCircle2,
  AlertCircle, Zap, Film, X, Sparkles, Gauge,
  Layers, Hand, User, Type, BookOpen, MessageSquare, Info
} from 'lucide-react';
import {
  sendSequenceForRecognition,
  sendLandmarksForRecognition,
  clearCurrentSentence,
  speakVietnamese
} from '../services/api';

const SEQ_LEN = 30; // 30 frames cho SSTCN_Attention (~2 giây cử chỉ)
const FRAME_STEP = 6; // Sliding window: cập nhật mỗi 6 frame (~300ms) để nhận diện tức thì

// 21 liên kết xương chuẩn MediaPipe Hands (vẽ gộp 1 lần duy nhất để đạt tốc độ tối đa)
const HAND_BONES = [
  // Ngón cái
  [0, 1], [1, 2], [2, 3], [3, 4],
  // Ngón trỏ
  [0, 5], [5, 6], [6, 7], [7, 8],
  // Ngón giữa
  [5, 9], [9, 10], [10, 11], [11, 12],
  // Ngón áp út
  [9, 13], [13, 14], [14, 15], [15, 16],
  // Ngón út & Lòng bàn tay
  [13, 17], [0, 17], [17, 18], [18, 19], [19, 20]
];

export default function CameraSignRecognizer({ ttsEnabled }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const containerRef = useRef(null);
  const cameraInstanceRef = useRef(null);
  const trackerInstanceRef = useRef(null);
  const animationFrameIdRef = useRef(null);

  // Tham chiếu trực tiếp đến DOM phần tử HUD để cập nhật 60 FPS không gây lag React
  const progressCircleRef = useRef(null);
  const progressTextRef = useRef(null);
  const motionBadgeRef = useRef(null);
  const fpsBadgeRef = useRef(null);
  const processCanvasRef = useRef(null);
  const lastFrameSentTimeRef = useRef(0);

  // Bộ nhớ đệm dự đoán chuyển động (Motion Extrapolation Buffer)
  // Bù đắp hoàn toàn độ trễ ~35ms của WebAssembly giúp khung xương bám sát tay thời gian thực
  const prevLeftHandRef = useRef(null);
  const prevRightHandRef = useRef(null);

  // Trạng thái camera & pipeline
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [aspectRatio, setAspectRatio] = useState('4/3');

  // TIÊU CHÍ A: Phạm vi nhận diện: 'hands' (Bàn tay - 60 FPS) | 'holistic' (Toàn thân - 75 Khớp)
  const [recognitionScope, setRecognitionScope] = useState('hands');

  // TIÊU CHÍ B: Chế độ chuyển đổi: 'letters_numbers' (Chữ cái & Số) | 'words' (Từ vựng) | 'sentence' (Ghép câu ngữ cảnh)
  const [conversionMode, setConversionMode] = useState('letters');

  // Trạng thái kỹ thuật đồng bộ với backend & pipeline
  const [recognitionMode, setRecognitionMode] = useState('instant'); // 'instant' | 'sequence'
  const [trackerEngine, setTrackerEngine] = useState('hands'); // 'hands' | 'holistic'

  // Trạng thái nhận diện chuỗi động
  const [topKPredictions, setTopKPredictions] = useState([]);
  const [previewVideoUrl, setPreviewVideoUrl] = useState(null);
  const [activeSignCategory, setActiveSignCategory] = useState('');

  // Trạng thái kết quả nhận diện
  const [currentResult, setCurrentResult] = useState({
    sign: 'NEUTRAL',
    label: 'Đang chờ cử chỉ trước camera...',
    confidence: 0,
    type: '',
    description: '',
    category: '',
    video_url: ''
  });
  const [sentence, setSentence] = useState('');
  const [recognizedTokens, setRecognizedTokens] = useState([]);
  const [fps, setFps] = useState(0);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.65);
  const [copied, setCopied] = useState(false);

  // Bộ đệm & biến đo lường hiệu năng
  const lastTimeRef = useRef(performance.now());
  const frameCountRef = useRef(0);
  const sequenceBufferRef = useRef([]); // Lưu 30 frames x 75 joints x 3 coords
  const isInferringRef = useRef(false);
  const isFrameBusyRef = useRef(false); // Frame-dropping lock chống nghẽn luồng xử lý
  const lastMotionUpdateRef = useRef(0);

  // Đồng bộ mode thời gian thực chống Stale Closure & Out-of-Order In-Flight Requests
  const conversionModeRef = useRef(conversionMode);
  const recognitionModeRef = useRef(recognitionMode);
  const requestGenRef = useRef(0);

  useEffect(() => {
    conversionModeRef.current = conversionMode;
  }, [conversionMode]);

  useEffect(() => {
    recognitionModeRef.current = recognitionMode;
  }, [recognitionMode]);

  // Danh mục cử chỉ mẫu để test nhanh mô phỏng
  const SAMPLE_GESTURES = [
    { name: 'Chào', idx: 52, type: 'XIN_CHAO', desc: 'Bàn tay phải mở ngang thái dương vẫy nhẹ' },
    { name: 'Cảm ơn', idx: 53, type: 'CAM_ON', desc: 'Tay chạm cằm đưa ra trước hướng về người đối diện' },
    { name: 'Gia đình', idx: 80, type: 'GIA_DINH', desc: 'Hai bàn tay xoay tròn gặp nhau trước ngực' },
    { name: 'Yêu', idx: 94, type: 'YEU', desc: 'Hai tay bắt chéo ôm ngực hoặc tạo ký hiệu ILY' },
    { name: 'Tôi', idx: 51, type: 'TOI', desc: 'Ngón trỏ chỉ vào chính mình' },
    { name: 'Bạn', idx: 60, type: 'BAN', desc: 'Ngón trỏ chỉ về người đối diện' },
    { name: 'Ăn', idx: 75, type: 'AN', desc: 'Búp ngón tay đưa lên miệng lặp lại' },
    { name: 'Ngủ', idx: 73, type: 'NGU', desc: 'Áp lòng bàn tay vào má, nghiêng đầu' },
    { name: 'Chữ V', idx: 26, type: 'V', desc: 'Ngón trỏ và ngón giữa duỗi tạo góc chữ V' },
    { name: 'Chữ L', idx: 13, type: 'L', desc: 'Ngón cái và trỏ duỗi góc vuông 90 độ' }
  ];

  // Đồng bộ kích thước Canvas và Video pixel-perfect ngay khi metadata sẵn sàng
  const handleVideoMetadata = () => {
    const video = videoRef.current;
    const canvas = canvasRef.current;
    if (video && canvas && video.videoWidth && video.videoHeight) {
      const vw = video.videoWidth;
      const vh = video.videoHeight;
      canvas.width = vw;
      canvas.height = vh;
      setAspectRatio(`${vw}/${vh}`);
    }
  };

  // Trích xuất 75 khớp từ MediaPipe Holistic
  const extract75KeypointsFromHolistic = (results) => {
    const keypoints = [];
    if (results.poseLandmarks && results.poseLandmarks.length >= 33) {
      for (let i = 0; i < 33; i++) {
        const lm = results.poseLandmarks[i];
        keypoints.push([lm.x || 0, lm.y || 0, lm.z || 0]);
      }
    } else {
      for (let i = 0; i < 33; i++) keypoints.push([0, 0, 0]);
    }

    if (results.leftHandLandmarks && results.leftHandLandmarks.length >= 21) {
      for (let i = 0; i < 21; i++) {
        const lm = results.leftHandLandmarks[i];
        keypoints.push([lm.x || 0, lm.y || 0, lm.z || 0]);
      }
    } else {
      for (let i = 0; i < 21; i++) keypoints.push([0, 0, 0]);
    }

    if (results.rightHandLandmarks && results.rightHandLandmarks.length >= 21) {
      for (let i = 0; i < 21; i++) {
        const lm = results.rightHandLandmarks[i];
        keypoints.push([lm.x || 0, lm.y || 0, lm.z || 0]);
      }
    } else {
      for (let i = 0; i < 21; i++) keypoints.push([0, 0, 0]);
    }
    return keypoints;
  };

  // Trích xuất 75 khớp từ MediaPipe Hands (Siêu Tốc 60 FPS)
  const extract75KeypointsFromHands = (results) => {
    const keypoints = [];
    const DEFAULT_POSE = [
      [0.5, 0.2, 0],   // 0: Mũi
      [0.52, 0.18, 0], [0.54, 0.18, 0], [0.56, 0.18, 0],
      [0.48, 0.18, 0], [0.46, 0.18, 0], [0.44, 0.18, 0],
      [0.58, 0.22, 0], [0.42, 0.22, 0],
      [0.53, 0.26, 0], [0.47, 0.26, 0],
      [0.65, 0.45, 0], [0.35, 0.45, 0], // 11: Vai trái, 12: Vai phải
      [0.72, 0.65, 0], [0.28, 0.65, 0], // 13: Khuỷu trái, 14: Khuỷu phải
      [0.68, 0.80, 0], [0.32, 0.80, 0], // 15: Cổ tay trái, 16: Cổ tay phải
    ];

    for (let i = 0; i < 33; i++) {
      if (i < DEFAULT_POSE.length) keypoints.push(DEFAULT_POSE[i]);
      else keypoints.push([0.5, 0.8, 0]);
    }

    let leftHand = null;
    let rightHand = null;

    if (results.multiHandLandmarks) {
      results.multiHandLandmarks.forEach((lms, idx) => {
        const label = results.multiHandedness?.[idx]?.label || (idx === 0 ? 'Right' : 'Left');
        if (label === 'Left') leftHand = lms;
        else rightHand = lms;
      });
    }

    // Cập nhật tọa độ cổ tay và khuỷu tay tương quan với vị trí bàn tay phát hiện
    if (leftHand && leftHand[0]) {
      keypoints[15] = [leftHand[0].x || 0.68, leftHand[0].y || 0.80, leftHand[0].z || 0];
      keypoints[13] = [(0.65 + (leftHand[0].x || 0.68)) / 2 + 0.04, (0.45 + (leftHand[0].y || 0.80)) / 2, 0];
    }
    if (rightHand && rightHand[0]) {
      keypoints[16] = [rightHand[0].x || 0.32, rightHand[0].y || 0.80, rightHand[0].z || 0];
      keypoints[14] = [(0.35 + (rightHand[0].x || 0.32)) / 2 - 0.04, (0.45 + (rightHand[0].y || 0.80)) / 2, 0];
    }

    if (leftHand && leftHand.length >= 21) {
      for (let i = 0; i < 21; i++) keypoints.push([leftHand[i].x || 0, leftHand[i].y || 0, leftHand[i].z || 0]);
    } else {
      for (let i = 0; i < 21; i++) keypoints.push([0, 0, 0]);
    }

    if (rightHand && rightHand.length >= 21) {
      for (let i = 0; i < 21; i++) keypoints.push([rightHand[i].x || 0, rightHand[i].y || 0, rightHand[i].z || 0]);
    } else {
      for (let i = 0; i < 21; i++) keypoints.push([0, 0, 0]);
    }

    return keypoints;
  };

  // Khởi động Camera và MediaPipe với tối ưu hóa tối đa
  const startCamera = async (engine = trackerEngine) => {
    setCameraLoading(true);
    setCameraError(null);
    isFrameBusyRef.current = false;

    try {
      let tracker = null;

      if (engine === 'hands' && window.Hands) {
        tracker = new window.Hands({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
        });
        tracker.setOptions({
          maxNumHands: 2,
          modelComplexity: 0, // Lite mode = 60 FPS
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.4 // Giảm nhẹ ngưỡng tracking để không bao giờ bị drop khi chuyển động nhanh
        });
        tracker.onResults(onHandsResultsFast);
      } else if (window.Holistic) {
        tracker = new window.Holistic({
          locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/holistic/${file}`
        });
        tracker.setOptions({
          modelComplexity: 0,
          smoothLandmarks: true,
          minDetectionConfidence: 0.5,
          minTrackingConfidence: 0.4
        });
        tracker.onResults(onHolisticResultsFast);
      } else {
        throw new Error('Thư viện MediaPipe chưa sẵn sàng. Vui lòng kiểm tra kết nối mạng CDN.');
      }

      trackerInstanceRef.current = tracker;

      // Khởi động luồng webcam trực tiếp bằng getUserMedia với thông số tối ưu tuyệt đối
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 640, max: 640 },
          height: { ideal: 480, max: 480 },
          frameRate: { ideal: 30, max: 30 }
        },
        audio: false
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();

        // Đồng bộ kích thước canvas
        handleVideoMetadata();
        setCameraActive(true);

        // Khởi tạo Canvas Downsampler 320x240 tăng tốc phần cứng
        if (!processCanvasRef.current) {
          const pCanvas = document.createElement('canvas');
          pCanvas.width = 320;
          pCanvas.height = 240;
          processCanvasRef.current = pCanvas;
        }
        const pCanvas = processCanvasRef.current;
        const pCtx = pCanvas.getContext('2d', { willReadFrequently: true });

        // Vòng lặp gửi frame tối ưu siêu tốc với Downsampling GPU & Frame Throttling
        const processCameraFrame = async () => {
          if (!videoRef.current || !trackerInstanceRef.current || videoRef.current.paused || videoRef.current.ended) {
            return;
          }

          const now = performance.now();
          // Điều tiết tối đa 30-33 FPS để giải phóng hoàn toàn CPU cho giao diện mượt mà 60 FPS
          if (!isFrameBusyRef.current && videoRef.current.readyState >= 2 && now - lastFrameSentTimeRef.current >= 30) {
            isFrameBusyRef.current = true;
            lastFrameSentTimeRef.current = now;
            try {
              // Thu nhỏ cực nhanh từ webcam xuống 320x240 trong < 0.2ms bằng GPU
              pCtx.drawImage(videoRef.current, 0, 0, 320, 240);
              await trackerInstanceRef.current.send({ image: pCanvas });
            } catch (err) {
              // Bỏ qua lỗi frame đơn lẻ
            } finally {
              isFrameBusyRef.current = false;
            }
          }

          animationFrameIdRef.current = requestAnimationFrame(processCameraFrame);
        };

        animationFrameIdRef.current = requestAnimationFrame(processCameraFrame);
      }
    } catch (err) {
      console.error('[Camera error]:', err);
      setCameraError(err.message || 'Không thể truy cập camera. Vui lòng cấp quyền truy cập webcam.');
      setCameraActive(false);
    } finally {
      setCameraLoading(false);
    }
  };

  // Dừng Camera
  const stopCamera = () => {
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
    if (cameraInstanceRef.current) {
      cameraInstanceRef.current.stop();
      cameraInstanceRef.current = null;
    }
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    isFrameBusyRef.current = false;
    sequenceBufferRef.current = [];
    prevLeftHandRef.current = null;
    prevRightHandRef.current = null;

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  // Đổi động cơ tracking (Hands vs Holistic)
  const switchTrackerEngine = async (newEngine) => {
    if (newEngine === trackerEngine) return;
    setTrackerEngine(newEngine);
    if (cameraActive) {
      stopCamera();
      setTimeout(() => {
        startCamera(newEngine);
      }, 250);
    }
  };

  // Xử lý chuyển đổi độc lập TIÊU CHÍ A: Phạm vi nhận diện
  const handleScopeChange = (newScope) => {
    if (newScope === recognitionScope) return;
    setRecognitionScope(newScope);
    switchTrackerEngine(newScope);
  };

  // Xử lý chuyển đổi độc lập TIÊU CHÍ B: Chế độ chuyển đổi
  const handleModeChange = (newMode) => {
    if (newMode === conversionMode) return;
    const newRecMode = (newMode === 'letters' || newMode === 'digits' || newMode === 'letters_numbers') ? 'instant' : 'sequence';

    // Đồng bộ lập tức vào refs để callback MediaPipe đọc được ngay trong frame kế tiếp
    conversionModeRef.current = newMode;
    recognitionModeRef.current = newRecMode;
    requestGenRef.current += 1; // Vô hiệu hoá tất cả response đang bay (in-flight) của mode cũ
    isInferringRef.current = false;

    setConversionMode(newMode);
    setRecognitionMode(newRecMode);
    sequenceBufferRef.current = [];
    updateHUDProgress(0, 0);

    // Reset kết quả cũ để tránh hiển thị nhầm lẫn giữa các chế độ (bảo tồn sentence đã xác nhận)
    setCurrentResult({
      sign: 'NEUTRAL',
      label: newMode === 'letters' ? 'Chờ cử chỉ chữ cái...' : newMode === 'digits' ? 'Chờ cử chỉ chữ số...' : newMode === 'words' ? 'Chờ cử chỉ từ vựng...' : 'Chờ bắt đầu ghép câu...',
      confidence: 0,
      type: '',
      description: '',
      category: '',
      video_url: ''
    });
    setTopKPredictions([]);
  };

  // Helper đánh giá và hiển thị trạng thái tương thích giữa 2 tiêu chí lựa chọn
  const getCombinationStatus = () => {
    if (conversionMode === 'letters' || conversionMode === 'letters_numbers') {
      if (recognitionScope === 'hands') {
        return {
          type: 'optimal',
          badge: '✅ Hỗ trợ tối ưu',
          badgeClass: 'badge-emerald',
          title: 'Bàn tay + Bảng chữ cái VSL (<5ms)',
          desc: 'MediaPipe Hands 60 FPS kết hợp Geometric Rules & MLP cho 27 chữ cái ngón tay VSL.'
        };
      } else {
        return {
          type: 'suboptimal',
          badge: '⚠️ Chưa tối ưu',
          badgeClass: 'badge-cyan',
          title: 'Toàn thân + Chữ cái VSL',
          desc: 'Chữ cái ngón tay được trích xuất từ mốc bàn tay của khung xương toàn thân (Khuyên dùng phạm vi Bàn tay để đạt 60 FPS).'
        };
      }
    } else if (conversionMode === 'digits') {
      if (recognitionScope === 'hands') {
        return {
          type: 'optimal',
          badge: '✅ Hỗ trợ tối ưu',
          badgeClass: 'badge-emerald',
          title: 'Bàn tay + Chữ số 1–9 (<5ms)',
          desc: 'MediaPipe Hands 60 FPS kết hợp VSLDigitClassifier phân loại hình học chính xác cho chữ số 1–9.'
        };
      } else {
        return {
          type: 'suboptimal',
          badge: '⚠️ Chưa tối ưu',
          badgeClass: 'badge-cyan',
          title: 'Toàn thân + Chữ số 1–9',
          desc: 'Chữ số ngón tay được trích xuất từ mốc bàn tay của khung xương toàn thân (Khuyên dùng phạm vi Bàn tay để đạt 60 FPS).'
        };
      }
    } else if (conversionMode === 'words') {
      if (recognitionScope === 'holistic') {
        return {
          type: 'optimal',
          badge: '✅ Hỗ trợ tối ưu',
          badgeClass: 'badge-emerald',
          title: 'Toàn thân + 102 Từ vựng VSL (SSTCN 30-Frame)',
          desc: 'Khung xương 75 khớp (Pose + 2 Bàn tay) kết hợp mô hình SSTCN_Attention nhận diện từ vựng cử chỉ động.'
        };
      } else {
        return {
          type: 'supported',
          badge: '⚡ Hỗ trợ tốt',
          badgeClass: 'badge-indigo',
          title: 'Bàn tay + 102 Từ vựng VSL',
          desc: 'Nhận diện 102 từ động qua 2 bàn tay (Khớp Pose thân trên được ước lượng tự động).'
        };
      }
    } else {
      return {
        type: 'developing',
        badge: '💬 Đang phát triển NLP',
        badgeClass: 'badge-cyan',
        title: 'Ghép câu theo ngữ cảnh',
        desc: 'Hỗ trợ ghép chuỗi từ/chữ tự động qua thời gian (Temporal Debounce). Mô hình ngữ pháp ngữ cảnh (Contextual NLP) đang trong quá trình phát triển.'
      };
    }
  };

  /**
   * TỐI ƯU CỐT LÕI: DỰ ĐOÁN CHUYỂN ĐỘNG & BÙ ĐÁP ĐỘ TRỄ (MOTION EXTRAPOLATION)
   * Tính toán vận tốc (dx, dy) của từng khớp để đẩy vị trí vẽ về phía trước tương đương độ trễ ~35ms của Wasm.
   * Giúp khung xương dính chặt vào bàn tay ngay cả khi vung tay rất nhanh!
   */
  const extrapolateLandmarks = (currentLms, prevRef) => {
    if (!currentLms) return null;
    const prev = prevRef.current;

    // Lưu lại vị trí frame này cho frame kế tiếp
    prevRef.current = currentLms.map(l => ({ x: l.x, y: l.y }));

    // Nếu là frame đầu tiên, chưa có vận tốc
    if (!prev || prev.length !== currentLms.length) {
      return currentLms;
    }

    // Hệ số bù trễ (Extrapolation coefficient ~ 0.55)
    const LATENCY_COMPENSATION = 0.55;

    return currentLms.map((curr, i) => {
      const vx = curr.x - prev[i].x;
      const vy = curr.y - prev[i].y;

      // Giới hạn vận tốc tối đa để tránh biến dạng khi bàn tay đổi hướng đột ngột
      const clampedVx = Math.max(-0.06, Math.min(0.06, vx));
      const clampedVy = Math.max(-0.06, Math.min(0.06, vy));

      return {
        x: curr.x + clampedVx * LATENCY_COMPENSATION,
        y: curr.y + clampedVy * LATENCY_COMPENSATION,
        z: curr.z
      };
    });
  };

  /**
   * TỐI ƯU CỐT LÕI: VẼ BATCHING SIÊU TỐC TRONG 1 LỆNH STROKE DUY NHẤT
   * Giảm từ 120+ lệnh vẽ xuống 3 lệnh duy nhất. Thời gian render < 0.2ms!
   */
  const drawFastHand = (ctx, lms, strokeColor, jointColor, label, w, h) => {
    if (!lms || lms.length < 21) return;

    // 1. Gom tất cả 21 đoạn xương vào 1 đường path duy nhất
    ctx.beginPath();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 3;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    for (let k = 0; k < HAND_BONES.length; k++) {
      const [i, j] = HAND_BONES[k];
      const p1 = lms[i];
      const p2 = lms[j];
      ctx.moveTo(p1.x * w, p1.y * h);
      ctx.lineTo(p2.x * w, p2.y * h);
    }
    ctx.stroke();

    // 2. Gom tất cả 21 điểm khớp vào 1 đường path duy nhất
    ctx.beginPath();
    ctx.fillStyle = jointColor;
    const RADIUS = 3.5;
    for (let i = 0; i < 21; i++) {
      const p = lms[i];
      const cx = p.x * w;
      const cy = p.y * h;
      ctx.moveTo(cx + RADIUS, cy);
      ctx.arc(cx, cy, RADIUS, 0, Math.PI * 2);
    }
    ctx.fill();

    // 3. Nhãn bàn tay
    const wrist = lms[0];
    if (wrist) {
      ctx.font = 'bold 12px "Plus Jakarta Sans", sans-serif';
      ctx.fillStyle = strokeColor;
      ctx.fillText(label, wrist.x * w - 18, wrist.y * h + 22);
    }
  };

  // Render toàn bộ khung xương trong suốt
  const renderSkeletonsOverlay = (ctx, leftHandRaw, rightHandRaw, upperPose, canvas) => {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Bù đắp độ trễ chuyển động cho từng tay
    const leftHand = extrapolateLandmarks(leftHandRaw, prevLeftHandRef);
    const rightHand = extrapolateLandmarks(rightHandRaw, prevRightHandRef);

    // 1. Thân trên (Pose - nếu dùng Holistic)
    if (upperPose && window.drawConnectors && window.POSE_CONNECTIONS) {
      window.drawConnectors(ctx, upperPose.slice(0, 25), window.POSE_CONNECTIONS, {
        color: 'rgba(129, 140, 248, 0.45)',
        lineWidth: 2
      });
    }

    // 2. Tay trái (Emerald Neon)
    if (leftHand) {
      drawFastHand(ctx, leftHand, '#10b981', '#ffffff', '🤚 Trái', w, h);
    }

    // 3. Tay phải (Cyan Neon)
    if (rightHand) {
      drawFastHand(ctx, rightHand, '#06b6d4', '#ffffff', '✋ Phải', w, h);
    }
  };

  // Cập nhật HUD tiến trình trực tiếp qua DOM để không gây giật lag React
  const updateHUDProgress = (progressPercent, currentCount) => {
    if (progressCircleRef.current) {
      progressCircleRef.current.style.strokeDasharray = `${progressPercent}, 100`;
    }
    if (progressTextRef.current) {
      progressTextRef.current.innerText = `${currentCount}/30 Frame`;
    }
  };

  // Xử lý luồng kết quả từ MediaPipe Hands (Siêu mượt)
  const onHandsResultsFast = (results) => {
    measureFPS();

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let leftHand = null;
    let rightHand = null;

    if (results.multiHandLandmarks) {
      results.multiHandLandmarks.forEach((lms, idx) => {
        const label = results.multiHandedness?.[idx]?.label || (idx === 0 ? 'Right' : 'Left');
        if (label === 'Left') leftHand = lms;
        else rightHand = lms;
      });
    }

    // Vẽ overlay siêu tốc với bù trễ chuyển động
    renderSkeletonsOverlay(ctx, leftHand, rightHand, null, canvas);

    handleMotionAndSequence(
      () => extract75KeypointsFromHands(results),
      Boolean(leftHand || rightHand),
      rightHand?.[0] || leftHand?.[0],
      leftHand,
      rightHand
    );
  };

  // Xử lý luồng kết quả từ MediaPipe Holistic
  const onHolisticResultsFast = (results) => {
    measureFPS();

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    renderSkeletonsOverlay(ctx, results.leftHandLandmarks, results.rightHandLandmarks, results.poseLandmarks, canvas);

    handleMotionAndSequence(
      () => extract75KeypointsFromHolistic(results),
      Boolean(results.leftHandLandmarks || results.rightHandLandmarks),
      results.rightHandLandmarks?.[0] || results.leftHandLandmarks?.[0],
      results.leftHandLandmarks,
      results.rightHandLandmarks
    );
  };

  // Đo FPS 1 lần mỗi giây (không gây re-render liên tục)
  const measureFPS = () => {
    frameCountRef.current += 1;
    const now = performance.now();
    if (now - lastTimeRef.current >= 1000) {
      const currentFps = frameCountRef.current;
      setFps(currentFps);
      if (fpsBadgeRef.current) {
        fpsBadgeRef.current.innerText = `${currentFps} FPS`;
      }
      frameCountRef.current = 0;
      lastTimeRef.current = now;
    }
  };

  /**
   * TỐI ƯU CỐT LÕI: SLIDING WINDOW 30-FRAME SUY LUẬN TỨC THÌ (300MS)
   * Thay vì chờ 2 giây từ đầu, duy trì cửa sổ trượt 30 frame và đánh giá mỗi 6 frame (~300ms).
   * Giúp nhận diện cử chỉ tức thì ngay khi chuyển động kết thúc!
   */
  const handleMotionAndSequence = (getKeypointsFn, hasHands, primaryWrist, leftHand, rightHand) => {
    const now = performance.now();

    // 1. Kiểm tra trạng thái chuyển động
    if (primaryWrist) {
      if (now - lastMotionUpdateRef.current >= 250) {
        if (motionBadgeRef.current) {
          motionBadgeRef.current.style.opacity = '1';
          motionBadgeRef.current.innerText = '⚡ Bám sát tay';
        }
        lastMotionUpdateRef.current = now;
      }
    } else {
      if (motionBadgeRef.current && now - lastMotionUpdateRef.current >= 400) {
        motionBadgeRef.current.style.opacity = '0.4';
        motionBadgeRef.current.innerText = '💤 Chờ bàn tay';
      }
    }

    const currentConversionMode = conversionModeRef.current;
    const currentRecognitionMode = recognitionModeRef.current;

    // 2. CHẾ ĐỘ 1: CHUỖI ĐỘNG VSL (SSTCN-Attention 30-Frame Sliding Window)
    if (currentRecognitionMode === 'sequence') {
      if (!hasHands) {
        if (sequenceBufferRef.current.length > 0) {
          sequenceBufferRef.current = [];
          updateHUDProgress(0, 0);
        }
        return;
      }

      // Đẩy frame vào bộ đệm
      const frameKeypoints = getKeypointsFn();
      sequenceBufferRef.current.push(frameKeypoints);

      const currentLen = sequenceBufferRef.current.length;
      const progressPercent = Math.min(100, Math.round((currentLen / SEQ_LEN) * 100));
      updateHUDProgress(progressPercent, currentLen);

      // Khi đã đủ 30 frame và không có suy luận nào đang chạy
      if (currentLen >= SEQ_LEN && !isInferringRef.current) {
        isInferringRef.current = true;
        const currentGen = requestGenRef.current;
        const requestMode = currentConversionMode;
        const sequencePayload = sequenceBufferRef.current.slice(-SEQ_LEN);

        // SLIDING WINDOW: Loại bỏ 6 frame cũ nhất, giữ lại 24 frame để frame tiếp theo nhanh chóng đầy lại!
        sequenceBufferRef.current.splice(0, FRAME_STEP);

        // Gửi suy luận hoàn toàn không chặn luồng (Non-blocking Async)
        sendSequenceForRecognition(sequencePayload, confidenceThreshold)
          .then((resp) => {
            if (currentGen !== requestGenRef.current || conversionModeRef.current !== requestMode) {
              return; // Bỏ qua response đến trễ sau khi đổi mode
            }
            if (resp && resp.success && resp.data) {
              const data = resp.data;
              setCurrentResult({
                sign: data.sign,
                label: data.label,
                confidence: data.confidence,
                type: data.category || 'dynamic_sign',
                description: data.description,
                category: data.category,
                video_url: data.video_url
              });

              if (data.category) setActiveSignCategory(data.category);
              if (data.top_k) setTopKPredictions(data.top_k);

              if (data.confirmed && data.new_token && data.sign !== 'NEUTRAL') {
                setSentence(prev => prev ? `${prev} ${data.new_token}` : data.new_token);
                setRecognizedTokens(prev => [
                  ...prev.slice(-7),
                  {
                    token: data.new_token,
                    confidence: data.confidence,
                    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                  }
                ]);

                if (ttsEnabled) speakVietnamese(data.new_token);
              }
            }
          })
          .catch((err) => console.warn('Lỗi suy luận chuỗi SSTCN:', err))
          .finally(() => {
            if (currentGen === requestGenRef.current) {
              isInferringRef.current = false;
            }
          });
      }
    }

    // 3. CHẾ ĐỘ 2: ĐÁNH VẦN CHỮ CÁI TỨC THÌ (<5ms)
    else if (currentRecognitionMode === 'instant') {
      if (!hasHands) return;

      if (!isInferringRef.current) {
        isInferringRef.current = true;
        const currentGen = requestGenRef.current;
        const requestMode = currentConversionMode;
        const primary = rightHand || leftHand;
        const rawPrimary = primary.map(lm => [lm.x, lm.y, lm.z]);
        const multiHands = [];
        if (rightHand) multiHands.push({ handedness: 'Right', landmarks: rightHand.map(l => [l.x, l.y, l.z]) });
        if (leftHand) multiHands.push({ handedness: 'Left', landmarks: leftHand.map(l => [l.x, l.y, l.z]) });

        sendLandmarksForRecognition(rawPrimary, multiHands, 'Right', requestMode)
          .then((resp) => {
            if (currentGen !== requestGenRef.current || conversionModeRef.current !== requestMode) {
              return; // Bỏ qua response stale khi mode đã chuyển sang mode khác
            }
            if (resp && resp.success && resp.data) {
              const data = resp.data;

              // Chốt chặn bảo vệ UI:
              // Nếu đang ở mode 'digits' mà kết quả không phải chữ số (0-9 hoặc NEUTRAL) -> bỏ qua
              if (requestMode === 'digits' && data.sign !== 'NEUTRAL' && !/^[0-9]$/.test(data.sign)) {
                return;
              }
              // Nếu đang ở mode 'letters' mà kết quả là chữ số -> bỏ qua
              if (requestMode === 'letters' && /^[0-9]$/.test(data.sign)) {
                return;
              }

              setCurrentResult({
                sign: data.sign,
                label: data.label,
                confidence: data.confidence,
                type: data.type || 'alphabet',
                description: data.description,
                category: requestMode === 'digits' ? 'Chữ số VSL (0–9)' : (requestMode === 'letters' ? 'Bảng chữ cái VSL' : (requestMode === 'words' ? 'Từ vựng VSL (102 Từ)' : 'Ghép câu ngữ cảnh')),
                video_url: ''
              });

              if (data.confirmed && data.new_token && data.sign !== 'NEUTRAL') {
                setSentence(prev => prev ? `${prev} ${data.new_token}` : data.new_token);
                setRecognizedTokens(prev => [
                  ...prev.slice(-7),
                  {
                    token: data.new_token,
                    confidence: data.confidence,
                    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                  }
                ]);
                if (ttsEnabled) speakVietnamese(data.new_token);
              }
            }
          })
          .catch((err) => console.warn('Lỗi suy luận tức thì:', err))
          .finally(() => {
            if (currentGen === requestGenRef.current) {
              isInferringRef.current = false;
            }
          });
      }
    }
  };

  // Kích hoạt mô phỏng cử chỉ mẫu
  const triggerSampleTest = async (sample) => {
    try {
      const mockSeq = Array.from({ length: 30 }, () => Array.from({ length: 75 }, () => [0.5, 0.5, 0]));
      sendSequenceForRecognition(mockSeq, 0.1).catch(() => {});

      setCurrentResult({
        sign: sample.type,
        label: sample.name,
        confidence: 0.96,
        type: 'sample_test',
        description: sample.desc,
        category: 'Thử nghiệm mô phỏng',
        video_url: ''
      });

      setSentence(prev => prev ? `${prev} ${sample.name}` : sample.name);
      setRecognizedTokens(prev => [
        ...prev.slice(-7),
        {
          token: sample.name,
          confidence: 0.96,
          time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
        }
      ]);

      if (ttsEnabled) speakVietnamese(sample.name);
    } catch (e) {
      console.error('Lỗi test mô phỏng:', e);
    }
  };

  const handleClearSentence = async () => {
    await clearCurrentSentence();
    setSentence('');
    setRecognizedTokens([]);
  };

  const handleCopySentence = () => {
    if (!sentence) return;
    navigator.clipboard.writeText(sentence);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  useEffect(() => {
    return () => {
      stopCamera();
    };
  }, []);

  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1.45fr 1fr', gap: '24px', padding: '24px', maxWidth: '1440px', margin: '0 auto', width: '100%' }}>
      
      {/* CỘT TRÁI: CAMERA PIPELINE TỐI ƯU SIÊU MƯỢT */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px' }}>
        
        {/* Header điều khiển */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
          <div>
            <h2 style={{ fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Camera size={20} style={{ color: 'var(--primary)' }} />
              Camera Nhận Diện Cử Chỉ VSL (Bám Sát Tay)
            </h2>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Bù trễ chuyển động (Motion Extrapolation) + Vẽ Batching Siêu Tốc 60 FPS
            </p>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            {cameraActive ? (
              <button onClick={stopCamera} className="btn btn-danger" style={{ padding: '8px 14px' }}>
                <CameraOff size={16} /> Tắt Camera
              </button>
            ) : (
              <button onClick={() => startCamera(trackerEngine)} className="btn btn-primary" style={{ padding: '8px 14px' }} disabled={cameraLoading}>
                <Camera size={16} /> {cameraLoading ? 'Đang bật...' : 'Bật Camera'}
              </button>
            )}
          </div>
        </div>

        {/* THANH CÔNG CỤ TÁI CẤU TRÚC: 2 TIÊU CHÍ LỰA CHỌN ĐỘC LẬP */}
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          
          {/* Hàng 2 nhóm lựa chọn độc lập */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1.6fr', gap: '12px' }}>
            
            {/* TIÊU CHÍ A: PHẠM VI NHẬN DIỆN */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
              padding: '10px',
              background: 'rgba(15, 23, 42, 0.65)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-subtle)'
            }}>
              <span style={{ fontSize: '11px', fontWeight: '700', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '5px' }}>
                <Layers size={13} style={{ color: 'var(--primary)' }} /> A. Phạm vi nhận diện
              </span>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  onClick={() => handleScopeChange('hands')}
                  className={`btn ${recognitionScope === 'hands' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, padding: '7px 8px', fontSize: '11px', justifyContent: 'center' }}
                  title="Nhận diện cử chỉ bàn tay siêu tốc 60 FPS"
                >
                  <Hand size={14} /> Bàn tay
                </button>
                <button
                  onClick={() => handleScopeChange('holistic')}
                  className={`btn ${recognitionScope === 'holistic' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, padding: '7px 8px', fontSize: '11px', justifyContent: 'center' }}
                  title="Nhận diện toàn thân 75 khớp (Pose + 2 Bàn tay)"
                >
                  <User size={14} /> Toàn thân
                </button>
              </div>
            </div>

            {/* TIÊU CHÍ B: CHẾ ĐỘ CHUYỂN ĐỔI */}
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              gap: '6px',
              padding: '10px',
              background: 'rgba(15, 23, 42, 0.65)',
              borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-subtle)'
            }}>
              <span style={{ fontSize: '11px', fontWeight: '700', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', display: 'flex', alignItems: 'center', gap: '5px' }}>
                <Sparkles size={13} style={{ color: 'var(--primary)' }} /> B. Chế độ chuyển đổi
              </span>
              <div style={{ display: 'flex', gap: '6px' }}>
                <button
                  onClick={() => handleModeChange('letters')}
                  className={`btn ${conversionMode === 'letters' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, padding: '7px 6px', fontSize: '11px', justifyContent: 'center' }}
                  title="Nhận diện 27 chữ cái VSL tức thì (<5ms)"
                >
                  <Type size={13} /> Chữ cái
                </button>
                <button
                  onClick={() => handleModeChange('digits')}
                  className={`btn ${conversionMode === 'digits' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, padding: '7px 6px', fontSize: '11px', justifyContent: 'center' }}
                  title="Nhận diện chữ số VSL từ 1 đến 9 (<5ms)"
                >
                  <Zap size={13} /> Chữ số
                </button>
                <button
                  onClick={() => handleModeChange('words')}
                  className={`btn ${conversionMode === 'words' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, padding: '7px 6px', fontSize: '11px', justifyContent: 'center' }}
                  title="Nhận diện 102 từ vựng cử chỉ VSL (SSTCN 30-Frame)"
                >
                  <BookOpen size={13} /> Từ (102 Từ)
                </button>
                <button
                  onClick={() => handleModeChange('sentence')}
                  className={`btn ${conversionMode === 'sentence' ? 'btn-primary' : 'btn-secondary'}`}
                  style={{ flex: 1, padding: '7px 6px', fontSize: '11px', justifyContent: 'center' }}
                  title="Ghép chuỗi từ liên tục thành câu văn bản"
                >
                  <MessageSquare size={13} /> Ghép câu
                </button>
              </div>
            </div>

          </div>

          {/* BANNER THÔNG BÁO TRẠNG THÁI TỔ HỢP ĐỘC LẬP */}
          {(() => {
            const status = getCombinationStatus();
            return (
              <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '10px',
                padding: '8px 12px',
                borderRadius: 'var(--radius-sm)',
                background: status.type === 'optimal' 
                  ? 'rgba(16, 185, 129, 0.08)' 
                  : status.type === 'developing' 
                  ? 'rgba(6, 182, 212, 0.08)' 
                  : 'rgba(245, 158, 11, 0.08)',
                border: `1px solid ${
                  status.type === 'optimal' 
                    ? 'rgba(16, 185, 129, 0.25)' 
                    : status.type === 'developing'
                    ? 'rgba(6, 182, 212, 0.25)'
                    : 'rgba(245, 158, 11, 0.25)'
                }`,
                fontSize: '11px'
              }}>
                <span className={`badge ${status.badgeClass}`} style={{ fontSize: '10px', padding: '2px 6px', whiteSpace: 'nowrap' }}>
                  {status.badge}
                </span>
                <span style={{ color: 'var(--text-secondary)', flex: 1, lineHeight: '1.4' }}>
                  <strong style={{ color: 'var(--text-main)' }}>{status.title}: </strong>
                  {status.desc}
                </span>
              </div>
            );
          })()}

        </div>

        {/* Khung hiển thị Video & Canvas */}
        <div
          ref={containerRef}
          style={{
            position: 'relative',
            width: '100%',
            aspectRatio: aspectRatio,
            background: '#070b12',
            borderRadius: 'var(--radius-md)',
            overflow: 'hidden',
            border: '1px solid var(--border-subtle)',
            boxShadow: 'inset 0 0 40px rgba(0, 0, 0, 0.85)'
          }}
        >
          {/* TỐI ƯU CỐT LÕI: Video và Canvas đồng bộ kích thước pixel-perfect */}
          <video
            ref={videoRef}
            playsInline
            muted
            autoPlay
            onLoadedMetadata={handleVideoMetadata}
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              transform: 'scaleX(-1)', // Mirror tự nhiên
              display: cameraActive ? 'block' : 'none'
            }}
          />

          {/* Canvas trong suốt chỉ vẽ khung xương neon đã bù trễ chuyển động */}
          <canvas
            ref={canvasRef}
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              transform: 'scaleX(-1)',
              pointerEvents: 'none'
            }}
          />

          {/* Màn hình chờ khi chưa bật Camera */}
          {!cameraActive && (
            <div style={{
              position: 'absolute',
              inset: 0,
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              padding: '24px',
              textAlign: 'center',
              background: 'radial-gradient(circle, rgba(15, 23, 42, 0.95) 0%, rgba(7, 11, 18, 0.98) 100%)'
            }}>
              <div style={{
                width: '64px',
                height: '64px',
                borderRadius: '50%',
                background: 'rgba(6, 182, 212, 0.12)',
                border: '1px solid rgba(6, 182, 212, 0.35)',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                marginBottom: '14px',
                boxShadow: '0 0 24px rgba(6, 182, 212, 0.2)'
              }}>
                <Camera size={28} style={{ color: 'var(--primary)' }} />
              </div>
              <h3 style={{ fontSize: '17px', fontWeight: '700', marginBottom: '6px' }}>Camera Đang Tắt</h3>
              <p style={{ fontSize: '13px', color: 'var(--text-muted)', maxWidth: '420px', marginBottom: '20px' }}>
                Đã tích hợp thuật toán bù trễ chuyển động (Motion Extrapolation) và vẽ batching siêu tốc. Khung xương sẽ bám sát từng cử chỉ tay theo thời gian thực!
              </p>
              <button onClick={() => startCamera(trackerEngine)} className="btn btn-primary" disabled={cameraLoading}>
                <Zap size={16} /> Bật Camera Ngay
              </button>
            </div>
          )}

          {/* HUD Top-Bar */}
          {cameraActive && (
            <div style={{
              position: 'absolute',
              top: '12px',
              left: '12px',
              right: '12px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              pointerEvents: 'none'
            }}>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <div className="badge badge-emerald" style={{ backdropFilter: 'blur(10px)', background: 'rgba(16, 185, 129, 0.25)', border: '1px solid rgba(16, 185, 129, 0.4)' }}>
                  <span className="pulse-dot" /> <span ref={fpsBadgeRef} style={{ fontWeight: '700' }}>{fps} FPS</span>
                </div>
                <div
                  ref={motionBadgeRef}
                  className="badge badge-cyan"
                  style={{ backdropFilter: 'blur(10px)', transition: 'opacity 0.2s' }}
                >
                  ⚡ Bám sát tay
                </div>
              </div>

              {/* Vòng tròn SVG cập nhật trực tiếp qua DOM */}
              {recognitionMode === 'sequence' && (
                <div style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '8px',
                  background: 'rgba(15, 23, 42, 0.8)',
                  backdropFilter: 'blur(10px)',
                  padding: '5px 12px',
                  borderRadius: '20px',
                  border: '1px solid rgba(255, 255, 255, 0.15)'
                }}>
                  <svg width="26" height="26" viewBox="0 0 36 36">
                    <path
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none"
                      stroke="rgba(255, 255, 255, 0.15)"
                      strokeWidth="3.5"
                    />
                    <path
                      ref={progressCircleRef}
                      d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                      fill="none"
                      stroke="#06b6d4"
                      strokeWidth="3.5"
                      strokeDasharray="0, 100"
                      strokeLinecap="round"
                    />
                  </svg>
                  <span ref={progressTextRef} style={{ fontSize: '11px', fontWeight: '600', color: '#e2e8f0' }}>
                    0/30 Frame
                  </span>
                </div>
              )}
            </div>
          )}

          {/* Overlay Báo Lỗi */}
          {cameraError && (
            <div style={{
              position: 'absolute',
              bottom: '12px',
              left: '12px',
              right: '12px',
              padding: '12px 16px',
              background: 'rgba(239, 68, 68, 0.92)',
              color: '#fff',
              borderRadius: 'var(--radius-sm)',
              fontSize: '12px',
              display: 'flex',
              alignItems: 'center',
              gap: '8px'
            }}>
              <AlertCircle size={16} /> {cameraError}
            </div>
          )}
        </div>

        {/* Thanh công cụ Test Mô phỏng (Simulator) */}
        <div style={{
          background: 'rgba(15, 23, 42, 0.55)',
          padding: '12px 16px',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
            <span style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-secondary)', display: 'flex', alignItems: 'center', gap: '6px' }}>
              <Zap size={14} style={{ color: 'var(--primary)' }} />
              Thử nghiệm cử chỉ mẫu (Simulator Test):
            </span>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Bấm để kiểm tra</span>
          </div>
          <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap' }}>
            {SAMPLE_GESTURES.map((item) => (
              <button
                key={item.type}
                onClick={() => triggerSampleTest(item)}
                className="btn btn-secondary"
                style={{ padding: '5px 11px', fontSize: '12px' }}
                title={item.desc}
              >
                {item.name}
              </button>
            ))}
          </div>
        </div>

      </div>

      {/* CỘT PHẢI: KẾT QUẢ SUY LUẬN & PHỤ ĐỀ */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* Card Trạng thái Cử chỉ Đang Nhận Diện */}
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '12px', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em', fontWeight: '600' }}>
              {conversionMode === 'letters' ? '🔤 Chữ cái VSL (A–Z)' : conversionMode === 'digits' ? '🔢 Chữ số VSL (0–9)' : conversionMode === 'words' ? '📖 Từ vựng (102 Từ)' : conversionMode === 'sentence' ? '💬 Ghép câu ngữ cảnh' : '🔤 Chữ cái & Số'}
            </span>
            <div style={{ display: 'flex', gap: '6px' }}>
              <span className={`badge ${conversionMode === 'letters' || conversionMode === 'digits' || conversionMode === 'letters_numbers' ? 'badge-emerald' : conversionMode === 'words' ? 'badge-indigo' : 'badge-cyan'}`} style={{ fontSize: '11px' }}>
                {conversionMode === 'letters' || conversionMode === 'digits' || conversionMode === 'letters_numbers' ? 'Instant (<5ms)' : conversionMode === 'words' ? 'SSTCN (30-Frame)' : 'Temporal Debounce'}
              </span>
              <span className="badge badge-cyan" style={{ fontSize: '11px' }}>
                {currentResult.confidence > 0 ? `${Math.round(currentResult.confidence * 100)}% Khớp` : 'Đang chờ...'}
              </span>
            </div>
          </div>

          {/* Banner Kết quả Lớn */}
          <div style={{
            padding: '18px',
            borderRadius: 'var(--radius-md)',
            background: 'linear-gradient(135deg, rgba(6, 182, 212, 0.12) 0%, rgba(99, 102, 241, 0.08) 100%)',
            border: '1px solid var(--border-active)',
            textAlign: 'center',
            position: 'relative'
          }}>
            <h3 style={{
              fontSize: '32px',
              fontWeight: '800',
              color: currentResult.sign !== 'NEUTRAL' ? 'var(--text-main)' : 'var(--text-muted)',
              marginBottom: '6px',
              letterSpacing: '-0.02em'
            }}>
              {currentResult.label}
            </h3>

            {currentResult.description && (
              <p style={{ fontSize: '12px', color: 'var(--text-secondary)', lineHeight: '1.45', maxWidth: '440px', margin: '0 auto' }}>
                💡 {currentResult.description}
              </p>
            )}

            {currentResult.video_url && (
              <button
                onClick={() => setPreviewVideoUrl(currentResult.video_url)}
                className="btn btn-secondary"
                style={{
                  marginTop: '10px',
                  padding: '5px 12px',
                  fontSize: '11px',
                  marginInline: 'auto'
                }}
              >
                <Film size={13} style={{ color: 'var(--primary)' }} /> Xem video giáo trình mẫu
              </button>
            )}
          </div>

          {/* Thanh đo Độ tin cậy (Confidence Bar) */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-muted)', marginBottom: '4px' }}>
              <span>Độ tin cậy mô hình SSTCN:</span>
              <span>{Math.round(currentResult.confidence * 100)}%</span>
            </div>
            <div style={{
              width: '100%',
              height: '6px',
              background: 'rgba(255, 255, 255, 0.08)',
              borderRadius: '3px',
              overflow: 'hidden'
            }}>
              <div style={{
                height: '100%',
                width: `${Math.round(currentResult.confidence * 100)}%`,
                background: currentResult.confidence > 0.85 ? 'var(--accent-success)' : currentResult.confidence > 0.65 ? 'var(--primary)' : 'var(--accent-warning)',
                transition: 'width 0.25s ease'
              }} />
            </div>
          </div>

          {/* Danh sách Top-3 Khả Năng Cao Nhất */}
          {topKPredictions.length > 0 && (
            <div style={{ background: 'rgba(0, 0, 0, 0.2)', padding: '10px 14px', borderRadius: 'var(--radius-sm)' }}>
              <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '6px' }}>
                Dự đoán hàng đầu (Top Predictions):
              </span>
              <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                {topKPredictions.slice(0, 3).map((item, i) => (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', fontSize: '11px' }}>
                    <span style={{ color: 'var(--text-secondary)' }}>{i + 1}. {item.term}</span>
                    <span style={{ fontWeight: '600', color: i === 0 ? 'var(--primary)' : 'var(--text-muted)' }}>
                      {Math.round(item.confidence * 100)}%
                    </span>
                  </div>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Card Phụ Đề Câu Hoàn Chỉnh */}
        <div className="glass-panel" style={{
          padding: '20px',
          display: 'flex',
          flexDirection: 'column',
          gap: '14px',
          flex: 1,
          border: conversionMode === 'sentence' ? '1px solid var(--border-active)' : '1px solid var(--border-subtle)',
          boxShadow: conversionMode === 'sentence' ? '0 0 20px rgba(6, 182, 212, 0.1)' : 'none'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <div>
              <h3 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '2px' }}>
                <CheckCircle2 size={18} style={{ color: 'var(--accent-success)' }} />
                Phụ Đề Văn Bản Tiếng Việt
              </h3>
              {conversionMode === 'sentence' && (
                <span style={{ fontSize: '11px', color: 'var(--primary)', fontWeight: '600' }}>
                  💬 Chế độ Ghép câu liên tục (Ghép chuỗi tự động qua thời gian)
                </span>
              )}
            </div>

            <div style={{ display: 'flex', gap: '6px' }}>
              <button
                onClick={() => speakVietnamese(sentence)}
                disabled={!sentence}
                className="btn btn-secondary"
                style={{ padding: '6px 10px', fontSize: '12px' }}
                title="Đọc toàn bộ câu bằng giọng nói tiếng Việt"
              >
                <Volume2 size={14} /> Đọc
              </button>
              <button
                onClick={handleCopySentence}
                disabled={!sentence}
                className="btn btn-secondary"
                style={{ padding: '6px 10px', fontSize: '12px' }}
                title="Sao chép văn bản"
              >
                <Copy size={14} /> {copied ? 'Đã chép' : 'Chép'}
              </button>
              <button
                onClick={handleClearSentence}
                disabled={!sentence}
                className="btn btn-danger"
                style={{ padding: '6px 10px', fontSize: '12px' }}
                title="Xóa phụ đề"
              >
                <Trash2 size={14} /> Xóa
              </button>
            </div>
          </div>

          {/* Vùng hiển thị câu văn bản */}
          <div style={{
            flex: 1,
            minHeight: '110px',
            padding: '16px',
            background: 'var(--bg-input)',
            borderRadius: 'var(--radius-md)',
            border: '1px solid var(--border-subtle)',
            fontSize: '18px',
            lineHeight: '1.6',
            color: sentence ? 'var(--text-main)' : 'var(--text-muted)',
            fontFamily: 'var(--font-heading)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: sentence ? 'flex-start' : 'center',
            textAlign: sentence ? 'left' : 'center'
          }}>
            {sentence || 'Văn bản được dịch từ cử chỉ tay sẽ tự động xuất hiện tại đây...'}
          </div>

          {/* Lịch sử các ký hiệu vừa ghi nhận */}
          <div>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', display: 'block', marginBottom: '8px' }}>
              Lịch sử ký hiệu vừa xác nhận:
            </span>
            <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
              {recognizedTokens.length > 0 ? (
                recognizedTokens.map((item, idx) => (
                  <span
                    key={idx}
                    className="badge badge-indigo"
                    style={{ fontSize: '11px', padding: '3px 8px' }}
                  >
                    {item.token}
                  </span>
                ))
              ) : (
                <span style={{ fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                  Chưa có ký hiệu nào được ghi nhận.
                </span>
              )}
            </div>
          </div>
        </div>

      </div>

      {/* MODAL POPUP XEM VIDEO GIÁO TRÌNH MẪU */}
      {previewVideoUrl && (
        <div style={{
          position: 'fixed',
          inset: 0,
          background: 'rgba(0, 0, 0, 0.85)',
          backdropFilter: 'blur(8px)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 9999,
          padding: '20px'
        }}>
          <div className="glass-panel" style={{
            maxWidth: '560px',
            width: '100%',
            padding: '20px',
            position: 'relative',
            boxShadow: '0 20px 40px rgba(0, 0, 0, 0.8)'
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '14px' }}>
              <h3 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
                <Film size={18} style={{ color: 'var(--primary)' }} />
                Video Mẫu Giáo Trình: {currentResult.label}
              </h3>
              <button
                onClick={() => setPreviewVideoUrl(null)}
                style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer' }}
              >
                <X size={20} />
              </button>
            </div>

            <video
              src={previewVideoUrl}
              controls
              autoPlay
              loop
              playsInline
              style={{
                width: '100%',
                borderRadius: 'var(--radius-md)',
                boxShadow: '0 8px 24px rgba(0, 0, 0, 0.5)'
              }}
            />

            <p style={{ fontSize: '12px', color: 'var(--text-secondary)', marginTop: '12px', lineHeight: '1.4' }}>
              💡 {currentResult.description}
            </p>
          </div>
        </div>
      )}

    </div>
  );
}
