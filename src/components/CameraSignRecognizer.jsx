import React, { useState, useEffect, useRef } from 'react';
import {
  Camera, CameraOff, Volume2, Copy, Trash2, AlertCircle, Zap, Film,
  Hand, User, Type, BookOpen, ScanLine, FlaskConical, SlidersHorizontal
} from 'lucide-react';
import {
  sendSequenceForRecognition,
  sendLandmarksForRecognition,
  clearCurrentSentence,
  speakVietnamese,
  cancelVietnameseSpeech,
  initSpeechSynthesisVoices,
  mediaUrl,
  mediaId,
  requestQipedcPractice,
  prepareQipedcLive,
  fetchSSTCNTopics,
  saveCalibrationSample,
  fetchCalibrationSummary
} from '../services/api';
import { createTtsOrchestrator } from '../utils/ttsOrchestrator';
import WordDebugPanel from './WordDebugPanel';
import CameraDiagnostic from './CameraDiagnostic';
import CaptionTokens from './CaptionTokens';
import { useCameraDiagnostic } from './useCameraDiagnostic';
import { isWordDebugEnabled, isWordDebugActive, logWordDebugReset, logWordDebugEvent } from '../services/wordDebugApi';
import PilotTestingHUD from './PilotTestingHUD';
import { Workspace, WorkspaceDialog } from './Workspace';
import { getEpochSecondsNow, createMonotonicTimestampGenerator } from './pilotProtocol.js';
import { HandTrackerStabilizer } from '../utils/handTrackerStabilizer';
import { createAcceleratedTracker } from '../utils/fastVisionTracker';
import { evaluateCameraFraming, AdaptivePerfController } from '../utils/cameraFraming';
import { isValidWordRecognitionData, extractRejectReason, sanitizeWordTopK } from '../utils/wordScopeGuard';

// Bảng ánh xạ phát âm TTS chuẩn cho từng chữ số (chỉ đọc tên số, không có tiền tố "số")
const DIGIT_TTS_MAP = {
  '0': 'không',
  '1': 'một',
  '2': 'hai',
  '3': 'ba',
  '4': 'bốn',
  '5': 'năm',
  '6': 'sáu',
  '7': 'bảy',
  '8': 'tám',
  '9': 'chín',
  '10': 'mười'
};

/**
 * Định dạng câu văn bản cho TTS: chuyển đổi các chữ số thành tên chữ tiếng Việt tương ứng
 * (ví dụ: '10 1 1 2' -> 'mười một một hai') và loại bỏ hoàn toàn từ 'Số'/'số'.
 * Giữ nguyên các chữ cái, từ vựng và câu thông thường.
 */
function formatSentenceForTTS(text) {
  if (!text) return '';
  // Loại bỏ tiền tố "Số " hoặc "số " nếu có
  let cleaned = text.replace(/(?:Số|số)\s+(\d+)/g, '$1');
  // Chuyển đổi các chữ số riêng biệt 0-10 thành tên số tiếng Việt
  return cleaned.replace(/\b(10|[0-9])\b/g, (match) => {
    return DIGIT_TTS_MAP[match] || match;
  });
}

const SEQ_LEN = 30; // 30 frames cho SSTCN_Attention (~2 giây cử chỉ)
const FRAME_STEP = 6; // Sliding window: cập nhật mỗi 6 frame (~300ms) để nhận diện tức thì
const LIVE_QIPEDC_WINDOW_MS = 2800;
const WORD_SUGGESTION_MIN = 0.60;
const WORD_SUGGESTION_LIMIT = 5;

export function isShortGesturesEnabled() {
  if (typeof window === 'undefined') return true;
  const urlParams = new URLSearchParams(window.location.search);
  if (urlParams.has('shortGestures')) {
    return urlParams.get('shortGestures') !== '0';
  }
  const saved = localStorage.getItem('vsl_allow_short_sequences');
  return saved !== '0';
}

function newLiveWordState() {
  return { lastHand: -Infinity, lastRequest: -Infinity, firstMotion: null,
    candidate: null, votes: 0, token: null, template: false, quiet: 0,
    templateCandidate: null, templateVotes: 0, pending: null,
    hadFastMotion: false, stillSince: null, flushed: false,
    cooldownUntil: -Infinity, lastCommittedToken: null, lastCommittedAt: -Infinity };
}

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

export default function CameraSignRecognizer({ ttsEnabled, practiceEntry, liveQipedcWords, onBrowsePractice, onExitPractice }) {
  const diagnostic = useCameraDiagnostic();
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const captionRef = useRef(null);
  const nextCaptionIdRef = useRef(0);
  const lastAutomaticCaptionRef = useRef({});
  const diagnosticDialogRef = useRef(null);
  const testingDialogRef = useRef(null);
  const calibrationDialogRef = useRef(null);
  const previewDialogRef = useRef(null);
  const cameraInstanceRef = useRef(null);
  const trackerInstanceRef = useRef(null);
  const animationFrameIdRef = useRef(null);

  // Tham chiếu trực tiếp đến DOM phần tử HUD để cập nhật 60 FPS không gây lag React
  const progressCircleRef = useRef(null);
  const progressTextRef = useRef(null);
  const motionBadgeRef = useRef(null);
  const fpsBadgeRef = useRef(null);
  const framingBadgeRef = useRef(null);
  const framingNoticeRef = useRef(null);
  const lastFramingUpdateRef = useRef(0);
  const framingStateRef = useRef({ ok: true, status: 'optimal' });
  const processCanvasRef = useRef(null);
  const lastFrameSentTimeRef = useRef(0);

  // Bộ nhớ đệm dự đoán chuyển động (Motion Extrapolation Buffer)
  // Bù đắp hoàn toàn độ trễ ~35ms của WebAssembly giúp khung xương bám sát tay thời gian thực
  const prevLeftHandRef = useRef(null);
  const prevRightHandRef = useRef(null);
  const stabilizerRef = useRef(null);
  if (!stabilizerRef.current) {
    stabilizerRef.current = new HandTrackerStabilizer();
  }
  const perfControllerRef = useRef(null);
  if (!perfControllerRef.current) {
    perfControllerRef.current = new AdaptivePerfController();
  }

  // Trạng thái camera & pipeline
  const [cameraActive, setCameraActive] = useState(false);
  const [cameraLoading, setCameraLoading] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [detailedTracking, setDetailedTracking] = useState(false);
  const detailedTrackingRef = useRef(false);

  // TIÊU CHÍ A: Phạm vi nhận diện: 'hands' (Bàn tay - 60 FPS) | 'holistic' (Toàn thân - 75 Khớp)
  const [recognitionScope, setRecognitionScope] = useState(practiceEntry ? 'holistic' : 'hands');

  // TIÊU CHÍ B: Chế độ chuyển đổi: 'letters' (Chữ cái) | 'digits' (Chữ số) | 'words' (Từ vựng)
  const [conversionMode, setConversionMode] = useState(practiceEntry ? 'words' : 'letters');

  // Trạng thái kỹ thuật đồng bộ với backend & pipeline
  const [recognitionMode, setRecognitionMode] = useState(practiceEntry ? 'sequence' : 'instant');
  const [trackerEngine, setTrackerEngine] = useState(recognitionScope);
  const [holisticStatus, setHolisticStatus] = useState('ready'); // 'ready' | 'unavailable' | 'pose_missing'
  const trackerEngineRef = useRef(trackerEngine);
  const activeInstanceEngineRef = useRef(null); // Engine thực tế của instance đang chạy ('hands' | 'holistic' | null)
  const holisticUnavailableRef = useRef(false);
  const cameraActiveRef = useRef(false);
  const holisticCanvasRef = useRef(null);
  const missingHandsCountRef = useRef(0);
  const missingPoseCountRef = useRef(0);
  const poseZeroHistoryRef = useRef([]);
  const lastFrameSendTimeRef = useRef(0);
  const frameLatencyHistoryRef = useRef([]);
  const windowTimestampsRef = useRef([]);
  const isStandardFlushedRef = useRef(false);
  const lastFlushTimeRef = useRef(0);

  useEffect(() => {
    trackerEngineRef.current = trackerEngine;
  }, [trackerEngine]);

  useEffect(() => {
    cameraActiveRef.current = cameraActive;
  }, [cameraActive]);

  // Trạng thái nhận diện chuỗi động
  const [wordSuggestions, setWordSuggestions] = useState([]);
  const [previewVideoUrl, setPreviewVideoUrl] = useState(null);

  // Tích hợp nâng cao từ KVSL + SignLang + LNT: Lọc Chủ đề, Ghi 3s (Record-and-Check), Personal Calibration
  const [sstcnTopicsInfo, setSstcnTopicsInfo] = useState(null);
  const [selectedTopic, setSelectedTopic] = useState('');
  const selectedTopicRef = useRef('');
  const [vocabScope, setVocabScope] = useState('all'); // 'all' (2-tier ưu tiên thông dụng) | 'common' (~730 từ thông dụng)
  const vocabScopeRef = useRef('all');
  const [wordCaptureMode, setWordCaptureMode] = useState('continuous'); // 'continuous' | 'record'
  const wordCaptureModeRef = useRef('continuous');
  const [recordPhase, setRecordPhase] = useState('idle'); // 'idle' | 'preparing' | 'recording'
  const recordPhaseRef = useRef('idle');
  const recordPurposeRef = useRef(null);
  const [recordCountdown, setRecordCountdown] = useState(0);
  const recordFramesRef = useRef([]);
  const recordTimestampsRef = useRef([]);
  const recordTimersRef = useRef([]);
  const [calibrationLabel, setCalibrationLabel] = useState('');
  const [calibrationMsg, setCalibrationMsg] = useState('');
  const [calibrationTotal, setCalibrationTotal] = useState(0);
  const [practiceStatus, setPracticeStatus] = useState('idle');
  const [practiceFeedback, setPracticeFeedback] = useState('');
  const [practiceVideo, setPracticeVideo] = useState(practiceEntry?.video_url);
  const [mirrorPractice, setMirrorPractice] = useState(false);
  const [practiceTrackingReady, setPracticeTrackingReady] = useState(false);
  const practiceTrackingReadyRef = useRef(false);
  const practiceActive = Boolean(practiceEntry) && conversionMode === 'words';
  const practiceVariants = practiceEntry?.variants?.filter((variant) => mediaId(variant.url)) || [];
  const liveWordsRef = useRef(newLiveWordState());
  const [liveReferenceStatus, setLiveReferenceStatus] = useState(null);

  useEffect(() => {
    selectedTopicRef.current = selectedTopic;
  }, [selectedTopic]);

  useEffect(() => {
    vocabScopeRef.current = vocabScope;
  }, [vocabScope]);

  useEffect(() => {
    wordCaptureModeRef.current = wordCaptureMode;
  }, [wordCaptureMode]);

  useEffect(() => {
    if (practiceEntry || conversionMode !== 'words' || wordCaptureMode !== 'continuous' || !cameraActive) return;
    let active = true;
    const extra = (liveQipedcWords || []).map(({ id }) => id);
    Promise.allSettled([prepareQipedcLive(), ...extra.map((id) => prepareQipedcLive(id))]).then((results) => {
      if (!active) return;
      const entries = results.flatMap((result) => result.status === 'fulfilled' ? result.value.entries : []);
      const indexedCount = results.find((result) => result.status === 'fulfilled')?.value?.indexed_count || 0;
      setLiveReferenceStatus({
        entries,
        indexedCount,
        failed: results.some((result) => result.status === 'rejected') || entries.some((item) => !item.ready),
      });
    });
    return () => { active = false; };
  }, [practiceEntry, conversionMode, wordCaptureMode, cameraActive, liveQipedcWords]);

  useEffect(() => {
    recordPhaseRef.current = recordPhase;
  }, [recordPhase]);

  useEffect(() => {
    let active = true;
    fetchSSTCNTopics().then((res) => {
      if (active && res && res.success) {
        setSstcnTopicsInfo(res);
      }
    });
    fetchCalibrationSummary().then((res) => {
      if (active && res && res.success) {
        setCalibrationTotal(res.total_samples || 0);
      }
    });
    return () => {
      active = false;
      recordTimersRef.current.forEach((t) => clearTimeout(t));
      recordTimersRef.current = [];
    };
  }, []);

  const preparePractice = async () => {
    setPracticeStatus('loading');
    setPracticeFeedback('');
    try {
      await requestQipedcPractice(practiceEntry.id, mediaId(practiceVideo));
      setPracticeStatus('ready');
    } catch (error) {
      setPracticeFeedback(error.message);
      setPracticeStatus('idle');
    }
  };

  const startTimedCapture = (purpose = 'predict') => {
    if (!cameraActiveRef.current) {
      setCalibrationMsg('Vui lòng bật camera (chế độ Từ vựng / Holistic) trước khi ghi.');
      return;
    }
    if (purpose === 'calibrate' && !calibrationLabel.trim()) {
      setCalibrationMsg('Nhập tên từ cần hiệu chỉnh (vd: Xin chào) trước khi ghi mẫu.');
      return;
    }
    recordTimersRef.current.forEach((t) => clearTimeout(t));
    recordPurposeRef.current = purpose;
    recordTimersRef.current = [];
    recordFramesRef.current = [];
    recordTimestampsRef.current = [];
    sequenceBufferRef.current = [];
    windowTimestampsRef.current = [];
    liveWordsRef.current = newLiveWordState();
    setWordSuggestions([]);
    setCalibrationMsg('');
    setPracticeFeedback('');
    requestGenRef.current += 1;
    isInferringRef.current = false;
    setRecordPhase('preparing');
    recordPhaseRef.current = 'preparing';
    setRecordCountdown(2);

    const t1 = setTimeout(() => setRecordCountdown(1), 1000);
    const t2 = setTimeout(() => {
      recordFramesRef.current = [];
      recordTimestampsRef.current = [];
      setRecordPhase('recording');
      recordPhaseRef.current = 'recording';
      setRecordCountdown(3);
    }, 2000);
    const t3 = setTimeout(() => setRecordCountdown(2), 3000);
    const t4 = setTimeout(() => setRecordCountdown(1), 4000);
    const t5 = setTimeout(async () => {
      setRecordPhase('idle');
      recordPhaseRef.current = 'idle';
      setRecordCountdown(0);
      const frames = recordFramesRef.current.slice();
      const tsList = recordTimestampsRef.current.slice();
      recordFramesRef.current = [];
      recordTimestampsRef.current = [];

      if (frames.length < 4) {
        const message = 'Không thu đủ khung hình có tay. Hãy thử lại.';
        if (purpose === 'practice') setPracticeFeedback(message);
        else setCalibrationMsg(message);
        return;
      }

      if (purpose === 'practice') {
        setPracticeStatus('scoring');
        try {
          const canvas = holisticCanvasRef.current;
          const result = await requestQipedcPractice(practiceEntry.id, mediaId(practiceVideo), {
            sequence: frames,
            aspect_ratio: canvas.width / canvas.height,
            mirror: mirrorPractice,
          });
          diagnostic.addPrediction(tsList.at(-1), {
            sign: result.matched ? result.word : 'NEUTRAL', confidence: result.similarity,
            diagnostic: { status: result.reason || (result.matched ? 'practice_matched' : 'practice_mismatch') },
          }, tsList, 'qipedc_practice');
          if (result.reason === 'tracking_gaps') {
            setPracticeFeedback('Camera mất dấu tay hoặc vai trong lúc ghi. Đặt đèn phía trước và giữ hai vai trong khung rồi thử lại.');
          } else if (result.reason === 'missing_both_hands') {
            setPracticeFeedback('Mẫu này dùng hai tay. Giữ cả hai bàn tay và cổ tay trong khung hình rồi thử lại.');
          } else {
            const scorePct = result.similarity != null ? `${Math.round(result.similarity * 100)}%` : null;
            setPracticeFeedback(result.matched
              ? `Khớp mẫu: ${result.word} (Độ tương đồng: ${scorePct})`
              : `Chưa khớp mẫu “${result.word}”${scorePct ? ` (Đạt ${scorePct} / cần ≥ 72%)` : ''}. Xem lại video và làm trọn động tác sau khi hết đếm ngược.`);
          }
        } catch (error) {
          setPracticeFeedback(error.message);
        } finally {
          setPracticeStatus('ready');
        }
      } else if (purpose === 'calibrate') {
        const res = await saveCalibrationSample({
          label: calibrationLabel.trim(),
          sequence: frames,
          timestamps: tsList,
          aspect_ratio: videoRef.current?.videoWidth / videoRef.current?.videoHeight || 1,
        });
        if (res && res.success) {
          setCalibrationMsg(`Đã lưu mẫu #${res.sample_count} cho “${res.label}”. Camera Tự do sẽ dùng ở lượt tiếp theo.`);
          fetchCalibrationSummary().then((s) => {
            if (s && s.success) setCalibrationTotal(s.total_samples || 0);
          });
        } else {
          setCalibrationMsg(`Lỗi lưu mẫu: ${res?.error || 'unknown'}`);
        }
      } else {
        const captureGeneration = requestGenRef.current;
        const captureTopic = selectedTopicRef.current;
        const resp = await sendSequenceForRecognition(
          frames,
          confidenceThreshold,
          'words',
          {
            tracker_engine: trackerEngineRef.current,
            capture_mode: 'record_3s',
            raw_frames: frames.length,
          },
          {
            topic_id: captureTopic || null,
            timestamps: tsList,
            aspect_ratio: videoRef.current?.videoWidth / videoRef.current?.videoHeight || 1,
            include_word_candidates: true,
          }
        );
        if (captureGeneration !== requestGenRef.current || captureTopic !== selectedTopicRef.current) return;
        if (resp && resp.success && resp.data) {
          const data = resp.data;
          updateWordSuggestions(data);
          const validWord = isValidWordRecognitionData(data);
          diagnostic.addPrediction(
            tsList.at(-1),
            data,
            tsList,
            'sstcn',
            validWord && data.confirmed ? data.new_token : null
          );
          if (!validWord) {
            setCurrentResult({
              sign: 'NEUTRAL',
              label: data.label || 'Từ nằm ngoài phạm vi nhận diện cho phép',
              confidence: 0,
              type: data.category || 'dynamic_sign',
              description: '',
              category: data.category || 'Trạng thái',
              video_url: '',
              region: data.region || null,
              scopeRejected: Boolean(data.scope_rejected),
              rejectReason: extractRejectReason(data),
            });
          } else {
            setCurrentResult({
              sign: data.sign,
              label: data.label,
              confidence: data.confidence,
              type: data.category || 'dynamic_sign',
              description: data.description,
              category: data.category,
              video_url: data.video_url,
              region: data.region,
            });
            if (data.confirmed && data.new_token && data.sign !== 'NEUTRAL') {
              commitCaption(data.new_token);
              if (ttsEnabledRef.current) speakVietnamese(data.new_token);
            }
          }
        }
      }
    }, 5000);
    recordTimersRef.current = [t1, t2, t3, t4, t5];
  };

  const changeWordCaptureMode = (mode) => {
    if (wordCaptureModeRef.current === mode || recordPhaseRef.current !== 'idle') return;
    diagnostic.stop();
    wordCaptureModeRef.current = mode;
    requestGenRef.current += 1;
    isInferringRef.current = false;
    liveWordsRef.current = newLiveWordState();
    setWordSuggestions([]);
    sequenceBufferRef.current = [];
    windowTimestampsRef.current = [];
    setWordCaptureMode(mode);
  };

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
  const [captionTokens, setCaptionTokens] = useState([]);
  const [selectedCaptionId, setSelectedCaptionId] = useState(null);
  const sentence = captionTokens.map(({ text }) => text).join(' ');
  const [fps, setFps] = useState(0);
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.65);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    const caption = captionRef.current;
    if (caption) caption.scrollTop = caption.scrollHeight;
  }, [captionTokens.length]);

  const commitCaption = (text, { stream, replaced } = {}) => {
    const originalId = lastAutomaticCaptionRef.current[stream];
    if (replaced && originalId !== undefined) {
      setCaptionTokens((previous) => previous.map((item) =>
        item.id === originalId && item.automatic && item.text === replaced ? { ...item, text } : item
      ));
      return;
    }
    const item = { id: ++nextCaptionIdRef.current, text, automatic: true };
    if (stream) lastAutomaticCaptionRef.current[stream] = item.id;
    setCaptionTokens((previous) => [...previous, item]);
  };

  const updateWordSuggestions = (data, templateOnly = false) => {
    if (data.scope_rejected || data.word_scope_valid === false) return;
    const uniqueWords = new Map();
    for (const item of sanitizeWordTopK(data.word_candidates)) {
      const key = item.word?.trim().toLocaleLowerCase('vi');
      if (!key || item.word_scope_valid === false || !Number.isFinite(item.confidence) ||
          item.confidence < WORD_SUGGESTION_MIN || item.confidence > 1) continue;
      if (templateOnly && item.source !== 'qipedc_template') continue;
      const previous = uniqueWords.get(key);
      if (!previous || item.confidence > previous.confidence) uniqueWords.set(key, item);
    }
    const suggestions = [...uniqueWords.values()]
      .sort((a, b) => b.confidence - a.confidence)
      .slice(0, WORD_SUGGESTION_LIMIT);
    if (suggestions.length) setWordSuggestions(suggestions);
  };

  // Bộ đệm & biến đo lường hiệu năng
  const lastTimeRef = useRef(performance.now());
  const frameCountRef = useRef(0);
  const sequenceBufferRef = useRef([]); // Lưu 30 frames x 75 joints x 3 coords
  const isInferringRef = useRef(false);
  const isFrameBusyRef = useRef(false); // Frame-dropping lock chống nghẽn luồng xử lý
  const lastMotionUpdateRef = useRef(0);

  // Đồng bộ mode & TTS thời gian thực chống Stale Closure & Out-of-Order In-Flight Requests
  const conversionModeRef = useRef(conversionMode);
  const recognitionModeRef = useRef(recognitionMode);
  const ttsEnabledRef = useRef(ttsEnabled);
  const requestGenRef = useRef(0);
  const instantHadHandsRef = useRef(false);
  const instantNoHandCountRef = useRef(0);
  const instantReleasePendingRef = useRef(false);
  const ttsOrchestratorRef = useRef(null);
  if (!ttsOrchestratorRef.current) {
    ttsOrchestratorRef.current = createTtsOrchestrator({
      speakFn: speakVietnamese,
      cancelFn: cancelVietnameseSpeech,
    });
  }
  const instantCameraSessionRef = useRef(null);
  const instantFramesRef = useRef([]);
  const instantHandAtRef = useRef(-Infinity);
  const monotonicTsGenRef = useRef(createMonotonicTimestampGenerator());
  const lastPilotTrialKeyRef = useRef(null);
  const resetInstantCameraSession = (active) => {
    instantCameraSessionRef.current = active ? (globalThis.crypto?.randomUUID?.() ??
      `camera-${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`) : null;
    instantFramesRef.current = [];
    instantHandAtRef.current = -Infinity;
  };

  useEffect(() => {
    initSpeechSynthesisVoices();
  }, []);

  useEffect(() => {
    ttsEnabledRef.current = ttsEnabled;
    if (!ttsEnabled) {
      ttsOrchestratorRef.current?.reset({ cancelAudio: true });
    }
  }, [ttsEnabled]);

  useEffect(() => {
    conversionModeRef.current = conversionMode;
  }, [conversionMode]);

  useEffect(() => {
    recognitionModeRef.current = recognitionMode;
  }, [recognitionMode]);

  // Controlled Webcam Pilot State (Task 7F.5B / Task 11.2.2)
  const [showPilotHUD, setShowPilotHUD] = useState(false);
  const showPilotHUDRef = useRef(showPilotHUD);
  showPilotHUDRef.current = showPilotHUD;
  const cameraFpsRef = useRef(0);
  const pilotMetadataProviderRef = useRef(null);
  const onPilotFrameResultRef = useRef(null);
  const [latestPilotResult, setLatestPilotResult] = useState(null);

  const handlePilotMetadataUpdate = ({ getFrameMetadata, onFrameResult }) => {
    pilotMetadataProviderRef.current = getFrameMetadata;
    onPilotFrameResultRef.current = onFrameResult;
  };

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
    }
  };

  // Trích xuất 75 khớp từ MediaPipe Holistic (làm tròn 4 chữ số thập phân để giảm ~68% dung lượng JSON)
  const extract75KeypointsFromHolistic = (results) => {
    const safeCoord = (v) => (Number.isFinite(v) ? Math.round(v * 10000) / 10000 : 0);
    const keypoints = [];
    if (results.poseLandmarks && results.poseLandmarks.length >= 33) {
      for (let i = 0; i < 33; i++) {
        const lm = results.poseLandmarks[i];
        keypoints.push([safeCoord(lm?.x), safeCoord(lm?.y), safeCoord(lm?.z)]);
      }
    } else {
      for (let i = 0; i < 33; i++) keypoints.push([0, 0, 0]);
    }

    if (results.leftHandLandmarks && results.leftHandLandmarks.length >= 21) {
      for (let i = 0; i < 21; i++) {
        const lm = results.leftHandLandmarks[i];
        keypoints.push([safeCoord(lm?.x), safeCoord(lm?.y), safeCoord(lm?.z)]);
      }
    } else {
      for (let i = 0; i < 21; i++) keypoints.push([0, 0, 0]);
    }

    if (results.rightHandLandmarks && results.rightHandLandmarks.length >= 21) {
      for (let i = 0; i < 21; i++) {
        const lm = results.rightHandLandmarks[i];
        keypoints.push([safeCoord(lm?.x), safeCoord(lm?.y), safeCoord(lm?.z)]);
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

  // Tạo bộ theo dõi tăng tốc GPU (HandLandmarker + PoseLandmarker Lite) với fallback WASM an toàn
  const createTracker = async (engine) => {
    return createAcceleratedTracker({
      engine,
      detailed: detailedTrackingRef.current,
      onHandsResults: onHandsResultsFast,
      onHolisticResults: onHolisticResultsFast,
    });
  };

  // Hiển thị trạng thái khi Holistic không khả dụng ở mode 'words' (ngừng dự đoán, không crash)
  const applyHolisticUnavailableState = (reasonMsg) => {
    holisticUnavailableRef.current = true;
    activeInstanceEngineRef.current = null;
    setHolisticStatus('unavailable');
    sequenceBufferRef.current = [];
    windowTimestampsRef.current = [];
    poseZeroHistoryRef.current = [];
    updateHUDProgress(0, 0);
    if (motionBadgeRef.current) {
      motionBadgeRef.current.style.opacity = '1';
      motionBadgeRef.current.innerText = '🚫 Cần Holistic';
    }
    setCurrentResult({
      sign: 'NEUTRAL',
      label: 'Holistic không khả dụng — Ngừng dự đoán',
      confidence: 0,
      type: 'holistic_unavailable',
      description: reasonMsg || 'Chế độ Từ vựng yêu cầu MediaPipe Holistic (75 khớp thật). Đã ngừng dự đoán để không đưa Pose giả vào SSTCN.',
      category: 'Kiểm soát Landmark',
      video_url: ''
    });
    setWordSuggestions([]);
    logWordDebugEvent('HOLISTIC_UNAVAILABLE', {
      reason: reasonMsg || 'holistic_init_failed',
      tracker_engine: trackerEngineRef.current,
      active_instance_engine: 'none'
    });
  };

  // Khởi động Camera và MediaPipe với tối ưu hóa tối đa
  const startCamera = async (engine = trackerEngineRef.current) => {
    setCameraLoading(true);
    resetInstantCameraSession(true);
    setCameraError(null);
    setWordSuggestions([]);
    isFrameBusyRef.current = false;

    const initialEngine = conversionModeRef.current === 'words' ? 'holistic' : engine;

    try {
      let tracker = null;
      try {
        perfControllerRef.current.reset();
        tracker = await createTracker(initialEngine);
        trackerInstanceRef.current = tracker;
        activeInstanceEngineRef.current = initialEngine;
        trackerEngineRef.current = initialEngine;
        setTrackerEngine(initialEngine);
        setRecognitionScope(initialEngine);
        if (initialEngine === 'holistic') {
          holisticUnavailableRef.current = false;
          setHolisticStatus('ready');
        }
      } catch (trackerErr) {
        if (conversionModeRef.current === 'words' || initialEngine === 'holistic') {
          // Chế độ words: không fallback sang Hands, đánh dấu Holistic không khả dụng và ngừng dự đoán an toàn
          trackerInstanceRef.current = null;
          trackerEngineRef.current = 'holistic';
          setTrackerEngine('holistic');
          setRecognitionScope('holistic');
          applyHolisticUnavailableState(trackerErr.message);
        } else {
          throw trackerErr;
        }
      }

      // Khởi động luồng webcam trực tiếp bằng getUserMedia với thông số tối ưu tuyệt đối
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          width: { ideal: 960 },
          height: { ideal: 720 },
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
        cameraActiveRef.current = true;

        // A01.1: Kiểm tra lại mode sau khi getUserMedia/video.play hoàn tất (chống race condition khi bấm đổi sang 'words' lúc camera đang mở)
        const postStartRequiredEngine = conversionModeRef.current === 'words' ? 'holistic' : trackerEngineRef.current;
        if (
          !holisticUnavailableRef.current &&
          (!trackerInstanceRef.current || trackerInstanceRef.current._engineType !== postStartRequiredEngine)
        ) {
          try {
            const oldTracker = trackerInstanceRef.current;
            const syncedTracker = await createTracker(postStartRequiredEngine);
            trackerInstanceRef.current = syncedTracker;
            activeInstanceEngineRef.current = postStartRequiredEngine;
            trackerEngineRef.current = postStartRequiredEngine;
            setTrackerEngine(postStartRequiredEngine);
            setRecognitionScope(postStartRequiredEngine);
            if (oldTracker && typeof oldTracker.close === 'function') {
              try { oldTracker.close(); } catch (e) {}
            }
          } catch (syncErr) {
            if (postStartRequiredEngine === 'holistic') {
              if (trackerInstanceRef.current && typeof trackerInstanceRef.current.close === 'function') {
                try { trackerInstanceRef.current.close(); } catch (e) {}
              }
              trackerInstanceRef.current = null;
              applyHolisticUnavailableState(syncErr.message);
            }
          }
        }

        // Khởi tạo Canvas Downsampler GPU-friendly (không ép willReadFrequently để giữ tăng tốc phần cứng)
        if (!processCanvasRef.current) {
          const pCanvas = document.createElement('canvas');
          pCanvas.width = 480;
          pCanvas.height = 360;
          processCanvasRef.current = pCanvas;
        }
        if (!holisticCanvasRef.current) {
          const hCanvas = document.createElement('canvas');
          hCanvas.width = 480;
          hCanvas.height = 360;
          holisticCanvasRef.current = hCanvas;
        }
        const pCanvas = processCanvasRef.current;
        const pCtx = pCanvas.getContext('2d');
        const hCanvas = holisticCanvasRef.current;
        const hCtx = hCanvas.getContext('2d');

        // Vòng lặp gửi frame thích ứng phần cứng (Zero-Copy GPU Texture trên máy có WebGL, tự động giảm tải Pose trên máy yếu)
        const processCameraFrame = async () => {
          if (!videoRef.current || videoRef.current.paused || videoRef.current.ended) {
            return;
          }

          const activeTracker = trackerInstanceRef.current;
          const activeMode = conversionModeRef.current;

          // Chốt chặn A01.1: Ở chế độ 'words', nếu Holistic không khả dụng hoặc instance không phải 'holistic' -> tạm ngừng gửi frame, không crash
          if (activeMode === 'words' && (holisticUnavailableRef.current || !activeTracker || activeTracker._engineType !== 'holistic')) {
            animationFrameIdRef.current = requestAnimationFrame(processCameraFrame);
            return;
          }

          if (!activeTracker) {
            animationFrameIdRef.current = requestAnimationFrame(processCameraFrame);
            return;
          }

          const now = performance.now();
          const detailed = detailedTrackingRef.current;
          const perfPlan = perfControllerRef.current.nextFramePlan({ detailed });
          if (!isFrameBusyRef.current && videoRef.current.readyState >= 2 && now - lastFrameSentTimeRef.current >= perfPlan.minIntervalMs) {
            isFrameBusyRef.current = true;
            lastFrameSentTimeRef.current = now;
            try {
              const video = videoRef.current;
              const currentEngine = activeTracker._engineType || trackerEngineRef.current;
              const isGpuDirect = activeTracker._delegate === 'GPU' && perfPlan.tier !== 'eco';
              let imageToSend;

              if (currentEngine === 'holistic') {
                const vw = video.videoWidth || 640;
                const vh = video.videoHeight || 480;
                const scale = Math.min(1, perfPlan.targetMaxDim / Math.max(vw, vh));
                const width = Math.round(vw * scale);
                const height = Math.round(vh * scale);
                if (hCanvas.width !== width || hCanvas.height !== height) {
                  hCanvas.width = width;
                  hCanvas.height = height;
                }
                if (isGpuDirect) {
                  imageToSend = video;
                } else {
                  hCtx.drawImage(video, 0, 0, hCanvas.width, hCanvas.height);
                  imageToSend = hCanvas;
                }
              } else if (isGpuDirect) {
                imageToSend = video;
              } else {
                const vw = video.videoWidth || 640;
                const vh = video.videoHeight || 480;
                const scale = Math.min(1, perfPlan.targetMaxDim / Math.max(vw, vh));
                const width = Math.round(vw * scale);
                const height = Math.round(vh * scale);
                if (pCanvas.width !== width || pCanvas.height !== height) {
                  pCanvas.width = width;
                  pCanvas.height = height;
                }
                pCtx.drawImage(video, 0, 0, pCanvas.width, pCanvas.height);
                imageToSend = pCanvas;
              }
              const sendStart = performance.now();
              lastFrameSendTimeRef.current = sendStart;
              await activeTracker.send({ image: imageToSend, runPose: perfPlan.runPose });
              perfControllerRef.current.recordFrameMs(performance.now() - sendStart);
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
      cameraActiveRef.current = false;
    } finally {
      setCameraLoading(false);
    }
  };

  // Dừng Camera
  const stopCamera = () => {
    requestGenRef.current += 1;
    resetInstantCameraSession(false);
    isInferringRef.current = false;
    diagnostic.stop();
    practiceTrackingReadyRef.current = false;
    setPracticeTrackingReady(false);
    recordTimersRef.current.forEach(clearTimeout);
    recordTimersRef.current = [];
    recordPhaseRef.current = 'idle';
    setRecordPhase('idle');
    setRecordCountdown(0);
    if (animationFrameIdRef.current) {
      cancelAnimationFrame(animationFrameIdRef.current);
      animationFrameIdRef.current = null;
    }
    if (cameraInstanceRef.current) {
      cameraInstanceRef.current.stop();
      cameraInstanceRef.current = null;
    }
    if (trackerInstanceRef.current) {
      try {
        if (typeof trackerInstanceRef.current.close === 'function') {
          trackerInstanceRef.current.close();
        }
      } catch (e) {}
      trackerInstanceRef.current = null;
    }
    activeInstanceEngineRef.current = null;
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
    cameraActiveRef.current = false;
    isFrameBusyRef.current = false;
    isInferringRef.current = false;
    requestGenRef.current += 1;
    instantHadHandsRef.current = false;
    instantNoHandCountRef.current = 0;
    instantReleasePendingRef.current = false;
    ttsOrchestratorRef.current?.reset({ cancelAudio: true });
    sequenceBufferRef.current = [];
    windowTimestampsRef.current = [];
    liveWordsRef.current = newLiveWordState();
    poseZeroHistoryRef.current = [];
    missingHandsCountRef.current = 0;
    missingPoseCountRef.current = 0;
    prevLeftHandRef.current = null;
    prevRightHandRef.current = null;
    stabilizerRef.current?.reset();
    monotonicTsGenRef.current.reset(0);
    lastPilotTrialKeyRef.current = null;

    const canvas = canvasRef.current;
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
    }
  };

  // Đổi động cơ tracking (Hands vs Holistic)
  const switchTrackerEngine = async (requestedEngine, reload = false) => {
    // A01.1: Chế độ 'words' bắt buộc khóa trên 'holistic', không cho phép chuyển sang 'hands'
    let newEngine = requestedEngine;
    if (conversionModeRef.current === 'words' && newEngine !== 'holistic') {
      logWordDebugEvent('ENGINE_FALLBACK_BLOCKED', {
        reason: 'words_mode_locks_holistic',
        requested_engine: newEngine,
        tracker_engine: trackerEngineRef.current,
        active_instance_engine: activeInstanceEngineRef.current || 'none'
      });
      newEngine = 'holistic';
    }

    const isAlreadySynced =
      newEngine === trackerEngineRef.current &&
      (!cameraActiveRef.current || activeInstanceEngineRef.current === newEngine) &&
      !(newEngine === 'holistic' && holisticUnavailableRef.current);

    if (isAlreadySynced && !reload) return;

    trackerEngineRef.current = newEngine;
    setTrackerEngine(newEngine);
    setRecognitionScope(newEngine);

    if (newEngine !== 'holistic') {
      holisticUnavailableRef.current = false;
      setHolisticStatus('ready');
    }

    if (cameraActiveRef.current && videoRef.current) {
      isFrameBusyRef.current = true;
      try {
        perfControllerRef.current.reset();
        const oldTracker = trackerInstanceRef.current;
        const newTracker = await createTracker(newEngine);
        trackerInstanceRef.current = newTracker;
        activeInstanceEngineRef.current = newEngine;
        if (newEngine === 'holistic') {
          holisticUnavailableRef.current = false;
          setHolisticStatus('ready');
        }
        if (oldTracker) {
          try {
            if (typeof oldTracker.close === 'function') oldTracker.close();
          } catch (e) {}
        }
        sequenceBufferRef.current = [];
        windowTimestampsRef.current = [];
        poseZeroHistoryRef.current = [];
        missingHandsCountRef.current = 0;
        missingPoseCountRef.current = 0;
        prevLeftHandRef.current = null;
        prevRightHandRef.current = null;
        stabilizerRef.current?.reset();
      } catch (err) {
        console.warn('Lỗi chuyển đổi động cơ tracking:', err);
        if (newEngine === 'holistic') {
          // Tuyệt đối không giữ lại oldTracker (Hands) chạy ngầm sinh DEFAULT_POSE giả!
          const oldTracker = trackerInstanceRef.current;
          trackerInstanceRef.current = null;
          if (oldTracker) {
            try {
              if (typeof oldTracker.close === 'function') oldTracker.close();
            } catch (e) {}
          }
          applyHolisticUnavailableState(err.message);
        }
      } finally {
        isFrameBusyRef.current = false;
      }
    }
  };

  const changeDetailedTracking = (enabled) => {
    detailedTrackingRef.current = enabled;
    setDetailedTracking(enabled);
    requestGenRef.current += 1;
    isInferringRef.current = false;
    liveWordsRef.current = newLiveWordState();
    setWordSuggestions([]);
    setCurrentResult({ sign: 'NEUTRAL', label: 'Đang chờ động tác từ vựng…', confidence: 0 });
    if (cameraActiveRef.current) switchTrackerEngine('holistic', true);
  };

  // Xử lý chuyển đổi độc lập TIÊU CHÍ A: Phạm vi nhận diện
  const handleScopeChange = (newScope) => {
    if (conversionModeRef.current === 'words' && newScope !== 'holistic') {
      logWordDebugEvent('ENGINE_FALLBACK_BLOCKED', {
        reason: 'manual_hands_scope_blocked_in_words_mode',
        requested_engine: newScope,
        tracker_engine: trackerEngineRef.current,
        active_instance_engine: activeInstanceEngineRef.current || 'none'
      });
      return;
    }
    if (newScope === recognitionScope && (!cameraActiveRef.current || activeInstanceEngineRef.current === newScope)) return;
    switchTrackerEngine(newScope);
  };

  // Xử lý chuyển đổi độc lập TIÊU CHÍ B: Chế độ chuyển đổi
  const handleModeChange = (newMode) => {
    if (newMode === conversionMode) return;
    diagnostic.stop();
    const newRecMode = (newMode === 'letters' || newMode === 'digits' || newMode === 'letters_numbers') ? 'instant' : 'sequence';

    // Đồng bộ lập tức vào refs để callback MediaPipe đọc được ngay trong frame kế tiếp
    conversionModeRef.current = newMode;
    recognitionModeRef.current = newRecMode;
    requestGenRef.current += 1; // Vô hiệu hoá tất cả response đang bay (in-flight) của mode cũ
    isInferringRef.current = false;
    instantHadHandsRef.current = false;
    instantNoHandCountRef.current = 0;
    instantReleasePendingRef.current = false;
    ttsOrchestratorRef.current?.reset({ cancelAudio: true });
    monotonicTsGenRef.current.reset(0);
    lastPilotTrialKeyRef.current = null;

    setConversionMode(newMode);
    resetInstantCameraSession(cameraActiveRef.current);
    setRecognitionMode(newRecMode);
    sequenceBufferRef.current = [];
    windowTimestampsRef.current = [];
    liveWordsRef.current = newLiveWordState();
    poseZeroHistoryRef.current = [];
    missingHandsCountRef.current = 0;
    missingPoseCountRef.current = 0;
    stabilizerRef.current?.reset();
    updateHUDProgress(0, 0);

    // CHỈ áp dụng Holistic cho duy nhất chế độ 'words' (Từ vựng SSTCN 102)
    // Các chế độ khác ('letters', 'digits') giữ nguyên tuyệt đối trên MediaPipe Hands
    const targetScope = (newMode === 'words') ? 'holistic' : 'hands';
    switchTrackerEngine(targetScope);

    // Nếu chuyển sang chế độ digits, chuẩn hóa phụ đề hiện tại để bỏ từ "Số "
    if (newMode === 'digits') {
      setCaptionTokens((previous) => previous.map((item) => ({
        ...item, text: item.text.replace(/(?:Số|số)\s+(\d+)/g, '$1'),
      })));
    }

    // Reset kết quả cũ để tránh hiển thị nhầm lẫn giữa các chế độ (bảo tồn sentence đã xác nhận)
    if (!(newMode === 'words' && holisticUnavailableRef.current)) {
      setCurrentResult({
        sign: 'NEUTRAL',
        label: newMode === 'letters' ? 'Chờ cử chỉ chữ cái...' : newMode === 'digits' ? 'Chờ cử chỉ chữ số...' : 'Chờ cử chỉ từ vựng...',
        confidence: 0,
        type: '',
        description: '',
        category: '',
        video_url: ''
      });
    }
    setWordSuggestions([]);
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

    const isHolistic = trackerEngineRef.current === 'holistic';
    // Holistic đã tắt smoothing nên chỉ bù nhẹ 0.20; Hands giữ 0.55
    const LATENCY_COMPENSATION = isHolistic ? 0.20 : 0.55;
    const maxV = isHolistic ? 0.12 : 0.06;

    return currentLms.map((curr, i) => {
      const vx = curr.x - prev[i].x;
      const vy = curr.y - prev[i].y;

      const clampedVx = Math.max(-maxV, Math.min(maxV, vx));
      const clampedVy = Math.max(-maxV, Math.min(maxV, vy));

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
  const drawFastHand = (ctx, lms, strokeColor, jointColor, label, w, h, bridged = false) => {
    if (!lms || lms.length < 21) return;

    ctx.save();
    if (bridged) {
      ctx.globalAlpha = 0.65;
      ctx.setLineDash([5, 4]);
    }
    // 1. Gom tất cả 21 đoạn xương vào 1 đường path duy nhất
    ctx.beginPath();
    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = bridged ? 2.5 : 3;
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
    ctx.setLineDash([]);
    ctx.beginPath();
    ctx.fillStyle = jointColor;
    const RADIUS = bridged ? 2.8 : 3.5;
    for (let i = 0; i < 21; i++) {
      const p = lms[i];
      const cx = p.x * w;
      const cy = p.y * h;
      ctx.moveTo(cx + RADIUS, cy);
      ctx.arc(cx, cy, RADIUS, 0, Math.PI * 2);
    }
    ctx.fill();

    // 3. Nhãn bàn tay (đảo ngược trục X cục bộ để chữ không bị lật ngược khi canvas dùng CSS scaleX(-1))
    const wrist = lms[0];
    if (wrist) {
      ctx.save();
      ctx.translate(wrist.x * w, wrist.y * h + 22);
      ctx.scale(-1, 1);
      ctx.font = 'bold 12px "Plus Jakarta Sans", sans-serif';
      ctx.fillStyle = strokeColor;
      ctx.textAlign = 'center';
      ctx.fillText(bridged ? `${label} (bù)` : label, 0, 0);
      ctx.restore();
    }
    ctx.restore();
  };

  // Cập nhật trực tiếp trạng thái khung hình camera qua DOM (0 re-render React)
  const updateFramingHUD = (pose, leftHand, rightHand, mode) => {
    const now = performance.now();
    if (now - lastFramingUpdateRef.current < 180) return;
    lastFramingUpdateRef.current = now;

    const check = evaluateCameraFraming(pose, leftHand, rightHand, mode);
    framingStateRef.current = check;
    const perfTier = perfControllerRef.current.tier;
    const isGpu = trackerInstanceRef.current?._delegate === 'GPU';
    const engineSuffix = perfTier === 'eco' ? ' · Tiết kiệm' : (isGpu ? ' · GPU' : '');

    if (framingBadgeRef.current) {
      framingBadgeRef.current.className = check.ok ? 'badge badge-emerald' : 'badge badge-amber';
      framingBadgeRef.current.innerText = `${check.ok ? '✓' : '⚠'} ${check.badge}${engineSuffix}`;
    }
    if (framingNoticeRef.current) {
      if (!check.ok && check.message) {
        framingNoticeRef.current.hidden = false;
        framingNoticeRef.current.innerText = check.message;
      } else {
        framingNoticeRef.current.hidden = true;
      }
    }
  };

  // Render toàn bộ khung xương trong suốt + khung căn chỉnh góc camera siêu nhẹ
  const renderSkeletonsOverlay = (ctx, leftHandRaw, rightHandRaw, upperPose, canvas, bridgeFlags = null) => {
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);

    // Vẽ 4 góc căn khung an toàn (đổi màu hổ phách nếu người dùng khuất vai hoặc tay chạm rìa)
    const frameOk = framingStateRef.current?.ok !== false;
    const padX = Math.round(w * 0.03);
    const padY = Math.round(h * 0.03);
    const corner = Math.max(14, Math.round(Math.min(w, h) * 0.045));
    ctx.save();
    ctx.strokeStyle = frameOk ? 'rgba(16, 185, 129, 0.32)' : 'rgba(245, 158, 11, 0.80)';
    ctx.lineWidth = frameOk ? 1.5 : 2.5;
    ctx.beginPath();
    ctx.moveTo(padX, padY + corner); ctx.lineTo(padX, padY); ctx.lineTo(padX + corner, padY);
    ctx.moveTo(w - padX - corner, padY); ctx.lineTo(w - padX, padY); ctx.lineTo(w - padX, padY + corner);
    ctx.moveTo(padX, h - padY - corner); ctx.lineTo(padX, h - padY); ctx.lineTo(padX + corner, h - padY);
    ctx.moveTo(w - padX - corner, h - padY); ctx.lineTo(w - padX, h - padY); ctx.lineTo(w - padX, h - padY - corner);
    ctx.stroke();
    ctx.restore();

    const leftHand = leftHandRaw;
    const rightHand = rightHandRaw;

    // 1. Thân trên (Pose - nếu dùng Holistic)
    if (upperPose && window.drawConnectors && window.POSE_CONNECTIONS) {
      window.drawConnectors(ctx, upperPose.slice(0, 25), window.POSE_CONNECTIONS, {
        color: 'rgba(129, 140, 248, 0.45)',
        lineWidth: 2
      });
    }

    // 2. Tay trái (Emerald Neon)
    if (leftHand) {
      drawFastHand(ctx, leftHand, '#10b981', '#ffffff', '🤚 Trái', w, h, Boolean(bridgeFlags?.leftBridged));
    }

    // 3. Tay phải (Cyan Neon)
    if (rightHand) {
      drawFastHand(ctx, rightHand, '#06b6d4', '#ffffff', '✋ Phải', w, h, Boolean(bridgeFlags?.rightBridged));
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
    const extractTs = getEpochSecondsNow();
    measureFPS();

    // Chốt chặn an toàn A01.1: Chế độ 'words' tuyệt đối không nhận frame từ Hands
    if (conversionModeRef.current === 'words') {
      return;
    }

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    let leftHand = null;
    let rightHand = null;
    let leftScore = null;
    let rightScore = null;

    if (results.multiHandLandmarks) {
      results.multiHandLandmarks.forEach((lms, idx) => {
        const label = results.multiHandedness?.[idx]?.label || (idx === 0 ? 'Right' : 'Left');
        const score = typeof results.multiHandedness?.[idx]?.score === 'number'
          ? results.multiHandedness[idx].score
          : null;
        if (label === 'Left') {
          leftHand = lms;
          leftScore = score;
        } else {
          rightHand = lms;
          rightScore = score;
        }
      });
    }

    // Ổn định tay qua Zero-Lag One-Euro Filter + dự đoán động học khi vung nhanh
    const stabilizedHands = stabilizerRef.current.processHands(leftHand, rightHand, extractTs * 1000);
    const drawLeft = stabilizedHands.left;
    const drawRight = stabilizedHands.right;

    updateFramingHUD(null, drawLeft, drawRight, conversionModeRef.current);
    diagnostic.trackFrame(drawLeft, drawRight, null, {
      fps: cameraFpsRef.current,
      leftBridged: stabilizedHands.leftBridged,
      rightBridged: stabilizedHands.rightBridged,
    });
    renderSkeletonsOverlay(ctx, drawLeft, drawRight, null, canvas, {
      leftBridged: stabilizedHands.leftBridged,
      rightBridged: stabilizedHands.rightBridged,
    });

    handleMotionAndSequence(
      () => extract75KeypointsFromHands(results),
      Boolean(drawLeft || drawRight),
      drawRight?.[0] || drawLeft?.[0],
      drawLeft,
      drawRight,
      {
        landmarkExtractTs: extractTs,
        observedHands: { left: leftHand, right: rightHand },
        handednessScore: rightHand ? rightScore : (leftHand ? leftScore : null),
        modelComplexity: 1,
        processingResolution: '640x480'
      }
    );
  };

  // Xử lý luồng kết quả từ MediaPipe Holistic (Zero-Lag Stabilizer & Chống rung giật)
  const onHolisticResultsFast = (results) => {
    measureFPS();
    const nowPerf = performance.now();

    // Ổn định thị giác và tách đôi tay qua Zero-Lag Stabilizer
    const stabilized = stabilizerRef.current.process(
      results.poseLandmarks,
      results.leftHandLandmarks,
      results.rightHandLandmarks,
      nowPerf
    );
    const modelResults = {
      poseLandmarks: stabilized.pose || results.poseLandmarks,
      leftHandLandmarks: stabilized.modelLeft,
      rightHandLandmarks: stabilized.modelRight,
    };
    const usesLiveWords = conversionModeRef.current === 'words' && wordCaptureModeRef.current === 'continuous' &&
      recordPhaseRef.current === 'idle';
    // Keep extrapolation for drawing. Real observations delimit the recognition sequence;
    // the backend can interpolate short gaps between observed landmarks.
    const isCalibration = recordPhaseRef.current === 'recording' && recordPurposeRef.current === 'calibrate';
    const recognitionResults = usesLiveWords || isCalibration ? {
      ...modelResults,
      leftHandLandmarks: stabilized.leftBridged ? null : stabilized.modelLeft,
      rightHandLandmarks: stabilized.rightBridged ? null : stabilized.modelRight,
    } : modelResults;

    const diagBufLen = sequenceBufferRef.current.length;
    const diagBufDurationMs = diagBufLen >= 2 ? Math.round(nowPerf - windowTimestampsRef.current[0]) : 0;
    diagnostic.trackFrame(stabilized.modelLeft, stabilized.modelRight, modelResults.poseLandmarks || [], {
      fps: cameraFpsRef.current,
      leftBridged: stabilized.leftBridged,
      rightBridged: stabilized.rightBridged,
      maxWristSpeed: stabilized.maxWristSpeed,
      handsClasped: stabilized.handsClasped,
      bufLen: diagBufLen,
      bufDurationMs: diagBufDurationMs,
    });
    if (practiceEntry) {
      const left = modelResults.poseLandmarks?.[11];
      const right = modelResults.poseLandmarks?.[12];
      if (!practiceTrackingReadyRef.current && left && right && (left.visibility ?? 1) >= 0.5 && (right.visibility ?? 1) >= 0.5) {
        practiceTrackingReadyRef.current = true;
        setPracticeTrackingReady(true);
      }
    }
    const extractTs = getEpochSecondsNow();

    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');

    updateFramingHUD(modelResults.poseLandmarks, stabilized.drawLeft, stabilized.drawRight, conversionModeRef.current);

    // Vẽ trực tiếp landmarks đã lọc Zero-Lag (không rung giật khi dừng tay, bám sát khi vung)
    renderSkeletonsOverlay(
      ctx,
      stabilized.drawLeft,
      stabilized.drawRight,
      modelResults.poseLandmarks,
      canvas,
      { leftBridged: stabilized.leftBridged, rightBridged: stabilized.rightBridged }
    );

    // POSE WRIST GUARD: Phát hiện cổ tay còn giơ cao trong không gian tương tác (ngăn Flush non khi bị occlusion giữa cử chỉ)
    const poseLms = modelResults.poseLandmarks || results.poseLandmarks;
    let poseWristsRaised = false;
    if (poseLms && poseLms.length >= 17) {
      const leftShoulderY = poseLms[11]?.y;
      const rightShoulderY = poseLms[12]?.y;
      const leftWristY = (poseLms[15]?.visibility ?? 1) >= 0.25 ? poseLms[15]?.y : undefined;
      const rightWristY = (poseLms[16]?.visibility ?? 1) >= 0.25 ? poseLms[16]?.y : undefined;
      if (typeof leftShoulderY === 'number' && typeof rightShoulderY === 'number') {
        const shoulderY = (leftShoulderY + rightShoulderY) / 2;
        const shoulderWidth = Math.max(0.08, Math.hypot((poseLms[11]?.x || 0) - (poseLms[12]?.x || 0), leftShoulderY - rightShoulderY));
        const maxRaisedY = shoulderY + 0.85 * shoulderWidth;
        if ((typeof leftWristY === 'number' && leftWristY <= maxRaisedY) ||
            (typeof rightWristY === 'number' && rightWristY <= maxRaisedY)) {
          poseWristsRaised = true;
        }
      }
    }

    handleMotionAndSequence(
      () => extract75KeypointsFromHolistic(recognitionResults),
      Boolean(recognitionResults.leftHandLandmarks || recognitionResults.rightHandLandmarks),
      stabilized.drawRight?.[0] || stabilized.drawLeft?.[0],
      recognitionResults.leftHandLandmarks,
      recognitionResults.rightHandLandmarks,
      {
        landmarkExtractTs: extractTs,
        observedHands: {
          left: stabilized.leftBridged ? null : recognitionResults.leftHandLandmarks,
          right: stabilized.rightBridged ? null : recognitionResults.rightHandLandmarks,
        },
        handednessScore: null,
        maxWristSpeed: stabilized.maxWristSpeed,
        maxFingerSpeed: stabilized.maxFingerSpeed,
        handsClasped: stabilized.handsClasped,
        modelComplexity: detailedTrackingRef.current ? 1 : 0,
        processingResolution: holisticCanvasRef.current
          ? `${holisticCanvasRef.current.width}x${holisticCanvasRef.current.height}` : null,
        poseWristsRaised,
      }
    );
  };

  // Đo FPS 1 lần mỗi giây & ghi nhận thời gian xử lý frame (phục vụ debug)
  const measureFPS = () => {
    frameCountRef.current += 1;
    const now = performance.now();
    if (lastFrameSendTimeRef.current > 0) {
      const lat = now - lastFrameSendTimeRef.current;
      frameLatencyHistoryRef.current.push(lat);
      if (frameLatencyHistoryRef.current.length > 30) {
        frameLatencyHistoryRef.current.shift();
      }
    }
    if (now - lastTimeRef.current >= 1000) {
      const currentFps = frameCountRef.current;
      cameraFpsRef.current = currentFps;
      setFps(currentFps);
      if (fpsBadgeRef.current) {
        fpsBadgeRef.current.innerText = `${currentFps} FPS`;
      }
      frameCountRef.current = 0;
      lastTimeRef.current = now;
    } else if (cameraFpsRef.current === 0 && now - lastTimeRef.current >= 250) {
      cameraFpsRef.current = Math.round((frameCountRef.current * 1000) / Math.max(1, now - lastTimeRef.current));
    }
  };

  /**
   * TỐI ƯU CỐT LÕI: SLIDING WINDOW 30-FRAME SUY LUẬN TỨC THÌ (300MS)
   * Thay vì chờ 2 giây từ đầu, duy trì cửa sổ trượt 30 frame và đánh giá mỗi 6 frame (~300ms).
   * Giúp nhận diện cử chỉ tức thì ngay khi chuyển động kết thúc!
   */
  const handleMotionAndSequence = (getKeypointsFn, hasHands, primaryWrist, leftHand, rightHand, diagMeta = null) => {
    const now = performance.now();
    const allowShort = isShortGesturesEnabled();
    const poseWristsRaised = Boolean(diagMeta?.poseWristsRaised);

    if (hasHands) {
      isStandardFlushedRef.current = false;
    }

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

    if (currentConversionMode === 'words' && practiceEntry) return;
    if (currentConversionMode === 'words' && recordPhaseRef.current === 'preparing') return;

    // 2. CHẾ ĐỘ 1: CHUỖI ĐỘNG VSL (SSTCN-Attention 30-Frame Sliding Window)
    if (currentRecognitionMode === 'sequence') {
      const liveWords = currentConversionMode === 'words' && wordCaptureModeRef.current === 'continuous' &&
        recordPhaseRef.current === 'idle';
      let liveState = liveWordsRef.current;
      let isEndOfGestureFlush = false;
      if (liveWords) {
        const wristSpeed = diagMeta?.maxWristSpeed ?? 0;
        const fingerSpeed = diagMeta?.maxFingerSpeed ?? 0;
        if (hasHands) {
          if (wristSpeed >= 0.025 || fingerSpeed >= 0.018) {
            liveState.hadFastMotion = true;
            liveState.stillSince = null;
            liveState.flushed = false;
          } else if (wristSpeed >= 0.018 || fingerSpeed >= 0.012) {
            liveState.hadFastMotion = true;
            liveState.stillSince = null;
          } else if (liveState.hadFastMotion) {
            liveState.stillSince ??= now;
          }
        }
        // Kết thúc động tác (hạ tay hoặc dừng yên >= 240ms sau khi vung tay): đánh giá ngay 1 lượt cuối
        // thay vì chờ 500ms rồi xóa bỏ buffer — giúp cử chỉ nhanh/vừa nhận đúng ngay lần đầu!
        const bufLen = sequenceBufferRef.current.length;
        const bufDuration = bufLen >= 2 ? now - windowTimestampsRef.current[0] : 0;
        const inPostCommitCooldown = now < liveState.cooldownUntil;
        const settledAfterMotion = Boolean(
          hasHands && liveState.hadFastMotion && liveState.stillSince !== null && (now - liveState.stillSince >= 240)
        );
        const releasedAfterMotion = Boolean(!hasHands && !poseWristsRaised && Number.isFinite(liveState.lastHand) && (now - liveState.lastHand >= 140));
        if (
          !inPostCommitCooldown && !liveState.token && !liveState.template && !liveState.flushed &&
          now - lastFlushTimeRef.current >= 450 &&
          bufLen >= 8 && bufDuration >= 420 && (settledAfterMotion || releasedAfterMotion)
        ) {
          isEndOfGestureFlush = true;
          liveState.flushed = true;
          lastFlushTimeRef.current = now;
          diagnostic.addPipelineEvent(now, 'end_of_gesture_flush', {
            reason: settledAfterMotion ? 'settled_after_motion' : 'released_after_motion',
            bufLen,
            bufDurationMs: Math.round(bufDuration),
            wristSpeed: Number(wristSpeed.toFixed(4)),
          });
        }

        if (!isEndOfGestureFlush && now - liveState.lastHand > 500) {
          // Cho lượt nhận diện đang chạy trả kết quả sau khi hạ tay.
          // Khi tay xuất hiện trở lại, bắt đầu cửa sổ mới ngay cả khi còn yêu cầu cũ.
          if (Number.isFinite(liveState.lastHand) && (hasHands || !isInferringRef.current)) {
            diagnostic.addPipelineEvent(now, 'buffer_reset', {
              reason: 'hands_absent_gt_500ms',
              prevBufLen: sequenceBufferRef.current.length,
              absentMs: Math.round(now - liveState.lastHand),
              prevToken: liveState.token,
              prevTemplateLocked: liveState.template,
              reenteredWithHand: Boolean(hasHands),
            });
            liveWordsRef.current = newLiveWordState();
            sequenceBufferRef.current = [];
            windowTimestampsRef.current = [];
            updateHUDProgress(0, 0);
            if (!liveState.token) {
              setCurrentResult({ sign: 'NEUTRAL', label: 'Đang chờ động tác từ vựng…', confidence: 0 });
            }
            liveState = liveWordsRef.current;
          }
          if (!hasHands) return;
        }
        if (hasHands) liveState.lastHand = now;
      } else if (!hasHands) {
        missingHandsCountRef.current = (missingHandsCountRef.current || 0) + 1;
        const bufLen = sequenceBufferRef.current.length;
        const bufDuration = bufLen >= 2 ? now - windowTimestampsRef.current[0] : 0;

        // End-of-Gesture Flush cho Standard mode:
        // Khi hạ tay (!hasHands && !poseWristsRaised), có >= 12 frame, và mất tay >= 3 frame (~100ms)
        // nhưng chưa tới maxMissingHands (trước khi reset buffer)
        if (
          allowShort &&
          !isStandardFlushedRef.current &&
          !isInferringRef.current &&
          now - lastFlushTimeRef.current >= 450 &&
          bufLen >= 12 &&
          bufDuration >= 350 &&
          missingHandsCountRef.current >= 3 &&
          !poseWristsRaised
        ) {
          isEndOfGestureFlush = true;
          isStandardFlushedRef.current = true;
          lastFlushTimeRef.current = now;
        }

        // Chịu được mất tay liên tiếp khi xoay cạnh tay hoặc chuyển pha cử chỉ nhiều pha:
        // Nếu cổ tay vẫn giơ trong không gian tương tác (poseWristsRaised), cho phép tới 20 frame (~650ms).
        // Nếu đã hạ tay xuống (!poseWristsRaised), chịu 8 frame (~250ms) rồi mới reset buffer.
        const maxMissingHands = poseWristsRaised ? 20 : 8;
        if (missingHandsCountRef.current >= maxMissingHands) {
          if (sequenceBufferRef.current.length > 0) {
            if (currentConversionMode === 'words') {
              logWordDebugReset(
                sequenceBufferRef.current.length,
                'no_hands_detected',
                missingHandsCountRef.current,
                trackerEngineRef.current,
                maxMissingHands
              );
            }
            sequenceBufferRef.current = [];
            windowTimestampsRef.current = [];
            updateHUDProgress(0, 0);
          }
          if (!isEndOfGestureFlush) return;
        } else if (!isEndOfGestureFlush) {
          return;
        }
      }
      missingHandsCountRef.current = 0;

      // Đẩy frame vào bộ đệm và lưu timestamp frame (bỏ qua push frame trống khi đang flush cuối động tác)
      if (!isEndOfGestureFlush || hasHands) {
        const frameKeypoints = getKeypointsFn();

        // Nếu đang ghi theo nhịp 3s (Record-and-Check hoặc Personal Calibration)
        if (recordPhaseRef.current === 'recording') {
          recordFramesRef.current.push(frameKeypoints);
          recordTimestampsRef.current.push(now);
          const recPct = Math.min(100, Math.round((recordFramesRef.current.length / SEQ_LEN) * 100));
          updateHUDProgress(recPct, recordFramesRef.current.length);
          return;
        }
        if (currentConversionMode === 'words' && wordCaptureModeRef.current === 'record') {
          return;
        }

        sequenceBufferRef.current.push(frameKeypoints);
        windowTimestampsRef.current.push(now);
      }

      if (liveWords) {
        while (windowTimestampsRef.current[0] < now - LIVE_QIPEDC_WINDOW_MS || sequenceBufferRef.current.length > 120) {
          sequenceBufferRef.current.shift();
          windowTimestampsRef.current.shift();
        }
      }

      const currentLen = sequenceBufferRef.current.length;
      const progressPercent = Math.min(100, Math.round((currentLen / SEQ_LEN) * 100));
      updateHUDProgress(progressPercent, Math.min(currentLen, SEQ_LEN));

      const enoughFrames = liveWords
        ? (now >= liveState.cooldownUntil && (
          isEndOfGestureFlush ||
          (currentLen >= 8 && now - windowTimestampsRef.current[0] >= 550 && now - liveState.lastRequest >= 220)
        ))
        : (isEndOfGestureFlush || currentLen >= SEQ_LEN);
      const canQueue = liveWords && (hasHands || isEndOfGestureFlush) && !liveState.template && now >= liveState.cooldownUntil;
      if (enoughFrames && (!isInferringRef.current || canQueue)) {
        const currentGen = requestGenRef.current;
        const requestMode = currentConversionMode;
        const requestTopic = requestMode === 'words' ? selectedTopicRef.current || null : null;
        const requestVocabScope = vocabScopeRef.current;
        const finalFlushRequest = isEndOfGestureFlush;
        const isStandardShortFlush = !liveWords && isEndOfGestureFlush;
        const sequencePayload = (liveWords || isStandardShortFlush)
          ? sequenceBufferRef.current.slice(0)
          : sequenceBufferRef.current.slice(-SEQ_LEN);

        // Tính thời lượng thực tế của cửa sổ
        const recentTimestamps = (liveWords || isStandardShortFlush)
          ? windowTimestampsRef.current.slice(0)
          : windowTimestampsRef.current.slice(-SEQ_LEN);
        const windowDurationMs = recentTimestamps.length >= 2
          ? Math.round(recentTimestamps[recentTimestamps.length - 1] - recentTimestamps[0])
          : 0;

        // SLIDING WINDOW: Loại bỏ 6 frame cũ nhất, giữ lại 24 frame để frame tiếp theo nhanh chóng đầy lại!
        if (liveWords) {
          liveState.lastRequest = now;
        } else if (isStandardShortFlush) {
          // Sau khi flush cử chỉ ngắn ở standard mode, xóa sạch buffer ngay để không gửi lặp lại
          sequenceBufferRef.current = [];
          windowTimestampsRef.current = [];
          updateHUDProgress(0, 0);
        } else {
          sequenceBufferRef.current.splice(0, FRAME_STEP);
          windowTimestampsRef.current.splice(0, FRAME_STEP);
        }

        // Chuẩn bị metadata nhận diện / debug cho Word Mode
        let debugMeta = null;
        if (requestMode === 'words') {
          const latencies = frameLatencyHistoryRef.current;
          const avgLat = latencies.length > 0
            ? Math.round((latencies.reduce((a, b) => a + b, 0) / latencies.length) * 10) / 10
            : null;
          const maxLat = latencies.length > 0
            ? Math.round(Math.max(...latencies) * 10) / 10
            : null;

          const currentClientFps = fpsBadgeRef.current
            ? (parseInt(fpsBadgeRef.current.innerText, 10) || fps)
            : fps;

          debugMeta = {
            tracker_engine: trackerEngineRef.current,
            active_instance_engine: activeInstanceEngineRef.current || trackerInstanceRef.current?._engineType || 'unknown',
            client_fps: currentClientFps,
            avg_frame_process_ms: avgLat,
            max_frame_process_ms: maxLat,
            window_duration_ms: windowDurationMs,
            consecutive_missing_hands: missingHandsCountRef.current
          };
        }

        const submit = () => {
          isInferringRef.current = true;
          const diagRecording = diagnostic.includesTime(now);
          const reqId = diagRecording ? diagnostic.nextRequestId() : null;
          const sentAt = performance.now();
          const payloadKB = diagRecording
            ? Math.round((JSON.stringify(sequencePayload).length / 1024) * 10) / 10
            : null;
          if (diagRecording) {
            diagnostic.addPipelineEvent(sentAt, 'request_sent', {
              reqId,
              mode: requestMode,
              liveWords,
              topic: requestTopic,
              windowFrames: sequencePayload.length,
              windowDurationMs,
              payloadKB,
              isFlush: finalFlushRequest,
            });
          }
          sendSequenceForRecognition(
            sequencePayload,
            confidenceThreshold,
            requestMode,
            debugMeta,
            {
              topic_id: requestTopic,
              timestamps: recentTimestamps,
              live_words: liveWords,
              gesture_complete: liveWords && finalFlushRequest,
              vocab_scope: liveWords ? requestVocabScope : undefined,
              include_word_candidates: requestMode === 'words',
              qipedc_entries: liveWords ? liveQipedcWords?.map(({ id }) => id) : null,
              aspect_ratio: holisticCanvasRef.current?.width / holisticCanvasRef.current?.height || 1,
              include_diagnostic: diagRecording,
              allow_short_sequences: liveWords ? (allowShort && finalFlushRequest) : allowShort,
            }
          )
            .then((resp) => {
              const receivedAt = performance.now();
              const rttMs = Math.round(receivedAt - sentAt);
              const baseMeta = {
                reqId,
                receivedAt,
                rttMs,
                windowFrames: sequencePayload.length,
                windowDurationMs,
                payloadKB,
                isFlush: finalFlushRequest,
              };
              if (currentGen !== requestGenRef.current || conversionModeRef.current !== requestMode) {
                if (diagRecording) {
                  diagnostic.addPipelineEvent(receivedAt, 'response_discarded', {
                    reqId,
                    rttMs,
                    reason: 'mode_or_generation_changed',
                  });
                }
                return; // Bỏ qua response đến trễ sau khi đổi mode
              }
              if (!resp?.success || resp.data?.success === false) {
                diagnostic.addPrediction(now, null, recentTimestamps, 'sstcn', null, true, {
                  ...baseMeta,
                  frontendDecision: {
                    action: 'backend_error',
                    reason: resp?.error || resp?.data?.error || 'Lỗi kết nối hoặc máy chủ trả về thất bại',
                  },
                });
              }
              if (resp && resp.success && resp.data) {
                const data = resp.data;
                const validWord = isValidWordRecognitionData(data);
                let token = data.confirmed && data.sign !== 'NEUTRAL' && validWord ? data.new_token : null;
                let replaced = null;
                let frontendDecision = {
                  action: token ? 'confirmed_backend' : 'unconfirmed',
                  reason: token
                    ? `Backend xác nhận trực tiếp “${token}”`
                    : (data.sign === 'NEUTRAL' ? 'Backend trả về NEUTRAL' : 'Chưa xác nhận ghi phụ đề'),
                };
                if (liveWords) {
                  if (liveWordsRef.current !== liveState || wordCaptureModeRef.current !== 'continuous' ||
                      (selectedTopicRef.current || null) !== requestTopic || vocabScopeRef.current !== requestVocabScope) {
                    if (diagRecording) {
                      diagnostic.addPrediction(now, data, recentTimestamps, 'sstcn', null, false, {
                        ...baseMeta,
                        frontendDecision: {
                          action: 'discarded_stale_window',
                          reason: 'Bỏ qua phản hồi vì cửa sổ tay đã bị reset (hoặc đổi chủ đề) trong lúc chờ server',
                        },
                      });
                    }
                    return;
                  }
                  token = null;
                  updateWordSuggestions(data, !finalFlushRequest);
                  liveState.quiet = data.motion_detected === false ? liveState.quiet + 1 : 0;
                  if (liveState.quiet >= 2) {
                    Object.assign(liveState, {
                      token: null, template: false, firstMotion: null, candidate: null, votes: 0,
                      hadFastMotion: false, stillSince: null, flushed: false,
                      lastCommittedToken: null, lastCommittedAt: 0,
                    });
                    if (diagRecording) {
                      diagnostic.addPipelineEvent(receivedAt, 'quiet_reset', { reqId, quietCount: liveState.quiet });
                    }
                  }
                  if (liveState.template) {
                    diagnostic.addPrediction(now, data, recentTimestamps, 'sstcn', null, false, {
                      ...baseMeta,
                      frontendDecision: {
                        action: 'ignored_template_locked',
                        reason: `Bỏ qua vì cửa sổ đang khóa sau khi đã chốt mẫu “${liveState.token}” (cần hạ tay >500ms hoặc 2 lượt đứng yên để mở khóa)`,
                        lockedToken: liveState.token,
                        quietCount: liveState.quiet,
                      },
                    });
                    return;
                  }
                  if (data.source === 'qipedc_template' && validWord) {
                    liveState.templateVotes = liveState.templateCandidate === data.entry_id ? liveState.templateVotes + 1 : 1;
                    liveState.templateCandidate = data.entry_id;
                    const singleShotFloor = data.match_method === 'hand_shape' ? 0.78 : 0.75;
                    const strongSingleShot = typeof data.confidence === 'number' && data.confidence >= singleShotFloor;
                    const waitForCompletion = data.requires_user_confirmation ||
                      (data.hand_count === 1 && (!finalFlushRequest || !strongSingleShot));
                    if (waitForCompletion || (!strongSingleShot && liveState.templateVotes < 2)) {
                      liveState.candidate = null;
                      liveState.votes = 0;
                      diagnostic.addPrediction(now, data, recentTimestamps, 'sstcn', null, false, {
                        ...baseMeta,
                        frontendDecision: {
                          action: waitForCompletion ? 'waiting_qipedc_completion' : 'waiting_qipedc_2nd_vote',
                          reason: waitForCompletion
                            ? `Đợi hoàn thành động tác và đủ điểm khớp mẫu “${data.sign}”`
                            : `Chờ một lượt khớp nữa cho “${data.sign}”`,
                          templateVotes: liveState.templateVotes,
                          templateCandidate: liveState.templateCandidate,
                        },
                      });
                      setCurrentResult({
                        sign: 'NEUTRAL',
                        label: waitForCompletion && finalFlushRequest ? 'Chọn từ gợi ý nếu đúng động tác.' : 'Đang xác nhận mẫu QIPEDC…',
                        confidence: 0,
                      });
                      return;
                    }
                  } else {
                    liveState.templateCandidate = null;
                    liveState.templateVotes = 0;
                  }
                  let committedLiveWord = false;
                  if (data.source === 'qipedc_template' && validWord) {
                    const isRecentDup = data.sign === liveState.lastCommittedToken && (receivedAt - liveState.lastCommittedAt) < 2600;
                    if (isRecentDup) {
                      if (sequenceBufferRef.current.length > 6) {
                        sequenceBufferRef.current = sequenceBufferRef.current.slice(-4);
                        windowTimestampsRef.current = windowTimestampsRef.current.slice(-4);
                      }
                      liveState.templateCandidate = null;
                      liveState.templateVotes = 0;
                      frontendDecision = {
                        action: 'ignored_recent_duplicate',
                        reason: `Bỏ qua lặp cùng từ “${data.sign}” trong vòng 2,6s (và cắt đuôi bộ đệm cũ)`,
                      };
                    } else {
                      token = data.sign;
                      replaced = liveState.token;
                      committedLiveWord = true;
                      frontendDecision = {
                        action: 'accepted_qipedc',
                        reason: `Frontend chốt từ QIPEDC “${token}” (${data.match_method || 'trajectory'}, phiếu=${liveState.templateVotes}${finalFlushRequest ? ', flush cuối' : ''}) và làm sạch cửa sổ cho từ tiếp theo`,
                        templateVotes: liveState.templateVotes,
                        emittedToken: token,
                        replaced,
                      };
                    }
                  } else if (data.sign === 'NEUTRAL' || !data.sign || !validWord) {
                    liveState.candidate = null;
                    liveState.votes = 0;
                    frontendDecision = {
                      action: 'neutral',
                      reason: `Chưa có từ được chốt (quiet=${liveState.quiet})`,
                      quietCount: liveState.quiet,
                    };
                  } else if (data.requires_user_confirmation) {
                    liveState.candidate = null;
                    liveState.votes = 0;
                    frontendDecision = {
                      action: 'waiting_manual_word',
                      reason: 'Chờ người dùng chọn trong bảng Từ gợi ý.',
                    };
                  } else {
                    liveState.firstMotion ??= now;
                    liveState.votes = liveState.candidate === data.sign ? liveState.votes + 1 : 1;
                    liveState.candidate = data.sign;
                    const conf = typeof data.confidence === 'number' ? data.confidence : 0;
                    const elapsedMotion = Math.round(now - liveState.firstMotion);
                    const flushThreshold = Math.max(confidenceThreshold, 0.85);
                    const readyWord =
                      (liveState.votes >= 2 && (elapsedMotion >= 500 || conf >= 0.85)) ||
                      (finalFlushRequest && !data.requires_live_consensus && conf >= flushThreshold);
                    const isRecentDup = data.sign === liveState.lastCommittedToken && (receivedAt - liveState.lastCommittedAt) < 2600;
                    if (!liveState.token && readyWord && !isRecentDup) {
                      token = data.sign;
                      committedLiveWord = true;
                      frontendDecision = {
                        action: 'accepted_sstcn',
                        reason: `Frontend chốt từ SSTCN “${token}” (phiếu=${liveState.votes}, ${Math.round(conf * 100)}%, sau ${elapsedMotion}ms) và làm sạch cửa sổ`,
                        votes: liveState.votes,
                        elapsedMotionMs: elapsedMotion,
                        emittedToken: token,
                      };
                    } else if (isRecentDup) {
                      frontendDecision = {
                        action: 'ignored_recent_duplicate',
                        reason: `Bỏ qua lặp cùng từ SSTCN “${data.sign}” trong vòng 2,6s`,
                        votes: liveState.votes,
                      };
                    } else if (liveState.token) {
                      frontendDecision = {
                        action: 'ignored_already_has_token',
                        reason: `Đã chốt từ “${liveState.token}” trong lượt tay này nên không ghi thêm “${data.sign}”`,
                        votes: liveState.votes,
                        existingToken: liveState.token,
                      };
                    } else {
                      frontendDecision = {
                        action: 'waiting_sstcn_vote',
                        reason: `SSTCN ra “${data.sign}” (${Math.round(conf * 100)}%), cần >=2 phiếu & >=500ms (hiện ${liveState.votes}/2, ${elapsedMotion}ms)`,
                        votes: liveState.votes,
                        elapsedMotionMs: elapsedMotion,
                      };
                    }
                  }
                  if (committedLiveWord && token && validWord) {
                    // Làm sạch bộ đệm sau khi chốt từ để người dùng thực hiện liền từ tiếp theo
                    // mà không bị khóa cứng hay trộn đuôi động tác cũ (chỉ chờ cooldown ngắn 550ms).
                    sequenceBufferRef.current = sequenceBufferRef.current.slice(-2);
                    windowTimestampsRef.current = windowTimestampsRef.current.slice(-2);
                    updateHUDProgress(0, sequenceBufferRef.current.length);
                    Object.assign(liveState, {
                      token: null,
                      template: false,
                      templateCandidate: null,
                      templateVotes: 0,
                      candidate: null,
                      votes: 0,
                      firstMotion: null,
                      hadFastMotion: false,
                      stillSince: null,
                      flushed: false,
                      pending: null,
                      cooldownUntil: receivedAt + 550,
                      lastCommittedToken: token,
                      lastCommittedAt: receivedAt,
                    });
                    if (diagRecording) {
                      diagnostic.addPipelineEvent(receivedAt, 'post_commit_window_reset', {
                        reqId,
                        committedToken: token,
                        cooldownMs: 550,
                      });
                    }
                  } else if (token && validWord) {
                    liveState.token = token;
                  }
                }
                diagnostic.addPrediction(now, data, recentTimestamps, 'sstcn', validWord && token && token !== replaced ? token : null, false, {
                  ...baseMeta,
                  replaced,
                  frontendDecision,
                });
                setLatestPilotResult(data);
                const waitingForVote = data.requires_user_confirmation || frontendDecision.action === 'waiting_sstcn_vote';
                const showWord = validWord && !waitingForVote;
                setCurrentResult({
                  sign: showWord ? data.sign : 'NEUTRAL',
                  label: data.requires_user_confirmation ? 'Chọn từ gợi ý nếu đúng động tác.'
                    : waitingForVote ? 'Đang xác nhận động tác…' : data.label,
                  confidence: showWord ? data.confidence : 0,
                  type: data.category || 'dynamic_sign',
                  description: showWord ? data.description : '',
                  category: data.category,
                  video_url: showWord ? data.video_url : '',
                  region: data.region,
                  source: data.source,
                  matchMethod: data.match_method,
                });

                if (requestMode === 'words' && !liveWords) updateWordSuggestions(data);
                if (validWord && token && token !== replaced) {
                  commitCaption(token, { stream: 'sequence', replaced });

                  if (ttsEnabledRef.current) {
                    let textToSpeak = data.tts_text;
                    if (!textToSpeak) {
                      if (data.type === 'digit' || requestMode === 'digits') {
                        textToSpeak = DIGIT_TTS_MAP[data.sign] || (data.sign ? DIGIT_TTS_MAP[data.sign.replace(/^Số\s*/, '')] : null) || data.sign;
                      } else {
                        textToSpeak = token;
                      }
                    }
                    speakVietnamese(textToSpeak);
                  }
                }
              }
            })
            .catch((err) => {
              console.warn('Lỗi suy luận chuỗi SSTCN:', err);
              if (diagRecording) {
                diagnostic.addPrediction(now, null, recentTimestamps, 'sstcn', null, true, {
                  reqId,
                  receivedAt: performance.now(),
                  rttMs: Math.round(performance.now() - sentAt),
                  windowFrames: sequencePayload.length,
                  windowDurationMs,
                  payloadKB,
                  isFlush: finalFlushRequest,
                  frontendDecision: { action: 'network_exception', reason: err?.message || 'Lỗi mạng khi gọi API' },
                });
              }
            })
            .finally(() => {
              if (currentGen === requestGenRef.current) {
                isInferringRef.current = false;
                if (liveWords && liveWordsRef.current === liveState && wordCaptureModeRef.current === 'continuous') {
                  const pending = liveState.pending;
                  liveState.pending = null;
                  if (pending && !liveState.template && (selectedTopicRef.current || null) === requestTopic &&
                      vocabScopeRef.current === requestVocabScope) pending.submit();
                }
              }
            });
        };

        if (isInferringRef.current) {
          // Luôn thay thế bằng cửa sổ MỚI NHẤT để không bao giờ bỏ lỡ pha mở ngón / kết thúc động tác
          // (chỉ giữ nguyên nếu cửa sổ đang chờ là flush cuối động tác còn cửa sổ mới không phải flush).
          if (!liveState.pending || finalFlushRequest || !liveState.pending.isFlush) {
            liveState.pending = { isFlush: finalFlushRequest, submit };
            diagnostic.addPipelineEvent(now, 'request_pending_queued', {
              bufLen: sequencePayload.length,
              windowDurationMs,
              isFlush: finalFlushRequest,
            });
          } else {
            diagnostic.addPipelineEvent(now, 'request_pending_kept_flush', {
              bufLen: sequencePayload.length,
              windowDurationMs,
            });
          }
        } else {
          submit();
        }
      }
    }

    // 3. CHẾ ĐỘ 2: ĐÁNH VẦN CHỮ CÁI & CHỮ SỐ TỨC THÌ (<5ms)
    else if (currentRecognitionMode === 'instant') {
      const dispatchInstantRelease = (releaseMode, releaseGen) => {
        if (!cameraActiveRef.current) return;
        isInferringRef.current = true;
        const releaseTs = monotonicTsGenRef.current.next(getEpochSecondsNow());
        sendLandmarksForRecognition([], [], 'Right', releaseMode, {
          timestamp: releaseTs,
          landmark_extract_ts: releaseTs,
          request_send_ts: releaseTs,
        })
          .then((relResp) => {
            if (releaseGen !== requestGenRef.current || conversionModeRef.current !== releaseMode) {
              return;
            }
            if (relResp && !relResp.backend_error && relResp.success && relResp.data) {
              const relData = relResp.data;
              if (relData.tts_speak) {
                ttsOrchestratorRef.current?.speakConfirmedEvent({
                  enabled: ttsEnabledRef.current,
                  eventId: relData.tts_event_id || relData.confirm_event_id || null,
                  text: relData.tts_speak,
                  mode: releaseMode,
                  digitMap: DIGIT_TTS_MAP,
                });
              }
              setCurrentResult({
                sign: 'NEUTRAL',
                label: releaseMode === 'digits' ? 'Chờ cử chỉ chữ số...' : 'Chờ cử chỉ chữ cái...',
                confidence: 0,
                type: '',
                description: '',
                category: releaseMode === 'digits' ? 'Chữ số VSL' : 'Bảng chữ cái VSL',
                video_url: '',
              });
            }
          })
          .catch(() => {
            // Ignore transient release network error
          })
          .finally(() => {
            if (releaseGen === requestGenRef.current) {
              isInferringRef.current = false;
            }
          });
      };

      const pilotProvider = showPilotHUDRef.current ? pilotMetadataProviderRef.current : null;
      const batchCamera = !pilotProvider &&
        instantCameraSessionRef.current && ['letters', 'digits'].includes(currentConversionMode);
      if (batchCamera) {
        // A predicted overlay must not count as another observation of a sign.
        const { left, right } = diagMeta?.observedHands || { left: leftHand, right: rightHand };
        const observed = Boolean(left || right);
        const released = !observed && now - instantHandAtRef.current >= 180 &&
          Number.isFinite(instantHandAtRef.current);
        if (observed || released) {
          instantHandAtRef.current = observed ? now : -Infinity;
          const hands = [];
          if (right) hands.push({ handedness: 'Right', landmarks: right.map(({ x, y, z }) => [x, y, z]) });
          if (left) hands.push({ handedness: 'Left', landmarks: left.map(({ x, y, z }) => [x, y, z]) });
          instantFramesRef.current.push({
            timestamp: monotonicTsGenRef.current.next(diagMeta?.landmarkExtractTs ?? getEpochSecondsNow()),
            landmarks: hands[0]?.landmarks || [],
            multi_hands: hands,
          });
          if (instantFramesRef.current.length > 16) instantFramesRef.current.shift();
        }
        if (!instantFramesRef.current.length) return;
      } else {
        if (!hasHands && !pilotProvider) {
          if (!instantHadHandsRef.current) return;
          instantNoHandCountRef.current += 1;
          // Sau 2 frame liên tiếp mất bàn tay (~60ms): gửi ngay 1 bản tin release để Backend xả TTS chữ nền đang chờ (A/D/E/O/U)
          if (instantNoHandCountRef.current >= 2) {
            instantHadHandsRef.current = false;
            instantNoHandCountRef.current = 0;
            if (!isInferringRef.current) {
              instantReleasePendingRef.current = false;
              dispatchInstantRelease(currentConversionMode, requestGenRef.current);
            } else {
              instantReleasePendingRef.current = true;
            }
          }
          return;
        }

        if (hasHands) {
          instantHadHandsRef.current = true;
          instantNoHandCountRef.current = 0;
          instantReleasePendingRef.current = false;
        }
      }

      if (!isInferringRef.current) {
        isInferringRef.current = true;
        const currentGen = requestGenRef.current;
        const requestMode = currentConversionMode;
        const primary = rightHand || leftHand;
        const primaryHandedness = rightHand ? 'Right' : (leftHand ? 'Left' : 'Right');
        const rawPrimary = primary ? primary.map(lm => [lm.x, lm.y, lm.z]) : [];
        const multiHands = [];
        if (rightHand) multiHands.push({ handedness: 'Right', landmarks: rightHand.map(l => [l.x, l.y, l.z]) });
        if (leftHand) multiHands.push({ handedness: 'Left', landmarks: leftHand.map(l => [l.x, l.y, l.z]) });

        let pilotMeta = null;
        if (pilotProvider) {
          try {
            pilotMeta = pilotProvider();
          } catch (e) {
            pilotMeta = null;
          }
        }

        if (!hasHands && !pilotMeta && !batchCamera) {
          isInferringRef.current = false;
          return;
        }

        const currentTrialKey = pilotMeta?.trial_id ? `${pilotMeta.session_id || ''}:${pilotMeta.trial_id}` : null;
        if (currentTrialKey !== lastPilotTrialKeyRef.current) {
          lastPilotTrialKeyRef.current = currentTrialKey;
          monotonicTsGenRef.current.reset(0);
        }

        const frameTs = monotonicTsGenRef.current.next(diagMeta?.landmarkExtractTs ?? getEpochSecondsNow());
        const rawSendTs = getEpochSecondsNow();
        const reqSendTs = rawSendTs >= frameTs ? rawSendTs : frameTs;

        if (pilotMeta) {
          pilotMeta.timestamp = frameTs;
          pilotMeta.landmark_extract_ts = frameTs;
          pilotMeta.request_send_ts = reqSendTs;
          pilotMeta.has_hands = Boolean(hasHands);
          pilotMeta.hand_detected = Boolean(hasHands);
          pilotMeta.handedness = hasHands ? primaryHandedness : null;
          if (diagMeta?.handednessScore !== undefined && diagMeta?.handednessScore !== null) {
            pilotMeta.handedness_score = Number(diagMeta.handednessScore.toFixed(4));
          }
          pilotMeta.mediapipe_model_complexity = diagMeta?.modelComplexity ?? (trackerEngineRef.current === 'holistic' ? 0 : 1);
          pilotMeta.processing_resolution = diagMeta?.processingResolution ?? (trackerEngineRef.current === 'holistic' ? '480x360' : '640x480');
          pilotMeta.camera_fps = cameraFpsRef.current || fps;
          if (videoRef.current) {
            pilotMeta.image_width = videoRef.current.videoWidth || 640;
            pilotMeta.image_height = videoRef.current.videoHeight || 480;
            pilotMeta.video_pts_sec = Number.isFinite(videoRef.current.currentTime)
              ? Number(videoRef.current.currentTime.toFixed(4))
              : 'UNKNOWN';
            pilotMeta.video_frame_idx = pilotMeta.frame_idx;
            pilotMeta.raw_landmarks = rawPrimary;
          }
        }

        const requestPilotMeta = pilotMeta ? Object.freeze({ ...pilotMeta }) : null;
        const requestApiMeta = requestPilotMeta || Object.freeze({
          timestamp: frameTs,
          landmark_extract_ts: frameTs,
          request_send_ts: reqSendTs,
          ...(batchCamera ? {
            camera_session_id: instantCameraSessionRef.current,
            camera_frames: instantFramesRef.current.splice(0),
          } : {}),
        });

        const instantSentAt = performance.now();
        sendLandmarksForRecognition(rawPrimary, multiHands, primaryHandedness, requestMode, requestApiMeta)
          .then((resp) => {
            const instantReceivedAt = performance.now();
            const instantRttMs = Math.round(instantReceivedAt - instantSentAt);
            if (currentGen !== requestGenRef.current || conversionModeRef.current !== requestMode) {
              // Phản hồi đã quá hạn do đổi mode: Bỏ qua dứt điểm, không gửi frame rỗng vào HUD
              return;
            }

            // Xử lý lỗi backend hoặc local fallback: Báo lỗi minh bạch cho HUD, không coi fallback là dự đoán backend hợp lệ
            if (resp && resp.backend_error) {
              diagnostic.addPrediction(now, null, null, 'landmarks', null, true, {
                receivedAt: instantReceivedAt,
                rttMs: instantRttMs,
                frontendDecision: { action: 'backend_error', reason: resp.error || 'Lỗi backend nhận diện bàn tay' },
              });
              if (requestPilotMeta && onPilotFrameResultRef.current) {
                try {
                  onPilotFrameResultRef.current(null, requestPilotMeta, {
                    backendError: true,
                    backendErrorMessage: resp.error || 'Backend recognition error',
                  });
                } catch (e) {
                  // Ignore pilot subscriber errors
                }
              }
              return;
            }

            if (resp && resp.success && resp.data) {
              for (const data of resp.results || [resp.data]) {
                setLatestPilotResult(data);
                if (onPilotFrameResultRef.current) {
                  try {
                    onPilotFrameResultRef.current(data, requestPilotMeta, {
                      backendError: false,
                      backendErrorMessage: null,
                    });
                  } catch (e) {
                    // Ignore pilot subscriber errors
                  }
                }

                // Chốt chặn bảo vệ UI:
                // Nếu đang ở mode 'digits' mà kết quả không phải chữ số (0-9, số nhiều chữ số, hoặc NEUTRAL) -> bỏ qua
                if (requestMode === 'digits' && data.sign !== 'NEUTRAL' && !/^[0-9]+$/.test(data.sign)) {
                  diagnostic.addPrediction(now, data, null, 'landmarks', null, false, {
                    receivedAt: instantReceivedAt,
                    rttMs: instantRttMs,
                    frontendDecision: { action: 'filtered_out_mode', reason: `Bỏ qua “${data.sign}” vì không phải chữ số` },
                  });
                  continue;
                }
                // Nếu đang ở mode 'letters': chỉ cho phép 29 chữ cái tiếng Việt chuẩn VSL (A..Y, Ă, Â, Đ, Ê, Ô, Ơ, Ư; không có F, J, W, Z) hoặc NEUTRAL
                if (requestMode === 'letters' && data.sign !== 'NEUTRAL' && !/^[A-EG-IK-VXYĂÂĐÊÔƠƯ]$/u.test(data.sign)) {
                  diagnostic.addPrediction(now, data, null, 'landmarks', null, false, {
                    receivedAt: instantReceivedAt,
                    rttMs: instantRttMs,
                    frontendDecision: { action: 'filtered_out_mode', reason: `Bỏ qua “${data.sign}” vì không thuộc 29 chữ cái` },
                  });
                  continue;
                }

                const caption = data.confirmed && data.sign !== 'NEUTRAL' ? data.new_token : null;
                const srcTag = data.prediction_source || data.source || 'landmarks';
                const clfReason = data.classifier_details?.reason || data.classifier_details?.status || null;
                diagnostic.addPrediction(now, data, null, 'landmarks', requestMode === 'digits' ? caption?.replace(/^Số\s*/, '') : caption, false, {
                  receivedAt: instantReceivedAt,
                  rttMs: instantRttMs,
                  frontendDecision: {
                    action: caption ? 'confirmed_caption' : (data.sign === 'NEUTRAL' ? 'neutral' : 'unconfirmed_smoothing'),
                    reason: caption
                      ? `Nguồn ${srcTag} → xác nhận ghi “${caption}”${clfReason ? ` (${clfReason})` : ''}`
                      : `Nguồn ${srcTag}, kết quả “${data.sign}”${clfReason ? ` (${clfReason})` : ''} — chờ ổn định/cooldown`,
                  },
                });

                setCurrentResult({
                  sign: data.sign,
                  label: data.label,
                  confidence: data.confidence,
                  type: data.type || 'alphabet',
                  description: data.description,
                  category: requestMode === 'digits' ? 'Chữ số VSL' : (requestMode === 'letters' ? 'Bảng chữ cái VSL' : 'Từ vựng VSL (102 Từ)'),
                  video_url: ''
                });

                // Nguồn điều phối TTS duy nhất trên Frontend dựa trên sự kiện xác nhận từ Backend (Checkpoint 5)
                if (data.tts_speak) {
                  ttsOrchestratorRef.current?.speakConfirmedEvent({
                    enabled: ttsEnabledRef.current,
                    eventId: data.tts_event_id || data.confirm_event_id || null,
                    text: data.tts_speak,
                    mode: requestMode,
                    digitMap: DIGIT_TTS_MAP,
                  });
                }

                if (data.confirmed && data.new_token && data.sign !== 'NEUTRAL') {
                  if (requestMode === 'letters' && !/^[A-EG-IK-VXYĂÂĐÊÔƠƯ]$/u.test(data.sign)) {
                    continue;
                  }
                  // Loại bỏ hoàn toàn tiền tố "Số " nếu kết quả là chữ số
                  const cleanToken = (requestMode === 'digits' && typeof data.new_token === 'string')
                    ? data.new_token.replace(/^Số\s*/, '')
                    : data.new_token;

                  const cleanReplaced = data.replaced_token?.replace(/^Số\s*/, '');
                  commitCaption(cleanToken, { stream: 'instant', replaced: cleanReplaced });
                }
              }
            }
          })
          .catch((err) => {
            console.warn('Lỗi suy luận tức thì:', err);
            if (requestPilotMeta && onPilotFrameResultRef.current) {
              try {
                onPilotFrameResultRef.current(null, requestPilotMeta, {
                  backendError: true,
                  backendErrorMessage: err?.message || 'Network exception in recognition request',
                });
              } catch (e) {
                // Ignore pilot subscriber errors
              }
            }
          })
          .finally(() => {
            if (currentGen === requestGenRef.current) {
              isInferringRef.current = false;
              if (!batchCamera && instantReleasePendingRef.current) {
                instantReleasePendingRef.current = false;
                dispatchInstantRelease(conversionModeRef.current, currentGen);
              }
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

      commitCaption(sample.name);

      if (ttsEnabled) speakVietnamese(sample.name);
    } catch (e) {
      console.error('Lỗi test mô phỏng:', e);
    }
  };

  const handleRemoveCaption = (id) => {
    setCaptionTokens((previous) => previous.filter((item) => item.id !== id));
    setSelectedCaptionId((previous) => previous === id ? null : previous);
  };

  const handleEditCaption = (id, text) => {
    setCaptionTokens((previous) => previous.map((item) =>
      item.id === id ? { ...item, text, automatic: false } : item
    ));
  };

  const handleMoveCaption = (sourceId, targetId) => {
    if (sourceId === targetId) return;
    setCaptionTokens((previous) => {
      const from = previous.findIndex(({ id }) => id === sourceId);
      const to = previous.findIndex(({ id }) => id === targetId);
      if (from < 0 || to < 0) return previous;
      const reordered = previous.slice();
      const [item] = reordered.splice(from, 1);
      reordered.splice(to, 0, item);
      return reordered;
    });
  };

  const handlePickWord = (item) => {
    const { word, confidence, source } = item;
    const selection = selectedCaptionId;
    const newItem = { id: ++nextCaptionIdRef.current, text: word, automatic: false };
    setCaptionTokens((previous) => previous.some(({ id }) => id === selection)
      ? previous.map((token) => token.id === selection ? { ...newItem, id: selection } : token)
      : [...previous, newItem]);
    setSelectedCaptionId(null);
    setWordSuggestions([]);
    setCurrentResult({
      sign: word,
      label: word,
      confidence,
      type: 'manual_word',
      video_url: item.video_url,
      region: item.region,
      source,
    });
    resetCaptionRecognition();
    const now = performance.now();
    Object.assign(liveWordsRef.current, {
      cooldownUntil: now + 550,
      lastCommittedToken: word,
      lastCommittedAt: now,
    });
    if (ttsEnabled) speakVietnamese(word);
  };

  const resetCaptionRecognition = () => {
    requestGenRef.current += 1;
    isInferringRef.current = false;
    sequenceBufferRef.current = [];
    windowTimestampsRef.current = [];
    liveWordsRef.current = newLiveWordState();
    updateHUDProgress(0, 0);
  };

  const handleClearSentence = () => {
    resetCaptionRecognition();
    resetInstantCameraSession(cameraActiveRef.current);
    ttsOrchestratorRef.current?.reset({ cancelAudio: true });
    instantHadHandsRef.current = false;
    instantNoHandCountRef.current = 0;
    instantReleasePendingRef.current = false;
    setCaptionTokens([]);
    setSelectedCaptionId(null);
    setWordSuggestions([]);
    clearCurrentSentence();
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
      ttsOrchestratorRef.current?.reset({ cancelAudio: true, clearHistory: true });
    };
  }, []);

  const cameraButton = cameraActive ? (
    <button type="button" onClick={stopCamera} className="btn btn-danger camera-power">
      <CameraOff size={17} /> Tắt camera
    </button>
  ) : (
    <button type="button" onClick={() => startCamera(trackerEngine)} className="btn btn-primary camera-power" disabled={cameraLoading}>
      <Camera size={17} /> {cameraLoading ? 'Đang bật…' : 'Bật camera'}
    </button>
  );

  const controls = (
    <>
      {cameraButton}

      {!practiceActive && (
        <>
          <section className="control-section" aria-labelledby="camera-mode-heading">
            <h3 id="camera-mode-heading">Chế độ nhận diện</h3>
            <div className="control-grid">
              {[
                ['letters', 'Chữ cái', Type],
                ['digits', 'Chữ số', Zap],
                ['words', 'Từ vựng', BookOpen],
              ].map(([mode, label, Icon]) => (
                <button type="button" key={mode} aria-pressed={conversionMode === mode}
                  className={`btn ${conversionMode === mode ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => handleModeChange(mode)}>
                  <Icon size={15} /> {label}
                </button>
              ))}
            </div>
          </section>

          <section className="control-section" aria-labelledby="camera-scope-heading">
            <h3 id="camera-scope-heading">Phạm vi theo dõi</h3>
            <div className="control-grid">
              <button type="button" onClick={() => handleScopeChange('hands')} disabled={conversionMode === 'words'}
                aria-pressed={recognitionScope === 'hands'} title={conversionMode === 'words' ? 'Từ vựng cần theo dõi toàn thân' : undefined}
                className={`btn ${recognitionScope === 'hands' ? 'btn-primary' : 'btn-secondary'}`}>
                <Hand size={15} /> Bàn tay
              </button>
              <button type="button" onClick={() => handleScopeChange('holistic')}
                aria-pressed={recognitionScope === 'holistic'}
                className={`btn ${recognitionScope === 'holistic' ? 'btn-primary' : 'btn-secondary'}`}>
                <User size={15} /> Toàn thân
              </button>
            </div>
          </section>
        </>
      )}

      {conversionMode === 'words' && (
        <section className="control-section" aria-label="Tùy chọn từ vựng">
          {!practiceEntry && (
            <>
              <label className="control-field">
                Chủ đề
                <select value={selectedTopic} onChange={(event) => {
                  const next = event.target.value;
                  selectedTopicRef.current = next;
                  setSelectedTopic(next);
                  setWordSuggestions([]);
                }}>
                  <option value="">Tất cả ({sstcnTopicsInfo?.word_classes_count ?? 27} từ SSTCN + QIPEDC)</option>
                  {(sstcnTopicsInfo?.topics || []).map((topic) => (
                    <option key={topic.id} value={topic.id}>{topic.label} ({topic.count})</option>
                  ))}
                </select>
              </label>
              {wordCaptureMode === 'continuous' && !selectedTopic && (
                <label className="control-field">
                  Phạm vi QIPEDC
                  <select value={vocabScope} onChange={(event) => {
                    const next = event.target.value;
                    vocabScopeRef.current = next;
                    setVocabScope(next);
                    setWordSuggestions([]);
                  }}>
                    <option value="all">Thông minh (ưu tiên thông dụng · 3.240 từ)</option>
                    <option value="common">Chỉ từ thông dụng (~730 từ · nhanh nhất)</option>
                  </select>
                </label>
              )}
              <div className="control-grid">
                <button type="button" aria-pressed={wordCaptureMode === 'continuous'}
                  className={`btn ${wordCaptureMode === 'continuous' ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => changeWordCaptureMode('continuous')}>Tự do</button>
                <button type="button" aria-pressed={wordCaptureMode === 'record'}
                  className={`btn ${wordCaptureMode === 'record' ? 'btn-primary' : 'btn-secondary'}`}
                  onClick={() => changeWordCaptureMode('record')}>Theo nhịp 3 giây</button>
              </div>
              {wordCaptureMode === 'record' && (
                <button type="button" className="btn btn-secondary" disabled={recordPhase !== 'idle'}
                  onClick={() => startTimedCapture('predict')}>
                  {recordPhase === 'idle' ? 'Ghi và nhận diện' : `Còn ${recordCountdown} giây`}
                </button>
              )}
            </>
          )}
          <label className="control-checkbox" title="Xử lý hình đến 640 px với MediaPipe Holistic Full">
            <input type="checkbox" checked={detailedTracking}
              disabled={cameraLoading || recordPhase !== 'idle' || diagnostic.status !== 'idle'}
              onChange={(event) => changeDetailedTracking(event.target.checked)} />
            Bám tay chi tiết
          </label>
        </section>
      )}

      {practiceActive && (
        <section className="control-section">
          <h3>Luyện “{practiceEntry.word}”</h3>
          {practiceVariants.length > 1 && (
            <label className="control-field">
              Video theo vùng miền
              <select value={practiceVideo} disabled={practiceStatus === 'loading' || practiceStatus === 'scoring' || recordPhase !== 'idle'}
                onChange={(event) => {
                  setPracticeVideo(event.target.value);
                  setPracticeStatus('idle');
                  setPracticeFeedback('');
                }}>
                {practiceVariants.map((variant) => <option key={variant.url} value={variant.url}>{variant.label}</option>)}
              </select>
            </label>
          )}
          <label className="control-checkbox">
            <input type="checkbox" checked={mirrorPractice} disabled={recordPhase !== 'idle' || practiceStatus === 'scoring'}
              onChange={(event) => {
                setMirrorPractice(event.target.checked);
                setPracticeFeedback('');
              }} />
            Tôi dùng tay ngược với video mẫu
          </label>
          {practiceStatus === 'idle' || practiceStatus === 'loading' ? (
            <button type="button" className="btn btn-primary" disabled={practiceStatus === 'loading'} onClick={preparePractice}>
              {practiceStatus === 'loading' ? 'Đang chuẩn bị mẫu…' : 'Chuẩn bị mẫu'}
            </button>
          ) : (
            <button type="button" className="btn btn-primary"
              disabled={!cameraActive || !practiceTrackingReady || recordPhase !== 'idle' || practiceStatus === 'scoring'}
              onClick={() => startTimedCapture('practice')}>
              {recordPhase === 'preparing' ? `Chuẩn bị ${recordCountdown} giây`
                : recordPhase === 'recording' ? `Đang ghi ${recordCountdown} giây`
                : practiceStatus === 'scoring' ? 'Đang so khớp…'
                : cameraActive && !practiceTrackingReady ? 'Đang tìm hai vai…' : 'Ghi 3 giây và so khớp'}
            </button>
          )}
          <button type="button" className="btn btn-secondary" onClick={onExitPractice}>Thoát luyện tập</button>
        </section>
      )}

      <div className="control-section workspace-tool-list" aria-label="Công cụ camera">
        {!practiceActive && onBrowsePractice && (
          <button type="button" className="btn btn-secondary" onClick={onBrowsePractice}>
            <BookOpen size={15} /> Từ điển QIPEDC
          </button>
        )}
        <button type="button" className="btn btn-secondary" onClick={() => diagnosticDialogRef.current?.showModal()}>
          <ScanLine size={15} /> Chẩn đoán video
        </button>
        {conversionMode === 'words' && !practiceActive && (
          <button type="button" className="btn btn-secondary" onClick={() => calibrationDialogRef.current?.showModal()}>
            <SlidersHorizontal size={15} /> Mẫu và hiệu chỉnh
          </button>
        )}
        {!practiceActive && (
          <button type="button" className="btn btn-secondary" onClick={() => testingDialogRef.current?.showModal()}>
            <FlaskConical size={15} /> Kiểm thử
          </button>
        )}
      </div>
    </>
  );

  const showWordSuggestions = conversionMode === 'words' && !practiceActive;

  return (
    <Workspace controls={controls}>
      <div className="workspace-content camera-workspace" data-word-suggestions={showWordSuggestions}>
        <section className="glass-panel workspace-stage camera-stage" aria-labelledby="camera-workspace-title">
          <header className="workspace-stage-heading">
            <div className="workspace-heading-title">
              <Camera size={19} aria-hidden="true" />
              <h1 id="camera-workspace-title">Ký hiệu → Văn bản</h1>
            </div>
            <div className="camera-stage-status">
              <span className={`badge ${cameraActive ? 'badge-emerald' : 'badge-neutral'}`}>
                {cameraActive ? 'Đang bật' : 'Đang tắt'}
              </span>
              {cameraActive && <button type="button" className="icon-button camera-mobile-stop" onClick={stopCamera}
                aria-label="Tắt camera" title="Tắt camera"><CameraOff size={16} /></button>}
            </div>
          </header>

          <div className="camera-view">
            <video ref={videoRef} playsInline muted autoPlay onLoadedMetadata={handleVideoMetadata}
              aria-label="Hình ảnh từ camera" hidden={!cameraActive} />
            <canvas ref={canvasRef} aria-hidden="true" />

            {!cameraActive && (
              <div className="camera-placeholder">
                <div className="camera-placeholder-icon"><Camera size={31} aria-hidden="true" /></div>
                <h2>Camera đang tắt</h2>
                <p>Bật camera để bắt đầu nhận diện ký hiệu.</p>
                <div className="camera-mobile-power">{cameraButton}</div>
              </div>
            )}

            {cameraActive && (
              <div className="camera-hud">
                <div className="camera-hud-status">
                  <span className="badge badge-emerald"><span className="pulse-dot" />
                    <span ref={fpsBadgeRef}>{fps} FPS</span>
                  </span>
                  <span ref={motionBadgeRef} className="badge badge-cyan">Bám sát tay</span>
                  <span ref={framingBadgeRef} className="badge badge-emerald">✓ Khung chuẩn</span>
                </div>
                {recognitionMode === 'sequence' && (
                  <div className="camera-hud-progress">
                    <svg width="22" height="22" viewBox="0 0 36 36" aria-hidden="true">
                      <path d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none" stroke="rgba(255,255,255,.15)" strokeWidth="3.5" />
                      <path ref={progressCircleRef}
                        d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831"
                        fill="none" stroke="#22d3ee" strokeWidth="3.5" strokeDasharray="0, 100" strokeLinecap="round" />
                    </svg>
                    <span ref={progressTextRef}>0/30 Frame</span>
                  </div>
                )}
              </div>
            )}

            {cameraActive && !cameraError && (
              <p ref={framingNoticeRef} className="camera-framing-hint" role="status" hidden />
            )}

            {recordPhase !== 'idle' && (
              <span className="camera-capture-state" role="status">
                {recordPhase === 'preparing' ? 'Chuẩn bị' : 'Đang ghi'} · {recordCountdown} giây
              </span>
            )}
            {diagnostic.status === 'recording' && (
              <span className="camera-diagnostic-recording">● Đang ghi chẩn đoán ({diagnostic.remainingSec}s)</span>
            )}
            {cameraError && <p className="camera-error" role="alert"><AlertCircle size={16} /> {cameraError}</p>}
            {!cameraError && conversionMode === 'words' && holisticStatus === 'pose_missing' && (
              <p className="camera-error" role="status">Đặt vai, khuỷu tay và bàn tay trong khung hình.</p>
            )}
          </div>
        </section>

        {practiceActive ? (
          <section className="glass-panel workspace-inspector practice-inspector" aria-label="Kết quả luyện tập">
            <h2>Mẫu “{practiceEntry.word}”</h2>
            <video controls preload="none" poster={practiceVideo === practiceEntry.video_url ? practiceEntry.thumb_url : undefined}
              src={mediaUrl(practiceVideo)} aria-label={`Video QIPEDC: ${practiceEntry.word}`} />
            <p role="status">
              {practiceFeedback || (practiceStatus === 'scoring' ? 'Đang so khớp động tác…' : 'Xem mẫu và thực hiện ký hiệu sau khi hết đếm ngược.')}
            </p>
          </section>
        ) : (
          <div className="camera-results">
            <section className="glass-panel workspace-inspector camera-prediction" aria-label="Ký hiệu đang nhận diện">
              <div className="workspace-inspector-heading">
                <h2>Đang nhận diện</h2>
                {currentResult.matchMethod === 'phases' ? (
                  <span className="badge badge-cyan">Khớp hai pha</span>
                ) : currentResult.confidence > 0 ? (
                  <span className="badge badge-cyan">{Math.round(currentResult.confidence * 100)}%</span>
                ) : null}
              </div>
              <h3 className="camera-result-word" data-neutral={currentResult.sign === 'NEUTRAL'}>{currentResult.label}</h3>
              {currentResult.description && <p className="camera-result-description">{currentResult.description}</p>}
              {currentResult.video_url && (
                <button type="button" className="btn btn-secondary camera-result-video" onClick={() => {
                  setPreviewVideoUrl(currentResult.video_url);
                  previewDialogRef.current?.showModal();
                }}>
                  <Film size={15} /> Xem ký hiệu mẫu
                </button>
              )}
            </section>

            {showWordSuggestions && (
              <section className="glass-panel workspace-inspector camera-word-suggestions" aria-labelledby="camera-suggestions-title">
                <div className="workspace-inspector-heading">
                  <h2 id="camera-suggestions-title">Từ gợi ý</h2>
                  <span className="camera-suggestion-floor">Từ 60%</span>
                </div>
                {wordSuggestions.length ? (
                  <>
                    <p className="workspace-empty">
                      {selectedCaptionId !== null ? 'Chọn gợi ý để thay cụm từ đã chọn.' : 'Chọn gợi ý để thêm vào phụ đề.'}
                    </p>
                    <ul className="camera-word-list" aria-label="Gợi ý nhận diện gần đây">
                      {wordSuggestions.map((item) => (
                        <li key={item.entry_id}>
                          <button type="button" onClick={() => handlePickWord(item)}>
                            <span>{item.word}</span>
                            <span className="camera-word-score"
                              title={item.match_method === 'personal_template' ? 'Độ tương đồng mẫu cá nhân'
                                : item.match_method === 'unilateral_core' ? 'Độ tương đồng dáng ngón tay và quỹ đạo'
                                : item.source === 'qipedc_template' ? 'Độ tương đồng mẫu QIPEDC' : 'Điểm dự đoán SSTCN'}>
                              {Math.round(item.confidence * 100)}%
                            </span>
                          </button>
                        </li>
                      ))}
                    </ul>
                  </>
                ) : <p className="workspace-empty">Chưa có gợi ý đạt 60%.</p>}
              </section>
            )}

            <section className="glass-panel workspace-inspector camera-subtitles" aria-labelledby="camera-subtitles-title">
              <div className="workspace-inspector-heading">
                <h2 id="camera-subtitles-title">Phụ đề</h2>
                <div className="caption-actions">
                  <button type="button" onClick={() => ttsOrchestratorRef.current?.speakManualSubtitle({ text: formatSentenceForTTS(sentence) })}
                    disabled={!sentence} className="icon-button" aria-label="Đọc phụ đề" title="Đọc phụ đề"><Volume2 size={16} /></button>
                  <button type="button" onClick={handleCopySentence} disabled={!sentence} className="icon-button"
                    aria-label={copied ? 'Đã chép phụ đề' : 'Chép phụ đề'} title={copied ? 'Đã chép' : 'Chép phụ đề'}><Copy size={16} /></button>
                  <button type="button" onClick={handleClearSentence} disabled={!sentence} className="icon-button caption-clear"
                    aria-label="Xóa phụ đề" title="Xóa phụ đề"><Trash2 size={16} /></button>
                </div>
              </div>
              <CaptionTokens tokens={captionTokens} selectedId={selectedCaptionId}
                onSelect={setSelectedCaptionId} onRemove={handleRemoveCaption} onMove={handleMoveCaption}
                onEdit={handleEditCaption} captionRef={captionRef} />
            </section>
          </div>
        )}
      </div>

      <WorkspaceDialog dialogRef={diagnosticDialogRef} title="Chẩn đoán camera (10 giây · Full Log)">
        <CameraDiagnostic capture={diagnostic} cameraActive={cameraActive}
          stream={() => videoRef.current?.srcObject} mode={conversionMode}
          captureMode={conversionMode === 'words' && wordCaptureMode === 'continuous' && !practiceEntry ? 'live_words' : 'other'}
          timed={recordPhase !== 'idle'} topic={selectedTopic} expectedSign={practiceEntry?.word}
          trackerEngine={trackerEngine} detailedTracking={detailedTracking}
          getResolution={() => (holisticCanvasRef.current ? `${holisticCanvasRef.current.width}x${holisticCanvasRef.current.height}` : null)} />
      </WorkspaceDialog>

      {!practiceActive && (
        <WorkspaceDialog dialogRef={testingDialogRef} title="Kiểm thử nhận diện">
          <div className="dialog-control-group">
            <h3>Mô phỏng cử chỉ</h3>
            <div className="dialog-button-list">{SAMPLE_GESTURES.map((item) => (
              <button type="button" key={item.type} onClick={() => {
                triggerSampleTest(item);
                testingDialogRef.current?.close();
              }} className="btn btn-secondary" title={item.desc}>{item.name}</button>
            ))}</div>
          </div>
          {conversionMode === 'words' && isWordDebugEnabled() && <WordDebugPanel />}
          <PilotTestingHUD isActive={showPilotHUD}
            onToggleActive={() => setShowPilotHUD((active) => !active)}
            onMetadataUpdate={handlePilotMetadataUpdate} cameraActive={cameraActive}
            latestRecognition={latestPilotResult} videoElementRef={videoRef} />
        </WorkspaceDialog>
      )}

      <WorkspaceDialog dialogRef={calibrationDialogRef} title="Mẫu và hiệu chỉnh">
        <div className="dialog-control-group">
          <h3>Mẫu QIPEDC đang sẵn sàng{liveReferenceStatus?.indexedCount ? ` · ${liveReferenceStatus.indexedCount + 1} từ` : ''}</h3>
          <p>{liveReferenceStatus?.indexedCount
            ? 'Camera đã nạp sẵn bộ mẫu QIPEDC và có thể dùng thêm các mục bạn chọn trong từ điển.'
            : liveReferenceStatus?.entries.filter((item) => item.ready).map((item) => item.word).join(' · ') ||
              (cameraActive ? 'Các mẫu chưa sẵn sàng.' : 'Bật camera ở chế độ Tự do để chuẩn bị mẫu.')}</p>
          {liveReferenceStatus?.failed && <p>Một số video mẫu chưa sẵn sàng.</p>}
        </div>
        <div className="dialog-control-group">
          <h3>Hiệu chỉnh cá nhân · {calibrationTotal} mẫu</h3>
          <label className="control-field">
            Tên ký hiệu
            <input value={calibrationLabel} onChange={(event) => setCalibrationLabel(event.target.value)}
              placeholder="Ví dụ: Chào, Cảm ơn" />
          </label>
          <button type="button" className="btn btn-primary" disabled={recordPhase !== 'idle'}
            onClick={() => startTimedCapture('calibrate')}>
            {recordPhase === 'idle' ? 'Ghi mẫu của tôi (3 giây)' : `Còn ${recordCountdown} giây`}
          </button>
          {calibrationMsg && <p role="status">{calibrationMsg}</p>}
        </div>
      </WorkspaceDialog>

      <WorkspaceDialog dialogRef={previewDialogRef} title="Video ký hiệu mẫu" onClose={() => setPreviewVideoUrl(null)}>
        {previewVideoUrl && (
          <video className="workspace-sample-video" src={mediaUrl(previewVideoUrl)} controls autoPlay loop playsInline />
        )}
        {currentResult.description && <p>{currentResult.description}</p>}
      </WorkspaceDialog>
    </Workspace>
  );
}
