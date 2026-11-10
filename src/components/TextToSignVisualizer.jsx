import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, SkipBack, SkipForward, Mic, MicOff, Send, Sparkles, Info, Check, Layers } from 'lucide-react';
import RobotSigner3D, { SPELL_LETTER_MS } from './RobotSigner3D';
import MixamoAvatarSigner from './MixamoAvatarSigner';
import { resolveVietnameseSign } from '../utils/MixamoPoseAdapter';
import AvatarSequenceCaption from './AvatarSequenceCaption';
import SignDescription from './SignDescription';
import { convertTextToVSL, fetchMotion } from '../services/api';
import { learnMotion } from '../services/motionLearner';
import { getHandPoseByGloss } from '../utils/handPoses';

const VIETNAMESE_MARKED_LETTERS = {
  Ă: { base: 'A', mark: 'Dấu breve (˘): bổ sung bằng chuyển động ngắn theo mẫu đánh vần VSL.' },
  Â: { base: 'A', mark: 'Dấu mũ (^): thêm tư thế dấu mũ tĩnh sau hình tay chữ A.' },
  Ê: { base: 'E', mark: 'Dấu mũ (^): thêm tư thế dấu mũ tĩnh sau hình tay chữ E.' },
  Ô: { base: 'O', mark: 'Dấu mũ (^): thêm tư thế dấu mũ tĩnh sau hình tay chữ O.' },
  Ơ: { base: 'O', mark: 'Dấu râu (’): thêm tư thế dấu râu tĩnh sau hình tay chữ O.' },
  Ư: { base: 'U', mark: 'Dấu râu (’): thêm tư thế dấu râu tĩnh sau hình tay chữ U.' }
};

// Nội dung Avatar theo các clip đã đối chiếu trong báo cáo bảng chữ cái.
const AVATAR_FINGER_DESCRIPTIONS = {
  A: ['Khép dọc cạnh ngón trỏ, đầu ngón nhô lên khỏi nắm tay', 'Gập ở gốc và hai đốt ngoài, thu đầu ngón về lòng bàn tay', 'Gập thành nắm tay, sát ngón trỏ', 'Gập thành nắm tay, sát ngón giữa', 'Gập vào lòng bàn tay, khép sát ngón áp út'],
  B: ['Gập ngang vào lòng bàn tay, nằm dưới gốc bốn ngón duỗi', 'Duỗi từ gốc đến đầu ngón, khép sát ngón giữa', 'Duỗi lên, khép giữa ngón trỏ và ngón áp út', 'Duỗi lên, khép giữa ngón giữa và ngón út', 'Duỗi lên, khép sát ngón áp út'],
  C: ['Cong đối diện bốn ngón, để hở khoảng trống hình chữ C', 'Gập nhẹ ở gốc và hai đốt ngoài, tạo mép trên của vòng cung', 'Khum theo vòng cung cùng ngón trỏ', 'Khum về phía ngón cái, cùng chiều với ngón giữa', 'Khum theo vòng cung, đầu ngón hướng về phía ngón cái'],
  D: ['Hướng đầu ngón về nhóm ngón đang khum', 'Duỗi từ gốc đến đầu ngón và hướng lên', 'Khum hai đốt ngoài, hướng đầu ngón về ngón cái', 'Co theo ngón giữa về phía lòng bàn tay', 'Co vào lòng bàn tay, sát ngón áp út'],
  Đ: ['Hướng đầu ngón về nhóm ngón đang co', 'Hướng lên ở gốc, gập hai đốt ngoài thành móc', 'Khum về phía ngón cái, thấp hơn đầu ngón trỏ', 'Co vào lòng bàn tay, sát ngón giữa', 'Co vào lòng bàn tay, sát ngón áp út'],
  E: ['Gập ngang vào lòng bàn tay, nằm bên dưới bốn đầu ngón', 'Gập đốt giữa và đốt đầu, hướng đầu ngón xuống', 'Gập hai đốt ngoài cùng chiều ngón trỏ', 'Gập hai đốt ngoài, thu đầu ngón về lòng bàn tay', 'Gập hai đốt ngoài, khép cạnh ngón áp út'],
  G: ['Nhô lên gần gốc ngón trỏ', 'Duỗi thẳng hướng lên', 'Ngón giữa, áp út và út gập về lòng bàn tay'],
  H: ['Khép trước ngón áp út và ngón út đang co', 'Duỗi lên, hai đầu ngón tách nhẹ thành hai nhánh', 'Gập về lòng bàn tay và nằm dưới hai ngón duỗi'],
  I: ['Khép về phía ngón trỏ, giữa và áp út đang gập', 'Duỗi từ gốc đến đầu ngón, hướng lên khỏi nắm tay', 'Ngón trỏ, giữa và áp út co về lòng bàn tay'],
  K: ['Đặt đầu ngón gần chỗ tách giữa ngón trỏ và ngón giữa', 'Duỗi thẳng và hướng lên', 'Gập ở gốc để thân ngón hướng ngang', 'Gập về lòng bàn tay, nằm dưới hai ngón mở'],
  L: ['Mở ngang sang bên, tạo nét ngang của chữ L', 'Duỗi lên tạo nét đứng của chữ L', 'Ngón giữa, áp út và út gập vào lòng bàn tay'],
  M: [null, 'Gập ở gốc, giữ ba đầu ngón sát nhau và hướng xuống', 'Co vào lòng bàn tay, tách khỏi nhóm ba ngón hướng xuống'],
  N: [null, 'Gập ở gốc, giữ hai đầu ngón sát nhau và hướng xuống', 'Co vào lòng bàn tay, tách khỏi hai ngón hướng xuống'],
  O: ['Bốn ngón khum về phía ngón cái, tạo khoảng tròn nhỏ giữa các đầu ngón'],
  P: ['Gập cổ tay, lòng bàn tay hướng xuống', 'Gập gần các ngón đang co', 'Duỗi ngang sang bên','Duỗi hướng xuống dưới','Gập vào lòng bàn tay','Gập vào lòng bàn tay'],
  Q: ['Gập cổ tay, lòng bàn tay hướng xuống', 'Cong về phía ngón trỏ', 'Cong xuống, hướng về ngón cái', 'Gập vào lòng bàn tay'],
  R: ['Duỗi và bắt chéo nhau', 'Ngón áp út và út co vào lòng bàn tay, ngón cái áp gần hai ngón co'],
  S: ['Duỗi ngang ra bên cạnh nắm tay', 'Gập các đốt về lòng bàn tay để tạo nắm tay'],
  T: ['Đầu ngón cái chụm với đầu ngón trỏ thành vòng', 'Ngón trỏ khum về ngón cái. Ngón giữa, áp út và út duỗi lên và hơi xòe'],
  U: ['Duỗi lên song song, giữ hai ngón khép sát dọc theo thân ngón', 'Ngón áp út và út gập vào lòng bàn tay, ngón cái giữ gần hai ngón gập'],
  V: ['Khép vào phía ngón áp út và út đang co', 'Duỗi lên và tách khỏi ngón giữa, tạo một nhánh chữ V', 'Duỗi lên và mở về phía đối diện ngón trỏ', 'Gập về lòng bàn tay, nằm dưới ngón cái'],
  X: ['Giữ gốc ngón hướng lên, gập đốt giữa và đốt đầu thành móc'],
  Y: ['Mở khỏi lòng bàn tay về phía đối diện ngón út', 'Gập vào lòng bàn tay, nằm giữa ngón cái và ngón giữa', 'Gập vào lòng bàn tay cùng ngón trỏ', 'Duỗi và mở sang phía đối diện ngón cái']
};

export default function TextToSignVisualizer({ ttsEnabled, requestText }) {
  const [inputText, setInputText] = useState('Xin chào bạn');
  const [sequence, setSequence] = useState([]);
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [playbackSpeed, setPlaybackSpeed] = useState(1.25); // mặc định nhanh hơn một chút cho câu liền mạch
  const [isListening, setIsListening] = useState(false);
  const [loading, setLoading] = useState(false);
  const [translationId, setTranslationId] = useState(0); // mỗi lần dịch câu mới
  const [viewModel, setViewModel] = useState('avatar');
  const [directSelectedLetter, setDirectSelectedLetter] = useState('');

  const timerRef = useRef(null);
  const recognitionRef = useRef(null);

  // Bảng chữ cái ngón tay đầy đủ 29 chữ cái tiếng Việt chuẩn VSL (không có F, J, W, Z)
  const ALPHABET_LIST = [
    'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 'H', 'I', 'K', 'L', 'M', 
    'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y'
  ];
  const VIETNAMESE_ALPHABET_LIST = [
    'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G', 'H', 'I', 'K', 'L', 'M',
    'N', 'O', 'Ô', 'Ơ', 'P', 'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y'
  ];
  const PHRASE_SUGGESTIONS = [
    'Xin chào',
    'Cảm ơn',
    'Tôi yêu bạn',
    'Bạn tên là gì?',
    'Mẹ tôi là bác sĩ',
    'Hôm nay tôi đi học',
    'Gia đình tôi',
    'Xin lỗi',
    'Buổi tối ăn cơm',
    'Bạn khỏe không?',
    'Giúp đỡ',
    'Địa chỉ'
  ];

  // Chuyển đổi Text sang Ký hiệu
  const handleTranslate = async (textToTranslate) => {
    const text = textToTranslate || inputText;
    if (!text || !text.trim()) return;

    setLoading(true);
    setDirectSelectedLetter('');
    try {
      const resp = await convertTextToVSL(text);
      if (resp && resp.success && resp.sequence) {
        setSequence(resp.sequence);
        setTranslationId((n) => n + 1);
        setCurrentIndex(0);
        setIsPlaying(true);
      }
    } catch (err) {
      console.error('Error translating text to VSL:', err);
    } finally {
      setLoading(false);
    }
  };

  // Chọn trực tiếp 1 chữ cái để kiểm tra giải phẫu ngón tay
  const handleSelectDirectLetter = (char) => {
    setDirectSelectedLetter(char);
    setInputText(`Chữ ${char}`);
    setSequence([{
      token: char,
      gloss: char,
      label: `Chữ ${char}`,
      type: 'fingerspelling',
      duration_ms: 2200,
      description: `Bảng chữ cái ngón tay VSL: Chữ ${char}`,
      avatar_pose: {}
    }]);
    setCurrentIndex(0);
    setIsPlaying(false);
  };

  // Quản lý phát tự động
  useEffect(() => {
    if (!isPlaying || sequence.length === 0) {
      if (timerRef.current) clearTimeout(timerRef.current);
      return;
    }

    const currentStep = sequence[currentIndex];
    // Thời lượng mỗi ký hiệu = đoạn có động tác robot đã học từ video (đánh vần: duration_ms)
    const clipMs = currentStep?.clip ? Math.round((currentStep.clip[1] - currentStep.clip[0]) * 1000) + 250 : 0;
    // Chưa có chuyển động (đang học) → robot đánh vần tạm: mỗi chữ cái ~0,55 giây
    const spellMs = !currentStep?.motion_id && !currentStep?.pose_key && currentStep?.spell?.length
      ? currentStep.spell.length * SPELL_LETTER_MS + 300 : 0;
    const baseMs = currentStep ? (clipMs || spellMs || currentStep.duration_ms || 1400) : 1800;
    const symbol = ((viewModel === 'avatar' && currentStep?.avatar_symbol) || currentStep?.gloss || currentStep?.token || '').trim().normalize('NFC').toUpperCase().replace(/^ALPHA_/, '');
    // The Avatar plays these poses on its own clock. Let the mark finish before advancing.
    const avatarMs = viewModel === 'avatar' && (currentStep?.avatar_symbol || ['Ô', 'Ă', 'Ơ'].includes(symbol))
      ? resolveVietnameseSign(symbol, { hasNextStep: currentIndex < sequence.length - 1 })
        .reduce((ms, action) => ms + action.duration * 1000, 0)
      : 0;
    const duration = Math.max(baseMs / playbackSpeed, avatarMs);

    timerRef.current = setTimeout(() => {
      if (currentIndex < sequence.length - 1) {
        setCurrentIndex(prev => prev + 1);
      } else {
        setIsPlaying(false);
      }
    }, duration);

    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [isPlaying, currentIndex, sequence, playbackSpeed, viewModel]);

  // Tải trước chuyển động robot của cả câu để robot bắt đầu ngay khi tới lượt
  useEffect(() => {
    sequence.forEach((s) => { if (s.motion_id) fetchMotion(s.motion_id); });
  }, [sequence]);

  // Từ chưa có chuyển động → robot tự HỌC từ video QIPEDC của từ đó (robot đánh vần tạm trong lúc học).
  // Học xong: cập nhật câu và diễn lại từ ký hiệu vừa học.
  const seqVersionRef = useRef(0);
  const [learning, setLearning] = useState({}); // video_id → tiến độ 0..1 | 'done' | 'failed'
  useEffect(() => {
    const version = ++seqVersionRef.current;
    const todo = [...new Set(sequence.filter((s) => s.video_id && !s.motion_id).map((s) => s.video_id))];
    if (!todo.length) return;
    let firstLearned = null;
    Promise.all(todo.map((vid) => learnMotion(vid, (p) => {
      if (seqVersionRef.current === version) setLearning((m) => ({ ...m, [vid]: p }));
    }).then((r) => {
      if (seqVersionRef.current !== version) return;
      setLearning((m) => ({ ...m, [vid]: r.success ? 'done' : 'failed' }));
      if (!r.success) return;
      setSequence((seq) => {
        const next = seq.map((s) => (s.video_id === vid
          ? { ...s, motion_id: r.motion_id, clip: r.clip, sign_description: r.sign_description || s.sign_description, has_pose: true }
          : s));
        const idx = next.findIndex((s) => s.video_id === vid);
        if (idx >= 0 && (firstLearned === null || idx < firstLearned)) firstLearned = idx;
        return next;
      });
    }))).then(() => {
      if (seqVersionRef.current !== version || firstLearned === null) return;
      // diễn lại từ ký hiệu đầu tiên vừa học được
      setCurrentIndex(firstLearned);
      setIsPlaying(true);
    });
  }, [translationId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Nhận yêu cầu diễn một từ từ tab Từ điển
  useEffect(() => {
    if (requestText?.text) {
      setInputText(requestText.text);
      handleTranslate(requestText.text);
    }
  }, [requestText]); // eslint-disable-line react-hooks/exhaustive-deps

  // Micro Web Speech
  const toggleSpeechRecognition = () => {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert('Trình duyệt của bạn chưa hỗ trợ Web Speech API. Bạn có thể gõ trực tiếp văn bản.');
      return;
    }

    if (isListening) {
      if (recognitionRef.current) recognitionRef.current.stop();
      setIsListening(false);
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = 'vi-VN';
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onstart = () => setIsListening(true);
    recognition.onresult = (event) => {
      const transcript = event.results[0][0].transcript;
      setInputText(transcript);
      handleTranslate(transcript);
      setIsListening(false);
    };
    recognition.onerror = () => setIsListening(false);
    recognition.onend = () => setIsListening(false);

    recognitionRef.current = recognition;
    recognition.start();
  };

  useEffect(() => {
    if (!requestText?.text) handleTranslate('Xin chào bạn');
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const currentStep = sequence[currentIndex] || null;
  const stepLabel = (step) => viewModel === 'avatar' && step?.avatar_symbol
    ? `Chữ ${step.avatar_symbol}` : step?.label;
  const currentGloss = (viewModel === 'avatar' && currentStep?.avatar_symbol) || currentStep?.gloss || currentStep?.token || 'A';
  const activeLetter = directSelectedLetter || (isPlaying && /^[a-z]$/i.test(currentGloss.trim()) ? currentGloss.trim() : '');
  const poseData = getHandPoseByGloss(currentGloss);
  const commonProps = {
    currentStep,
    isPlaying,
    speed: playbackSpeed,
    sequence,
    currentIndex,
    activeLetter
  };

  // Phân tích trạng thái giải phẫu 5 ngón tay cho bước hiện tại
  const getFingerStatesDescription = (gloss) => {
    const g = gloss.toUpperCase().replace('ALPHA_', '');
    const markedLetter = VIETNAMESE_MARKED_LETTERS[g];
    if (markedLetter) {
      return [
        ...getFingerStatesDescription(markedLetter.base),
        { name: 'Dấu chữ', state: markedLetter.mark, color: 'var(--primary)' }
      ];
    }

    switch (g) {
      case 'A':
        return [
          { name: 'Ngón cái', state: 'Áp sát mép ngoài đốt ngón trỏ, hướng thẳng lên', color: 'var(--primary)' },
          { name: 'Ngón trỏ', state: 'Gập chặt 3 đốt (MCP, PIP, DIP = 90°)', color: 'var(--accent-warning)' },
          { name: 'Ngón giữa', state: 'Gập chặt 3 đốt vào lòng bàn tay', color: 'var(--accent-warning)' },
          { name: 'Ngón áp út', state: 'Gập chặt 3 đốt vào lòng bàn tay', color: 'var(--accent-warning)' },
          { name: 'Ngón út', state: 'Gập chặt 3 đốt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'B':
        return [
          { name: 'Ngón cái', state: 'Gập ngang qua mặt trước lòng bàn tay', color: 'var(--accent-warning)' },
          { name: 'Ngón trỏ', state: 'Duỗi thẳng đứng 100%, khép sát ngón giữa', color: 'var(--accent-success)' },
          { name: 'Ngón giữa', state: 'Duỗi thẳng đứng 100%, khép sát', color: 'var(--accent-success)' },
          { name: 'Ngón áp út', state: 'Duỗi thẳng đứng 100%, khép sát', color: 'var(--accent-success)' },
          { name: 'Ngón út', state: 'Duỗi thẳng đứng 100%, khép sát', color: 'var(--accent-success)' }
        ];
      case 'C':
        return [
          { name: 'Ngón cái', state: 'Uốn cong hình bán nguyệt đối diện 4 ngón', color: 'var(--primary)' },
          { name: 'Ngón trỏ', state: 'Uốn cong nhẹ tạo hình vòm chữ C', color: 'var(--primary)' },
          { name: 'Ngón giữa', state: 'Uốn cong đồng bộ theo vòm chữ C', color: 'var(--primary)' },
          { name: 'Ngón áp út', state: 'Uốn cong đồng bộ', color: 'var(--primary)' },
          { name: 'Ngón út', state: 'Uốn cong đồng bộ', color: 'var(--primary)' }
        ];
      case 'D':
        return [
          { name: 'Ngón cái', state: 'Đầu ngón chạm nhẹ vào đầu ngón giữa tạo vòng khép', color: 'var(--primary)' },
          { name: 'Ngón trỏ', state: 'Duỗi thẳng đứng chỉ lên trên (100%)', color: 'var(--accent-success)' },
          { name: 'Ngón giữa', state: 'Uốn cong tạo vòng tròn khép kín với ngón cái', color: 'var(--primary)' },
          { name: 'Ngón áp út', state: 'Gập co lại vào lòng bàn tay', color: 'var(--accent-warning)' },
          { name: 'Ngón út', state: 'Gập co lại vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'Đ':
        return [
          { name: 'Ngón cái', state: 'Giữ tư thế như chữ D, khép sát cạnh ngón trỏ', color: 'var(--primary)' },
          { name: 'Ngón trỏ', state: 'Hơi quặp đầu ngón để phân biệt với chữ D', color: 'var(--accent-success)' },
          { name: 'Ngón giữa', state: 'Khum cong như chữ D', color: 'var(--primary)' },
          { name: 'Ngón áp út', state: 'Gập vào lòng bàn tay như chữ D', color: 'var(--accent-warning)' },
          { name: 'Ngón út', state: 'Gập vào lòng bàn tay như chữ D', color: 'var(--accent-warning)' }
        ];
      case 'E':
        return [
          { name: 'Ngón cái', state: 'Gập ngang co bên dưới các đầu ngón tay', color: 'var(--accent-warning)' },
          { name: 'Ngón trỏ', state: 'Gập đốt giữa và mút, đầu mút tì mép ngón cái', color: 'var(--primary)' },
          { name: 'Ngón giữa', state: 'Gập đốt đồng bộ tì mép ngón cái', color: 'var(--primary)' },
          { name: 'Ngón áp út', state: 'Gập đốt đồng bộ tì mép ngón cái', color: 'var(--primary)' },
          { name: 'Ngón út', state: 'Gập đốt đồng bộ tì mép ngón cái', color: 'var(--primary)' }
        ];
      case 'F':
        return [
          { name: 'Ngón cái', state: 'Đầu ngón cái chạm đầu ngón trỏ tạo vòng tròn (OK)', color: 'var(--primary)' },
          { name: 'Ngón trỏ', state: 'Chạm đầu ngón cái tạo vòng tròn', color: 'var(--primary)' },
          { name: 'Ngón giữa', state: 'Duỗi thẳng đứng 100%, xòe nhẹ', color: 'var(--accent-success)' },
          { name: 'Ngón áp út', state: 'Duỗi thẳng đứng 100%, xòe nhẹ', color: 'var(--accent-success)' },
          { name: 'Ngón út', state: 'Duỗi thẳng đứng 100%, xòe nhẹ', color: 'var(--accent-success)' }
        ];
      case 'G':
        return [
          { name: 'Ngón cái', state: 'Duỗi thẳng hướng ngang sang bên song song ngón trỏ', color: 'var(--accent-success)' },
          { name: 'Ngón trỏ', state: 'Duỗi thẳng hướng ngang song song ngón cái', color: 'var(--accent-success)' },
          { name: '3 Ngón còn lại', state: 'Gập chặt 3 đốt vào lòng bàn tay (Cổ tay xoay ngang)', color: 'var(--accent-warning)' }
        ];
      case 'H':
        return [
          { name: 'Ngón cái', state: 'Gập giữ các ngón tay còn lại', color: 'var(--accent-warning)' },
          { name: 'Ngón trỏ & Giữa', state: 'Duỗi thẳng song song khép sát hướng ngang', color: 'var(--accent-success)' },
          { name: 'Ngón áp út & Út', state: 'Gập chặt vào lòng bàn tay (Cổ tay xoay ngang)', color: 'var(--accent-warning)' }
        ];
      case 'I':
        return [
          { name: 'Ngón cái', state: 'Gập ngang giữ ngón trỏ, giữa, áp út', color: 'var(--accent-warning)' },
          { name: 'Ngón út', state: 'Duy nhất ngón út duỗi thẳng đứng chỉ lên trên (100%)', color: 'var(--accent-success)' },
          { name: '3 Ngón còn lại', state: 'Gập chặt 3 đốt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'J':
        return [
          { name: 'Ngón út', state: 'Chuyển động Frame-by-frame: Duỗi thẳng vẽ nét móc chữ J', color: 'var(--accent-success)' },
          { name: 'Cổ tay', state: 'Xoay cổ tay chúc xuống rồi móc uốn lên', color: 'var(--primary)' }
        ];
      case 'K':
        return [
          { name: 'Ngón cái', state: 'Đầu ngón cái tì vào giữa ngón trỏ và ngón giữa', color: 'var(--primary)' },
          { name: 'Ngón trỏ', state: 'Duỗi thẳng đứng chỉ lên trên', color: 'var(--accent-success)' },
          { name: 'Ngón giữa', state: 'Duỗi hơi nghiêng ra phía trước', color: 'var(--accent-success)' },
          { name: 'Ngón áp út & Út', state: 'Gập chặt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'L':
        return [
          { name: 'Ngón cái', state: 'Mở ngang vuông góc 90° so với ngón trỏ (chữ L)', color: 'var(--accent-success)' },
          { name: 'Ngón trỏ', state: 'Duỗi thẳng đứng 100% chỉ lên trời', color: 'var(--accent-success)' },
          { name: '3 Ngón còn lại', state: 'Gập chặt 3 đốt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'M':
        return [
          { name: 'Ngón cái', state: 'Luồn thò đầu ngón ra dưới đốt ngón áp út', color: 'var(--primary)' },
          { name: '3 Ngón (Trỏ-Giữa-Áp út)', state: 'Gập phủ qua ngón cái', color: 'var(--accent-warning)' },
          { name: 'Ngón út', state: 'Gập chặt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'N':
        return [
          { name: 'Ngón cái', state: 'Luồn thò đầu ngón ra dưới đốt ngón giữa', color: 'var(--primary)' },
          { name: '2 Ngón (Trỏ & Giữa)', state: 'Gập phủ qua ngón cái', color: 'var(--accent-warning)' },
          { name: 'Ngón áp út & Út', state: 'Gập chặt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'O':
        return [
          { name: 'Tất cả ngón tay', state: 'Đầu 5 ngón chụm đều vào đầu ngón cái tạo hình tròn chữ O', color: 'var(--primary)' }
        ];
      case 'P':
        return [
          { name: 'Cổ tay', state: 'Cổ tay chúc xuống dưới hướng xuống đất (Tư thế K chúc)', color: 'var(--primary)' },
          { name: 'Ngón trỏ & Giữa', state: 'Ngón trỏ chỉ xuống, ngón cái tì vào giữa', color: 'var(--accent-success)' }
        ];
      case 'Q':
        return [
          { name: 'Cổ tay', state: 'Cổ tay chúc thẳng xuống đất (Tư thế G chúc)', color: 'var(--primary)' },
          { name: 'Ngón cái & Trỏ', state: 'Hai ngón duỗi song song chúc xuống đất', color: 'var(--accent-success)' }
        ];
      case 'R':
        return [
          { name: 'Ngón trỏ & Giữa', state: 'Hai ngón duỗi thẳng bắt chéo nhau (ngón giữa vắt qua ngón trỏ)', color: 'var(--accent-success)' },
          { name: 'Ngón cái & 2 ngón kia', state: 'Gập chặt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'S':
        return [
          { name: 'Ngón cái', state: 'Nắm đấm chặt, ngón cái vắt ngang qua mặt trước của cả 4 ngón', color: 'var(--accent-warning)' },
          { name: '4 Ngón tay', state: 'Gập chặt 3 đốt tạo thành nắm đấm', color: 'var(--accent-warning)' }
        ];
      case 'T':
        return [
          { name: 'Ngón cái', state: 'Kẹp thò đầu ngón cái lên giữa ngón trỏ và ngón giữa', color: 'var(--primary)' },
          { name: '4 Ngón tay', state: 'Nắm chặt lại thành quả đấm', color: 'var(--accent-warning)' }
        ];
      case 'U':
        return [
          { name: 'Ngón trỏ & Giữa', state: 'Duỗi thẳng đứng song song khép sát nhau 100%', color: 'var(--accent-success)' },
          { name: 'Ngón cái & 2 ngón kia', state: 'Gập chặt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'V':
        return [
          { name: 'Ngón cái', state: 'Gập giữ ngón áp út và út', color: 'var(--accent-warning)' },
          { name: 'Ngón trỏ', state: 'Duỗi thẳng, mở chếch sang trái hình chữ V', color: 'var(--accent-success)' },
          { name: 'Ngón giữa', state: 'Duỗi thẳng, mở chếch sang phải hình chữ V', color: 'var(--accent-success)' },
          { name: 'Ngón áp út & Út', state: 'Gập chặt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'W':
        return [
          { name: '3 Ngón (Trỏ, Giữa, Áp út)', state: 'Duỗi thẳng xòe đều hình chữ W', color: 'var(--accent-success)' },
          { name: 'Ngón cái & Út', state: 'Ngón cái giữ đầu ngón út gập', color: 'var(--accent-warning)' }
        ];
      case 'X':
        return [
          { name: 'Ngón trỏ', state: 'Uốn cong hình móc câu (đốt giữa PIP và mút DIP gập vuông góc)', color: 'var(--primary)' },
          { name: 'Các ngón còn lại', state: 'Nắm chặt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'Y':
      case 'YEU_THUONG':
        return [
          { name: 'Ngón cái', state: 'Mở rộng sang cạnh ngoài (Shaka)', color: 'var(--accent-success)' },
          { name: 'Ngón trỏ', state: g === 'YEU_THUONG' ? 'Duỗi thẳng (I Love You)' : 'Gập chặt vào lòng bàn tay', color: g === 'YEU_THUONG' ? 'var(--accent-success)' : 'var(--accent-warning)' },
          { name: 'Ngón giữa & Áp út', state: 'Gập chặt 3 đốt', color: 'var(--accent-warning)' },
          { name: 'Ngón út', state: 'Duỗi thẳng mở sang bên', color: 'var(--accent-success)' }
        ];
      case 'Z':
        return [
          { name: 'Ngón trỏ', state: 'Chuyển động Frame-by-frame: Duỗi thẳng vẽ nét chữ Z trong không gian', color: 'var(--accent-success)' },
          { name: 'Các ngón còn lại', state: 'Nắm chặt vào lòng bàn tay', color: 'var(--accent-warning)' }
        ];
      case 'XIN_CHAO':
        return [
          { name: 'Động tác', state: 'Chuyển động Frame-by-frame: Bàn tay phẳng từ thái dương vẫy mở ra trước', color: 'var(--primary)' },
          { name: '5 Ngón tay', state: 'Các đốt duỗi thẳng tự nhiên, khép sát thân thiện', color: 'var(--accent-success)' },
          { name: 'Cánh tay', state: 'Nâng cánh tay ngang trán, gập cẳng tay 45°', color: 'var(--primary)' }
        ];
      case 'CAM_ON':
        return [
          { name: 'Động tác', state: 'Chuyển động Frame-by-frame: Đầu ngón tay chạm cằm rồi đưa hạ ra trước', color: 'var(--primary)' },
          { name: '5 Ngón tay', state: 'Lòng bàn tay hướng vào trong, đầu ngón tay khép phẳng', color: 'var(--accent-success)' },
          { name: 'Ý nghĩa', state: 'Thể hiện sự trân trọng xuất phát từ lời nói/trái tim', color: 'var(--primary)' }
        ];
      case 'XIN_LOI':
        return [
          { name: 'Động tác', state: 'Chuyển động Frame-by-frame: Bàn tay áp ngực xoa tròn theo chiều kim đồng hồ', color: 'var(--primary)' },
          { name: 'Bàn tay', state: 'Nắm nhẹ hoặc áp phẳng lên vùng ngực trái', color: 'var(--accent-success)' }
        ];
      default:
        return [
          { name: 'Ngón cái', state: 'Được định vị theo chuẩn giải phẫu VSL', color: 'var(--primary)' },
          { name: '4 Ngón tay', state: 'Khớp MCP, PIP, DIP uốn góc theo hình thái chữ', color: 'var(--accent-success)' }
        ];
    }
  };

  const fingerSymbol = currentGloss.trim().normalize('NFC').toUpperCase().replace(/^ALPHA_/, '');
  const baseSymbol = VIETNAMESE_MARKED_LETTERS[fingerSymbol]?.base ?? fingerSymbol;
  const avatarDescriptions = viewModel === 'avatar' ? AVATAR_FINGER_DESCRIPTIONS[baseSymbol] : undefined;
  const fingerRows = viewModel === 'avatar' && fingerSymbol === 'P'
    ? [
      { name: 'Cổ tay', color: 'var(--primary)' },
      { name: 'Ngón cái', color: 'var(--accent-warning)' },
      { name: 'Ngón trỏ', color: 'var(--accent-success)' },
      { name: 'Ngón giữa', color: 'var(--accent-success)' },
      { name: 'Ngón áp út', color: 'var(--accent-warning)' },
      { name: 'Ngón út', color: 'var(--accent-warning)' }
    ]
    : viewModel === 'avatar' && fingerSymbol === 'Q'
      ? [
        { name: 'Cổ tay', color: 'var(--primary)' },
        { name: 'Ngón cái', color: 'var(--primary)' },
        { name: 'Ngón trỏ', color: 'var(--primary)' },
        { name: 'Các ngón còn lại', color: 'var(--accent-warning)' }
      ]
      : getFingerStatesDescription(currentGloss);
  const fingerDetails = fingerRows.map((detail, index) => ({
    ...detail,
    state: avatarDescriptions?.[index] ?? detail.state
  }));

  return (
    <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0, 1.25fr) minmax(0, 1.2fr)', gap: '24px', padding: '24px', maxWidth: '1440px', margin: '0 auto', width: '100%' }}>
      
      {/* CỘT TRÁI: ROBOT 3D — dính trên màn hình khi cuộn cột phải */}
      <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '16px', position: 'sticky', top: '16px', alignSelf: 'start' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
          <div>
            <h2 style={{ fontSize: '18px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sparkles size={20} style={{ color: 'var(--primary)' }} />
              Mô phỏng Ký hiệu VSL 3D
            </h2>
            <p style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
              Chuyển đổi giữa avatar Mixamo và robot diễn hoạt ký hiệu
            </p>
          </div>

          <span className="badge badge-emerald">
            {isPlaying ? 'Đang Diễn Hoạt' : 'Tạm Dừng'}
          </span>
        </div>

        {/* Khung Canvas 3D */}
        <div style={{
          height: '600px',
          width: '100%',
          background: 'radial-gradient(circle at center, rgba(30, 41, 59, 0.4) 0%, rgba(7, 11, 18, 0.95) 100%)',
          borderRadius: 'var(--radius-md)',
          overflow: 'hidden',
          border: '1px solid var(--border-subtle)',
          position: 'relative'
        }}>
          <div style={{ position: 'absolute', top: '12px', left: '12px', zIndex: 5, display: 'flex', gap: '4px', padding: '4px', borderRadius: '10px', background: 'rgba(8, 15, 28, 0.88)', border: '1px solid var(--border-subtle)', backdropFilter: 'blur(10px)' }}>
            <button type="button" onClick={() => setViewModel('avatar')} className={`btn ${viewModel === 'avatar' ? 'btn-primary' : 'btn-secondary'}`} style={{ padding: '7px 10px', fontSize: '12px' }} aria-pressed={viewModel === 'avatar'}>
              👧 Avatar 3D
            </button>
            <button type="button" onClick={() => setViewModel('robot')} className={`btn ${viewModel === 'robot' ? 'btn-primary' : 'btn-secondary'}`} style={{ padding: '7px 10px', fontSize: '12px' }} aria-pressed={viewModel === 'robot'}>
              🤖 Robot
            </button>
          </div>
          {viewModel === 'robot' ? (
            <RobotSigner3D {...commonProps} />
          ) : (
            <MixamoAvatarSigner {...commonProps} />
          )}
        </div>

        {viewModel === 'avatar' && (
          <AvatarSequenceCaption sequence={sequence} currentIndex={currentIndex} />
        )}

        {/* Điều khiển Playback */}
        <div style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          background: 'rgba(15, 23, 42, 0.6)',
          padding: '12px 18px',
          borderRadius: 'var(--radius-md)',
          border: '1px solid var(--border-subtle)',
          flexWrap: 'wrap',
          gap: '12px'
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
            <button
              onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}
              disabled={currentIndex === 0}
              className="btn btn-secondary"
              style={{ padding: '8px 12px' }}
              title="Ký hiệu trước"
            >
              <SkipBack size={16} />
            </button>

            <button
              onClick={() => {
                if (isPlaying) setDirectSelectedLetter('');
                setIsPlaying(!isPlaying);
              }}
              className="btn btn-primary"
              style={{ padding: '8px 16px' }}
            >
              {isPlaying ? <Pause size={16} /> : <Play size={16} />}
              {isPlaying ? 'Tạm Dừng' : 'Phát'}
            </button>

            <button
              onClick={() => setCurrentIndex(Math.min(sequence.length - 1, currentIndex + 1))}
              disabled={currentIndex >= sequence.length - 1}
              className="btn btn-secondary"
              style={{ padding: '8px 12px' }}
              title="Ký hiệu tiếp theo"
            >
              <SkipForward size={16} />
            </button>

            <button
              onClick={() => { setCurrentIndex(0); setIsPlaying(true); }}
              className="btn btn-secondary"
              style={{ padding: '8px 12px' }}
              title="Phát lại từ đầu"
            >
              <RotateCcw size={16} />
            </button>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '8px', fontSize: '12px', color: 'var(--text-secondary)' }}>
            <span>Tốc độ:</span>
            {[0.75, 1.0, 1.25, 1.5].map((spd) => (
              <button
                key={spd}
                onClick={() => setPlaybackSpeed(spd)}
                style={{
                  padding: '4px 8px',
                  borderRadius: '4px',
                  border: '1px solid var(--border-subtle)',
                  background: playbackSpeed === spd ? 'var(--primary)' : 'rgba(255, 255, 255, 0.05)',
                  color: playbackSpeed === spd ? '#fff' : 'var(--text-muted)',
                  cursor: 'pointer',
                  fontWeight: '600'
                }}
              >
                {spd}x
              </button>
            ))}
          </div>
        </div>
      </div>

      {/* CỘT PHẢI: BẢNG SOI GIẢI PHẪU CƠ NGÓN TAY VÀ CHỌN CHỮ CÁI A-Z */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: '20px' }}>
        
        {/* Diễn tả cụ thể ký hiệu đang diễn */}
        {currentStep && (
          <div className="glass-panel" style={{ padding: '16px 20px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
            <div style={{ display: 'flex', alignItems: 'baseline', gap: '8px', flexWrap: 'wrap' }}>
              <strong style={{ fontSize: '20px', fontFamily: 'Outfit, sans-serif' }}>{stepLabel(currentStep)}</strong>
              {currentStep.pos && <span style={{ fontSize: '12px', color: 'var(--text-muted)' }}>{currentStep.pos}</span>}
              <span style={{ flex: 1 }} />
              <span className="badge badge-emerald" style={{ fontSize: '10.5px' }}>Diễn tả ký hiệu</span>
            </div>
            {currentStep.definition && (
              <p style={{ margin: 0, fontSize: '12.5px', color: 'var(--text-secondary)', lineHeight: 1.5 }}>
                <strong style={{ color: 'var(--text-main)' }}>Nghĩa: </strong>{currentStep.definition}
              </p>
            )}
            {currentStep.video_id && !currentStep.motion_id && (() => {
              const st = learning[currentStep.video_id];
              const pct = typeof st === 'number' ? Math.round(st * 100) : 0;
              return (
                <div style={{ padding: '10px 12px', borderRadius: 10, background: 'rgba(34, 211, 238, 0.08)', border: '1px solid rgba(34, 211, 238, 0.3)', display: 'flex', flexDirection: 'column', gap: 6 }}>
                  <span style={{ fontSize: 12.5, color: 'var(--text-main)' }}>
                    {st === 'failed'
                      ? 'Chưa học được ký hiệu này từ video — robot đánh vần thay thế.'
                      : `Robot đang học ký hiệu này từ video người thật… ${pct}% (trong lúc chờ, robot đánh vần từ)`}
                  </span>
                  {st !== 'failed' && (
                    <div style={{ height: 4, borderRadius: 4, background: 'rgba(255,255,255,0.08)', overflow: 'hidden' }}>
                      <div style={{ width: `${pct}%`, height: '100%', background: 'var(--primary)', transition: 'width .2s' }} />
                    </div>
                  )}
                </div>
              );
            })()}
            <SignDescription step={currentStep} />
          </div>
        )}

        {/* Bảng Kiểm tra Chi tiết Trạng thái Đốt Ngón Tay (Finger Anatomical Inspector) — chỉ khi robot có tư thế */}
        {(currentStep?.has_pose ?? true) && (
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h3 style={{ fontSize: '16px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Layers size={18} style={{ color: 'var(--primary)' }} />
              Trạng Thái Giải Phẫu Cơ Ngón Tay (Cử chỉ: {stepLabel(currentStep)})
            </h3>
            <span className="badge badge-cyan" style={{ fontSize: '11px' }}>
              Khớp MCP / PIP / DIP
            </span>
          </div>

          {/* Danh sách trạng thái từng đốt ngón tay */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            {fingerDetails.map((f, idx) => (
              <div
                key={idx}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  padding: '8px 12px',
                  borderRadius: 'var(--radius-sm)',
                  background: 'rgba(15, 23, 42, 0.65)',
                  border: '1px solid var(--border-subtle)',
                  fontSize: '12px'
                }}
              >
                <strong style={{ color: f.color, minWidth: '100px', display: 'flex', alignItems: 'center', gap: '6px' }}>
                  <span style={{ width: '6px', height: '6px', borderRadius: '50%', background: f.color }} />
                  {f.name}:
                </strong>
                <span style={{ color: 'var(--text-secondary)', textAlign: 'right', flex: 1 }}>
                  {f.state}
                </span>
              </div>
            ))}
          </div>
        </div>

        )}

        {/* Bảng chữ cái riêng theo chế độ avatar/robot */}
        <div className="glass-panel" style={{ padding: '16px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <span style={{ fontSize: '12px', fontWeight: '700', color: 'var(--text-secondary)' }}>
              {viewModel === 'avatar' ? '🔤 Bảng chữ cái tiếng Việt' : '🎯 Bảng chữ cái tiếng Việt (Robot)'}
            </span>
          </div>
          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(8, 1fr)', gap: '6px' }}>
            {(viewModel === 'avatar' ? VIETNAMESE_ALPHABET_LIST : ALPHABET_LIST).map((char) => {
              const isSelected = currentGloss === char;
              return (
                <button
                  key={char}
                  onClick={() => handleSelectDirectLetter(char)}
                  style={{
                    padding: '8px 0',
                    borderRadius: 'var(--radius-sm)',
                    border: isSelected ? '2px solid var(--primary)' : '1px solid var(--border-subtle)',
                    background: isSelected ? 'linear-gradient(135deg, var(--primary) 0%, #0284c7 100%)' : 'rgba(255, 255, 255, 0.05)',
                    color: isSelected ? '#fff' : 'var(--text-main)',
                    fontWeight: '700',
                    fontSize: '14px',
                    cursor: 'pointer',
                    transition: 'all 0.15s ease'
                  }}
                  title={`Chọn chữ ${char} cho ${viewModel === 'avatar' ? 'avatar' : 'robot'}`}
                >
                  {char}
                </button>
              );
            })}
          </div>
        </div>

        {/* Nhập Văn Bản & Giọng Nói Hỗ Trợ Câu Dài Liên Tục */}
        <div className="glass-panel" style={{ padding: '20px', display: 'flex', flexDirection: 'column', gap: '14px' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
            <h3 style={{ fontSize: '15px', display: 'flex', alignItems: 'center', gap: '8px' }}>
              <Sparkles size={16} style={{ color: 'var(--primary)' }} />
              Dịch Đoạn Văn & Diễn Tả Liên Tục (Continuous VSL Discourse)
            </h3>
            <span className="badge badge-cyan" style={{ fontSize: '11px' }}>
              {sequence.length > 0 ? `${currentIndex + 1} / ${sequence.length} ký hiệu` : 'Sẵn sàng'}
            </span>
          </div>

          {/* Nút bấm nhanh đưa toàn bộ câu văn dài của người dùng vào */}
          <div style={{
            background: 'rgba(56, 189, 248, 0.08)',
            border: '1px solid rgba(56, 189, 248, 0.25)',
            borderRadius: 'var(--radius-md)',
            padding: '10px 14px',
            display: 'flex',
            flexDirection: 'column',
            gap: '8px'
          }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <strong style={{ fontSize: '12px', color: 'var(--primary)' }}>
                ⭐ Đoạn văn dài mẫu thực tế (Câu chuyện cuối tuần):
              </strong>
              <button
                onClick={() => {
                  const longSample = "Những ngày cuối tuần của tôi không có lịch trình cụ thể vì chúng phụ thuộc vào mức độ bận rộn của tôi trong tuần. Tôi thường dành một ít thời gian cho việc học tập hoặc các dự án cá nhân, nhưng tôi cũng đảm bảo mình có đủ thời gian để thư giãn. Nếu có thể, tôi thích gặp gỡ bạn bè hoặc khám phá những địa điểm mới, vì điều đó cho tôi cơ hội nghỉ ngơi khỏi thói quen thường ngày.";
                  setInputText(longSample);
                  handleTranslate(longSample);
                }}
                className="btn btn-primary"
                style={{ padding: '4px 12px', fontSize: '11px', fontWeight: '700' }}
              >
                🚀 Nạp & Diễn Tả Liên Tục Cả Câu
              </button>
            </div>
            <p style={{ fontSize: '11px', color: 'var(--text-secondary)', margin: 0, lineHeight: '1.4' }}>
              "Những ngày cuối tuần của tôi không có lịch trình cụ thể... Tôi thường dành một ít thời gian cho việc học tập... thích gặp gỡ bạn bè hoặc khám phá những địa điểm mới..."
            </p>
          </div>

          {/* Ô nhập Textarea hỗ trợ đoạn văn dài nhiều câu */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
            <textarea
              rows={3}
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              placeholder="Nhập hoặc dán đoạn văn bản dài tiếng Việt muốn nhân vật diễn tả liên tục..."
              style={{
                width: '100%',
                padding: '10px 14px',
                borderRadius: 'var(--radius-md)',
                background: 'var(--bg-input)',
                border: '1px solid var(--border-subtle)',
                color: 'var(--text-main)',
                fontFamily: 'var(--font-heading)',
                fontSize: '13px',
                lineHeight: '1.5',
                outline: 'none',
                resize: 'vertical'
              }}
            />

            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: '8px' }}>
              <button
                onClick={toggleSpeechRecognition}
                className={`btn ${isListening ? 'btn-danger' : 'btn-secondary'}`}
                style={{ padding: '8px 12px', fontSize: '12px' }}
                title={isListening ? "Đang lắng nghe... Bấm để dừng" : "Nói vào micro"}
              >
                {isListening ? <MicOff size={14} /> : <Mic size={14} />}
                {isListening ? 'Đang nghe...' : 'Giọng nói'}
              </button>

              <button
                onClick={() => handleTranslate(inputText)}
                disabled={loading}
                className="btn btn-primary"
                style={{ padding: '8px 20px', fontSize: '13px', fontWeight: '700' }}
              >
                <Send size={14} /> {loading ? 'Đang phân tích...' : 'Dịch & Bắt Đầu Diễn Tả'}
              </button>
            </div>
          </div>

          {/* Thanh Tiến Trình & Danh Sách Token Đang Diễn Tả Liên Tục */}
          {sequence.length > 0 && (
            <div style={{
              background: 'rgba(15, 23, 42, 0.7)',
              borderRadius: 'var(--radius-md)',
              padding: '12px',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              flexDirection: 'column',
              gap: '10px'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', color: 'var(--text-secondary)' }}>
                <span>Tiến trình diễn tả liên tục:</span>
                <strong style={{ color: 'var(--primary)' }}>
                  {Math.round(((currentIndex + 1) / sequence.length) * 100)}% ({currentIndex + 1}/{sequence.length})
                </strong>
              </div>

              {/* Progress Bar */}
              <div style={{ width: '100%', height: '6px', background: 'rgba(255, 255, 255, 0.1)', borderRadius: '3px', overflow: 'hidden' }}>
                <div style={{
                  width: `${((currentIndex + 1) / sequence.length) * 100}%`,
                  height: '100%',
                  background: 'linear-gradient(90deg, var(--primary) 0%, #10b981 100%)',
                  transition: 'width 0.3s ease'
                }} />
              </div>

              {/* Danh sách các cụm từ đang được phát */}
              <div style={{
                display: 'flex',
                gap: '6px',
                overflowX: 'auto',
                paddingBottom: '4px',
                scrollbarWidth: 'thin'
              }}>
                {sequence.map((step, idx) => {
                  const isActive = idx === currentIndex;
                  return (
                    <button
                      key={idx}
                      onClick={() => {
                        setCurrentIndex(idx);
                        setIsPlaying(true);
                      }}
                      style={{
                        padding: '4px 10px',
                        borderRadius: 'var(--radius-sm)',
                        fontSize: '11px',
                        whiteSpace: 'nowrap',
                        border: isActive ? '1px solid var(--primary)' : '1px solid var(--border-subtle)',
                        background: isActive ? 'rgba(56, 189, 248, 0.25)' : 'rgba(255, 255, 255, 0.04)',
                        color: isActive ? '#38bdf8' : 'var(--text-muted)',
                        fontWeight: isActive ? '700' : '500',
                        cursor: 'pointer',
                        transition: 'all 0.2s ease'
                      }}
                      title={`Bấm để nhảy đến: ${stepLabel(step)}`}
                    >
                      {idx + 1}. {stepLabel(step)}
                    </button>
                  );
                })}
              </div>
            </div>
          )}

          {/* Các cụm từ mẫu sinh hoạt ngắn */}
          <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap', marginTop: '4px' }}>
            <span style={{ fontSize: '11px', color: 'var(--text-muted)', alignSelf: 'center', marginRight: '4px' }}>
              Gợi ý nhanh:
            </span>
            {PHRASE_SUGGESTIONS.map((phrase, i) => (
              <button
                key={i}
                onClick={() => {
                  setInputText(phrase);
                  handleTranslate(phrase);
                }}
                className="btn btn-secondary"
                style={{ padding: '3px 8px', fontSize: '11px' }}
              >
                {phrase}
              </button>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
