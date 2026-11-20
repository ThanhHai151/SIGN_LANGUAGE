import React, { useState, useEffect, useRef } from 'react';
import { Play, Pause, RotateCcw, SkipBack, SkipForward, Mic, MicOff, Send, Box, Type, Layers, List } from 'lucide-react';
import RobotSigner3D, { SPELL_LETTER_MS } from './RobotSigner3D';
import MixamoAvatarSigner from './MixamoAvatarSigner';
import { resolveVietnameseSign } from '../utils/MixamoPoseAdapter';
import AvatarSequenceCaption from './AvatarSequenceCaption';
import SignDescription from './SignDescription';
import { Workspace, WorkspaceDialog } from './Workspace';
import { convertTextToVSL, fetchLetter, fetchLetters, fetchMotion } from '../services/api';
import { learnMotion } from '../services/motionLearner';

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

export default function TextToSignVisualizer({ requestText }) {
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
  const [controlsTarget, setControlsTarget] = useState(null);
  const alphabetDialogRef = useRef(null);
  const detailsDialogRef = useRef(null);
  const phrasesDialogRef = useRef(null);
  const tunerHostRef = useRef(null);

  const timerRef = useRef(null);
  const recognitionRef = useRef(null);

  // Bảng chữ cái ngón tay tiếng Việt (QIPEDC): chữ gốc; rà chuột vào chữ gốc → hiện các chữ có dấu để chọn
  const ALPHABET_LIST = [
    'A', 'B', 'C', 'D', 'E', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O',
    'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'X', 'Y', 'F', 'J', 'W', 'Z',
  ];
  const LETTER_VARIANTS = { A: ['Ă', 'Â'], D: ['Đ'], E: ['Ê'], O: ['Ô', 'Ơ'], U: ['Ư'] };
  const TONE_LIST = [
    { key: 'sắc', mark: '´' }, { key: 'huyền', mark: '`' }, { key: 'hỏi', mark: '?' },
    { key: 'ngã', mark: '~' }, { key: 'nặng', mark: '.' },
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

  // Bảng chữ cái tải sẵn MỘT LẦN (kèm chuyển động robot) → bấm chữ nào robot diễn ngay, không chờ mạng
  const lettersRef = useRef(null);
  useEffect(() => {
    let cancelled = false;
    fetchLetters()
      .then((data) => {
        if (cancelled || !data?.letters) return;
        lettersRef.current = data.letters;
        Object.values(data.letters).forEach((it) => { if (it.motion_id) fetchMotion(it.motion_id); });
      })
      .catch(() => {});
    return () => { cancelled = true; };
  }, []);

  // Chọn trực tiếp 1 chữ cái: robot diễn hình chữ cái học từ video bảng chữ cái QIPEDC
  const handleSelectDirectLetter = async (char) => {
    const key = char.toLowerCase();
    const isTone = char === key;
    // Avatar Mixamo diễn theo ký hiệu: chữ cái (Ă, Ơ…) hoặc dấu thanh TONE_SAC…
    const toneSymbol = { 'sắc': 'TONE_SAC', 'huyền': 'TONE_HUYEN', 'hỏi': 'TONE_HOI', 'ngã': 'TONE_NGA', 'nặng': 'TONE_NANG' }[key];
    setDirectSelectedLetter(isTone ? toneSymbol : char);
    setInputText(isTone ? `Dấu ${char}` : `Chữ ${char}`);
    setIsPlaying(false);
    let item = lettersRef.current?.[key];
    if (!item) {
      try {
        item = await fetchLetter(char);
      } catch { /* mất kết nối backend */ }
    }
    if (item) {
      setSequence([item]);
      setCurrentIndex(0);
      return;
    }
    // Backend chưa có bảng chữ cái học từ video → tư thế vẽ tay của chữ không dấu (Ă, Â → A; Đ → D)
    const base = char.normalize('NFD').replace(/[\u0300-\u036f]/g, '').replace(/đ/gi, 'D').toUpperCase();
    setSequence([{
      token: char,
      gloss: base,
      label: isTone ? `Dấu ${char}` : `Chữ ${char}`,
      type: 'fingerspelling',
      duration_ms: 2200,
      pose_key: isTone ? null : base,
      word: key,
      description: `Bảng chữ cái ngón tay VSL: Chữ ${char} (tư thế tham khảo — hãy khởi động lại backend để dùng bảng chữ cái học từ video)`,
      avatar_pose: {}
    }]);
    setCurrentIndex(0);
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
  const stepLabel = (step) => viewModel === 'avatar' && step?.avatar_symbol && !step.avatar_symbol.startsWith('TONE_')
    ? `Chữ ${step.avatar_symbol}` : step?.label;
  const currentGloss = (viewModel === 'avatar' && currentStep?.avatar_symbol) || currentStep?.gloss || currentStep?.token || 'A';
  const activeLetter = directSelectedLetter || (isPlaying && /^[a-z]$/i.test(currentGloss.trim()) ? currentGloss.trim() : '');
  const learningProgress = currentStep?.video_id ? learning[currentStep.video_id] : null;
  const commonProps = {
    currentStep,
    isPlaying,
    speed: playbackSpeed,
    sequence,
    currentIndex,
    activeLetter,
    controlsTarget,
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

  const editor = (
    <div className="control-section text-editor">
      <label className="control-field">
        Văn bản tiếng Việt
        <textarea rows={3} value={inputText} onChange={(event) => setInputText(event.target.value)}
          placeholder="Nhập hoặc dán văn bản…" />
      </label>
      <div className="control-grid text-editor-actions">
        <button type="button" onClick={toggleSpeechRecognition}
          className={`btn ${isListening ? 'btn-danger' : 'btn-secondary'}`}
          aria-pressed={isListening} title={isListening ? 'Dừng nghe' : 'Nói vào micro'}>
          {isListening ? <MicOff size={15} /> : <Mic size={15} />}
          {isListening ? 'Dừng nghe' : 'Giọng nói'}
        </button>
        <button type="button" onClick={() => handleTranslate(inputText)}
          disabled={loading || !inputText.trim()} className="btn btn-primary">
          <Send size={15} /> {loading ? 'Đang dịch…' : 'Dịch'}
        </button>
      </div>
    </div>
  );

  const playback = (
    <div className="playback-controls">
      <button type="button" onClick={() => setCurrentIndex(Math.max(0, currentIndex - 1))}
        disabled={currentIndex === 0} className="icon-button" aria-label="Ký hiệu trước" title="Ký hiệu trước"><SkipBack size={16} /></button>
      <button type="button" disabled={!sequence.length} onClick={() => {
        if (isPlaying) setDirectSelectedLetter('');
        setIsPlaying(!isPlaying);
      }} className="btn btn-primary play-signs">
        {isPlaying ? <Pause size={15} /> : <Play size={15} />} {isPlaying ? 'Tạm dừng' : 'Phát'}
      </button>
      <button type="button" onClick={() => setCurrentIndex(Math.min(sequence.length - 1, currentIndex + 1))}
        disabled={currentIndex >= sequence.length - 1} className="icon-button" aria-label="Ký hiệu tiếp theo" title="Ký hiệu tiếp theo"><SkipForward size={16} /></button>
      <button type="button" disabled={!sequence.length} onClick={() => { setCurrentIndex(0); setIsPlaying(true); }}
        className="icon-button" aria-label="Phát lại từ đầu" title="Phát lại từ đầu"><RotateCcw size={16} /></button>
    </div>
  );

  const controls = (
    <>
      <div className="desktop-text-editor">{editor}</div>
      <section className="control-section" aria-label="Mô hình và góc nhìn">
        <h3>Nhân vật 3D</h3>
        <div className="control-grid">
          {['avatar', 'robot'].map((model) => (
            <button key={model} type="button" onClick={() => setViewModel(model)}
              className={`btn ${viewModel === model ? 'btn-primary' : 'btn-secondary'}`} aria-pressed={viewModel === model}>
              {model === 'avatar' ? 'Avatar' : 'Robot'}
            </button>
          ))}
        </div>
        <div ref={setControlsTarget} className="model-view-target" />
      </section>

      <section className="control-section" aria-label="Phát ký hiệu">
        <h3>Phát ký hiệu</h3>
        <div className="desktop-playback">{playback}</div>
        <label className="control-field">
          Tốc độ
          <select value={playbackSpeed} onChange={(event) => setPlaybackSpeed(Number(event.target.value))}>
            {[0.75, 1.0, 1.25, 1.5].map((speed) => <option key={speed} value={speed}>{speed}×</option>)}
          </select>
        </label>
      </section>

      <div className="control-section workspace-tool-list text-tools" aria-label="Công cụ văn bản">
        <button type="button" className="btn btn-secondary" onClick={() => phrasesDialogRef.current?.showModal()}>
          <List size={15} /> Câu mẫu
        </button>
        <button type="button" className="btn btn-secondary" onClick={() => alphabetDialogRef.current?.showModal()}>
          <Type size={15} /> Bảng chữ cái
        </button>
        <button type="button" className="btn btn-secondary" onClick={() => detailsDialogRef.current?.showModal()}>
          <Layers size={15} /> Chi tiết ký hiệu
        </button>
      </div>
    </>
  );

  const selectSample = (phrase) => {
    setInputText(phrase);
    handleTranslate(phrase);
    phrasesDialogRef.current?.close();
  };

  return (
    <Workspace controls={controls}>
      <div className="mobile-text-editor">{editor}</div>
      <div className="workspace-content text-workspace">
        <section className="glass-panel workspace-stage text-stage" aria-labelledby="text-workspace-title">
          <header className="workspace-stage-heading">
            <div className="workspace-heading-title">
              <Box size={19} aria-hidden="true" />
              <h1 id="text-workspace-title">Văn bản → Ký hiệu</h1>
            </div>
            <span className={`badge ${isPlaying ? 'badge-emerald' : 'badge-neutral'}`}>{isPlaying ? 'Đang diễn' : 'Tạm dừng'}</span>
          </header>
          <div className="sign-viewport" aria-label="Mô phỏng 3D">
            {viewModel === 'robot' ? <RobotSigner3D {...commonProps} />
              : <MixamoAvatarSigner {...commonProps} tunerHostRef={tunerHostRef} />}
          </div>
          {viewModel === 'avatar' && <AvatarSequenceCaption sequence={sequence} currentIndex={currentIndex} />}
          <div className="mobile-playback">{playback}</div>
        </section>

        <section className="glass-panel workspace-inspector sign-inspector" aria-labelledby="sign-inspector-title">
          <div className="workspace-inspector-heading">
            <h2 id="sign-inspector-title">Ký hiệu đang diễn</h2>
            {sequence.length > 0 && <span className="badge badge-neutral">{currentIndex + 1}/{sequence.length}</span>}
          </div>
          {currentStep ? (
            <>
              <h3 className="sign-current-word" aria-live="polite">{stepLabel(currentStep)}</h3>
              <div className="sign-description" tabIndex={0} aria-label="Mô tả ký hiệu">
                {currentStep.type === 'vocabulary' && currentStep.region_label && (
                  <span className="badge badge-cyan">{currentStep.region_label}</span>
                )}
                {currentStep.definition && <p>{currentStep.definition}</p>}
                {currentStep.type === 'vocabulary' && !currentStep.video_id && !currentStep.motion_id && !currentStep.pose_key && (
                  <p className="sign-notice">Đang diễn từ này bằng bảng chữ cái ngón tay.</p>
                )}
                {currentStep.video_id && !currentStep.motion_id && (
                  <p className="sign-notice" role="status">
                    {learningProgress === 'failed' ? 'Đang thay thế bằng đánh vần.'
                      : `Đang chuẩn bị ký hiệu từ video · ${typeof learningProgress === 'number' ? Math.round(learningProgress * 100) : 0}%`}
                  </p>
                )}
              </div>
            </>
          ) : (
            <p className="workspace-empty">{loading ? 'Đang phân tích văn bản…' : 'Nhập văn bản và bấm Dịch để bắt đầu.'}</p>
          )}
          {sequence.length > 0 && (
            <div className="sign-sequence">
              <h3>Chuỗi ký hiệu</h3>
              <ol aria-label="Chọn ký hiệu để phát">
                {sequence.map((step, index) => (
                  <li key={index}>
                    <button type="button" aria-current={index === currentIndex ? 'step' : undefined}
                      onClick={() => { setCurrentIndex(index); setIsPlaying(true); }}>
                      <span>{index + 1}</span>{stepLabel(step)}
                    </button>
                  </li>
                ))}
              </ol>
            </div>
          )}
        </section>
      </div>

      <WorkspaceDialog dialogRef={phrasesDialogRef} title="Câu mẫu">
        <div className="dialog-button-list">
          {PHRASE_SUGGESTIONS.map((phrase) => (
            <button type="button" key={phrase} className="btn btn-secondary" onClick={() => selectSample(phrase)}>{phrase}</button>
          ))}
        </div>
        <button type="button" className="btn btn-primary" onClick={() => selectSample(
          'Những ngày cuối tuần của tôi không có lịch trình cụ thể vì chúng phụ thuộc vào mức độ bận rộn của tôi trong tuần. Tôi thường dành một ít thời gian cho việc học tập hoặc các dự án cá nhân, nhưng tôi cũng đảm bảo mình có đủ thời gian để thư giãn. Nếu có thể, tôi thích gặp gỡ bạn bè hoặc khám phá những địa điểm mới, vì điều đó cho tôi cơ hội nghỉ ngơi khỏi thói quen thường ngày.'
        )}>Diễn đoạn văn mẫu</button>
      </WorkspaceDialog>

      <WorkspaceDialog dialogRef={alphabetDialogRef} title="Bảng chữ cái">
        <div className="alphabet-grid">
          {ALPHABET_LIST.flatMap((letter) => [letter, ...(LETTER_VARIANTS[letter] || [])]).map((letter) => {
            const selected = currentStep?.type === 'fingerspelling' && currentStep.word === letter.toLowerCase();
            return (
              <button type="button" key={letter} className={`btn ${selected ? 'btn-primary' : 'btn-secondary'}`}
                aria-pressed={selected} onClick={() => {
                  handleSelectDirectLetter(letter);
                  alphabetDialogRef.current?.close();
                }}>{letter}</button>
            );
          })}
        </div>
        <div className="dialog-control-group">
          <h3>Dấu thanh</h3>
          <div className="dialog-button-list">
            {TONE_LIST.map(({ key, mark }) => (
              <button type="button" key={key} className="btn btn-secondary" onClick={() => {
                handleSelectDirectLetter(key);
                alphabetDialogRef.current?.close();
              }}>{key} {mark}</button>
            ))}
          </div>
        </div>
      </WorkspaceDialog>

      <WorkspaceDialog dialogRef={detailsDialogRef} title="Chi tiết ký hiệu">
        <h3>{stepLabel(currentStep) || 'Tư thế hiện tại'}</h3>
        {currentStep?.definition && <p>{currentStep.definition}</p>}
        {currentStep && <SignDescription step={currentStep} />}
        {(currentStep?.has_pose ?? true) && (
          <dl className="finger-details">
            {fingerDetails.map((finger, index) => (
              <div key={index}>
                <dt style={{ color: finger.color }}>{finger.name}</dt>
                <dd>{finger.state}</dd>
              </div>
            ))}
          </dl>
        )}
        {viewModel === 'avatar' && (
          <details className="avatar-tuner">
            <summary>Tinh chỉnh bàn tay Avatar</summary>
            <div ref={tunerHostRef} className="avatar-tuner-host" />
          </details>
        )}
      </WorkspaceDialog>
    </Workspace>
  );
}
