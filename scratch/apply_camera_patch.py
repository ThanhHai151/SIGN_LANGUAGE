import re

with open("src/components/CameraSignRecognizer.jsx", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Add refs
ref_target = "  const lastMotionUpdateRef = useRef(0);"
ref_replacement = """  const lastMotionUpdateRef = useRef(0);

  // Đồng bộ mode thời gian thực chống Stale Closure & Out-of-Order In-Flight Requests
  const conversionModeRef = useRef(conversionMode);
  const recognitionModeRef = useRef(recognitionMode);
  const requestGenRef = useRef(0);

  useEffect(() => {
    conversionModeRef.current = conversionMode;
  }, [conversionMode]);

  useEffect(() => {
    recognitionModeRef.current = recognitionMode;
  }, [recognitionMode]);"""

assert ref_target in code, "Could not find ref_target"
code = code.replace(ref_target, ref_replacement, 1)

# 2. Update handleModeChange
mode_change_pattern = re.compile(
    r"  const handleModeChange = \(newMode\) => \{.*?\n  \};\n",
    re.DOTALL
)

mode_change_replacement = """  const handleModeChange = (newMode) => {
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
"""

assert mode_change_pattern.search(code) is not None, "Could not find handleModeChange"
code = mode_change_pattern.sub(mode_change_replacement, code, count=1)

# 3. Update handleMotionAndSequence
motion_seq_pattern = re.compile(
    r"    // 2\. CHẾ ĐỘ 1: CHUỖI ĐỘNG VSL.*?    else if \(recognitionMode === 'instant'\) \{.*?\n    \}\n  \};",
    re.DOTALL
)

motion_seq_replacement = """    const currentConversionMode = conversionModeRef.current;
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
  };"""

assert motion_seq_pattern.search(code) is not None, "Could not find motion_seq_pattern"
code = motion_seq_pattern.sub(motion_seq_replacement, code, count=1)

with open("src/components/CameraSignRecognizer.jsx", "w", encoding="utf-8") as f:
    f.write(code)

print("Patch applied successfully!")

