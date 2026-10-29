# -*- coding: utf-8 -*-
path = 'src/components/CameraSignRecognizer.jsx'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Thêm hàm formatSentenceForTTS ngay sau DIGIT_TTS_MAP
old_map_section = """// Bảng ánh xạ phát âm TTS chuẩn cho từng chữ số (chỉ đọc tên số, không có tiền tố "số")
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
};"""

new_map_section = """// Bảng ánh xạ phát âm TTS chuẩn cho từng chữ số (chỉ đọc tên số, không có tiền tố "số")
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
  return cleaned.replace(/\\b(10|[0-9])\\b/g, (match) => {
    return DIGIT_TTS_MAP[match] || match;
  });
}"""

assert old_map_section in content, "old_map_section not found"
content = content.replace(old_map_section, new_map_section, 1)

# 2. Làm sạch phụ đề hiện tại khi đổi sang mode digits trong handleModeChange
old_mode_change = """    // Reset kết quả cũ để tránh hiển thị nhầm lẫn giữa các chế độ (bảo tồn sentence đã xác nhận)
    setCurrentResult({"""

new_mode_change = """    // Nếu chuyển sang chế độ digits, chuẩn hóa phụ đề hiện tại để bỏ từ "Số "
    if (newMode === 'digits') {
      setSentence(prev => prev ? prev.replace(/(?:Số|số)\\s+(\\d+)/g, '$1') : '');
      setRecognizedTokens(prev => prev.map(item => ({
        ...item,
        token: (typeof item.token === 'string') ? item.token.replace(/^Số\\s*/, '') : item.token
      })));
    }

    // Reset kết quả cũ để tránh hiển thị nhầm lẫn giữa các chế độ (bảo tồn sentence đã xác nhận)
    setCurrentResult({"""

assert old_mode_change in content, "old_mode_change not found"
content = content.replace(old_mode_change, new_mode_change, 1)

# 3. Cập nhật phần setSentence và setRecognizedTokens trong callback nhận diện tức thì
old_infer_callback = """              if (data.confirmed && data.new_token && data.sign !== 'NEUTRAL') {
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
              }"""

new_infer_callback = """              if (data.confirmed && data.new_token && data.sign !== 'NEUTRAL') {
                // Loại bỏ hoàn toàn tiền tố "Số " nếu kết quả là chữ số
                const cleanToken = (requestMode === 'digits' && typeof data.new_token === 'string')
                  ? data.new_token.replace(/^Số\\s*/, '')
                  : data.new_token;

                setSentence(prev => {
                  // Làm sạch tiền tố "Số " trong chuỗi phụ đề phiên hiện tại nếu có
                  const normalizedPrev = prev ? prev.replace(/(?:Số|số)\\s+(\\d+)/g, '$1') : '';
                  return normalizedPrev ? `${normalizedPrev} ${cleanToken}` : cleanToken;
                });

                setRecognizedTokens(prev => [
                  ...prev.slice(-7),
                  {
                    token: cleanToken,
                    confidence: data.confidence,
                    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
                  }
                ]);

                if (ttsEnabled) {
                  let textToSpeak = data.tts_text;
                  if (!textToSpeak) {
                    if (data.type === 'digit' || requestMode === 'digits') {
                      textToSpeak = DIGIT_TTS_MAP[cleanToken] || cleanToken;
                    } else {
                      textToSpeak = cleanToken;
                    }
                  }
                  speakVietnamese(textToSpeak);
                }
              }"""

assert old_infer_callback in content, "old_infer_callback not found"
content = content.replace(old_infer_callback, new_infer_callback, 1)

# 4. Sửa nút Đọc phụ đề
old_button_read = """              <button
                onClick={() => speakVietnamese(sentence)}
                disabled={!sentence}"""

new_button_read = """              <button
                onClick={() => speakVietnamese(formatSentenceForTTS(sentence))}
                disabled={!sentence}"""

assert old_button_read in content, "old_button_read not found"
content = content.replace(old_button_read, new_button_read, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated src/components/CameraSignRecognizer.jsx for clean digit subtitle and TTS formatting.")

