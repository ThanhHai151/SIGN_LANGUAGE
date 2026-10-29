# -*- coding: utf-8 -*-
path = 'src/components/CameraSignRecognizer.jsx'
with open(path, 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Thêm DIGIT_TTS_MAP ở gần đầu file (sau các import)
old_import = """import {
  sendSequenceForRecognition,
  sendLandmarksForRecognition,
  clearCurrentSentence,
  speakVietnamese
} from '../services/api';"""

new_import = """import {
  sendSequenceForRecognition,
  sendLandmarksForRecognition,
  clearCurrentSentence,
  speakVietnamese
} from '../services/api';

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
};"""

assert old_import in content, "old_import not found"
content = content.replace(old_import, new_import, 1)

# 2. Sửa dòng phát âm TTS
old_speak = """                if (ttsEnabled) speakVietnamese(data.new_token);"""

new_speak = """                if (ttsEnabled) {
                  let textToSpeak = data.tts_text;
                  if (!textToSpeak) {
                    if (data.type === 'digit' || requestMode === 'digits') {
                      textToSpeak = DIGIT_TTS_MAP[data.sign] || (data.sign ? DIGIT_TTS_MAP[data.sign.replace(/^Số\\s*/, '')] : null) || data.sign;
                    } else {
                      textToSpeak = data.new_token;
                    }
                  }
                  speakVietnamese(textToSpeak);
                }"""

assert old_speak in content, "old_speak not found"
content = content.replace(old_speak, new_speak, 1)

with open(path, 'w', encoding='utf-8') as f:
    f.write(content)

print("Updated src/components/CameraSignRecognizer.jsx successfully.")

