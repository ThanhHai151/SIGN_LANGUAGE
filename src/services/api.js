/**
 * VSL Connect API Service
 * Giao tiếp thời gian thực giữa Frontend React và Python FastAPI Backend
 */

const API_BASE_URL = 'http://127.0.0.1:8000';

export async function checkBackendHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/health`, { method: 'GET' });
    if (!res.ok) throw new Error('Backend offline');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Backend health check failed:', err.message);
    return { status: 'offline', error: err.message };
  }
}

export async function fetchDictionary() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/dictionary`);
    if (!res.ok) throw new Error('Cannot load dictionary');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Using fallback dictionary due to error:', err);
    return null;
  }
}

export async function fetchVSL102Classes() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/vsl_102_classes`);
    if (!res.ok) throw new Error('Cannot load VSL 102 classes');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Cannot fetch 102 classes:', err);
    return null;
  }
}

export async function sendSequenceForRecognition(sequence, threshold = 0.65) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/recognize_sequence`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sequence, threshold })
    });
    if (!res.ok) throw new Error('Sequence recognition failed');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Sequence recognition error:', err);
    return {
      success: false,
      error: err.message
    };
  }
}

export async function sendLandmarksForRecognition(landmarks, multi_hands = null, handedness = 'Right') {
  try {
    const res = await fetch(`${API_BASE_URL}/api/recognize_landmarks`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ landmarks, multi_hands, handedness })
    });
    if (!res.ok) throw new Error('Recognition failed');
    return await res.json();
  } catch (err) {
    // Fallback: Local Client-side heuristic if backend is temporarily disconnected
    return {
      success: true,
      data: clientSideLocalRecognize(landmarks)
    };
  }
}

export async function convertTextToVSL(text) {
  try {
    const res = await fetch(`${API_BASE_URL}/api/text_to_sign`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    if (!res.ok) throw new Error('Text to sign failed');
    return await res.json();
  } catch (err) {
    console.warn('[VSL API] Fallback to client-side text tokenizer:', err);
    return clientSideTextToSign(text);
  }
}

export async function clearCurrentSentence() {
  try {
    const res = await fetch(`${API_BASE_URL}/api/clear_sentence`, { method: 'POST' });
    return await res.json();
  } catch (err) {
    return { success: true };
  }
}

/**
 * Phát âm tiếng Việt (Text-to-Speech) sử dụng Web Speech API
 */
export function speakVietnamese(text) {
  if (!window.speechSynthesis || !text) return;

  window.speechSynthesis.cancel(); // Dừng câu trước nếu đang nói

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = 'vi-VN';
  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  // Cố gắng tìm giọng đọc tiếng Việt nếu có trong trình duyệt
  const voices = window.speechSynthesis.getVoices();
  const viVoice = voices.find(v => v.lang.includes('vi') || v.lang.includes('VN'));
  if (viVoice) {
    utterance.voice = viVoice;
  }

  window.speechSynthesis.speak(utterance);
}

/**
 * Fallback bộ nhận diện tại Client (Edge) khi mất kết nối Backend
 */
function clientSideLocalRecognize(landmarks) {
  if (!landmarks || landmarks.length < 21) {
    return { sign: 'NEUTRAL', label: 'Chờ cử chỉ...', confidence: 0.0, confirmed: false };
  }
  const p = landmarks;
  const thumbExt = p[4].x < p[3].x;
  const indexExt = p[8].y < p[6].y;
  const middleExt = p[12].y < p[10].y;
  const ringExt = p[16].y < p[14].y;
  const pinkyExt = p[20].y < p[18].y;

  if (indexExt && middleExt && !ringExt && !pinkyExt) {
    return { sign: 'V', label: 'Chữ V / Hòa bình', confidence: 0.94, confirmed: true };
  }
  if (indexExt && thumbExt && !middleExt && !ringExt && !pinkyExt) {
    return { sign: 'L', label: 'Chữ L', confidence: 0.95, confirmed: true };
  }
  if (indexExt && thumbExt && pinkyExt && !middleExt && !ringExt) {
    return { sign: 'YEU_THUONG', label: 'Yêu thương', confidence: 0.96, confirmed: true };
  }
  if (indexExt && middleExt && ringExt && pinkyExt) {
    return { sign: 'XIN_CHAO', label: 'Xin chào', confidence: 0.92, confirmed: true };
  }
  return { sign: 'GESTURE', label: 'Cử chỉ VSL', confidence: 0.85, confirmed: true };
}

/**
 * Fallback Text-to-Sign Client-side
 */
function clientSideTextToSign(text) {
  const words = text.trim().split(/\s+/);
  const sequence = words.map(w => ({
    token: w,
    gloss: w.toUpperCase(),
    label: w.charAt(0).toUpperCase() + w.slice(1),
    type: 'vocabulary',
    duration_ms: 1500,
    description: `Thực hiện cử chỉ VSL cho từ "${w}"`,
    avatar_pose: {
      right_arm: [30, 20, 0],
      right_wrist: [10, 0, 0],
      fingers: [1, 1, 1, 1, 1]
    }
  }));
  return {
    success: true,
    input_text: text,
    sequence,
    total_tokens: sequence.length,
    total_duration_ms: sequence.length * 1500
  };
}
