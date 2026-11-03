/**
 * Word Debug API (chỉ dùng cho công cụ test Word Mode).
 * Hoàn toàn tách khỏi luồng nhận diện: không import/đổi gì trong api.js.
 */

const API_BASE_URL = 'http://127.0.0.1:8000';

/** Word Debug chỉ bật khi mở web với ?wordDebug=1 hoặc build với VITE_WORD_DEBUG=true. */
export function isWordDebugEnabled() {
  try {
    if (import.meta.env && import.meta.env.VITE_WORD_DEBUG === 'true') return true;
    return new URLSearchParams(window.location.search).get('wordDebug') === '1';
  } catch {
    return false;
  }
}

// Cờ "đang có phiên test" — panel cập nhật, camera đọc để quyết định có gửi event hay không.
let wordDebugActive = false;
export function setWordDebugActive(value) {
  wordDebugActive = Boolean(value);
}
export function isWordDebugActive() {
  return wordDebugActive;
}

async function request(path, options) {
  const res = await fetch(`${API_BASE_URL}${path}`, options);
  let body = null;
  try {
    body = await res.json();
  } catch {
    body = null;
  }
  if (!res.ok) {
    throw new Error((body && body.detail) || `HTTP ${res.status}`);
  }
  return body;
}

export async function fetchWordTargets() {
  const body = await request('/api/debug/word/targets');
  return body.targets || [];
}

export function fetchWordDebugStatus() {
  return request('/api/debug/word/status');
}

export function startWordDebugSession(targetClass, targetWord, trial) {
  return request('/api/debug/word/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'start', target_class: targetClass, target_word: targetWord, trial })
  });
}

export function stopWordDebugSession() {
  return request('/api/debug/word/session', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ action: 'stop' })
  });
}

/**
 * Ghi nhận buffer frame bị reset (fire-and-forget, không chặn luồng camera).
 * Chỉ gọi được khi có phiên test đang chạy.
 */
export function logWordDebugReset(framesBeforeReset, reason) {
  if (!wordDebugActive) return;
  fetch(`${API_BASE_URL}/api/debug/word/event`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      event_type: 'SEQUENCE_RESET',
      frames_before_reset: framesBeforeReset,
      reason
    })
  }).catch(() => {});
}
