/**
 * Bộ điều phối phát âm (TTS Orchestrator) duy nhất trên Frontend.
 * - Chỉ phát âm dựa trên sự kiện đã được Backend xác nhận (tts_speak + tts_event_id).
 * - Chống đọc trùng theo mã sự kiện (eventId), không chặn nhầm hai ký hiệu giống nhau liên tiếp.
 * - Quản lý tập trung vòng đời hủy/dọn dẹp khi tắt giọng nói, tắt camera, đổi chế độ, xóa phụ đề hoặc unmount.
 */

export function normalizeSpeechToken(rawText, mode = null, digitMap = null) {
  if (typeof rawText !== 'string') return '';
  const trimmed = rawText.trim();
  if (!trimmed) return '';

  if (mode === 'digits' && digitMap) {
    const stripped = trimmed.replace(/^(?:Số|số)\s+/, '');
    return digitMap[trimmed] || digitMap[stripped] || stripped;
  }
  return trimmed;
}

export function createTtsOrchestrator({
  speakFn,
  cancelFn,
  maxRememberedEvents = 256,
} = {}) {
  const spokenEventIds = new Set();
  const eventOrder = [];
  const activeTimers = new Set();

  function rememberEventId(eventId) {
    spokenEventIds.add(eventId);
    eventOrder.push(eventId);
    while (eventOrder.length > maxRememberedEvents) {
      const oldest = eventOrder.shift();
      if (oldest !== undefined) {
        spokenEventIds.delete(oldest);
      }
    }
  }

  function clearTimers() {
    for (const timerId of activeTimers) {
      clearTimeout(timerId);
    }
    activeTimers.clear();
  }

  function speakConfirmedEvent({
    enabled = true,
    eventId = null,
    text = '',
    mode = null,
    digitMap = null,
  } = {}) {
    if (!enabled) {
      return { spoken: false, reason: 'tts_disabled', eventId };
    }
    const cleanText = normalizeSpeechToken(text, mode, digitMap);
    if (!cleanText) {
      return { spoken: false, reason: 'empty_text', eventId };
    }
    if (typeof eventId === 'string' && eventId.length > 0) {
      if (spokenEventIds.has(eventId)) {
        return { spoken: false, reason: 'duplicate_event_id', eventId };
      }
      rememberEventId(eventId);
    }
    if (typeof speakFn === 'function') {
      speakFn(cleanText, { interrupt: false });
    }
    return { spoken: true, reason: 'ok', text: cleanText, eventId };
  }

  function speakManualSubtitle({ text = '' } = {}) {
    const cleanText = typeof text === 'string' ? text.trim() : '';
    if (!cleanText) {
      return { spoken: false, reason: 'empty_text' };
    }
    if (typeof speakFn === 'function') {
      speakFn(cleanText, { interrupt: true });
    }
    return { spoken: true, reason: 'manual', text: cleanText };
  }

  function scheduleTimer(callback, delayMs) {
    const id = setTimeout(() => {
      activeTimers.delete(id);
      callback();
    }, delayMs);
    activeTimers.add(id);
    return id;
  }

  function reset({ cancelAudio = true, clearHistory = false } = {}) {
    clearTimers();
    if (clearHistory) {
      spokenEventIds.clear();
      eventOrder.length = 0;
    }
    if (cancelAudio && typeof cancelFn === 'function') {
      cancelFn();
    }
  }

  return {
    speakConfirmedEvent,
    speakManualSubtitle,
    scheduleTimer,
    clearTimers,
    reset,
    hasSpokenEvent: (eventId) => spokenEventIds.has(eventId),
    getRememberedCount: () => spokenEventIds.size,
  };
}
