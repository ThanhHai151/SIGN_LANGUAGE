/**
 * Client-side scope guardrail for Camera `words` recognition results.
 * Ensures rejected or out-of-scope predictions (including unverified `gt-*`
 * curriculum classes) never render as recognized words, never enter subtitles,
 * never trigger TTS, and never overwrite valid tokens, while preserving debug
 * rejection reasons.
 */

export function isGtEntryData(item) {
  if (!item || typeof item !== 'object') return false;
  if (item.region === 'GT') return true;
  if (typeof item.entry_id === 'string' && item.entry_id.startsWith('gt-')) return true;
  return false;
}

export function isValidWordRecognitionData(data) {
  if (!data || typeof data !== 'object') return false;
  if (data.scope_rejected === true || data.word_scope_valid === false) return false;
  if (isGtEntryData(data)) return false;
  if (!data.sign || data.sign === 'NEUTRAL') return false;
  return true;
}

export function extractRejectReason(data) {
  if (!data || typeof data !== 'object') return null;
  if (data.reject_reason || data.diagnostic?.reject_reason) {
    return data.reject_reason || data.diagnostic?.reject_reason;
  }
  if (isGtEntryData(data)) return 'unverified_gt_curriculum';
  return null;
}

export function sanitizeWordTopK(topK) {
  if (!Array.isArray(topK)) return [];
  return topK.filter((item) => item && !isGtEntryData(item) && item.scope_rejected !== true);
}

export function resolveWordRecognitionOutput(data, { currentToken = null } = {}) {
  const rejectReason = extractRejectReason(data);
  if (!isValidWordRecognitionData(data)) {
    return {
      accepted: false,
      displayResult: {
        sign: 'NEUTRAL',
        label: isGtEntryData(data)
          ? 'Từ nằm ngoài phạm vi nhận diện cho phép'
          : (data?.label || 'Từ nằm ngoài phạm vi nhận diện cho phép'),
        confidence: 0,
        scopeRejected: Boolean(data?.scope_rejected || data?.word_scope_valid === false || isGtEntryData(data)),
        rejectReason,
      },
      token: null,
      replaced: null,
      speakText: null,
      preserveToken: currentToken,
      debugReason: rejectReason,
    };
  }

  return {
    accepted: true,
    displayResult: {
      sign: data.sign,
      label: data.label,
      confidence: data.confidence,
      type: data.category || 'dynamic_sign',
      description: data.description,
      category: data.category,
      video_url: data.video_url,
      region: data.region,
      source: data.source,
      matchMethod: data.match_method,
      scopeRejected: false,
      rejectReason: null,
    },
    token: data.confirmed && data.new_token ? data.new_token : null,
    replaced: null,
    speakText: data.tts_text || (data.confirmed && data.new_token ? data.new_token : null),
    preserveToken: currentToken,
    debugReason: null,
  };
}
