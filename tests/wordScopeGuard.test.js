import test from 'node:test';
import assert from 'node:assert/strict';
import {
  extractRejectReason,
  isGtEntryData,
  isValidWordRecognitionData,
  resolveWordRecognitionOutput,
  sanitizeWordTopK,
} from '../src/utils/wordScopeGuard.js';

test('Case 11a: valid recognized word passes guard and permits token/TTS', () => {
  const data = {
    success: true,
    sign: 'xin lỗi',
    label: 'xin lỗi',
    entry_id: 'q-xin-loi',
    confidence: 0.91,
    region: '',
    source: 'qipedc_template',
    match_method: 'trajectory',
    word_scope_valid: true,
    scope_rejected: false,
    confirmed: true,
    new_token: 'xin lỗi',
  };

  assert.equal(isGtEntryData(data), false);
  assert.equal(isValidWordRecognitionData(data), true);
  const out = resolveWordRecognitionOutput(data, { currentToken: 'chào' });
  assert.equal(out.accepted, true);
  assert.equal(out.displayResult.sign, 'xin lỗi');
  assert.equal(out.token, 'xin lỗi');
  assert.equal(out.speakText, 'xin lỗi');
});

test('Case 11b: scope-rejected result becomes NEUTRAL, emits no subtitle/TTS, does not overwrite valid token, preserves rejectReason', () => {
  const rejectedPayload = {
    success: true,
    sign: 'NEUTRAL',
    label: 'Từ nằm ngoài phạm vi nhận diện cho phép',
    confidence: 0.0,
    confirmed: true,
    new_token: 'từ bị cấm',
    tts_text: 'từ bị cấm',
    scope_rejected: true,
    word_scope_valid: false,
    reject_reason: 'rejected_region_north',
    diagnostic: {
      status: 'scope_rejected',
      reject_reason: 'rejected_region_north',
      rejected_entry_id: 'q-only-north',
    },
  };

  assert.equal(isValidWordRecognitionData(rejectedPayload), false);
  assert.equal(extractRejectReason(rejectedPayload), 'rejected_region_north');

  const out = resolveWordRecognitionOutput(rejectedPayload, { currentToken: 'xin chào' });
  assert.equal(out.accepted, false);
  assert.equal(out.displayResult.sign, 'NEUTRAL');
  assert.equal(out.displayResult.confidence, 0);
  assert.equal(out.displayResult.scopeRejected, true);
  assert.equal(out.token, null);
  assert.equal(out.replaced, null);
  assert.equal(out.speakText, null);
  assert.equal(out.preserveToken, 'xin chào');
  assert.equal(out.debugReason, 'rejected_region_north');
});

test('Case 11c: NEUTRAL or missing sign never enters subtitles or TTS', () => {
  const neutralData = {
    success: true,
    sign: 'NEUTRAL',
    label: 'Đang chờ động tác từ vựng…',
    confidence: 0.0,
    confirmed: false,
    new_token: null,
    diagnostic: { status: 'no_motion' },
  };
  assert.equal(isValidWordRecognitionData(neutralData), false);
  const out = resolveWordRecognitionOutput(neutralData, { currentToken: 'bạn' });
  assert.equal(out.token, null);
  assert.equal(out.speakText, null);
  assert.equal(out.preserveToken, 'bạn');
});

test('Case 11d: gt-* curriculum predictions and GT items in top_k are blocked from display, subtitles, TTS, and Top-K', () => {
  const gtLeak = {
    success: true,
    sign: 'Tôi',
    label: 'Tôi',
    entry_id: 'gt-toi',
    region: 'GT',
    confidence: 0.96,
    confirmed: true,
    new_token: 'Tôi',
    top_k: [
      { index: 51, term: 'Tôi', entry_id: 'gt-toi', confidence: 0.96 },
      { index: 67, term: 'Đi', entry_id: 'q-di', confidence: 0.04 },
      { index: 100, term: 'Công an', entry_id: 'gt-cong-an', confidence: 0.0 },
    ],
  };

  assert.equal(isGtEntryData(gtLeak), true);
  assert.equal(isValidWordRecognitionData(gtLeak), false);
  assert.equal(extractRejectReason(gtLeak), 'unverified_gt_curriculum');

  const out = resolveWordRecognitionOutput(gtLeak, { currentToken: 'chào' });
  assert.equal(out.accepted, false);
  assert.equal(out.displayResult.sign, 'NEUTRAL');
  assert.equal(out.displayResult.scopeRejected, true);
  assert.equal(out.token, null);
  assert.equal(out.speakText, null);
  assert.equal(out.preserveToken, 'chào');
  assert.equal(out.debugReason, 'unverified_gt_curriculum');

  const cleanTopK = sanitizeWordTopK(gtLeak.top_k);
  assert.deepEqual(cleanTopK, [{ index: 67, term: 'Đi', entry_id: 'q-di', confidence: 0.04 }]);
});
