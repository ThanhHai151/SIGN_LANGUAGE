import test from 'node:test';
import assert from 'node:assert/strict';
import { createTtsOrchestrator, normalizeSpeechToken } from '../src/utils/ttsOrchestrator.js';

const DIGIT_MAP = {
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
  '10': 'mười',
};

test('normalizeSpeechToken strips Số prefix in digits mode and preserves letters', () => {
  assert.equal(normalizeSpeechToken('1', 'digits', DIGIT_MAP), 'một');
  assert.equal(normalizeSpeechToken('Số 10', 'digits', DIGIT_MAP), 'mười');
  assert.equal(normalizeSpeechToken('mười', 'digits', DIGIT_MAP), 'mười');
  assert.equal(normalizeSpeechToken('Ă', 'letters', DIGIT_MAP), 'Ă');
});

test('deduplicates by eventId while allowing consecutive identical signs with distinct eventIds', () => {
  const calls = [];
  const orch = createTtsOrchestrator({
    speakFn: (text, opts) => calls.push({ text, opts }),
    cancelFn: () => calls.push({ cancel: true }),
  });

  // First A confirmed & flushed
  const r1 = orch.speakConfirmedEvent({ enabled: true, eventId: 'tts_s1_1', text: 'A', mode: 'letters' });
  assert.equal(r1.spoken, true);

  // Duplicate delivery of same eventId must NOT speak again
  const rDup = orch.speakConfirmedEvent({ enabled: true, eventId: 'tts_s1_1', text: 'A', mode: 'letters' });
  assert.equal(rDup.spoken, false);
  assert.equal(rDup.reason, 'duplicate_event_id');

  // Second deliberate A with new eventId MUST speak
  const r2 = orch.speakConfirmedEvent({ enabled: true, eventId: 'tts_s1_2', text: 'A', mode: 'letters' });
  assert.equal(r2.spoken, true);

  // Consecutive identical digits '1' -> '1'
  orch.speakConfirmedEvent({ enabled: true, eventId: 'tts_s1_3', text: '1', mode: 'digits', digitMap: DIGIT_MAP });
  orch.speakConfirmedEvent({ enabled: true, eventId: 'tts_s1_4', text: '1', mode: 'digits', digitMap: DIGIT_MAP });

  assert.deepEqual(
    calls.map((c) => c.text),
    ['A', 'A', 'một', 'một']
  );
  assert.ok(calls.every((c) => c.opts?.interrupt === false));
});

test('respects ttsEnabled toggle and cleans up timers/audio on camera stop, mode switch, clear subtitle', () => {
  const events = [];
  const orch = createTtsOrchestrator({
    speakFn: (text, opts) => events.push({ type: 'speak', text, interrupt: opts?.interrupt }),
    cancelFn: () => events.push({ type: 'cancel' }),
  });

  // Voice disabled -> never speaks
  const muted = orch.speakConfirmedEvent({ enabled: false, eventId: 'tts_muted_1', text: 'B', mode: 'letters' });
  assert.equal(muted.spoken, false);
  assert.equal(muted.reason, 'tts_disabled');
  assert.equal(events.length, 0);

  // Manual speaker button uses interrupt: true even for full sentence
  const manual = orch.speakManualSubtitle({ text: 'A B C' });
  assert.equal(manual.spoken, true);
  assert.deepEqual(events[0], { type: 'speak', text: 'A B C', interrupt: true });

  // Scheduled timer is cleared on reset (camera off / mode switch / clear sentence)
  let timerFired = false;
  orch.scheduleTimer(() => { timerFired = true; }, 50);
  orch.reset({ cancelAudio: true });
  assert.equal(events.at(-1)?.type, 'cancel');
  assert.equal(timerFired, false);
});
