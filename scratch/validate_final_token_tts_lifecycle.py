import os
import sys
import json
import time
import collections
import numpy as np

sys.path.insert(0, ".")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["VSL_DIACRITIC_DEBUG"] = "0"
os.environ["VSL_PREFETCH"] = "0"

from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

timestamp_str = time.strftime('%Y%m%d_%H%M%S')
session_id = f"final_val_{timestamp_str}"

print("=" * 95)
print(f"FINAL TOKEN REPLACEMENT, TTS LIFECYCLE & BATCH REGRESSION VALIDATION")
print(f"Session ID: {session_id}")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 95)

# ------------------------------------------------------------------------------
# 1. 7 DIACRITIC REPLAY: TOKENS, REPLACEMENT & TTS TIMING SEQUENCE
# ------------------------------------------------------------------------------
print("\n--- 1. VERIFYING 7 DIACRITIC LETTERS (REPLAY, TOKEN REPLACEMENT & TTS) ---")

VIDEOS = {
    'Ă': ('backend/data/alphabet_v2/motion_sequences/Ă.json', 'A', 'D0490B'),
    'Â': ('backend/data/alphabet_v2/motion_sequences/Â.json', 'A', 'D0491B'),
    'Ê': ('backend/data/alphabet_v2/motion_sequences/Ê.json', 'E', 'D0497B'),
    'Đ': ('backend/data/alphabet_v2/motion_sequences/Đ.json', 'D', 'D0495'),
    'Ô': ('backend/data/alphabet_v2/motion_sequences/Ô.json', 'O', 'D0507B'),
    'Ơ': ('backend/data/alphabet_v2/motion_sequences/Ơ.json', 'O', 'D0508N'),
    'Ư': ('backend/data/alphabet_v2/motion_sequences/Ư.json', 'U', 'D0515N'),
}

replay_results = {}

for target, (path, exp_base, vid_id) in VIDEOS.items():
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    seq = data['sequence']
    trial_id = f"{session_id}_{target}"
    rec.reset()
    rec.reset_diacritic_fsm(trial_id)

    token_events = []
    replacement_events = []
    tts_events = []
    base_emitted_time = None
    diac_emitted_time = None

    for item in seq:
        idx = item.get('frame_index')
        t = item.get('timestamp_sec', 0.0)
        lms = item.get('landmarks')

        res = rec.predict_landmarks(
            lms,
            mode='letters',
            session_id=trial_id,
            trial_id=trial_id,
            frame_idx=idx,
            timestamp=t,
            expected_sign=target
        )

        tok = res.get('new_token')
        rep = res.get('replaced_token')
        tts = res.get('tts_speak')

        if rep:
            replacement_events.append((idx, round(t, 3), rep, tok))
        if tok:
            token_events.append((idx, round(t, 3), tok, rep))
            if tok == exp_base and base_emitted_time is None:
                base_emitted_time = t
            if tok == target and diac_emitted_time is None:
                diac_emitted_time = t
        if tts:
            tts_events.append((idx, round(t, 3), tts))

    final_sentence = rec.sentence_tokens.copy()
    spoken_utterances = [ev[2] for ev in tts_events]
    delta_t_ms = round((diac_emitted_time - base_emitted_time) * 1000, 1) if (base_emitted_time and diac_emitted_time) else None

    # Criteria:
    # 1. Final sentence == [target]
    # 2. Spoken utterances == [target] (base letter NEVER spoken!)
    # 3. Exactly 1 replacement event (exp_base -> target)
    is_token_pass = (final_sentence == [target])
    is_tts_pass = (spoken_utterances == [target])
    is_overall_pass = is_token_pass and is_tts_pass

    replay_results[target] = {
        'target': target,
        'video_id': vid_id,
        'expected_base': exp_base,
        'delta_base_to_diac_ms': delta_t_ms,
        'token_events': token_events,
        'replacement_events': replacement_events,
        'tts_events': tts_events,
        'final_sentence': final_sentence,
        'spoken_utterances': spoken_utterances,
        'status': 'PASS' if is_overall_pass else 'FAIL'
    }

    print(f"Target {target:<2} ({vid_id}): Status={replay_results[target]['status']} | "
          f"Δ(base->diac)={delta_t_ms}ms | Final={final_sentence} | TTS={spoken_utterances}")

# ------------------------------------------------------------------------------
# 2. GESTURE LIFECYCLE & INDEPENDENT GESTURE SCENARIOS
# ------------------------------------------------------------------------------
print("\n--- 2. VERIFYING GESTURE LIFECYCLE & INDEPENDENT GESTURE SCENARIOS ---")

npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
static_data = np.load(npz_path, allow_pickle=True)
lm_static = static_data["landmarks"]
y_static = static_data["y"]
cls_static = [str(c) for c in static_data["classes"]]

def get_good_sample(ch):
    indices = [i for i, c in enumerate(y_static) if cls_static[c] == ch]
    for idx in indices:
        t_res = rec.predict_landmarks(lm_static[idx].tolist(), mode='letters')
        if t_res.get('sign') == ch:
            return lm_static[idx].tolist()
    return lm_static[indices[0]].tolist() if indices else None

neutral_lms = [[0.5, 0.85, 0.0]] * 21  # Hand dropped / neutral position (wrist_y = 0.85 > 0.70)
lm_a = get_good_sample('A')
lm_o = get_good_sample('O')
lm_v = get_good_sample('V')

scenarios = {}

# Scenario 3: User signs A independently, drops hand, then signs Ă
print("\n[Scenario 3] Ký A độc lập -> Thả tay (Neutral) -> Ký Ă (Video D0490B):")
rec.reset()
s3_trial = f"{session_id}_scenario3"
rec.reset_diacritic_fsm(s3_trial)
s3_tts = []

# Phase 1: Sign A for 5 frames
t = 1.0
for f in range(5):
    res = rec.predict_landmarks(lm_a, mode='letters', session_id=s3_trial, trial_id=s3_trial, timestamp=t)
    if res.get('tts_speak'):
        s3_tts.append((round(t, 3), res['tts_speak']))
    t += 0.04

# Phase 2: Drop hand into neutral for 5 frames (should commit TTS 'A' on release!)
for f in range(5):
    res = rec.predict_landmarks(neutral_lms, mode='letters', session_id=s3_trial, trial_id=s3_trial, timestamp=t)
    if res.get('tts_speak'):
        s3_tts.append((round(t, 3), res['tts_speak']))
    t += 0.04

sentence_after_release = rec.sentence_tokens.copy()

# Phase 3: Sign Ă from video fixture
with open(VIDEOS['Ă'][0], encoding='utf-8') as f:
    seq_a_breve = json.load(f)['sequence']

for item in seq_a_breve:
    idx = item.get('frame_index')
    fr_t = t + item.get('timestamp_sec', 0.0)
    lms = item.get('landmarks')
    res = rec.predict_landmarks(lms, mode='letters', session_id=s3_trial, trial_id=s3_trial, frame_idx=idx, timestamp=fr_t, expected_sign='Ă')
    if res.get('tts_speak'):
        s3_tts.append((round(fr_t, 3), res['tts_speak']))

s3_spoken = [ev[1] for ev in s3_tts]
s3_success = (rec.sentence_tokens == ['A', 'Ă']) and (s3_spoken == ['A', 'Ă'])
scenarios['scenario_3_independent_a_then_a_breve'] = {
    'expected_sentence': ['A', 'Ă'],
    'actual_sentence': rec.sentence_tokens.copy(),
    'expected_tts': ['A', 'Ă'],
    'actual_tts': s3_spoken,
    'status': 'PASS' if s3_success else 'FAIL'
}
print(f"Scenario 3 Status: {scenarios['scenario_3_independent_a_then_a_breve']['status']} | Sentence={rec.sentence_tokens} | TTS={s3_spoken}")

# Scenario 4: User signs O independently, drops hand, then signs Ô
print("\n[Scenario 4] Ký O độc lập -> Thả tay (Neutral) -> Ký Ô (Video D0507B):")
rec.reset()
s4_trial = f"{session_id}_scenario4"
rec.reset_diacritic_fsm(s4_trial)
s4_tts = []

# Phase 1: Sign O for 5 frames
t = 1.0
for f in range(5):
    res = rec.predict_landmarks(lm_o, mode='letters', session_id=s4_trial, trial_id=s4_trial, timestamp=t)
    if res.get('tts_speak'):
        s4_tts.append((round(t, 3), res['tts_speak']))
    t += 0.04

# Phase 2: Drop hand into neutral for 5 frames
for f in range(5):
    res = rec.predict_landmarks(neutral_lms, mode='letters', session_id=s4_trial, trial_id=s4_trial, timestamp=t)
    if res.get('tts_speak'):
        s4_tts.append((round(t, 3), res['tts_speak']))
    t += 0.04

# Phase 3: Sign Ô from video fixture
with open(VIDEOS['Ô'][0], encoding='utf-8') as f:
    seq_o_hat = json.load(f)['sequence']

for item in seq_o_hat:
    idx = item.get('frame_index')
    fr_t = t + item.get('timestamp_sec', 0.0)
    lms = item.get('landmarks')
    res = rec.predict_landmarks(lms, mode='letters', session_id=s4_trial, trial_id=s4_trial, frame_idx=idx, timestamp=fr_t, expected_sign='Ô')
    if res.get('tts_speak'):
        s4_tts.append((round(fr_t, 3), res['tts_speak']))

s4_spoken = [ev[1] for ev in s4_tts]
s4_success = (rec.sentence_tokens == ['O', 'Ô']) and (s4_spoken == ['O', 'Ô'])
scenarios['scenario_4_independent_o_then_o_hat'] = {
    'expected_sentence': ['O', 'Ô'],
    'actual_sentence': rec.sentence_tokens.copy(),
    'expected_tts': ['O', 'Ô'],
    'actual_tts': s4_spoken,
    'status': 'PASS' if s4_success else 'FAIL'
}
print(f"Scenario 4 Status: {scenarios['scenario_4_independent_o_then_o_hat']['status']} | Sentence={rec.sentence_tokens} | TTS={s4_spoken}")

# Scenario 5: User signs A, drops hand, signs A again
print("\n[Scenario 5] Ký A độc lập -> Thả tay -> Ký A lần 2:")
rec.reset()
s5_trial = f"{session_id}_scenario5"
rec.reset_diacritic_fsm(s5_trial)
s5_tts = []

t = 1.0
for f in range(5):
    res = rec.predict_landmarks(lm_a, mode='letters', session_id=s5_trial, trial_id=s5_trial, timestamp=t)
    if res.get('tts_speak'):
        s5_tts.append((round(t, 3), res['tts_speak']))
    t += 0.04

for f in range(5):
    res = rec.predict_landmarks(neutral_lms, mode='letters', session_id=s5_trial, trial_id=s5_trial, timestamp=t)
    if res.get('tts_speak'):
        s5_tts.append((round(t, 3), res['tts_speak']))
    t += 0.04

t += 0.5  # Pauses past cooldown
for f in range(5):
    res = rec.predict_landmarks(lm_a, mode='letters', session_id=s5_trial, trial_id=s5_trial, timestamp=t)
    if res.get('tts_speak'):
        s5_tts.append((round(t, 3), res['tts_speak']))
    t += 0.04

for f in range(5):
    res = rec.predict_landmarks(neutral_lms, mode='letters', session_id=s5_trial, trial_id=s5_trial, timestamp=t)
    if res.get('tts_speak'):
        s5_tts.append((round(t, 3), res['tts_speak']))
    t += 0.04

s5_spoken = [ev[1] for ev in s5_tts]
s5_success = (rec.sentence_tokens == ['A', 'A']) and (s5_spoken == ['A', 'A'])
scenarios['scenario_5_consecutive_identical_signs'] = {
    'expected_sentence': ['A', 'A'],
    'actual_sentence': rec.sentence_tokens.copy(),
    'expected_tts': ['A', 'A'],
    'actual_tts': s5_spoken,
    'status': 'PASS' if s5_success else 'FAIL'
}
print(f"Scenario 5 Status: {scenarios['scenario_5_consecutive_identical_signs']['status']} | Sentence={rec.sentence_tokens} | TTS={s5_spoken}")

# Scenario 6: Reset between trials
print("\n[Scenario 6] Trial 1 kết thúc với A -> Reset -> Trial 2 ký Ă độc lập:")
rec.reset()
s6_t1 = f"{session_id}_trial1"
s6_t2 = f"{session_id}_trial2"

# Trial 1 signs A
for f in range(5):
    rec.predict_landmarks(lm_a, mode='letters', session_id=session_id, trial_id=s6_t1, timestamp=1.0 + f * 0.04)

rec.reset_diacritic_fsm(s6_t2)
rec.reset()

# Trial 2 signs Ă
s6_tts2 = []
with open(VIDEOS['Ă'][0], encoding='utf-8') as f:
    seq_a_breve = json.load(f)['sequence']

for item in seq_a_breve:
    idx = item.get('frame_index')
    fr_t = 10.0 + item.get('timestamp_sec', 0.0)
    lms = item.get('landmarks')
    res = rec.predict_landmarks(lms, mode='letters', session_id=session_id, trial_id=s6_t2, frame_idx=idx, timestamp=fr_t, expected_sign='Ă')
    if res.get('tts_speak'):
        s6_tts2.append((round(fr_t, 3), res['tts_speak']))

s6_spoken = [ev[1] for ev in s6_tts2]
s6_success = (rec.sentence_tokens == ['Ă']) and (s6_spoken == ['Ă'])
scenarios['scenario_6_trial_isolation_reset'] = {
    'expected_sentence': ['Ă'],
    'actual_sentence': rec.sentence_tokens.copy(),
    'expected_tts': ['Ă'],
    'actual_tts': s6_spoken,
    'status': 'PASS' if s6_success else 'FAIL'
}
print(f"Scenario 6 Status: {scenarios['scenario_6_trial_isolation_reset']['status']} | Sentence={rec.sentence_tokens} | TTS={s6_spoken}")

# Scenario 7: Standalone A held static for 1.5 seconds (Stationary hold commit without release)
print("\n[Scenario 7] Ký A và giữ yên 1.5s không hạ tay (Stationary hold commit):")
rec.reset()
s7_trial = f"{session_id}_scenario7"
rec.reset_diacritic_fsm(s7_trial)
s7_tts = []

t = 1.0
for f in range(35):  # 35 * 0.04s = 1.4s
    res = rec.predict_landmarks(lm_a, mode='letters', session_id=s7_trial, trial_id=s7_trial, timestamp=t)
    if res.get('tts_speak'):
        s7_tts.append((round(t, 3), res['tts_speak']))
    t += 0.04

s7_spoken = [ev[1] for ev in s7_tts]
s7_success = (rec.sentence_tokens == ['A']) and (s7_spoken == ['A'])
scenarios['scenario_7_stationary_hold_commit'] = {
    'expected_sentence': ['A'],
    'actual_sentence': rec.sentence_tokens.copy(),
    'expected_tts': ['A'],
    'actual_tts': s7_spoken,
    'status': 'PASS' if s7_success else 'FAIL'
}
print(f"Scenario 7 Status: {scenarios['scenario_7_stationary_hold_commit']['status']} | Sentence={rec.sentence_tokens} | TTS={s7_spoken}")

# ------------------------------------------------------------------------------
# 3. EXIT GUARD DETAILED VERIFICATION
# ------------------------------------------------------------------------------
print("\n--- 3. VERIFYING EXIT GUARD BEHAVIOR (Ă -> V) ---")

rec.reset()
eg_trial = f"{session_id}_exit_guard"
rec.reset_diacritic_fsm(eg_trial)
eg_tts = []

# Part A: Run Ă video replay
with open(VIDEOS['Ă'][0], encoding='utf-8') as f:
    seq_a_breve = json.load(f)['sequence']

t_curr = 0.0
for item in seq_a_breve:
    idx = item.get('frame_index')
    t_curr = item.get('timestamp_sec', 0.0)
    lms = item.get('landmarks')
    res = rec.predict_landmarks(lms, mode='letters', session_id=eg_trial, trial_id=eg_trial, frame_idx=idx, timestamp=t_curr, expected_sign='Ă')
    if res.get('tts_speak'):
        eg_tts.append((round(t_curr, 3), res['tts_speak']))

sentence_after_a_breve = rec.sentence_tokens.copy()
print(f"Exit Guard Part A (After Ă): Sentence={sentence_after_a_breve} (V must NOT appear)")

# Part B: User intentionally drops hand to neutral, then brings up deliberate V
t_curr += 0.2
for f in range(5):  # 5 frames of neutral release
    res = rec.predict_landmarks(neutral_lms, mode='letters', session_id=eg_trial, trial_id=eg_trial, timestamp=t_curr)
    if res.get('tts_speak'):
        eg_tts.append((round(t_curr, 3), res['tts_speak']))
    t_curr += 0.04

t_curr += 0.5  # Pause past cooldown
for f in range(8):  # 8 frames of deliberate V
    res = rec.predict_landmarks(lm_v, mode='letters', session_id=eg_trial, trial_id=eg_trial, timestamp=t_curr)
    if res.get('tts_speak'):
        eg_tts.append((round(t_curr, 3), res['tts_speak']))
    t_curr += 0.04

sentence_after_deliberate_v = rec.sentence_tokens.copy()
eg_spoken = [ev[1] for ev in eg_tts]
eg_success = (sentence_after_a_breve == ['Ă']) and (sentence_after_deliberate_v == ['Ă', 'V']) and (eg_spoken == ['Ă', 'V'])

exit_guard_check = {
    'sentence_after_a_breve': sentence_after_a_breve,
    'sentence_after_deliberate_v': sentence_after_deliberate_v,
    'spoken_utterances': eg_spoken,
    'status': 'PASS' if eg_success else 'FAIL'
}
print(f"Exit Guard Part B (After deliberate V): Status={exit_guard_check['status']} | Final Sentence={sentence_after_deliberate_v} | TTS={eg_spoken}")

# ------------------------------------------------------------------------------
# 4. STATIC 22-LETTER REGRESSION (551 SAMPLES)
# ------------------------------------------------------------------------------
print("\n--- 4. RUNNING FULL STATIC 22-LETTER REGRESSION (551 SAMPLES) ---")

correct_count = 0
total_samples = len(lm_static)
false_diacritics_single = 0
false_diacritics_hold = 0

for i in range(total_samples):
    lm = lm_static[i].tolist()
    gt = cls_static[y_static[i]]
    
    # 1-frame classification via full pipeline
    session_id_single = f"static_single_{i}"
    rec.reset_diacritic_fsm(session_id_single)
    rec.reset()
    res = rec.predict_landmarks(lm, mode="letters", session_id=session_id_single, timestamp=1.0)
    pred = res.get('sign')
    if pred == gt:
        correct_count += 1
    if pred in ('Ă', 'Â', 'Ê', 'Ô', 'Ơ', 'Ư', 'Đ'):
        false_diacritics_single += 1

    # 10-frame static hold test
    rec.reset()
    hold_sid = f"hold_test_{i}"
    rec.reset_diacritic_fsm(hold_sid)
    hold_tokens = []
    for f in range(10):
        t_res = rec.predict_landmarks(lm, mode='letters', session_id=hold_sid, trial_id=hold_sid, timestamp=1.0 + f * 0.04)
        if t_res.get('new_token') in ('Ă', 'Â', 'Ê', 'Ô', 'Ơ', 'Ư', 'Đ'):
            false_diacritics_hold += 1
            break

accuracy = round(correct_count / total_samples * 100, 2)
print(f"Static 22-letter Accuracy: {correct_count}/{total_samples} ({accuracy}%) [Baseline: 454/551 (82.40%)]")
print(f"Single-frame False Diacritics: {false_diacritics_single}/{total_samples}")
print(f"10-Frame Hold False Diacritics: {false_diacritics_hold}/{total_samples}")

# ------------------------------------------------------------------------------
# 5. COMPILE AND SAVE DIAGNOSTICS REPORT
# ------------------------------------------------------------------------------
report = {
    'session_id': session_id,
    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'replay_7_diacritics': replay_results,
    'lifecycle_scenarios': scenarios,
    'exit_guard': exit_guard_check,
    'static_regression_551': {
        'total': total_samples,
        'correct': correct_count,
        'accuracy_pct': accuracy,
        'baseline_pct': 82.40,
        'false_diacritics_single_frame': false_diacritics_single,
        'false_diacritics_10_frame_hold': false_diacritics_hold,
        'status': 'PASS' if accuracy >= 82.40 and false_diacritics_hold == 0 else 'FAIL'
    }
}

output_path = f"backend/data/diagnostics/final_lifecycle_token_tts_{timestamp_str}.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(report, f, ensure_ascii=False, indent=2)

print(f"\nSaved final lifecycle diagnostics report to: {output_path}")
print("=" * 95)
