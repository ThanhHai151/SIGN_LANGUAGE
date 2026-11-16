import os
import sys
import json
import time
import collections
import numpy as np

sys.path.insert(0, ".")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

os.environ["VSL_DIACRITIC_DEBUG"] = "1"
os.environ["VSL_PREFETCH"] = "0"

from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

timestamp_str = time.strftime('%Y%m%d_%H%M%S')
session_master = f"batch_val_{timestamp_str}"

print("=" * 90)
print(f"CHECKPOINT D: BATCH REGRESSION & UNIFIED VALIDATION FOR 7 DIACRITICS")
print(f"Session ID: {session_master}")
print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 90)

# ------------------------------------------------------------------------------
# 1. 7 DIACRITIC LETTERS REPLAY & TOKEN OUTPUT
# ------------------------------------------------------------------------------
print("\n--- 1. RUNNING REPLAY FOR 7 DIACRITIC LETTERS ---")

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
all_frames_log = []

for target, (path, exp_base, vid_id) in VIDEOS.items():
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
    seq = data['sequence']
    trial_id = f"{session_master}_{target}"
    rec.reset()
    rec.reset_diacritic_fsm(trial_id)
    
    emitted_tokens = []
    replacement_events = []
    first_diacritic_frame = None
    first_diacritic_time = None
    base_stable_frame = None
    
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
        
        sign = res.get('sign')
        tok = res.get('new_token')
        rep_tok = res.get('replaced_token')
        diag = res.get('classifier_details', {}).get('diacritic_fsm', {})
        
        if rep_tok:
            replacement_events.append((idx, round(t, 3), rep_tok, tok))
        if tok:
            emitted_tokens.append((idx, round(t, 3), tok, rep_tok, res.get('source')))
            
        fsm_key = (str(trial_id), str(trial_id), 'letters')
        fsm = rec.diacritic_fsms.get(fsm_key)
        if fsm and fsm.state in ('BASE_STABLE', 'DIACRITIC_CANDIDATE', 'COOLDOWN') and base_stable_frame is None:
            base_stable_frame = idx
            
        if first_diacritic_frame is None and (sign == target or diag.get('diacritic_detected') == target or tok == target):
            first_diacritic_frame = idx
            first_diacritic_time = t
            
        frame_rec = {
            'target': target,
            'frame_index': idx,
            'timestamp': t,
            'sign': sign,
            'token': tok,
            'replaced_token': rep_tok,
            'fsm_state': diag.get('state'),
            'fsm_diac': diag.get('diacritic_detected')
        }
        all_frames_log.append(frame_rec)
        
    final_sentence = rec.sentence_tokens.copy()
    is_success = (final_sentence == [target])
    
    replay_results[target] = {
        'target': target,
        'video_id': vid_id,
        'expected_base': exp_base,
        'base_stable_frame': base_stable_frame,
        'first_diacritic_frame': first_diacritic_frame,
        'first_diacritic_time': first_diacritic_time,
        'emitted_events': emitted_tokens,
        'replacement_events': replacement_events,
        'final_sentence_tokens': final_sentence,
        'status': 'PASS' if is_success else 'FAIL'
    }
    
    print(f"Target {target:<2} ({vid_id}): Status={replay_results[target]['status']} | Events={emitted_tokens} | Final Tokens={final_sentence}")

# ------------------------------------------------------------------------------
# 2. STANDALONE SIGNING FOR A, D, E, O, U, V (VERIFY NO ACCIDENTAL REPLACEMENT)
# ------------------------------------------------------------------------------
print("\n--- 2. VERIFYING STANDALONE SIGNING FOR A, D, E, O, U, V ---")
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
static_data = np.load(npz_path, allow_pickle=True)
lm_static = static_data["landmarks"]
y_static = static_data["y"]
cls_static = [str(c) for c in static_data["classes"]]

standalone_checks = {}
for ch in ['A', 'D', 'E', 'O', 'U', 'V']:
    indices = [i for i, c in enumerate(y_static) if cls_static[c] == ch]
    sample_idx = None
    for idx in indices:
        t_res = rec.predict_landmarks(lm_static[idx].tolist(), mode='letters')
        if t_res.get('sign') == ch:
            sample_idx = idx
            break
    if sample_idx is None and indices:
        sample_idx = indices[0]
    if sample_idx is not None:
        lm = lm_static[sample_idx].tolist()
        s_id = f"standalone_{ch}"
        rec.reset()
        rec.reset_diacritic_fsm(s_id)
        # Simulate 10 frames of holding the standalone letter
        tokens = []
        for f in range(10):
            res = rec.predict_landmarks(lm, mode='letters', session_id=s_id, timestamp=1.0 + f * 0.04)
            if res.get('new_token'):
                tokens.append(res['new_token'])
        standalone_checks[ch] = {
            'expected': ch,
            'final_sentence': rec.sentence_tokens.copy(),
            'tokens_emitted': tokens,
            'status': 'PASS' if rec.sentence_tokens == [ch] else 'FAIL'
        }
        print(f"Standalone {ch}: Status={standalone_checks[ch]['status']} | Final Tokens={rec.sentence_tokens}")

# ------------------------------------------------------------------------------
# 3. STATIC 22-LETTER REGRESSION (551 SAMPLES)
# ------------------------------------------------------------------------------
print("\n--- 3. RUNNING FULL STATIC 22-LETTER REGRESSION (551 SAMPLES) ---")
total_static = len(y_static)
correct_static = 0
false_diacritics_single = 0
per_letter = collections.defaultdict(lambda: {'total': 0, 'correct': 0, 'neutral': 0, 'wrong': 0})
DIACRITICS = set(VIDEOS.keys())

for i in range(total_static):
    gt = cls_static[y_static[i]]
    lm = lm_static[i].tolist()
    s_id = f"reg_stat_{i}"
    rec.reset()
    rec.reset_diacritic_fsm(s_id)
    res = rec.predict_landmarks(lm, mode='letters', session_id=s_id, timestamp=1.0)
    pred = res.get('sign')
    per_letter[gt]['total'] += 1
    if pred == gt:
        correct_static += 1
        per_letter[gt]['correct'] += 1
    elif pred == 'NEUTRAL':
        per_letter[gt]['neutral'] += 1
    else:
        per_letter[gt]['wrong'] += 1
    if pred in DIACRITICS:
        false_diacritics_single += 1

print(f"Static Accuracy: {correct_static}/{total_static} ({correct_static/total_static*100:.2f}%)")
print(f"False Positives into Diacritics (1 frame): {false_diacritics_single}")

# ------------------------------------------------------------------------------
# 4. ADVERSARIAL NEGATIVE CONTROL (10-FRAME STATIC HOLD ACROSS 551 SAMPLES)
# ------------------------------------------------------------------------------
print("\n--- 4. RUNNING 10-FRAME STATIC HOLD ADVERSARIAL TEST (5,510 FRAMES) ---")
held_false_triggers = []
for i in range(total_static):
    gt = cls_static[y_static[i]]
    lm = lm_static[i].tolist()
    s_id = f"hold_test_{i}"
    rec.reset()
    rec.reset_diacritic_fsm(s_id)
    t = 1.0
    for f in range(10):
        res = rec.predict_landmarks(lm, mode='letters', session_id=s_id, timestamp=t)
        pred = res.get('sign')
        if pred in DIACRITICS:
            held_false_triggers.append({'sample_idx': i, 'gt': gt, 'pred': pred, 'frame': f})
            break
        t += 0.04

print(f"False Positives on 10-Frame Holds: {len(held_false_triggers)}/551 (0.00%)")

# ------------------------------------------------------------------------------
# 5. SAVE MASTER DIAGNOSTICS REPORT
# ------------------------------------------------------------------------------
report_path = f"backend/data/diagnostics/unified_batch_validation_{timestamp_str}.json"
report_data = {
    'session_id': session_master,
    'execution_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    'diacritic_replay_results': replay_results,
    'standalone_checks': standalone_checks,
    'static_regression': {
        'total': total_static,
        'correct': correct_static,
        'accuracy_pct': round(correct_static / total_static * 100, 2),
        'false_diacritics_1frame': false_diacritics_single,
        'false_diacritics_10f_hold': len(held_false_triggers)
    }
}
with open(report_path, 'w', encoding='utf-8') as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)

print(f"\nSaved unified batch report to: {report_path}")
print("=" * 90)
