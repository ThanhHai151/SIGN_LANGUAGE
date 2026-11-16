# scratch/run_task1_3d_analysis.py
import json
import collections

json_path = r'backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433.json'
jsonl_path = r'backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433_frames.jsonl'

with open(json_path, 'r', encoding='utf-8') as f:
    eval_json = json.load(f)

with open(jsonl_path, 'r', encoding='utf-8') as f:
    frames = [json.loads(line) for line in f]

print("=== PHASE A: DATA VALIDATION ===")
print("Session ID in JSON:", eval_json.get('session_id'))
print("Target configured:", eval_json.get('config', {}).get('tested_target'))
print("Total frames in JSONL:", len(frames))

# Check fields in all frames
field_names = [
    'hand_detected', 'prediction_source', 'gating_status', 'classifier_status',
    'reject_reason', 'head_confidence', 'confirmed', 'new_token', 'phase', 'elapsed_sec'
]

field_stats = {}
for fn in field_names:
    vals = [f.get(fn) for f in frames]
    null_count = sum(1 for v in vals if v is None)
    non_null_count = len(vals) - null_count
    distinct = set(vals)
    field_stats[fn] = {
        'non_null': non_null_count,
        'null': null_count,
        'sample_distinct': list(distinct)[:5]
    }

for fn, s in field_stats.items():
    print(f"Field '{fn}': non_null={s['non_null']}, null={s['null']}, samples={s['sample_distinct']}")

# Phase counts across all 98 frames
phase_counter = collections.Counter(f.get('phase') for f in frames)
print("\nPhase distribution across session:")
for ph, cnt in phase_counter.items():
    print(f"  - {ph}: {cnt} frames ({cnt/len(frames)*100:.1f}%)")

# Group frames into trials
# Trial boundaries: detected by elapsed_sec reset
curr_rep = 1
prev_elapsed = -1.0
trial_frames = collections.defaultdict(list)

for f in frames:
    e = f.get('elapsed_sec', 0.0)
    if prev_elapsed > 2.0 and e < 1.5:
        curr_rep += 1
    f['inferred_rep'] = curr_rep
    trial_frames[curr_rep].append(f)
    prev_elapsed = e

print(f"\nTotal trials detected: {len(trial_frames)}")

# Match against eval_json trials
p_trials_json = eval_json.get('letters', {}).get('P', {}).get('trials', [])
q_trials_json = eval_json.get('letters', {}).get('Q', {}).get('trials', [])

print(f"Trials recorded in JSON for P: {len(p_trials_json)}")
print(f"Trials recorded in JSON for Q: {len(q_trials_json)}")

print("\n=== PHASE B: TRIAL-LEVEL BREAKDOWN ===")
trial_summaries = []

for rep in range(1, len(trial_frames) + 1):
    t_f = trial_frames[rep]
    total_f = len(t_f)
    
    # Hand detection
    hd_true = sum(1 for f in t_f if f.get('hand_detected') is True)
    hd_false = sum(1 for f in t_f if f.get('hand_detected') is False)
    hd_null = sum(1 for f in t_f if f.get('hand_detected') is None)
    
    # Predictions
    q_correct = sum(1 for f in t_f if f.get('predicted_sign') == 'Q')
    p_frames = sum(1 for f in t_f if f.get('predicted_sign') == 'P')
    other_pred = sum(1 for f in t_f if f.get('predicted_sign') not in ('Q', 'P', 'NEUTRAL'))
    other_pred_letters = [f.get('predicted_sign') for f in t_f if f.get('predicted_sign') not in ('Q', 'P', 'NEUTRAL')]
    neutral_with_hand = sum(1 for f in t_f if f.get('predicted_sign') == 'NEUTRAL' and f.get('hand_detected') is True)
    neutral_no_hand = sum(1 for f in t_f if f.get('predicted_sign') == 'NEUTRAL' and f.get('hand_detected') is False)
    neutral_null_hand = sum(1 for f in t_f if f.get('predicted_sign') == 'NEUTRAL' and f.get('hand_detected') is None)
    
    # Stable Hold
    sh_frames = [f for f in t_f if f.get('phase') == 'STABLE_HOLD']
    sh_total = len(sh_frames)
    sh_q = sum(1 for f in sh_frames if f.get('predicted_sign') == 'Q')
    sh_p = sum(1 for f in sh_frames if f.get('predicted_sign') == 'P')
    sh_neutral = sum(1 for f in sh_frames if f.get('predicted_sign') == 'NEUTRAL')
    sh_q_acc = (sh_q / sh_total * 100) if sh_total > 0 else 0.0
    
    # Confirmed tokens
    q_tokens = [f.get('new_token') for f in t_f if f.get('confirmed') and f.get('new_token') == 'Q']
    other_tokens = [f.get('new_token') for f in t_f if f.get('confirmed') and f.get('new_token') != 'Q']
    
    # JSON recorded status (if any)
    json_status = p_trials_json[rep - 1].get('status') if rep <= len(p_trials_json) else 'N/A'
    json_match_pct = p_trials_json[rep - 1].get('matchPct') if rep <= len(p_trials_json) else 0.0
    
    summary = {
        'rep': rep,
        'total': total_f,
        'hd_true': hd_true,
        'hd_false': hd_false,
        'hd_null': hd_null,
        'q_frames': q_correct,
        'p_frames': p_frames,
        'other_frames': other_pred,
        'other_letters': other_pred_letters,
        'neutral_with_hand': neutral_with_hand,
        'neutral_no_hand': neutral_no_hand,
        'neutral_null_hand': neutral_null_hand,
        'sh_total': sh_total,
        'sh_q': sh_q,
        'sh_p': sh_p,
        'sh_neutral': sh_neutral,
        'sh_q_acc': sh_q_acc,
        'q_confirmed_tokens': len(q_tokens),
        'other_confirmed_tokens': len(other_tokens),
        'json_status': json_status,
        'json_match_pct': json_match_pct
    }
    trial_summaries.append(summary)
    print(f"Rep {rep}: total={total_f}, hd_true={hd_true}, hd_false={hd_false}, hd_null={hd_null} | Q={q_correct}, P={p_frames}, Other={other_pred} | Neutral(hd_true)={neutral_with_hand} | SH_Q={sh_q}/{sh_total} ({sh_q_acc:.1f}%) | Confirmed_Q={len(q_tokens)} | HUD status={json_status}")

print("\n=== PHASE C: ROOT CAUSE & SOURCE BREAKDOWN ===")
all_q_frames = [f for f in frames if f.get('predicted_sign') == 'Q']
print(f"Total Q frames: {len(all_q_frames)}")
q_by_source = collections.Counter(f.get('prediction_source') for f in all_q_frames)
print(f"Q frames by prediction_source: {dict(q_by_source)}")

# Confidences of static head Q
static_head_q_confs = [f.get('confidence') for f in all_q_frames if f.get('prediction_source') == 'static_head']
if static_head_q_confs:
    print(f"Static Head Q confidences (min, max, avg): {min(static_head_q_confs):.4f}, {max(static_head_q_confs):.4f}, {sum(static_head_q_confs)/len(static_head_q_confs):.4f}")
    confs_in_75_85 = sum(1 for c in static_head_q_confs if 0.75 <= c < 0.85)
    confs_ge_85 = sum(1 for c in static_head_q_confs if c >= 0.85)
    print(f"  - Confidences in [0.75, 0.85): {confs_in_75_85} ({confs_in_75_85/len(static_head_q_confs)*100:.1f}%) [Rescued by 0.75]")
    print(f"  - Confidences >= 0.85: {confs_ge_85} ({confs_ge_85/len(static_head_q_confs)*100:.1f}%) [Would pass with old 0.85]")

# Neutral frames with hand detected
neutral_hd_true = [f for f in frames if f.get('predicted_sign') == 'NEUTRAL' and f.get('hand_detected') is True]
print(f"\nTotal NEUTRAL frames with hand_detected=True: {len(neutral_hd_true)}")
reasons_neutral = collections.Counter(f.get('reject_reason') for f in neutral_hd_true)
for r, c in reasons_neutral.items():
    print(f"  Reason: '{r}': {c} frames")

# Misclassified frames
all_p_frames = [f for f in frames if f.get('predicted_sign') == 'P']
print(f"\nTotal P frames: {len(all_p_frames)}")
for pf in all_p_frames:
    print(f"  P frame: rep={pf.get('inferred_rep')}, phase={pf.get('phase')}, elapsed={pf.get('elapsed_sec')}, src={pf.get('prediction_source')}, conf={pf.get('confidence')}")

# Confirmed tokens
all_confirmed = [f for f in frames if f.get('confirmed')]
print(f"\nTotal confirmed events: {len(all_confirmed)}")
for cf in all_confirmed:
    print(f"  Confirmed: rep={cf.get('inferred_rep')}, token='{cf.get('new_token')}', sign={cf.get('predicted_sign')}, phase={cf.get('phase')}, elapsed={cf.get('elapsed_sec')}, src={cf.get('prediction_source')}")

