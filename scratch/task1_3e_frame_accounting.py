# scratch/task1_3e_frame_accounting.py
import json
import collections

json_path = r'backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433.json'
jsonl_path = r'backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433_frames.jsonl'

with open(json_path, 'r', encoding='utf-8') as f:
    eval_json = json.load(f)

with open(jsonl_path, 'r', encoding='utf-8') as f:
    frames = [json.loads(line) for line in f]

total_frames = len(frames)
print(f"Total frames read from JSONL: {total_frames}")

# 1. Prediction labels
preds = collections.Counter(f.get('predicted_sign') for f in frames)
print(f"Prediction distribution: {dict(preds)}")
print(f"Sum of predictions: {sum(preds.values())}")
assert sum(preds.values()) == total_frames

# 2. Hand detected
hd = collections.Counter(f.get('hand_detected') for f in frames)
print(f"hand_detected distribution: {dict(hd)}")
print(f"Sum of hand_detected: {sum(hd.values())}")
assert sum(hd.values()) == total_frames

# 3. Phase
phases = collections.Counter(f.get('phase') for f in frames)
print(f"Phase distribution: {dict(phases)}")
print(f"Sum of phases: {sum(phases.values())}")
assert sum(phases.values()) == total_frames

# 4. Trials / Reps grouping
# How are reps determined? Let's check if frameRecord has rep or frame_idx, or elapsed_sec
curr_rep = 1
prev_elapsed = -1.0
rep_counts = collections.defaultdict(int)
rep_frames = collections.defaultdict(list)

for idx, f in enumerate(frames):
    e = f.get('elapsed_sec', 0.0)
    if prev_elapsed > 2.0 and e < 1.5:
        curr_rep += 1
    f['rep'] = curr_rep
    rep_counts[curr_rep] += 1
    rep_frames[curr_rep].append(f)
    prev_elapsed = e

print(f"Rep distribution: {dict(rep_counts)}")
print(f"Sum of rep frames: {sum(rep_counts.values())}")
assert sum(rep_counts.values()) == total_frames

# Detailed breakdown per rep:
print("\n--- DETAILED BREAKDOWN PER REP ---")
for r in range(1, curr_rep + 1):
    rf = rep_frames[r]
    r_preds = collections.Counter(f.get('predicted_sign') for f in rf)
    r_hd = collections.Counter(f.get('hand_detected') for f in rf)
    r_ph = collections.Counter(f.get('phase') for f in rf)
    r_src = collections.Counter(f.get('prediction_source') for f in rf)
    r_conf_tokens = [f.get('new_token') for f in rf if f.get('confirmed')]
    print(f"Rep {r:2d} ({len(rf):2d} frames):")
    print(f"   Predictions: {dict(r_preds)}")
    print(f"   Hand detected: True={r_hd.get(True, 0)}, False={r_hd.get(False, 0)}, Null={r_hd.get(None, 0)}")
    print(f"   Phases: {dict(r_ph)}")
    print(f"   Sources: {dict(r_src)}")
    print(f"   Confirmed tokens ({len(r_conf_tokens)}): {r_conf_tokens}")

# Check JSON letter P trials
p_trials_in_json = eval_json.get('letters', {}).get('P', {}).get('trials', [])
print(f"\nTrials recorded in JSON: {len(p_trials_in_json)}")
for t in p_trials_in_json:
    print(f"   JSON Rep {t.get('rep')}: status={t.get('status')}, matchPct={t.get('matchPct')}, matchFrames={t.get('matchFrames')}, mismatchFrames={t.get('mismatchFrames')}, confusions={t.get('confusions')}")

