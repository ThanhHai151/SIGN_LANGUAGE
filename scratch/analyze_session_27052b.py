# scratch/analyze_session_27052b.py
import json
import collections

jsonl_path = r'backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433_frames.jsonl'
json_path = r'backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433.json'

with open(json_path, 'r', encoding='utf-8') as f:
    eval_json = json.load(f)

with open(jsonl_path, 'r', encoding='utf-8') as f:
    frames = [json.loads(line) for line in f]

print("=== SESSION OVERVIEW ===")
print(f"Session ID: {eval_json.get('session_id')}")
print(f"Target in config: {eval_json.get('config', {}).get('tested_target')}")
print(f"Total frames: {len(frames)}")

# Check field presence in frames
sample = frames[0] if frames else {}
fields = list(sample.keys())
print("\nFields in frameRecord:")
for fld in fields:
    print(f"  - {fld}: {sample.get(fld)}")

# Group by trial / rep
# Since frames might not have explicit 'rep', let's check what identifies trials:
# timestamps, elapsed_sec, or frame_idx
print("\nTrial breakdown analysis:")
trials = collections.defaultdict(list)

# Notice how frames have elapsed_sec resetting or timestamps
# Let's inspect elapsed_sec and timestamp progression
curr_trial = 1
prev_elapsed = -1.0

for f in frames:
    e = f.get('elapsed_sec', 0.0)
    # If elapsed_sec drops significantly (e.g. from ~5.0 to < 1.0), it's a new trial
    if prev_elapsed > 2.0 and e < 1.5:
        curr_trial += 1
    f['trial_rep'] = curr_trial
    trials[curr_trial].append(f)
    prev_elapsed = e

print(f"Detected {len(trials)} trials based on elapsed_sec reset.")
for rep, t_frames in sorted(trials.items()):
    preds = collections.Counter(f.get('predicted_sign') for f in t_frames)
    hd = collections.Counter(f.get('hand_detected') for f in t_frames)
    phases = collections.Counter(f.get('phase') for f in t_frames)
    sources = collections.Counter(f.get('prediction_source') for f in t_frames)
    conf_tokens = [f for f in t_frames if f.get('confirmed')]
    stable_frames = [f for f in t_frames if f.get('phase') == 'STABLE_HOLD']
    stable_preds = collections.Counter(f.get('predicted_sign') for f in stable_frames)
    
    print(f"\n--- TRIAL REP {rep} (Total {len(t_frames)} frames) ---")
    print(f"  Predicted signs: {dict(preds)}")
    print(f"  Hand detected: {dict(hd)}")
    print(f"  Sources: {dict(sources)}")
    print(f"  Phases: {dict(phases)}")
    print(f"  Stable hold frames ({len(stable_frames)}): {dict(stable_preds)}")
    print(f"  Confirmed tokens ({len(conf_tokens)}): {[f.get('new_token') for f in conf_tokens]}")

