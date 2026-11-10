import json

with open('backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl', 'r', encoding='utf-8') as f:
    records = [json.loads(line) for line in f if json.loads(line).get('trial_id') and json.loads(line).get('expected_sign') == 'H']

# Group by trial
trials = {}
for r in records:
    tid = r['trial_id']
    if tid not in trials:
        trials[tid] = []
    trials[tid].append(r)

print("=== TIMELINE OF REPS FOR LETTER H ===")
for tid, frames in trials.items():
    signs = [f.get('final_sign') for f in frames]
    # run length encoding of signs
    rle = []
    curr_s = signs[0]
    curr_cnt = 0
    for s in signs:
        if s == curr_s:
            curr_cnt += 1
        else:
            rle.append(f"{curr_s}:{curr_cnt}")
            curr_s = s
            curr_cnt = 1
    rle.append(f"{curr_s}:{curr_cnt}")
    print(f"{tid} ({len(frames)} frames): {' -> '.join(rle)}")

