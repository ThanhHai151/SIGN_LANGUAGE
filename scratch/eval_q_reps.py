import json

with open('backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433_frames.jsonl', 'r') as f:
    frames = [json.loads(line) for line in f]

reps = {}
rep = 1
for i, f in enumerate(frames):
    elapsed = f.get('elapsed_sec', 0.0)
    if i > 0 and elapsed < frames[i-1].get('elapsed_sec', 0.0):
        rep += 1
    f['rep'] = rep
    reps.setdefault(rep, []).append(f)

# Evaluate Reps 6-10 with target Q
q_frames = []
for r in range(6, 11):
    q_frames.extend(reps[r])

total_q = len(q_frames)
q_pred = sum(1 for f in q_frames if f['predicted_sign'] == 'Q')
p_pred = sum(1 for f in q_frames if f['predicted_sign'] == 'P')
neutral_pred = sum(1 for f in q_frames if f['predicted_sign'] == 'NEUTRAL')
src_head = sum(1 for f in q_frames if f['prediction_source'] == 'static_head' and f['predicted_sign'] == 'Q')
src_geom = sum(1 for f in q_frames if f['prediction_source'] == 'geometric_rule' and f['predicted_sign'] == 'Q')

stable_q = [f for f in q_frames if f['phase'] == 'STABLE_HOLD']
stable_q_pred = sum(1 for f in stable_q if f['predicted_sign'] == 'Q')

print(f"Reps 6-10 Total Frames: {total_q}")
print(f"Q predictions: {q_pred} / {total_q} ({q_pred/total_q*100:.1f}%)")
print(f"  - from static_head: {src_head}")
print(f"  - from geometric_rule: {src_geom}")
print(f"P confusions: {p_pred}")
print(f"NEUTRAL: {neutral_pred}")
print(f"STABLE_HOLD frames: {len(stable_q)}")
print(f"STABLE_HOLD Q predictions: {stable_q_pred} / {len(stable_q)} ({stable_q_pred/len(stable_q)*100:.1f}%)")

