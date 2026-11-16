import json

lines = [json.loads(l) for l in open('backend/data/diagnostics/pilot_eval_29_pilot_20261112_041651_86b712_20261112_041820_frames.jsonl', encoding='utf-8')]

print("=== H WEBCAM FRAMES DETAILS ===")
print(f"Total lines: {len(lines)}")

preds = {}
rejects = {}
head_preds = {}

for i, f in enumerate(lines):
    p = f.get('predicted_sign')
    r = f.get('reject_reason')
    hc = f.get('head_confidence')
    preds[p] = preds.get(p, 0) + 1
    if r:
        rejects[r] = rejects.get(r, 0) + 1

print("\nPreds:", preds)
print("\nTop reject reasons:", sorted(rejects.items(), key=lambda x: -x[1])[:10])

# Inspect Rep 1 (where 9/10 passed) vs Rep 2 vs Rep 3-10
reps = {}
cur_rep = 1
prev_t = -1
for f in lines:
    t = f.get('elapsed_sec', 0)
    if t < prev_t:
        cur_rep += 1
    if cur_rep not in reps:
        reps[cur_rep] = []
    reps[cur_rep].append(f)
    prev_t = t

for r, flist in reps.items():
    r_preds = {}
    for f in flist:
        p = f.get('predicted_sign')
        r_preds[p] = r_preds.get(p, 0) + 1
    print(f"Rep {r:02d} ({len(flist)} frames): {r_preds}, head_confs: {[round(f.get('head_confidence') or 0, 2) for f in flist[:4]]}")

