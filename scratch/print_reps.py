import json

frames_path = "backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433_frames.jsonl"
with open(frames_path, "r", encoding="utf-8") as f:
    frames = [json.loads(line) for line in f]

reps = {}
rep = 1
for i, f in enumerate(frames):
    elapsed = f.get("elapsed_sec", 0.0)
    if i > 0 and elapsed < frames[i-1].get("elapsed_sec", 0.0):
        rep += 1
    f["rep"] = rep
    f["idx_in_rep"] = len(reps.get(rep, [])) + 1
    reps.setdefault(rep, []).append(f)

for r in [6, 7, 8, 9, 10]:
    print(f"=== REP {r} ===")
    for f in reps[r]:
        print(f"  Frame {f['frame_idx']} (idx {f['idx_in_rep']}) | t={f['elapsed_sec']:.2f}s | phase={f['phase']} | pred={f['predicted_sign']} ({f['confidence']}) | src={f['prediction_source']} | conf={f['confirmed']} | new_tok={f['new_token']}")

