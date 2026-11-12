import json

frames_path = "backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433_frames.jsonl"
with open(frames_path, "r", encoding="utf-8") as f:
    frames = [json.loads(line) for line in f]

# Assign rep_idx based on elapsed_sec resets
rep = 1
rep_frames = []
reps = {}

for i, f in enumerate(frames):
    elapsed = f.get("elapsed_sec", 0.0)
    # If elapsed drops significantly from previous frame, it is a new rep
    if i > 0 and elapsed < frames[i-1].get("elapsed_sec", 0.0):
        rep += 1
    f["calc_rep"] = rep
    f["calc_idx_in_rep"] = len(reps.get(rep, [])) + 1
    reps.setdefault(rep, []).append(f)

print(f"Total reps detected: {len(reps)}")
for r, r_frames in reps.items():
    print(f"\n--- REP {r} (Frames {r_frames[0]['frame_idx']}..{r_frames[-1]['frame_idx']}, Total: {len(r_frames)}) ---")
    for f in r_frames:
        print(f"  Frame {f['frame_idx']:2d} (rep_idx {f['calc_idx_in_rep']:2d}) | t={f['elapsed_sec']:.2f}s | phase={f['phase']:<11} | hand={str(f['hand_detected']):<5} | pred={f['predicted_sign']:<7} | conf={f['confidence']} | src={str(f['prediction_source']):<14} | gate={str(f['gating_status']):<8} | head_conf={f['head_confidence']} | confirmed={f['confirmed']} | new_token={f['new_token']} | rej={f['reject_reason']}")

