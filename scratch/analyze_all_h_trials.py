import json
from collections import Counter

file_path = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"

h_trials = []
with open(file_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, start=1):
        d = json.loads(line)
        if d.get("expected_sign") == "H":
            h_trials.append((line_idx, d))

print(f"Total H frames: {len(h_trials)}")

by_rep = Counter(d.get("trial_id") for _, d in h_trials)
for rep, count in sorted(by_rep.items()):
    frames = [d for _, d in h_trials if d.get("trial_id") == rep]
    sign_counts = Counter(d.get("final_sign") for d in frames)
    # Check failed conditions of h_webcam
    failed_conds = Counter()
    for d in frames:
        h_webcam = d.get("geometric", {}).get("h_rule_diagnostics", {}).get("h_webcam", {})
        if h_webcam.get("matched"):
            failed_conds["[MATCHED_H_WEBCAM]"] += 1
        else:
            for fc in h_webcam.get("failed_conditions", []):
                failed_conds[fc] += 1
    # Check avg features
    avg_ang = sum(d.get("geometric", {}).get("features", {}).get("ang_IM_dir", 0) for d in frames) / len(frames)
    avg_mid_h = sum(d.get("geometric", {}).get("features", {}).get("mid_horiz_ratio", 0) for d in frames) / len(frames)
    avg_idx_h = sum(d.get("geometric", {}).get("features", {}).get("idx_horiz_ratio", 0) for d in frames) / len(frames)
    print(f"\n{rep} ({count} frames):")
    print(f"  Signs: {dict(sign_counts)}")
    print(f"  Avg ang_IM_dir={avg_ang:.1f}°, mid_horiz={avg_mid_h:.2f}, idx_horiz={avg_idx_h:.2f}")
    print(f"  h_webcam conditions: {dict(failed_conds)}")

