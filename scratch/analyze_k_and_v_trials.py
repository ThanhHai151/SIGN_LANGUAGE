import json
from collections import Counter

file_path = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"

lines = []
with open(file_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, start=1):
        lines.append((line_idx, json.loads(line)))

for target in ["K", "V"]:
    target_frames = [item for item in lines if item[1].get("expected_sign") == target]
    print("=" * 60)
    print(f"ANALYSIS OF TARGET: {target} (Total frames: {len(target_frames)})")
    print("=" * 60)
    by_rep = Counter(d.get("trial_id") for _, d in target_frames)
    for rep, count in sorted(by_rep.items()):
        frames = [d for _, d in target_frames if d.get("trial_id") == rep]
        sign_counts = Counter(d.get("final_sign") for d in frames)
        src_counts = Counter(d.get("prediction_source") for d in frames)
        reasons = Counter(d.get("geometric", {}).get("reason") for d in frames)
        avg_conf = sum(d.get("final_confidence", 0) for d in frames) / len(frames)
        print(f"\n{rep} ({count} frames):")
        print(f"  Signs: {dict(sign_counts)}")
        print(f"  Sources: {dict(src_counts)}")
        print(f"  Avg Confidence: {avg_conf:.2f}")
        for r, c in reasons.items():
            print(f"    Reason ({c}): {r}")

    all_signs = Counter(d.get("final_sign") for _, d in target_frames)
    all_sources = Counter(d.get("prediction_source") for _, d in target_frames)
    print(f"\nOVERALL {target}:")
    print(f"  Total frames: {len(target_frames)}")
    print(f"  Signs distribution: {dict(all_signs)}")
    print(f"  Sources distribution: {dict(all_sources)}")
    print(f"  Accuracy: {all_signs[target] / len(target_frames) * 100:.2f}%")

