import json

file_path = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"

idle_h_frames = []
with open(file_path, "r", encoding="utf-8") as f:
    for line_idx, line in enumerate(f, start=1):
        d = json.loads(line)
        if d.get("session_id") is None and d.get("final_sign") == "H":
            idle_h_frames.append((line_idx, d))

print(f"Total idle frames predicting H: {len(idle_h_frames)}")
for line_idx, d in idle_h_frames[:10]:
    geom = d.get("geometric", {})
    reason = geom.get("reason")
    feats = geom.get("features", {})
    print(f"  Line {line_idx}: reason={reason}")
    print(f"    ang_IM={feats.get('ang_IM_dir')}, mid_h={feats.get('mid_horiz_ratio')}, idx_h={feats.get('idx_horiz_ratio')}")

