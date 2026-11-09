import json

file_path = "backend/data/diagnostics/pilot_pilot_20261108_0321.jsonl"

with open(file_path, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f]

print("--- Trial 1 (H_rep01_att01) ---")
for i in range(1, 11):
    d = lines[i]
    feats = d.get("geometric", {}).get("features", {})
    sign = d.get("final_sign")
    print(f"  Frame {d.get('frame_idx')}: sign={sign}, ang_IM={feats.get('ang_IM_dir'):.1f}, mid_h={feats.get('mid_horiz_ratio'):.2f}, idx_h={feats.get('idx_horiz_ratio'):.2f}, d_IW={feats.get('d_IW'):.2f}, d_MW={feats.get('d_MW'):.2f}")

print("\n--- Trial 2 (H_rep02_att01) ---")
for i in range(73, 81):
    d = lines[i]
    feats = d.get("geometric", {}).get("features", {})
    sign = d.get("final_sign")
    print(f"  Frame {d.get('frame_idx')}: sign={sign}, ang_IM={feats.get('ang_IM_dir'):.1f}, mid_h={feats.get('mid_horiz_ratio'):.2f}, idx_h={feats.get('idx_horiz_ratio'):.2f}, d_IW={feats.get('d_IW'):.2f}, d_MW={feats.get('d_MW'):.2f}")

