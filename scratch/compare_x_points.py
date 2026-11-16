import json

with open("backend/data/alphabet_v2/static_landmarks/X.json", "r", encoding="utf-8") as f:
    data = json.load(f)

for fidx in [40, 116]:
    sample = [s for s in data["samples"] if s["frame_index"] == fidx][0]
    lms = sample["landmarks"]
    print(f"\n--- FRAME {fidx} ---")
    for name, idx in [("wrist", 0), ("thumb_tip", 4), ("idx_mcp", 5), ("idx_pip", 6), ("idx_dip", 7), ("idx_tip", 8), ("mid_tip", 12), ("ring_tip", 16), ("pinky_tip", 20)]:
        print(f"  {name:10}: x={lms[idx][0]:.3f}, y={lms[idx][1]:.3f}, z={lms[idx][2]:.3f}")

