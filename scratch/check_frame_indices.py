import json

for letter in ["P", "Q", "R", "X"]:
    path = f"backend/data/alphabet_v2/static_landmarks/{letter}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"\n==================== LETTER {letter} ====================")
    samples = [s for s in data["samples"] if s.get("phase") == "stable_pose" and s.get("quality_status") == "ok"]
    for i, s in enumerate(samples):
        print(f"Sample {i:02d}: frame_index={s['frame_index']}, ts={s.get('timestamp_sec', 0):.2f}s")

