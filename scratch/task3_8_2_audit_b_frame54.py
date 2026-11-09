import json
import numpy as np

# 1. Inspect B.json
b_json_path = "backend/data/alphabet_v2/static_landmarks/B.json"
with open(b_json_path, "r", encoding="utf-8") as f:
    b_data = json.load(f)

b_samples = b_data.get("samples", [])
print(f"Total samples in B.json: {len(b_samples)}")

# Check video_id and frame_index
d0492_samples = [s for s in b_samples if s.get("video_id") == "D0492"]
print(f"Total D0492 samples in B.json: {len(d0492_samples)}")

frame54_json = [s for s in d0492_samples if s.get("frame_index") == 54]
print(f"D0492 frame 54 in B.json: {len(frame54_json)}")
if frame54_json:
    s54 = frame54_json[0]
    print("Frame 54 in B.json metadata:")
    for k, v in s54.items():
        if k != "landmarks":
            print(f"  {k}: {v}")

# 2. Inspect NPZ
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
npz = np.load(npz_path, allow_pickle=True)
classes = list(npz["classes"])
y = npz["y"]
meta = json.loads(str(npz["metadata_json"]))

print(f"\nTotal NPZ samples: {len(y)}")
print(f"Metadata samples in NPZ: {len(meta)}")

# Find D0492 frame 54 in NPZ
npz_d0492_54 = []
for i, m in enumerate(meta):
    if m.get("video_id") == "D0492" and m.get("frame_index") == 54:
        npz_d0492_54.append((i, m))

print(f"D0492 frame 54 in NPZ: {len(npz_d0492_54)} matches")
for idx, m in npz_d0492_54:
    print(f"  NPZ idx={idx}, label={classes[y[idx]]}, metadata={m}")

# Print all frames of D0492 in B.json to see the timeline
print("\n--- TIMELINE OF D0492 IN B.JSON ---")
for s in d0492_samples:
    f_idx = s.get("frame_index")
    ts = s.get("timestamp_sec")
    phase = s.get("phase")
    has_lm = s.get("landmarks") is not None
    print(f"frame={f_idx:02d} | ts={ts:.3f}s | phase={phase:<12} | has_lm={has_lm}")

