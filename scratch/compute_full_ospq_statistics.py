import sys
sys.path.insert(0, '.')
import json
import numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

# Load stable hold frames from static_landmarks JSONs
data_sources = {
    "O": ("backend/data/alphabet_v2/static_landmarks/O.json", "D0506", lambda s: s.get("phase") == "stable_pose" and (s.get("frame_index") or s.get("frame_idx", 0)) in [29, 36, 43]),
    "S": ("backend/data/alphabet_v2/static_landmarks/S.json", "D0512", lambda s: s.get("phase") == "stable_pose" and 24 <= (s.get("frame_index") or s.get("frame_idx", 0)) <= 40),
    "P": ("backend/data/alphabet_v2/static_landmarks/P.json", "D0509B", lambda s: s.get("phase") == "stable_pose" and 26 <= (s.get("frame_index") or s.get("frame_idx", 0)) <= 46),
    "Q": ("backend/data/alphabet_v2/static_landmarks/Q.json", "D0510", lambda s: s.get("phase") == "stable_pose" and 32 <= (s.get("frame_index") or s.get("frame_idx", 0)) <= 56),
}

features_to_calc = [
    "d_TI", "d_TM", "d_TW", "d_IM", "d_IW", "d_MW", "d_RW", "d_PW",
    "thumb_lat", "ang_TI", "ang_IM_dir", "idx_tip_y", "mid_tip_y",
    "idx_horiz_ratio", "mid_horiz_ratio", "idx_straight", "mid_straight"
]

all_stats = {}

for letter, (json_path, vid_id, filter_fn) in data_sources.items():
    with open(json_path, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    samples = [s for s in raw["samples"] if s.get("landmarks") and filter_fn(s)]
    feats = [clf.extract_features(s["landmarks"]) for s in samples]
    
    letter_stat = {
        "video_id": vid_id,
        "sample_count": len(samples),
        "frame_indices": [s.get("frame_index") or s.get("frame_idx") for s in samples],
        "metrics": {}
    }
    
    for k in features_to_calc:
        vals = [f[k] for f in feats if k in f and f[k] is not None]
        if vals:
            letter_stat["metrics"][k] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "median": float(np.median(vals)),
                "p25": float(np.percentile(vals, 25)),
                "p75": float(np.percentile(vals, 75)),
            }
    all_stats[letter] = letter_stat

with open("scratch/ospq_statistics.json", "w", encoding="utf-8") as f:
    json.dump(all_stats, f, indent=2, ensure_ascii=False)

print("Saved scratch/ospq_statistics.json successfully.")

# Print summary table
print("\n" + "=" * 110)
print(f"{'Feature':<18} | {'O (D0506)':<20} | {'S (D0512)':<20} | {'P (D0509B)':<20} | {'Q (D0510)':<20}")
print("=" * 110)

for k in features_to_calc:
    row = f"{k:<18} | "
    for let in ["O", "S", "P", "Q"]:
        m = all_stats[let]["metrics"].get(k)
        if m:
            val_str = f"{m['mean']:.2f}±{m['std']:.2f} [{m['min']:.2f}-{m['max']:.2f}]"
        else:
            val_str = "N/A"
        row += f"{val_str:<20} | "
    print(row)

