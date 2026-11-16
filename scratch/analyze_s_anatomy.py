import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
import pandas as pd
from backend.vision.alphabet_classifier import AlphabetClassifier

clf = AlphabetClassifier()

# Load dataset
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
npz = np.load(npz_path, allow_pickle=True)
classes = npz["classes"]
y = npz["y"]
lms = npz["landmarks"]
meta = json.loads(str(npz["metadata_json"]))

s_class_idx = int(np.where(classes == "S")[0][0])
s_mask = (y == s_class_idx)
s_lms = lms[s_mask]
s_meta = [m for m in meta if m.get("label") == "S"]

print(f"Total S samples in npz: {len(s_lms)}")

# Extract features for all S samples
s_features = []
for idx in range(len(s_lms)):
    lm = s_lms[idx].tolist()
    feat = clf.extract_features(lm)
    fidx = s_meta[idx].get("frame_index")
    is_rep = s_meta[idx].get("is_representative")
    feat["frame_index"] = fidx
    feat["is_representative"] = is_rep
    s_features.append(feat)

# Feature keys of interest
keys = [
    "d_TW", "d_IW", "d_MW", "d_RW", "d_PW",
    "d_TI", "d_TM", "d_TR", "d_TP", "d_IM",
    "thumb_lat", "ang_ti",
    "index_curl", "middle_curl", "ring_curl", "pinky_curl",
    "i_ext", "m_ext", "r_ext", "p_ext",
    "idx_straight", "mid_straight",
    "idx_tip_y", "mid_tip_y",
    "mcp_width_2d", "thumb_ring_dist"
]

print("\n=== PER-SAMPLE S METRICS ===")
header = f"{'Frame':6s} {'Rep':5s} | {'d_TW':5s} {'d_IW':5s} {'d_MW':5s} {'d_RW':5s} {'d_PW':5s} | {'d_TI':5s} {'d_TM':5s} {'th_lat':6s} | {'i_ext':5s} {'m_ext':5s} {'r_ext':5s} {'p_ext':5s}"
print(header)
print("-" * len(header))
for f in s_features:
    fidx = f["frame_index"]
    rep = str(f["is_representative"])
    dtw = f.get("d_TW", 0)
    diw = f.get("d_IW", 0)
    dmw = f.get("d_MW", 0)
    drw = f.get("d_RW", 0)
    dpw = f.get("d_PW", 0)
    dti = f.get("d_TI", 0)
    dtm = f.get("d_TM", 0)
    th_lat = f.get("thumb_lat", 0)
    ie = str(f.get("i_ext", False))[0]
    me = str(f.get("m_ext", False))[0]
    re_ = str(f.get("r_ext", False))[0]
    pe = str(f.get("p_ext", False))[0]
    print(f"{fidx:<6d} {rep:<5s} | {dtw:5.2f} {diw:5.2f} {dmw:5.2f} {drw:5.2f} {dpw:5.2f} | {dti:5.2f} {dtm:5.2f} {th_lat:6.2f} | {ie:<5s} {me:<5s} {re_:<5s} {pe:<5s}")

# Statistical summary (min, mean, std, max)
print("\n=== STATISTICAL DISTRIBUTION OF LETTER S (15 samples) ===")
stats = {}
for k in keys:
    vals = [float(f[k]) for f in s_features if k in f and not isinstance(f[k], bool)]
    if vals:
        stats[k] = {
            "min": np.min(vals),
            "p25": np.percentile(vals, 25),
            "median": np.median(vals),
            "mean": np.mean(vals),
            "p75": np.percentile(vals, 75),
            "max": np.max(vals),
            "std": np.std(vals)
        }
        print(f"{k:<15s}: min={stats[k]['min']:6.3f}, 25%={stats[k]['p25']:6.3f}, median={stats[k]['median']:6.3f}, mean={stats[k]['mean']:6.3f}, max={stats[k]['max']:6.3f}, std={stats[k]['std']:6.3f}")

# Save stats to json for reporting
with open("scratch/s_statistics.json", "w", encoding="utf-8") as f:
    json.dump(stats, f, indent=2)
print("\nSaved s_statistics.json")
