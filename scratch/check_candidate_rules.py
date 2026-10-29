import json
import numpy as np

with open("backend/reports/task_3b0_feature_analysis.json", "r", encoding="utf-8") as f:
    data = json.load(f)

features = data["features_by_letter"]

letters_to_check = ["s", "t", "c", "o", "d", "l", "a", "e", "b", "i", "y", "v"]

print(f"{'Letter':<6} | {'Samples':<7} | {'Key Features (mean ± std [min, max])'}")
print("-" * 75)

for l in letters_to_check:
    feats = features.get(l, [])
    if not feats:
        print(f"{l:<6} | 0")
        continue
    d_ti = [x["d_TI"] for x in feats]
    d_tm = [x["d_TM"] for x in feats]
    d_tw = [x["d_TW"] for x in feats]
    d_im = [x["d_IM"] for x in feats]
    c_idx = [x["curl_index"] for x in feats]
    c_mid = [x["curl_middle"] for x in feats]
    c_pky = [x["curl_pinky"] for x in feats]
    
    print(f"[{l.upper()}] ({len(feats)} mẫu):")
    print(f"  d_TI:       {np.mean(d_ti):.3f} ± {np.std(d_ti):.3f} [{np.min(d_ti):.2f}, {np.max(d_ti):.2f}]")
    print(f"  d_TM:       {np.mean(d_tm):.3f} ± {np.std(d_tm):.3f} [{np.min(d_tm):.2f}, {np.max(d_tm):.2f}]")
    print(f"  d_TW:       {np.mean(d_tw):.3f} ± {np.std(d_tw):.3f} [{np.min(d_tw):.2f}, {np.max(d_tw):.2f}]")
    print(f"  d_IM:       {np.mean(d_im):.3f} ± {np.std(d_im):.3f} [{np.min(d_im):.2f}, {np.max(d_im):.2f}]")
    print(f"  curl_index: {np.mean(c_idx):.3f} [{np.min(c_idx):.2f}, {np.max(c_idx):.2f}]")
    print(f"  curl_middle:{np.mean(c_mid):.3f} [{np.min(c_mid):.2f}, {np.max(c_mid):.2f}]")
    print(f"  curl_pinky: {np.mean(c_pky):.3f} [{np.min(c_pky):.2f}, {np.max(c_pky):.2f}]")
    print()

