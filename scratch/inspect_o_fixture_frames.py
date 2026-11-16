import sys
sys.path.append(".")
import json
from backend.vision.alphabet_classifier import AlphabetClassifier

clf = AlphabetClassifier()
data = json.load(open("backend/data/alphabet_v2/static_landmarks/O.json", encoding="utf-8"))
samples = [s for s in data["samples"] if s.get("phase") == "stable_pose" and s.get("landmarks")]

print(f"Total valid stable frames in O.json: {len(samples)}")
for i, s in enumerate(samples):
    f = clf.extract_features(s["landmarks"])
    sf = s.get("source_frame", i)
    idx_s = f.get("idx_straight", 0.0)
    d_ti = f.get("d_TI", 0.0)
    match_o = (idx_s <= 0.48) and (d_ti <= 0.22)
    print(f"Sample {i:2d} (frame {sf:2d}): idx_straight = {idx_s:.4f} {'[FAIL]' if idx_s > 0.48 else '[PASS]'}, d_TI = {d_ti:.4f} {'[FAIL]' if d_ti > 0.22 else '[PASS]'} -> Match: {match_o}")
