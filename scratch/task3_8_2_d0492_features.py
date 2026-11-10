import json
import sys
sys.path.insert(0, ".")
from backend.vision.alphabet_classifier import AlphabetClassifier

clf = AlphabetClassifier()
b_json_path = "backend/data/alphabet_v2/static_landmarks/B.json"
with open(b_json_path, "r", encoding="utf-8") as f:
    b_data = json.load(f)

d0492_valid = [s for s in b_data.get("samples", []) if s.get("video_id") == "D0492" and s.get("landmarks")]
print(f"Total valid landmark frames in D0492: {len(d0492_valid)}")

for s in d0492_valid:
    fidx = s["frame_index"]
    ts = s["timestamp_sec"]
    lm = s["landmarks"]
    f = clf.extract_features(lm)
    matches = clf.evaluate_rules(f)
    match_letters = [m[0] for m in matches]
    pred, conf = clf.classify(lm)
    idx_s = f["idx_straight"]
    d_ti = f["d_TI"]
    d_tm = f["d_TM"]
    d_iw = f["d_IW"]
    d_mw = f["d_MW"]
    d_rw = f["d_RW"]
    d_pw = f["d_PW"]
    d_tw = f["d_TW"]
    print(f"fidx={fidx:02d} | ts={ts:.3f}s | idx_s={idx_s:.4f} | d_ti={d_ti:.4f} | d_tm={d_tm:.4f} | d_iw={d_iw:.2f} | d_mw={d_mw:.2f} | matches={match_letters} | pred={pred} ({conf:.2f})")

