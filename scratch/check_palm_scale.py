import json
import numpy as np

# Webcam
recs = [json.loads(line) for line in open("backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl", encoding="utf-8") if json.loads(line).get("expected_sign") == "O"]
webcam_scales = [r["geometric"]["features"]["palm_scale"] for r in recs]

# Fixture O
import sys
sys.path.append(".")
from backend.vision.alphabet_classifier import AlphabetClassifier
clf = AlphabetClassifier()
odata = json.load(open("backend/data/alphabet_v2/static_landmarks/O.json", encoding="utf-8"))
fixture_scales = [clf.extract_features(s["landmarks"])["palm_scale"] for s in odata["samples"] if s.get("phase") == "stable_pose" and s.get("landmarks")]

print(f"Fixture palm_scale: mean={np.mean(fixture_scales):.4f}, median={np.median(fixture_scales):.4f}, min={np.min(fixture_scales):.4f}, max={np.max(fixture_scales):.4f}")
print(f"Webcam palm_scale: mean={np.mean(webcam_scales):.4f}, median={np.median(webcam_scales):.4f}, min={np.min(webcam_scales):.4f}, max={np.max(webcam_scales):.4f}")
