import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

fixture_path = "backend/tests/fixtures/real_alphabet_landmarks.json"
with open(fixture_path, "r", encoding="utf-8") as f:
    data = json.load(f)

samples = [s for s in data["samples"] if s.get("letter_upper") == "R"]
print(f"Total R samples in fixtures: {len(samples)}")

c = VSLAlphabetClassifier()
for s in samples:
    frame_idx = s.get("frame_idx")
    lms = s.get("landmarks") or s.get("landmarks_21")
    if lms is None:
        print(f"Frame {frame_idx}: lms is None")
        continue
    feat = c.extract_features(lms)
    diag = c.evaluate_rule_r_diagnostics(feat)
    res = c.classify_detailed(lms)
    print(f"Frame {frame_idx}: matched={diag.get('matched')}, failed={diag.get('failed_conditions')}, letter={res.get('letter')}, idx_horiz={feat.get('idx_horiz_ratio')}, idx_dy={feat.get('idx_dy')}")
