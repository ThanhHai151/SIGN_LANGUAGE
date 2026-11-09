import json, cv2
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

with open("backend/data/alphabet_v2/static_landmarks/P.json", "r", encoding="utf-8") as f:
    p_data = json.load(f)

c = VSLAlphabetClassifier()
samples = p_data.get("samples", [])
print(f"Total samples in P.json: {len(samples)}")

for i, s in enumerate(samples):
    lms = s.get("landmarks_21") or s.get("landmarks")
    if not lms:
        continue
    feat = c.extract_features(lms)
    phase = s.get("phase")
    print(f"Sample {i:2d} (phase={phase}): d_IW={feat.get('d_IW'):.2f}, d_MW={feat.get('d_MW'):.2f}, d_TI={feat.get('d_TI'):.2f}, d_TM={feat.get('d_TM'):.2f}, idx_tip_y={feat.get('idx_tip_y'):.2f}, idx_dy={feat.get('idx_dy'):.2f}, idx_horiz={feat.get('idx_horiz_ratio'):.2f}, ang_IM={feat.get('ang_IM_dir'):.1f}")
