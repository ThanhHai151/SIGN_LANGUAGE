import json
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

c = VSLAlphabetClassifier()

with open("backend/data/alphabet_v2/static_landmarks/R.json", "r", encoding="utf-8") as f:
    data = json.load(f)

stable = [s for s in data["samples"] if s.get("phase") == "stable_pose" and s.get("quality_status") == "ok"]

print(f"{'Idx':<4} {'Frame':<6} {'Pred':<8} {'d_IW':<7} {'d_MW':<7} {'d_IM':<7} {'idx_y':<7} {'idx_dy':<7} {'idx_h':<7} {'cross':<8} {'d_rw':<7} {'d_pw':<7} {'Failed Conds'}")
print("-" * 125)

for i, s in enumerate(stable):
    lms = s["landmarks"]
    feat = c.extract_features(lms)
    pred, _ = c.classify(lms)
    diag = c.evaluate_rule_r_diagnostics(feat)
    fails = diag.get("failed_conditions", [])
    print(f"{i:<4} {s['frame_index']:<6} {pred:<8} {feat['d_IW']:<7.2f} {feat['d_MW']:<7.2f} {feat['d_IM']:<7.2f} {feat['idx_tip_y']:<7.2f} {feat['idx_dy']:<7.2f} {feat['idx_horiz_ratio']:<7.2f} {feat.get('cross_indicator',0):<8.4f} {feat['d_RW']:<7.2f} {feat['d_PW']:<7.2f} {fails}")
