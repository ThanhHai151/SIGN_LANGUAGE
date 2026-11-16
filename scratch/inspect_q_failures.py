import json
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

c = VSLAlphabetClassifier()

with open("backend/data/alphabet_v2/static_landmarks/Q.json", "r", encoding="utf-8") as f:
    data = json.load(f)

stable = [s for s in data["samples"] if s.get("phase") == "stable_pose" and s.get("quality_status") == "ok"]

print(f"{'Idx':<4} {'Frame':<6} {'Pred':<8} {'d_IW':<7} {'d_TW':<7} {'d_TI':<7} {'idx_y':<7} {'idx_dy':<7} {'idx_h':<7} {'ang_TI':<8} {'ang_IM':<8} {'Failed Conds'}")
print("-" * 115)

for i, s in enumerate(stable):
    lms = s["landmarks"]
    feat = c.extract_features(lms)
    pred, _ = c.classify(lms)
    diag = c.evaluate_rule_q_diagnostics(feat)
    fails = diag.get("failed_conditions", [])
    print(f"{i:<4} {s['frame_index']:<6} {pred:<8} {feat['d_IW']:<7.2f} {feat['d_TW']:<7.2f} {feat['d_TI']:<7.2f} {feat['idx_tip_y']:<7.2f} {feat['idx_dy']:<7.2f} {feat['idx_horiz_ratio']:<7.2f} {feat['ang_TI']:<8.1f} {feat['ang_IM_dir']:<8.1f} {fails}")

