import json
import sys
sys.path.insert(0, ".")

from backend.vision.alphabet_classifier import alphabet_classifier, extract_dual_hand_features

print("=== P (D0509N) FEATURES & DIAGNOSTICS ===")
with open("scratch/d0509n_extracted_raw.json", encoding="utf-8") as f:
    samples_p = json.load(f)

for s in samples_p:
    lm = s.get("landmarks")
    if not lm or len(lm) < 21: continue
    fidx = s["frame_index"]
    feat = extract_dual_hand_features(lm)
    diag = alphabet_classifier.evaluate_rule_p_diagnostics(feat)
    res = alphabet_classifier.classify_detailed(lm)
    if 20 <= fidx <= 35:
        print(f"P f{fidx}: geom_status={res['status']} letter={res['letter']} | matched={diag['matched']} failed={diag['failed_conditions']}")
        print(f"   d_IW={feat.get('d_IW'):.2f}, d_MW={feat.get('d_MW'):.2f}, d_TI={feat.get('d_TI'):.2f}, d_TM={feat.get('d_TM'):.2f}, d_TW={feat.get('d_TW'):.2f}, ang_IM={feat.get('ang_IM_dir'):.1f}, idx_dy={feat.get('idx_dy'):.2f}, idx_horiz={feat.get('idx_horiz_ratio'):.2f}")

print("\n=== X (D0517N) FEATURES & DIAGNOSTICS ===")
with open("scratch/d0517n_extracted_raw.json", encoding="utf-8") as f:
    samples_x = json.load(f)

for s in samples_x:
    lm = s.get("landmarks")
    if not lm or len(lm) < 21: continue
    fidx = s["frame_index"]
    feat = extract_dual_hand_features(lm)
    diag = alphabet_classifier.evaluate_rule_x_diagnostics(feat)
    res = alphabet_classifier.classify_detailed(lm)
    if 30 <= fidx <= 65:
        print(f"X f{fidx}: geom_status={res['status']} letter={res['letter']} | matched={diag['matched']} failed={diag['failed_conditions']}")

