import json
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

c = VSLAlphabetClassifier()

for letter, diag_fn in [
    ("P", c.evaluate_rule_p_diagnostics),
    ("Q", c.evaluate_rule_q_diagnostics),
    ("R", c.evaluate_rule_r_diagnostics),
    ("X", c.evaluate_rule_x_diagnostics),
]:
    path = f"backend/data/alphabet_v2/static_landmarks/{letter}.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data["samples"]
    stable_samples = [s for s in samples if s.get("phase") == "stable_pose"]
    print(f"\n==================== {letter} DIAGNOSTICS ====================")
    print(f"Total samples: {len(samples)}, stable: {len(stable_samples)}")
    
    # Check stable samples
    fail_counts = {}
    preds = []
    for s in stable_samples:
        lms = s.get("landmarks_21") or s.get("landmarks")
        if not lms or len(lms) != 21:
            continue
        feat = c.extract_features(lms)
        pred, conf = c.classify(lms)
        preds.append(pred)
        diag = diag_fn(feat)
        if diag:
            for failed in diag.get("failed_conditions", []):
                fail_counts[failed] = fail_counts.get(failed, 0) + 1
    
    print(f"Predictions on stable: {dict((p, preds.count(p)) for p in set(preds))}")
    print("Failed conditions on stable samples:")
    for cond, count in sorted(fail_counts.items(), key=lambda x: -x[1]):
        print(f"  {cond}: {count}/{len(stable_samples)} ({count/len(stable_samples)*100:.1f}%)")
