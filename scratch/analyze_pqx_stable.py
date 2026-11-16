"""Analyze stable_pose features for P, Q, X from JSON landmark files."""
import json
import sys
sys.path.insert(0, "d:/SIGN_LANGUAGE")
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

c = VSLAlphabetClassifier()

for letter in ["P", "Q", "X"]:
    path = f"backend/data/alphabet_v2/static_landmarks/{letter}.json"
    with open(path) as f:
        data = json.load(f)
    
    print(f"\n{'='*60}")
    print(f"LETTER {letter}")
    print(f"{'='*60}")
    
    stable_samples = [s for s in data["samples"] if s.get("phase") == "stable_pose"]
    print(f"Total samples: {len(data['samples'])}, Stable: {len(stable_samples)}")
    
    features_list = []
    for i, s in enumerate(stable_samples):
        lms = s.get("landmarks_21") or s.get("landmarks")
        if lms:
            feat = c.extract_features(lms)
            features_list.append(feat)
    
    if not features_list:
        print("No features extracted!")
        continue
    
    # Print key features for each sample
    print(f"\n{'Sample':<8} {'d_IW':<8} {'d_MW':<8} {'d_TI':<8} {'d_TM':<8} {'idx_tip_y':<12} {'idx_horiz':<12} {'idx_curl':<10} {'idx_str':<10} {'ang_IM':<10} {'thumb_lat':<12} {'mid_tip_y':<12} {'pass?'}")
    print("-"*160)
    
    for i, feat in enumerate(features_list):
        pred, conf = c.classify(stable_samples[i].get("landmarks_21") or stable_samples[i].get("landmarks"))
        passed = "✓" if pred == letter else f"✗({pred})"
        
        print(
            f"{i:<8} "
            f"{feat.get('d_IW',0):<8.2f} "
            f"{feat.get('d_MW',0):<8.2f} "
            f"{feat.get('d_TI',0):<8.2f} "
            f"{feat.get('d_TM',0):<8.2f} "
            f"{feat.get('idx_tip_y',0):<12.2f} "
            f"{feat.get('idx_horiz_ratio',0):<12.2f} "
            f"{feat.get('idx_curl',0):<10.2f} "
            f"{feat.get('idx_straight',0):<10.2f} "
            f"{feat.get('ang_IM_dir',0):<10.1f} "
            f"{feat.get('thumb_lat',0):<12.2f} "
            f"{feat.get('mid_tip_y',0):<12.2f} "
            f"{passed}"
        )
    
    # Summary stats
    import statistics
    for key in ["d_IW", "d_MW", "d_TI", "d_TM", "idx_tip_y", "idx_horiz_ratio", "idx_curl", "idx_straight", "ang_IM_dir", "thumb_lat", "mid_tip_y"]:
        vals = [f.get(key, 0) for f in features_list]
        print(f"  {key}: min={min(vals):.3f}, max={max(vals):.3f}, mean={statistics.mean(vals):.3f}")
    
    # Classify all and summarize
    correct = sum(1 for s in stable_samples 
                  if s.get("landmarks_21") or s.get("landmarks"))
    preds = []
    for s in stable_samples:
        lms = s.get("landmarks_21") or s.get("landmarks")
        if lms:
            pred, _ = c.classify(lms)
            preds.append(pred)
    
    correct = sum(1 for p in preds if p == letter)
    print(f"\nClassification: {correct}/{len(preds)} correct ({100*correct/max(len(preds),1):.1f}%)")
    from collections import Counter
    print(f"Prediction distribution: {dict(Counter(preds))}")

