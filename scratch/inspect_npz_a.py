import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from backend.vision.alphabet_classifier import AlphabetClassifier
from scratch.task2_4_candidate_validation import eval_x_match_custom, simulate_pipeline

classifier = AlphabetClassifier()
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
classes = data["classes"]
y = data["y"]
landmarks = data["landmarks"]

print("--- Inspecting 'A' False Positives in vsl_alphabet_static_v2.npz ---")
for idx in range(len(y)):
    true_label = classes[y[idx]]
    if true_label != "A":
        continue
    lm = landmarks[idx].tolist()
    feat = classifier.extract_features(lm)
    if not feat:
        continue
    is_x = eval_x_match_custom(feat, 0.25, 1.55, 1.30, 0.65)
    if is_x:
        print(f"\nNPZ sample {idx} (True={true_label}):")
        print(f"  d_TI={feat.get('d_TI'):.4f}, d_IW={feat.get('d_IW'):.4f}, d_TW={feat.get('d_TW'):.4f}, d_PW={feat.get('d_PW'):.4f}")
        print(f"  i_curl={feat.get('index_curl'):.4f}, idx_straight={feat.get('idx_straight'):.4f}, d_RW={feat.get('d_RW'):.4f}, tip_y={feat.get('idx_tip_y'):.4f}")
        print(f"  d_MW={feat.get('d_MW'):.4f}, m_curl={feat.get('middle_curl'):.4f}, mid_straight={feat.get('mid_straight'):.4f}")
        matches_orig = classifier.evaluate_rules(feat)
        print(f"  Original matches: {matches_orig}")
        sim = simulate_pipeline(classifier, lm, feat, 0.25, 1.55, 1.30, 0.65)
        print(f"  Simulate with Cand 2: status={sim['status']}, letter={sim['letter']}, matches={sim['matches']}")

