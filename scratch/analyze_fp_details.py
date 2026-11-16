import sys
sys.path.insert(0, '.')
import numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier
from scratch.test_ospq_candidate_rules import eval_candidate_o, eval_candidate_q

clf = VSLAlphabetClassifier()
data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(data['classes'])
landmarks = data['landmarks']
y = data['y']

print("=== INSPECTING FP FOR O (Classes B and A) ===")
for i in range(len(y)):
    true_c = classes[y[i]]
    if true_c in ['B', 'A']:
        lm = landmarks[i].tolist()
        feat = clf.extract_features(lm)
        if eval_candidate_o(feat):
            print(f"Sample {i} (True={true_c}): d_TI={feat['d_TI']:.2f}, d_TM={feat['d_TM']:.2f}, d_IW={feat['d_IW']:.2f}, d_MW={feat['d_MW']:.2f}, d_RW={feat['d_RW']:.2f}, d_PW={feat['d_PW']:.2f}, pinky_curl={feat['pinky_curl']:.2f}")

print("\n=== INSPECTING FP FOR Q (Classes G, X, N) ===")
for i in range(len(y)):
    true_c = classes[y[i]]
    if true_c in ['G', 'X', 'N']:
        lm = landmarks[i].tolist()
        feat = clf.extract_features(lm)
        if eval_candidate_q(feat):
            print(f"Sample {i} (True={true_c}): "
                  f"IW={feat['d_IW']:.2f}, MW={feat['d_MW']:.2f}, TW={feat['d_TW']:.2f}, TI={feat['d_TI']:.2f}, "
                  f"idx_tip_y={feat['idx_tip_y']:.2f}, idx_horiz={feat['idx_horiz_ratio']:.2f}, idx_dx={feat['idx_dx']:.2f}, "
                  f"idx_straight={feat['idx_straight']:.2f}, mid_straight={feat['mid_straight']:.2f}, "
                  f"mid_curl={feat['middle_curl']:.2f}, ring_curl={feat['ring_curl']:.2f}")

