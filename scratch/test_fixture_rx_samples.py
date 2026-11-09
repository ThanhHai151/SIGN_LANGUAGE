import json
import sys
import numpy as np

sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for target in ['R', 'X']:
    samples = [s for s in d['samples'] if s.get('letter_upper') == target and s.get('landmarks') is not None]
    print(f"=== Fixture {target} ({len(samples)} valid samples) ===")
    for i, s in enumerate(samples):
        lm = s['landmarks']
        feat = clf.extract_features(lm)
        sh0 = lm[0]
        sh9 = lm[9]
        palm_scale = np.linalg.norm(np.array(sh9) - np.array(sh0))
        base_dx = (lm[9][0] - lm[5][0]) / palm_scale
        tip_dx = (lm[12][0] - lm[8][0]) / palm_scale
        cross = base_dx * tip_dx
        print(f"  sample {i} (frame {s.get('frame_idx')}):")
        print(f"    d_IW={feat['d_IW']:.2f}, d_MW={feat['d_MW']:.2f}, d_RW={feat['d_RW']:.2f}, d_PW={feat['d_PW']:.2f}, d_IM={feat['d_IM']:.2f}")
        print(f"    d_TI={feat['d_TI']:.2f}, d_TW={feat['d_TW']:.2f}, idx_str={feat['idx_straight']:.2f}, idx_curl={feat['index_curl']:.2f}")
        print(f"    idx_y={feat['idx_tip_y']:.2f}, mid_y={feat['mid_tip_y']:.2f}, cross={cross:.5f}")


