import json
import sys
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier
clf = VSLAlphabetClassifier()

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for target in ['R', 'X']:
    samples = [s for s in d['samples'] if s.get('letter_upper') == target]
    print(f"=== Fixture samples for {target} ({len(samples)} samples) ===")
    for i, s in enumerate(samples):
        lm = s['landmarks']
        feat = clf.extract_features(lm)
        fidx = s.get('frame_idx') or s.get('frame_index')
        print(f" sample {i} (frame {fidx}): d_IW={feat['d_IW']:.2f}, d_MW={feat['d_MW']:.2f}, "
              f"d_RW={feat['d_RW']:.2f}, d_PW={feat['d_PW']:.2f}, d_IM={feat['d_IM']:.2f}, "
              f"d_TI={feat['d_TI']:.2f}, idx_y={feat['idx_tip_y']:.2f}")


