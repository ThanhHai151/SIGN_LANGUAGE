import json
import sys, os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle, compute_finger_states
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

with open('backend/tests/fixtures/real_alphabet_landmarks.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

r_samples = [s for s in d['samples'] if s.get('letter') == 'r' and s.get('landmarks')]
print(f"Valid R samples in fixture: {len(r_samples)}")

for s in r_samples:
    p = s['landmarks']
    feat = clf.extract_features(p)
    print(f"R sample (frame {s.get('frame_idx')}): d_IM={feat['d_IM']}, ang_IM_dir={feat['ang_IM_dir']}, d_IW={feat['d_IW']}, d_MW={feat['d_MW']}, diff_x={feat['diff_x_IM']}")

