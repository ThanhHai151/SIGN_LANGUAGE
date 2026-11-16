import sys
sys.path.insert(0, ".")
import json
from backend.vision.feature_extractor import extract_hand_features

data = json.load(open('backend/data/alphabet_v2/static_landmarks/I.json', encoding='utf-8'))
samples = [s for s in data['samples'] if s['landmarks'] is not None]

for s in samples:
    f_idx = s['frame_index']
    phase = s['phase']
    lm = s['landmarks']
    feat = extract_hand_features(lm)
    print(f"f{f_idx:02d} ({phase:11s}): d_PW={feat['d_PW']:.2f} p_ext={feat['pinky_ext']} p_curl={feat['pinky_curl']:.3f} p_straight={feat.get('pinky_straight', 0.0):.3f} | idx_curl={feat['index_curl']:.3f} mid_curl={feat['middle_curl']:.3f} ring_curl={feat['ring_curl']:.3f} | d_IW={feat['d_IW']:.2f} d_MW={feat['d_MW']:.2f} d_RW={feat['d_RW']:.2f} d_TW={feat['d_TW']:.2f} d_TI={feat['d_TI']:.2f}")

