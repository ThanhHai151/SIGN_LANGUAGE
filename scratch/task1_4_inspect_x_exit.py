import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

with open('backend/data/alphabet_v2/static_landmarks/X.json', 'r', encoding='utf-8') as f:
    x_data = json.load(f)

for s in x_data['samples']:
    f_idx = s.get('frame_index') or s.get('frame_idx')
    if f_idx and f_idx >= 134 and s.get('landmarks'):
        feat = clf.extract_features(s['landmarks'])
        res = clf.classify_detailed(s['landmarks'])
        ph = s.get('phase')
        print(f"Frame {f_idx:3d} ({ph:11s}): pred={res['letter']:7s} | idx_tip_y={feat['idx_tip_y']:+.3f}, d_IW={feat['d_IW']:.3f}, d_MW={feat['d_MW']:.3f}, d_RW={feat['d_RW']:.3f}, d_PW={feat['d_PW']:.3f}, ang_IM={feat['ang_IM_dir']:4.1f}°")

