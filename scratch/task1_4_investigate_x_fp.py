import sys
sys.path.insert(0, '.')
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

with open('backend/data/alphabet_v2/static_landmarks/X.json', 'r', encoding='utf-8') as f:
    x_data = json.load(f)

for s in x_data['samples']:
    if not s.get('landmarks'):
        continue
    f_idx = s.get('frame_index') or s.get('frame_idx')
    res = clf.classify_detailed(s['landmarks'])
    if res['letter'] == 'P':
        print(f"ALERT! X frame {f_idx} predicted as P!")
        print(f"  Reason: {res['reason']}")
        feat = clf.extract_features(s['landmarks'])
        diag = clf.evaluate_rule_p_diagnostics(feat)
        print(f"  Diagnostics: {diag}")
        print("  Features of this frame:")
        for k in ['d_IW', 'idx_straight', 'idx_tip_y', 'idx_dy', 'idx_horiz_ratio', 'd_MW', 'mid_straight', 'mid_tip_y', 'd_TM', 'd_TI', 'd_TW', 'ang_IM_dir', 'd_RW', 'd_PW']:
            print(f"    {k:15s}: {feat.get(k)}")

