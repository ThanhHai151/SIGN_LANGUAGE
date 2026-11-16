import json
import sys
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

with open('backend/data/alphabet_v2/static_landmarks/D0500N.json', 'r', encoding='utf-8') as f:
    d = json.load(f)

for s in d['samples']:
    if s.get('frame_index') in [10, 20, 32]:
        lm = s['landmarks']
        feat = clf.extract_features(lm)
        res = clf.classify_detailed(lm)
        diag_r = clf.evaluate_rule_r_diagnostics(feat)
        print(f"H D0500N frame {s['frame_index']}:")
        print(f"  mcp_width={feat['mcp_width_2d']:.2f}, thumb_ring={feat['thumb_ring_dist']:.2f}, cross={feat['cross_indicator']:.5f}")
        print(f"  R matched={diag_r['matched']}, failed={diag_r['failed_conditions']}")
        print(f"  Result: letter={res['letter']}, status={res['status']}, reason={res['reason']}")

