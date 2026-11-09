import numpy as np
import json
import sys
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()
data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))
lms = data['landmarks']

for target in ['R', 'X']:
    print(f"=== TESTING {target} ===")
    for i, m in enumerate(meta):
        if m['label'] == target and m.get('is_representative'):
            lm = lms[i].tolist()
            res = clf.classify_detailed(lm)
            f = m['frame_index']
            print(f"frame {f:2d}: letter={res['letter']}, status={res['status']}")
            if target == 'R':
                diag = clf.evaluate_rule_r_diagnostics(res['features'])
                print(f"  R matched={diag['matched']}, failed={diag['failed_conditions']}")
            if target == 'X':
                diag = clf.evaluate_rule_x_diagnostics(res['features'])
                print(f"  X matched={diag['matched']}, failed={diag['failed_conditions']}")

