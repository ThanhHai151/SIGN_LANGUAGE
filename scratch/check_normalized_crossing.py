import numpy as np
import json
import sys
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))
classes = list(data['classes'])
y = data['y']
lms = data['landmarks']

for c_name in ['R', 'U', 'V', 'K']:
    c_indices = [i for i, m in enumerate(meta) if m['label'] == c_name]
    vals = []
    for idx in c_indices:
        lm = lms[idx]
        sh0 = lm[0]
        sh9 = lm[9]
        palm_scale = np.linalg.norm(np.array(sh9) - np.array(sh0))
        base_dx = (lm[9][0] - lm[5][0]) / palm_scale
        tip_dx = (lm[12][0] - lm[8][0]) / palm_scale
        cross_norm = base_dx * tip_dx
        vals.append(cross_norm)
    print(f"Class {c_name:2s}: min={min(vals):.4f}, max={max(vals):.4f}, mean={np.mean(vals):.4f}")


