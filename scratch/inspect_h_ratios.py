import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

classifier = VSLAlphabetClassifier()

for name in ['D0500N', 'H']:
    path = f'backend/data/alphabet_v2/static_landmarks/{name}.json'
    samples = json.load(open(path))['samples']
    print(f"\n=== INSPECTING {name} ===")
    for s in samples:
        lms = s.get('landmarks')
        if not lms:
            continue
        pred, conf = classifier.classify(lms)
        feat = classifier.extract_features(lms)
        idx_h = feat.get('idx_horiz_ratio', 0)
        mid_h = feat.get('mid_horiz_ratio', 0)
        ang_im = feat.get('ang_IM_dir', 0)
        d_im = feat.get('d_IM', 0)
        fidx = s.get('frame_index')
        if pred != 'H':
            print(f"Frame {fidx}: pred={pred}, idx_horiz={idx_h:.3f}, mid_horiz={mid_h:.3f}, ang_im={ang_im:.1f}, d_im={d_im:.3f}")

