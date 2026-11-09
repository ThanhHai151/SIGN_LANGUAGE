import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

classifier = VSLAlphabetClassifier()

with open('backend/data/alphabet_v2/static_landmarks/M.json', 'r', encoding='utf-8') as f:
    m_data = json.load(f)

with open('backend/data/alphabet_v2/static_landmarks/N.json', 'r', encoding='utf-8') as f:
    n_data = json.load(f)

def run_features(samples, label, frames):
    print(f"\n================ FEATURES FOR {label} ================")
    for s in samples:
        fidx = s['frame_index']
        if fidx not in frames:
            continue
        feat = classifier.extract_features(s['landmarks'])
        if not feat:
            continue
        print(f"Frame {fidx}:")
        for k in ['thumb_ext', 'index_ext', 'middle_ext', 'ring_ext', 'pinky_ext',
                  'index_curl', 'middle_curl', 'ring_curl', 'pinky_curl',
                  'd_TW', 'd_IW', 'd_MW', 'd_RW', 'd_PW', 'd_TI', 'd_TM', 'd_IM',
                  'palm_scale']:
            print(f"  {k:12s}: {feat.get(k)}")

run_features(m_data['samples'], 'M (D0504)', [46, 48, 50, 52, 54, 56])
run_features(n_data['samples'], 'N (D0505)', [20, 22, 24, 26, 28, 30])

