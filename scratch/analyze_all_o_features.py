import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.alphabet_classifier import alphabet_classifier

# Load O.json
with open('backend/data/alphabet_v2/static_landmarks/O.json', 'r', encoding='utf-8') as f:
    o_json = json.load(f)

# Load NPZ
npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes'].tolist()
o_idx = classes.index('O')
npz_o_lms = [npz['landmarks'][i] for i in range(len(npz['y'])) if npz['y'][i] == o_idx]

all_o = []
for s in o_json['samples']:
    if s.get('phase') == 'stable_pose' and s.get('landmarks'):
        all_o.append(('O.json', s['landmarks']))

for i, lm in enumerate(npz_o_lms):
    all_o.append((f'NPZ_{i}', lm.tolist()))

print(f"Total O samples: {len(all_o)}")

# Check feature ranges
records = []
for src, lm in all_o:
    feat = alphabet_classifier.extract_features(lm)
    records.append({
        'src': src,
        'd_TI': feat['d_TI'],
        'd_TM': feat['d_TM'],
        'd_IM': feat['d_IM'],
        'd_MR': feat['d_MR'],
        'd_IW': feat['d_IW'],
        'd_MW': feat['d_MW'],
        'd_RW': feat['d_RW'],
        'd_PW': feat['d_PW'],
        'idx_horiz': feat['idx_horiz_ratio'],
        'mid_horiz': feat['mid_horiz_ratio'],
        'idx_down': feat['idx_down_ratio'],
        'thumb_mid': feat['thumb_mid_dist'],
        'thumb_ring': feat['thumb_ring_dist'],
        'wrist_y': feat['wrist_y'],
    })

import pandas as pd
df = pd.DataFrame(records)
print(df.describe().to_string())

