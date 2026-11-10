import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from backend.vision.alphabet_classifier import AlphabetClassifier

# 1. Load user's live webcam S frames
webcam_file = 'backend/data/diagnostics/pilot_pilot_20261106_2347.jsonl'
with open(webcam_file, 'r', encoding='utf-8') as f:
    webcam_lines = [json.loads(l.strip()) for l in f if l.strip()]

webcam_s_features = [
    line['geometric']['features'] 
    for line in webcam_lines 
    if line.get('expected_sign') == 'S' and 'features' in line.get('geometric', {})
]
print(f"Total user webcam S frames: {len(webcam_s_features)}")

# 2. Load dataset
npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes']
y = npz['y']
lms = npz['landmarks']
meta = json.loads(str(npz['metadata_json']))

clf = AlphabetClassifier()

dataset_s_features = []
dataset_other_features = [] # (true_label, feat)

for i in range(len(y)):
    lm = lms[i].tolist()
    feat = clf.extract_features(lm)
    true_label = classes[y[i]]
    if true_label == 'S':
        fidx = meta[i].get('frame_index')
        dataset_s_features.append((fidx, feat))
    else:
        dataset_other_features.append((true_label, feat))

print(f"Total dataset S samples: {len(dataset_s_features)}")
print(f"Total dataset other samples: {len(dataset_other_features)}")

# Let's inspect the distribution of d_TW, d_IW, d_MW, d_RW, d_PW, d_TI, d_TM, thumb_lat across dataset other classes
print("\n--- DISTRIBUTION OF FIST-LIKE LETTERS IN DATASET ---")
fist_labels = ['A', 'E', 'T', 'O', 'X', 'M', 'N', 'C']
for lab in fist_labels:
    feats = [f for l, f in dataset_other_features if l == lab and f is not None]
    dtw = [f['d_TW'] for f in feats]
    diw = [f['d_IW'] for f in feats]
    th_lat = [f['thumb_lat'] for f in feats]
    dti = [f['d_TI'] for f in feats]
    dtm = [f['d_TM'] for f in feats]
    print(f"Class {lab:2s} ({len(feats)} samples): d_TW=[{min(dtw):.2f}, {max(dtw):.2f}], d_IW=[{min(diw):.2f}, {max(diw):.2f}], th_lat=[{min(th_lat):.2f}, {max(th_lat):.2f}], d_TI=[{min(dti):.2f}, {max(dti):.2f}], d_TM=[{min(dtm):.2f}, {max(dtm):.2f}]")
