import sys
import os
sys.path.insert(0, os.path.abspath('.'))

import numpy as np
import json
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

classifier = VSLAlphabetClassifier()

# Load static dataset v2
data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
X = data['X']
labels = data['y']
classes = data['classes']
landmarks = data['landmarks']

# labels are indices into classes
label_names = [classes[idx] for idx in labels]

print(f"Total samples in dataset v2: {len(labels)}")
unique_labels = sorted(list(set(label_names)))
print(f"Unique labels ({len(unique_labels)}): {unique_labels}")

# Analyze each label's landmarks with classifier.extract_features
label_stats = {}
for lbl in unique_labels:
    idxs = [i for i, l in enumerate(label_names) if l == lbl]
    feats = []
    down_count = 0
    for i in idxs:
        lm = landmarks[i].tolist()
        f = classifier.extract_features(lm)
        if f:
            feats.append(f)
            # check downward: index tip Y > index MCP Y and middle tip Y > middle MCP Y
            if lm[8][1] > lm[5][1] and lm[12][1] > lm[9][1]:
                down_count += 1
                
    if feats:
        d_iw = [f['d_IW'] for f in feats]
        d_mw = [f['d_MW'] for f in feats]
        d_rw = [f['d_RW'] for f in feats]
        d_pw = [f['d_PW'] for f in feats]
        d_tw = [f['d_TW'] for f in feats]
        
        down_ratio = down_count / len(idxs)
        print(f"Label {lbl:2s}: count={len(feats):2d}, down_ratio={down_ratio:.2f}, d_IW={np.mean(d_iw):.2f}, d_MW={np.mean(d_mw):.2f}, d_RW={np.mean(d_rw):.2f}, d_PW={np.mean(d_pw):.2f}, d_TW={np.mean(d_tw):.2f}")
