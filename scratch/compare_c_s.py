import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from backend.vision.alphabet_classifier import AlphabetClassifier

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
clf = AlphabetClassifier()

# C frame 52
c_lms = npz['landmarks'][npz['y'] == 2] # C is index 2
meta = json.loads(str(npz['metadata_json']))
c_meta = [m for m in meta if m.get('label') == 'C']
for i, m in enumerate(c_meta):
    if m.get('frame_index') == 52:
        f = clf.extract_features(c_lms[i].tolist())
        print("C frame 52 features:")
        for k, v in f.items():
            print(f"  {k}: {v}")

# S hold frame 28
s_lms = npz['landmarks'][npz['y'] == 16]
s_meta = [m for m in meta if m.get('label') == 'S']
for i, m in enumerate(s_meta):
    if m.get('frame_index') == 28:
        f = clf.extract_features(s_lms[i].tolist())
        print("\nS frame 28 features:")
        for k, v in f.items():
            print(f"  {k}: {v}")

