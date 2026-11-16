import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from backend.vision.alphabet_classifier import AlphabetClassifier

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
clf = AlphabetClassifier()

# A frame 14 of D0490B
a_lms = npz['landmarks'][npz['y'] == 0]
meta = json.loads(str(npz['metadata_json']))
a_meta = [m for m in meta if m.get('label') == 'A']
for i, m in enumerate(a_meta):
    if m.get('video_id') == 'D0490B' and m.get('frame_index') == 14:
        f = clf.extract_features(a_lms[i].tolist())
        print("A D0490B frame 14 features:")
        for k, v in f.items():
            print(f"  {k}: {v}")
        break

