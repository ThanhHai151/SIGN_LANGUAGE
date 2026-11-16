import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from backend.vision.alphabet_classifier import AlphabetClassifier

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
clf = AlphabetClassifier()
meta = json.loads(str(npz['metadata_json']))

s_lms = npz['landmarks'][npz['y'] == 16]
s_meta = [m for m in meta if m.get('label') == 'S']

for i in range(len(s_lms)):
    f = clf.extract_features(s_lms[i].tolist())
    fidx = s_meta[i].get('frame_index')
    print(f"Frame {fidx}: middle_curl={f['middle_curl']:.4f}, ring_curl={f['ring_curl']:.4f}, d_TW={f['d_TW']:.2f}, d_IW={f['d_IW']:.2f}")

