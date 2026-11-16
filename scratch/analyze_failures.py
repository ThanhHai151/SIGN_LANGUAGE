import json
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()
ds = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(ds['classes'])
y = ds['y']
lms = ds['landmarks']
meta = json.loads(str(ds['metadata_json']))

print("=== Analyzing T failures ===")
t_idx = classes.index('T')
for i in np.where(y == t_idx)[0]:
    p = lms[i].tolist()
    res = clf.classify_detailed(p)
    if res.get('letter') != 'T':
        m = meta[i]
        feat = res.get('features', {})
        print(f"Frame {m.get('frame_index')}: status={res.get('status')}, reason={res.get('reason')}")
        print(f"   d_TI={feat.get('d_TI')}, d_TM={feat.get('d_TM')}, d_IM={feat.get('d_IM')}, d_IW={feat.get('d_IW')}")

print("\n=== Analyzing S failures ===")
s_idx = classes.index('S')
for i in np.where(y == s_idx)[0]:
    p = lms[i].tolist()
    res = clf.classify_detailed(p)
    if res.get('letter') != 'S':
        m = meta[i]
        feat = res.get('features', {})
        print(f"Frame {m.get('frame_index')}: status={res.get('status')}, reason={res.get('reason')}")
        print(f"   d_TW={feat.get('d_TW')}, d_TI={feat.get('d_TI')}, d_TM={feat.get('d_TM')}, d_IW={feat.get('d_IW')}, d_MW={feat.get('d_MW')}")
        print(f"   exts: i={feat.get('index_ext')}, m={feat.get('middle_ext')}, r={feat.get('ring_ext')}, p={feat.get('pinky_ext')}")

