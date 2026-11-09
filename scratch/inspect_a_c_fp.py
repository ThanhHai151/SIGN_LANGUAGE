import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from backend.vision.alphabet_classifier import AlphabetClassifier

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes']
y = npz['y']
lms = npz['landmarks']
meta = json.loads(str(npz['metadata_json']))

clf = AlphabetClassifier()

for i in range(len(y)):
    lab = classes[y[i]]
    if lab in ['A', 'C']:
        lm = lms[i].tolist()
        f = clf.extract_features(lm)
        if f is None: continue
        i_ext = f.get('i_ext', False)
        m_ext = f.get('m_ext', False)
        r_ext = f.get('r_ext', False)
        p_ext = f.get('p_ext', False)
        if not i_ext and not m_ext and not r_ext and not p_ext:
            d_tw = f.get('d_TW', 99)
            d_iw = f.get('d_IW', 99)
            d_mw = f.get('d_MW', 99)
            d_rw = f.get('d_RW', 99)
            d_pw = f.get('d_PW', 99)
            d_ti = f.get('d_TI', 99)
            d_tm = f.get('d_TM', 99)
            th_lat = f.get('thumb_lat', 0)
            if (d_tw <= 1.40 and 0.10 <= d_ti <= 0.50 and
                0.35 <= d_tm <= 0.80 and d_iw <= 1.35 and
                d_mw <= 0.85 and d_rw <= 0.75 and d_pw <= 0.85 and
                th_lat >= 0.50):
                fidx = meta[i].get('frame_index')
                vid = meta[i].get('video_id')
                print(f"FP sample on {lab} ({vid}, frame {fidx}): d_TW={d_tw:.2f}, d_IW={d_iw:.2f}, d_MW={d_mw:.2f}, d_TI={d_ti:.2f}, d_TM={d_tm:.2f}, th_lat={th_lat:.2f}")

