import numpy as np
import json
import sys
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()
data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))
classes = list(data['classes'])
y = data['y']
lms = data['landmarks']

r_indices = [i for i, m in enumerate(meta) if m['label'] == 'R' and 22 <= m['frame_index'] <= 36]

for idx in r_indices:
    m = meta[idx]
    lm = lms[idx]
    f = m['frame_index']
    feat = clf.extract_features(lm.tolist())
    
    p5 = lm[5]
    p8 = lm[8]
    p9 = lm[9]
    p12 = lm[12]
    base_dx = p9[0] - p5[0]
    tip_dx = p12[0] - p8[0]
    cross_ind = base_dx * tip_dx
    
    d_iw = feat['d_IW']
    d_mw = feat['d_MW']
    d_rw = feat['d_RW']
    d_pw = feat['d_PW']
    d_im = feat['d_IM']
    idx_straight = feat['idx_straight']
    mid_straight = feat['mid_straight']
    idx_y = feat['idx_tip_y']
    mid_y = feat['mid_tip_y']
    
    c1 = (d_iw >= 1.10 and d_mw >= 1.30 and d_rw <= 0.65 and d_pw <= 0.55 and idx_straight >= 0.85 and mid_straight >= 0.85)
    c2 = (idx_y <= -0.85 and mid_y <= -1.05)
    c3 = (d_im <= 0.55 and cross_ind <= 0.00003 and d_mw >= d_iw - 0.25)
    
    print(f"Frame {f:2d}: c1(fingers)={c1}, c2(upright)={c2}, c3(cross)={c3}, all={c1 and c2 and c3}")
    if not c3:
        print(f"   c3 detail: d_im={d_im:.2f}<=0.55, cross_ind={cross_ind:.6f}<=0.00003, d_mw-d_iw={d_mw-d_iw:+.2f}>=-0.25")


