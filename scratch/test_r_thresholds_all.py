import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np

npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes']
y = npz['y']
lms = npz['landmarks']

from backend.vision.alphabet_classifier import AlphabetClassifier
clf = AlphabetClassifier()

samples = []
for i in range(len(y)):
    lm = lms[i].tolist()
    feat = clf.extract_features(lm)
    samples.append((classes[y[i]], feat))

def eval_r_thresholds(
    d_iw_min=1.10, d_mw_min=1.15,
    d_rw_max=0.80, d_pw_max=0.80,
    idx_str_min=0.72, mid_str_min=0.72,
    idx_y_max=-0.70, mid_y_max=-0.70,
    d_im_max=0.60, cross_ind_max=0.0055,
    mcp_w_min=0.40, th_ring_min=0.18
):
    r_tp = 0
    fps = {}
    for lab, f in samples:
        if f is None: continue
        d_iw = f.get('d_IW', 0)
        d_mw = f.get('d_MW', 0)
        d_rw = f.get('d_RW', 99)
        d_pw = f.get('d_PW', 99)
        idx_str = f.get('idx_straight', 0)
        mid_str = f.get('mid_straight', 0)
        idx_y = f.get('idx_tip_y', 0)
        mid_y = f.get('mid_tip_y', 0)
        d_im = f.get('d_IM', 99)
        cross_ind = f.get('cross_indicator', 1.0)
        mcp_w = f.get('mcp_width_2d')
        th_ring = f.get('thumb_ring_dist')

        is_r_fingers = (
            d_iw >= d_iw_min and d_mw >= d_mw_min and
            d_rw <= d_rw_max and d_pw <= d_pw_max and
            idx_str >= idx_str_min and mid_str >= mid_str_min
        )
        is_r_upright = (
            idx_y <= idx_y_max and mid_y <= mid_y_max
        )
        is_r_cross = (
            d_im <= d_im_max and cross_ind <= cross_ind_max
        )
        is_r_orient = (
            (mcp_w is None or mcp_w >= mcp_w_min) and
            (th_ring is None or th_ring >= th_ring_min)
        )
        matched = is_r_fingers and is_r_upright and is_r_cross and is_r_orient
        if lab == 'R':
            if matched: r_tp += 1
        else:
            if matched:
                fps[lab] = fps.get(lab, 0) + 1

    return r_tp, fps

tp, fps = eval_r_thresholds()
print(f"R TP: {tp} / 14, False Positives: {fps}")

