import json
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle, compute_finger_states
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()
ds = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(ds['classes'])
y = ds['y']
lms = ds['landmarks']
meta = json.loads(str(ds['metadata_json']))

a_idx = classes.index('A')
e_idx = classes.index('E')
s_idx = classes.index('S')

def extract_fist_features(p):
    sh, ps = _shift_and_scale(p)
    states = compute_finger_states(sh, ps)
    d_tw = calculate_distance(sh[4], sh[0]) / ps
    d_iw = calculate_distance(sh[8], sh[0]) / ps
    d_mw = calculate_distance(sh[12], sh[0]) / ps
    d_ti = calculate_distance(sh[4], sh[8]) / ps
    d_tm = calculate_distance(sh[4], sh[12]) / ps
    thumb_lat = abs(sh[4][0] - sh[0][0]) / ps
    t_curl = states.get("thumb_curl", 0.0)
    vy_t = (sh[4][1] - sh[2][1]) / ps # thumb vertical direction (negative is upright)
    return {
        "d_tw": d_tw, "d_iw": d_iw, "d_mw": d_mw, "d_ti": d_ti, "d_tm": d_tm,
        "thumb_lat": thumb_lat, "t_curl": t_curl, "vy_t": vy_t
    }

print("=== Letter A ===")
a_tw, a_lat, a_ti, a_tm = [], [], [], []
for i in np.where(y == a_idx)[0]:
    f = extract_fist_features(lms[i].tolist())
    a_tw.append(f['d_tw']); a_lat.append(f['thumb_lat']); a_ti.append(f['d_ti']); a_tm.append(f['d_tm'])
print(f"A: d_TW={np.mean(a_tw):.2f} [{min(a_tw):.2f}, {max(a_tw):.2f}], thumb_lat={np.mean(a_lat):.2f} [{min(a_lat):.2f}, {max(a_lat):.2f}], d_TI={np.mean(a_ti):.2f} [{min(a_ti):.2f}, {max(a_ti):.2f}], d_TM={np.mean(a_tm):.2f} [{min(a_tm):.2f}, {max(a_tm):.2f}]")

print("=== Letter E ===")
e_tw, e_lat, e_ti, e_tm = [], [], [], []
for i in np.where(y == e_idx)[0]:
    f = extract_fist_features(lms[i].tolist())
    e_tw.append(f['d_tw']); e_lat.append(f['thumb_lat']); e_ti.append(f['d_ti']); e_tm.append(f['d_tm'])
print(f"E: d_TW={np.mean(e_tw):.2f} [{min(e_tw):.2f}, {max(e_tw):.2f}], thumb_lat={np.mean(e_lat):.2f} [{min(e_lat):.2f}, {max(e_lat):.2f}], d_TI={np.mean(e_ti):.2f} [{min(e_ti):.2f}, {max(e_ti):.2f}], d_TM={np.mean(e_tm):.2f} [{min(e_tm):.2f}, {max(e_tm):.2f}]")

print("=== Letter S (stable) ===")
s_tw, s_lat, s_ti, s_tm = [], [], [], []
for i in np.where(y == s_idx)[0]:
    m = meta[i]
    if 18 <= m.get('frame_index') <= 40:
        f = extract_fist_features(lms[i].tolist())
        s_tw.append(f['d_tw']); s_lat.append(f['thumb_lat']); s_ti.append(f['d_ti']); s_tm.append(f['d_tm'])
print(f"S: d_TW={np.mean(s_tw):.2f} [{min(s_tw):.2f}, {max(s_tw):.2f}], thumb_lat={np.mean(s_lat):.2f} [{min(s_lat):.2f}, {max(s_lat):.2f}], d_TI={np.mean(s_ti):.2f} [{min(s_ti):.2f}, {max(s_ti):.2f}], d_TM={np.mean(s_tm):.2f} [{min(s_tm):.2f}, {max(s_tm):.2f}]")

