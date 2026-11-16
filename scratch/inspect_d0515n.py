import json
import math
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle, compute_finger_states

ds = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(ds['classes'])
y = ds['y']
lms = ds['landmarks']
meta = json.loads(str(ds['metadata_json']))

u_idx = classes.index('U')
print("=== D0515N (Ư base mapped to U in static_v2) ===")
for i in np.where(y == u_idx)[0]:
    m = meta[i]
    if m.get('video_id') == 'D0515N':
        p = lms[i].tolist()
        sh, ps = _shift_and_scale(p)
        states = compute_finger_states(sh, ps)
        d_iw = calculate_distance(sh[8], sh[0]) / ps
        d_mw = calculate_distance(sh[12], sh[0]) / ps
        d_rw = calculate_distance(sh[16], sh[0]) / ps
        d_pw = calculate_distance(sh[20], sh[0]) / ps
        d_im = calculate_distance(sh[8], sh[12]) / ps
        ang_im = calculate_angle(sh[8], sh[0], sh[12])

        vx_i, vy_i = sh[8][0] - sh[5][0], sh[8][1] - sh[5][1]
        len_i = math.hypot(vx_i, vy_i) + 1e-8
        vx_m, vy_m = sh[12][0] - sh[9][0], sh[12][1] - sh[9][1]
        len_m = math.hypot(vx_m, vy_m) + 1e-8
        cos_im = max(-1.0, min(1.0, (vx_i * vx_m + vy_i * vy_m) / (len_i * len_m)))
        ang_dir = math.degrees(math.acos(cos_im))
        print(f"Frm {m.get('frame_index'):<2}: d_IM={d_im:.3f}, ang_IM={ang_im:.1f}°, ang_dir={ang_dir:.1f}°, d_IW={d_iw:.2f}, d_MW={d_mw:.2f}, d_RW={d_rw:.2f}, d_PW={d_pw:.2f}")

