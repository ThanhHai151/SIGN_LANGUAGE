import json
import math
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

u_idx = classes.index('U')
v_idx = classes.index('V')
h_idx = classes.index('H')

print(f"Total U samples: {np.sum(y == u_idx)}")
print(f"Total V samples: {np.sum(y == v_idx)}")
print(f"Total H samples: {np.sum(y == h_idx)}")

def extract_uv_features(p):
    sh, ps = _shift_and_scale(p)
    states = compute_finger_states(sh, ps)
    d_iw = calculate_distance(sh[8], sh[0]) / ps
    d_mw = calculate_distance(sh[12], sh[0]) / ps
    d_rw = calculate_distance(sh[16], sh[0]) / ps
    d_pw = calculate_distance(sh[20], sh[0]) / ps
    d_im = calculate_distance(sh[8], sh[12]) / ps
    ang_im = calculate_angle(sh[8], sh[0], sh[12])
    diff_x_im = abs(sh[8][0] - sh[12][0]) / ps

    vx_i, vy_i = sh[8][0] - sh[5][0], sh[8][1] - sh[5][1]
    len_i = math.hypot(vx_i, vy_i) + 1e-8
    vx_m, vy_m = sh[12][0] - sh[9][0], sh[12][1] - sh[9][1]
    len_m = math.hypot(vx_m, vy_m) + 1e-8
    cos_im = max(-1.0, min(1.0, (vx_i * vx_m + vy_i * vy_m) / (len_i * len_m)))
    ang_im_dir = math.degrees(math.acos(cos_im))

    idx_horiz_ratio = abs(vx_i) / len_i

    return {
        "i_ext": states["index_extended"],
        "m_ext": states["middle_extended"],
        "r_ext": states["ring_extended"],
        "p_ext": states["pinky_extended"],
        "d_iw": d_iw,
        "d_mw": d_mw,
        "d_rw": d_rw,
        "d_pw": d_pw,
        "d_im": d_im,
        "ang_im": ang_im,
        "ang_im_dir": ang_im_dir,
        "diff_x_im": diff_x_im,
        "idx_horiz_ratio": idx_horiz_ratio
    }

print("\n--- U samples (D0514) ---")
u_dim, u_ang, u_diffx = [], [], []
for i in np.where(y == u_idx)[0]:
    m = meta[i]
    if m.get('video_id') == 'D0514':
        f = extract_uv_features(lms[i].tolist())
        u_dim.append(f['d_im'])
        u_ang.append(f['ang_im_dir'])
        u_diffx.append(f['diff_x_im'])
        print(f"Frm {m.get('frame_index'):<2}: d_IM={f['d_im']:.3f}, ang_IM={f['ang_im']:.1f}°, ang_dir={f['ang_im_dir']:.1f}°, diff_x={f['diff_x_im']:.3f}, d_IW={f['d_iw']:.2f}, d_MW={f['d_mw']:.2f}, d_RW={f['d_rw']:.2f}, d_PW={f['d_pw']:.2f}")

print("\n--- V samples ---")
v_dim, v_ang, v_diffx = [], [], []
for i in np.where(y == v_idx)[0]:
    m = meta[i]
    f = extract_uv_features(lms[i].tolist())
    v_dim.append(f['d_im'])
    v_ang.append(f['ang_im_dir'])
    v_diffx.append(f['diff_x_im'])
    print(f"Frm {m.get('frame_index'):<2} (vid {m.get('video_id')}): d_IM={f['d_im']:.3f}, ang_IM={f['ang_im']:.1f}°, ang_dir={f['ang_im_dir']:.1f}°, diff_x={f['diff_x_im']:.3f}, d_IW={f['d_iw']:.2f}, d_MW={f['d_mw']:.2f}")

print("\n=== U vs V Statistics ===")
print(f"U d_IM: min={min(u_dim):.3f}, max={max(u_dim):.3f}, mean={np.mean(u_dim):.3f}")
print(f"V d_IM: min={min(v_dim):.3f}, max={max(v_dim):.3f}, mean={np.mean(v_dim):.3f}")
print(f"U ang_dir: min={min(u_ang):.1f}°, max={max(u_ang):.1f}°, mean={np.mean(u_ang):.1f}°")
print(f"V ang_dir: min={min(v_ang):.1f}°, max={max(v_ang):.1f}°, mean={np.mean(v_ang):.1f}°")
print(f"U diff_x: min={min(u_diffx):.3f}, max={max(u_diffx):.3f}, mean={np.mean(u_diffx):.3f}")
print(f"V diff_x: min={min(v_diffx):.3f}, max={max(v_diffx):.3f}, mean={np.mean(v_diffx):.3f}")
