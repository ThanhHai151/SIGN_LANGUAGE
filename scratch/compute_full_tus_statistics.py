import json
import math
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle, compute_finger_states
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

ds = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(ds['classes'])
y = ds['y']
lms = ds['landmarks']
meta = json.loads(str(ds['metadata_json']))

def extract_rich_metrics(p):
    if not p or len(p) != 21:
        return None
    wrist = p[0]
    sh, ps = _shift_and_scale(p)
    if ps < 1e-5:
        return None

    states = compute_finger_states(sh, ps)

    # 1. Distances to wrist
    d_tw = calculate_distance(sh[4], sh[0]) / ps
    d_iw = calculate_distance(sh[8], sh[0]) / ps
    d_mw = calculate_distance(sh[12], sh[0]) / ps
    d_rw = calculate_distance(sh[16], sh[0]) / ps
    d_pw = calculate_distance(sh[20], sh[0]) / ps

    # 2. Inter-fingertip distances
    d_ti = calculate_distance(sh[4], sh[8]) / ps
    d_tm = calculate_distance(sh[4], sh[12]) / ps
    d_tr = calculate_distance(sh[4], sh[16]) / ps
    d_tp = calculate_distance(sh[4], sh[20]) / ps
    d_im = calculate_distance(sh[8], sh[12]) / ps
    d_mr = calculate_distance(sh[12], sh[16]) / ps
    d_rp = calculate_distance(sh[16], sh[20]) / ps

    # 3. Key thumb to joint distances
    d_t4_i5 = calculate_distance(sh[4], sh[5]) / ps   # Thumb tip to Index MCP
    d_t4_i6 = calculate_distance(sh[4], sh[6]) / ps   # Thumb tip to Index PIP
    d_t4_m9 = calculate_distance(sh[4], sh[9]) / ps   # Thumb tip to Middle MCP
    d_t4_m10 = calculate_distance(sh[4], sh[10]) / ps # Thumb tip to Middle PIP

    # 4. Joint angles
    ang_t_ip = calculate_angle(sh[2], sh[3], sh[4])
    ang_t_mcp = calculate_angle(sh[1], sh[2], sh[3])
    ang_i_mcp = calculate_angle(sh[0], sh[5], sh[6])
    ang_i_pip = calculate_angle(sh[5], sh[6], sh[7])
    ang_i_dip = calculate_angle(sh[6], sh[7], sh[8])
    ang_m_mcp = calculate_angle(sh[0], sh[9], sh[10])
    ang_m_pip = calculate_angle(sh[9], sh[10], sh[11])
    ang_m_dip = calculate_angle(sh[10], sh[11], sh[12])
    ang_r_pip = calculate_angle(sh[13], sh[14], sh[15])
    ang_p_pip = calculate_angle(sh[17], sh[18], sh[19])

    # Inter-finger angles
    ang_ti = calculate_angle(sh[4], sh[0], sh[8])
    ang_tm = calculate_angle(sh[4], sh[0], sh[12])
    ang_im = calculate_angle(sh[8], sh[0], sh[12])
    ang_tp = calculate_angle(sh[4], sh[0], sh[20])

    # 2D direction vectors
    vx_i, vy_i = sh[8][0] - sh[5][0], sh[8][1] - sh[5][1]
    len_i = math.hypot(vx_i, vy_i) + 1e-8
    vx_m, vy_m = sh[12][0] - sh[9][0], sh[12][1] - sh[9][1]
    len_m = math.hypot(vx_m, vy_m) + 1e-8
    cos_im = max(-1.0, min(1.0, (vx_i * vx_m + vy_i * vy_m) / (len_i * len_m)))
    ang_im_dir = math.degrees(math.acos(cos_im))

    diff_x_im = abs(sh[8][0] - sh[12][0]) / ps
    thumb_lat = abs(sh[4][0] - sh[0][0]) / ps

    # Straightness & curl
    t_curl = states.get("thumb_curl", 0.0)
    i_curl = states.get("index_curl", 0.0)
    m_curl = states.get("middle_curl", 0.0)
    r_curl = states.get("ring_curl", 0.0)
    p_curl = states.get("pinky_curl", 0.0)

    clf_feat = clf.extract_features(p)
    idx_straight = clf_feat.get("idx_straight", 0.0) if clf_feat else 0.0
    mid_straight = clf_feat.get("mid_straight", 0.0) if clf_feat else 0.0

    return {
        "thumb_ext": states["thumb_extended"],
        "index_ext": states["index_extended"],
        "middle_ext": states["middle_extended"],
        "ring_ext": states["ring_extended"],
        "pinky_ext": states["pinky_extended"],
        "t_curl": t_curl, "i_curl": i_curl, "m_curl": m_curl, "r_curl": r_curl, "p_curl": p_curl,
        "d_tw": d_tw, "d_iw": d_iw, "d_mw": d_mw, "d_rw": d_rw, "d_pw": d_pw,
        "d_ti": d_ti, "d_tm": d_tm, "d_tr": d_tr, "d_tp": d_tp,
        "d_im": d_im, "d_mr": d_mr, "d_rp": d_rp,
        "d_t4_i5": d_t4_i5, "d_t4_i6": d_t4_i6, "d_t4_m9": d_t4_m9, "d_t4_m10": d_t4_m10,
        "ang_t_ip": ang_t_ip, "ang_t_mcp": ang_t_mcp,
        "ang_i_mcp": ang_i_mcp, "ang_i_pip": ang_i_pip, "ang_i_dip": ang_i_dip,
        "ang_m_mcp": ang_m_mcp, "ang_m_pip": ang_m_pip, "ang_m_dip": ang_m_dip,
        "ang_r_pip": ang_r_pip, "ang_p_pip": ang_p_pip,
        "ang_ti": ang_ti, "ang_tm": ang_tm, "ang_im": ang_im, "ang_im_dir": ang_im_dir,
        "ang_tp": ang_tp,
        "diff_x_im": diff_x_im,
        "thumb_lat": thumb_lat,
        "idx_straight": idx_straight, "mid_straight": mid_straight
    }

# Gather all data for all 22 classes
dataset_metrics = {}
for cls_name in classes:
    idx = classes.index(cls_name)
    rows = []
    for i in np.where(y == idx)[0]:
        m = meta[i]
        met = extract_rich_metrics(lms[i].tolist())
        if met:
            met['frame_index'] = m.get('frame_index')
            met['video_id'] = m.get('video_id')
            rows.append(met)
    dataset_metrics[cls_name] = rows

# Also get pure stable frames for T, U, S from their specific json files
stable_frames = {}
for let, jpath in [('T', 'backend/data/alphabet_v2/static_landmarks/T.json'),
                   ('U', 'backend/data/alphabet_v2/static_landmarks/U.json'),
                   ('S', 'backend/data/alphabet_v2/static_landmarks/S.json')]:
    with open(jpath, 'r', encoding='utf-8') as f:
        js = json.load(f)
    st = []
    for s in js['samples']:
        if s.get('phase') == 'stable_pose' and s.get('landmarks'):
            # Only keep valid hold frames (e.g. before exiting drop)
            met = extract_rich_metrics(s['landmarks'])
            if met:
                met['frame_index'] = s['frame_index']
                st.append(met)
    stable_frames[let] = st

# Filter stable frames: drop exit transition where hand drops
# T: frames 18-44
stable_frames['T_hold'] = [r for r in stable_frames['T'] if 18 <= r['frame_index'] <= 44]
# U: frames 16-42
stable_frames['U_hold'] = [r for r in stable_frames['U'] if 16 <= r['frame_index'] <= 42]
# S: frames 18-40
stable_frames['S_hold'] = [r for r in stable_frames['S'] if 18 <= r['frame_index'] <= 40]

print(f"Sample counts: T_hold={len(stable_frames['T_hold'])}, U_hold={len(stable_frames['U_hold'])}, S_hold={len(stable_frames['S_hold'])}")

# Feature summary table
features_to_report = [
    "i_curl", "m_curl", "r_curl", "p_curl", "t_curl",
    "d_iw", "d_mw", "d_rw", "d_pw", "d_tw",
    "d_ti", "d_tm", "d_im", "d_mr", "d_rp",
    "d_t4_i6", "d_t4_m10",
    "ang_i_pip", "ang_m_pip", "ang_r_pip", "ang_p_pip", "ang_t_ip",
    "ang_im", "ang_im_dir", "ang_ti", "ang_tm",
    "diff_x_im", "thumb_lat", "idx_straight", "mid_straight"
]

res = {}
for let in ['T_hold', 'U_hold', 'S_hold']:
    res[let] = {}
    for feat in features_to_report:
        vals = [r[feat] for r in stable_frames[let] if r.get(feat) is not None]
        arr = np.array(vals)
        res[let][feat] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "p25": float(np.percentile(arr, 25)),
            "median": float(np.median(arr)),
            "p75": float(np.percentile(arr, 75))
        }

with open('scratch/tus_statistics.json', 'w', encoding='utf-8') as f:
    json.dump(res, f, indent=2)

print("\nSaved scratch/tus_statistics.json successfully!")

