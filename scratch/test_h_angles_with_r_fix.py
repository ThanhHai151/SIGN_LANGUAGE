import os, math, numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

g_idx = classes.index("G")
pts_h_real = [list(p) for p in landmarks[np.where(y == g_idx)[0][1]].tolist()]
for k_i, k_m in [(5, 9), (6, 10), (7, 11), (8, 12)]:
    dx = pts_h_real[k_i][0] - pts_h_real[5][0]
    dy = pts_h_real[k_i][1] - pts_h_real[5][1]
    dz = pts_h_real[k_i][2] - pts_h_real[5][2]
    pts_h_real[k_m] = [
        pts_h_real[9][0] + 1.22 * dx,
        pts_h_real[9][1] + 1.22 * dy,
        pts_h_real[9][2] + dz,
    ]

w_h = pts_h_real[0]

# Simulate with R_IDX_HORIZ_MAX = 0.68, R_IDX_DY_MAX = -0.20
# Let's see if H matches on all angles -45 to 40 deg
for deg in range(-45, 45, 5):
    rad = math.radians(deg)
    c, s_val = math.cos(rad), math.sin(rad)
    rot_h = [
        [
            w_h[0] + (pt[0] - w_h[0]) * c - (pt[1] - w_h[1]) * s_val,
            w_h[1] + (pt[0] - w_h[0]) * s_val + (pt[1] - w_h[1]) * c,
            pt[2],
        ]
        for pt in pts_h_real
    ]
    # Check what features rot_h has
    c_inst = VSLAlphabetClassifier()
    feat = c_inst.extract_features(rot_h)
    
    idx_horiz = feat.get("idx_horiz_ratio", 0.0)
    idx_dy = feat.get("idx_dy", 0.0)
    
    # Is R upright with this new condition?
    idx_tip_y = feat.get("idx_tip_y", 0.0)
    mid_tip_y = feat.get("mid_tip_y", 0.0)
    is_r_upright = (
        idx_tip_y <= -0.70
        and mid_tip_y <= -0.70
        and idx_horiz <= 0.68
        and idx_dy <= -0.20
    )
    if is_r_upright:
        print(f"Angle {deg}: WARNING - is_r_upright still True! idx_horiz={idx_horiz}, idx_dy={idx_dy}")
    else:
        # All good
        pass

print("Done checking all angles of H!")

