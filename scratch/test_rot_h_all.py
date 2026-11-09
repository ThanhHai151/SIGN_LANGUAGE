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
classifier = VSLAlphabetClassifier()

print("Testing rot_h across -45 to 40 degrees:")
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
    feat = classifier.extract_features(rot_h)
    diag_r = classifier.evaluate_rule_r_diagnostics(feat)
    res_h = classifier.classify_detailed(rot_h)
    print(f"Deg {deg:3d}: letter={res_h['letter']}, status={res_h['status']}, idx_horiz={feat.get('idx_horiz_ratio'):.2f}, idx_dy={feat.get('idx_dy'):.2f}, diag_r_match={diag_r.get('matched')}")

