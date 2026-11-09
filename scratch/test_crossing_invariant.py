import numpy as np
import json
import sys

data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))
classes = list(data['classes'])
y = data['y']
lms = data['landmarks']

print("CROSSING FINGER ANALYSIS: (MCP9_x - MCP5_x) * (Tip12_x - Tip8_x)")
# If Base dx and Tip dx have OPPOSITE signs, the fingers have crossed!
for c_name in ['R', 'U', 'V', 'K', 'H', 'D', 'B']:
    c_indices = [i for i, m in enumerate(meta) if m['label'] == c_name]
    vals = []
    crossed_count = 0
    for idx in c_indices:
        lm = lms[idx]
        p5 = lm[5]
        p8 = lm[8]
        p9 = lm[9]
        p12 = lm[12]
        base_dx = p9[0] - p5[0]
        tip_dx = p12[0] - p8[0]
        # Cross product or sign inversion
        cross_indicator = base_dx * tip_dx
        vals.append(cross_indicator)
        if cross_indicator < 0:
            crossed_count += 1
    print(f"Class {c_name:2s} ({len(c_indices):2d} samples): crossed={crossed_count}/{len(c_indices)}, "
          f"indicator: min={min(vals):.6f}, max={max(vals):.6f}, mean={np.mean(vals):.6f}")


