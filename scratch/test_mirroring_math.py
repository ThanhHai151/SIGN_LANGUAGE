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

# Check R sample 24
idx_r = [i for i, m in enumerate(meta) if m['label'] == 'R' and m['frame_index'] == 24][0]
lm_r = lms[idx_r].tolist()
feat_r = clf.extract_features(lm_r)

# Mirror landmarks: x -> -x
lm_r_mirrored = [[-pt[0], pt[1], pt[2]] for pt in lm_r]
feat_r_mirrored = clf.extract_features(lm_r_mirrored)

print("Original R frame 24:")
print(f"  d_IW={feat_r['d_IW']:.3f}, d_MW={feat_r['d_MW']:.3f}, d_IM={feat_r['d_IM']:.3f}")
print("Mirrored R frame 24:")
print(f"  d_IW={feat_r_mirrored['d_IW']:.3f}, d_MW={feat_r_mirrored['d_MW']:.3f}, d_IM={feat_r_mirrored['d_IM']:.3f}")

# Check base_dx * tip_dx
sh0 = lm_r[0]
sh9 = lm_r[9]
palm_scale = np.linalg.norm(np.array(sh9) - np.array(sh0))
base_dx = (lm_r[9][0] - lm_r[5][0]) / palm_scale
tip_dx = (lm_r[12][0] - lm_r[8][0]) / palm_scale
cross = base_dx * tip_dx

sh0_m = lm_r_mirrored[0]
sh9_m = lm_r_mirrored[9]
palm_scale_m = np.linalg.norm(np.array(sh9_m) - np.array(sh0_m))
base_dx_m = (lm_r_mirrored[9][0] - lm_r_mirrored[5][0]) / palm_scale_m
tip_dx_m = (lm_r_mirrored[12][0] - lm_r_mirrored[8][0]) / palm_scale_m
cross_m = base_dx_m * tip_dx_m

print(f"Original cross: {cross:.6f}, Mirrored cross: {cross_m:.6f}")
assert abs(cross - cross_m) < 1e-6, "Cross indicator must be invariant to mirroring!"
print("Assertion passed: cross indicator is perfectly mirroring invariant!")


