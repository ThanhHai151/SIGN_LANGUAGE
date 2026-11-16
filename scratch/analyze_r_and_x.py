import json
import sys
import numpy as np

sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

# 1. Inspect R samples
with open('backend/data/alphabet_v2/static_landmarks/R.json', 'r', encoding='utf-8') as f:
    r_data = json.load(f)

print("=" * 80)
print("PHÂN TÍCH ĐẶC TRƯNG CHỮ R (D0511)")
print("=" * 80)

r_samples = r_data.get('samples', [])
r_features = []
for s in r_samples:
    lm = s.get('landmarks')
    if not lm or len(lm) < 21:
        continue
    feat = clf.extract_features(lm)
    if not feat:
        continue
    fidx = s.get('frame_index') or s.get('frame_idx')
    phase = s.get('phase')
    feat['frame_idx'] = fidx
    feat['phase'] = phase
    # Compute cross-finger relation between Index (8) and Middle (12)
    # Check if tip of middle (12) crosses tip of index (8) in X coordinate
    # Tip 8 and Tip 12
    dx_im = lm[8][0] - lm[12][0]
    dy_im = lm[8][1] - lm[12][1]
    dist_8_12 = np.linalg.norm(np.array(lm[8]) - np.array(lm[12])) / feat['palm_scale']
    feat['dist_8_12'] = dist_8_12
    feat['dx_im'] = dx_im
    feat['dy_im'] = dy_im
    r_features.append(feat)

print(f"Tổng số frame R có landmark: {len(r_features)}")
stable_r = [f for f in r_features if f['phase'] == 'stable_pose']
print(f"Số frame stable_pose: {len(stable_r)}")

keys_to_print = [
    'd_IW', 'd_MW', 'd_RW', 'd_PW', 'd_TI', 'd_TM', 'd_TW', 'd_IM',
    'dist_8_12', 'dx_im', 'diff_x_IM', 'ang_IM_dir', 'idx_straight', 'mid_straight',
    'idx_tip_y', 'mid_tip_y', 'thumb_lat', 'index_curl', 'middle_curl'
]

print("\nThống kê stable_pose của R:")
for k in keys_to_print:
    vals = [f[k] for f in stable_r if k in f]
    if vals:
        print(f"  {k:15s}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, min={np.min(vals):.3f}, max={np.max(vals):.3f}")

# 2. Inspect X samples
with open('backend/data/alphabet_v2/static_landmarks/X.json', 'r', encoding='utf-8') as f:
    x_data = json.load(f)

print("\n" + "=" * 80)
print("PHÂN TÍCH ĐẶC TRƯNG CHỮ X (D0517B)")
print("=" * 80)

x_samples = x_data.get('samples', [])
x_features = []
for s in x_samples:
    lm = s.get('landmarks')
    if not lm or len(lm) < 21:
        continue
    feat = clf.extract_features(lm)
    if not feat:
        continue
    fidx = s.get('frame_index') or s.get('frame_idx')
    phase = s.get('phase')
    feat['frame_idx'] = fidx
    feat['phase'] = phase
    # Tip 8 to Tip 4
    dist_4_8 = np.linalg.norm(np.array(lm[4]) - np.array(lm[8])) / feat['palm_scale']
    feat['dist_4_8'] = dist_4_8
    x_features.append(feat)

print(f"Tổng số frame X có landmark: {len(x_features)}")
stable_x = [f for f in x_features if f['phase'] == 'stable_pose']
print(f"Số frame stable_pose: {len(stable_x)}")

print("\nThống kê stable_pose của X:")
for k in keys_to_print + ['dist_4_8']:
    vals = [f[k] for f in stable_x if k in f]
    if vals:
        print(f"  {k:15s}: mean={np.mean(vals):.3f}, std={np.std(vals):.3f}, min={np.min(vals):.3f}, max={np.max(vals):.3f}")
