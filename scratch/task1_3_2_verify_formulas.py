import sys
sys.path.insert(0, '.')
import json
import math
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

print("=== 1. FEATURE FORMULA AUDIT ON REPRESENTATIVE FRAMES ===")

# Test frame: P D0509B frame 38
with open('backend/data/alphabet_v2/static_landmarks/P.json', 'r', encoding='utf-8') as f:
    p_data = json.load(f)

p38 = [s for s in p_data['samples'] if (s.get('frame_index') or s.get('frame_idx')) == 38][0]
lms = p38['landmarks']

feat = clf.extract_features(lms)

# Manual calculations following production code:
p = lms
wrist = p[0]
sh = [[p[i][0] - wrist[0], p[i][1] - wrist[1], p[i][2] - wrist[2]] for i in range(21)]
palm_scale = math.sqrt(sh[9][0]**2 + sh[9][1]**2 + sh[9][2]**2)

# 1. d_iw
manual_d_iw = math.sqrt(sh[8][0]**2 + sh[8][1]**2 + sh[8][2]**2) / palm_scale

# 2. idx_straight (5->8)
vx_58 = (sh[8][0] - sh[5][0]) / palm_scale
vy_58 = (sh[8][1] - sh[5][1]) / palm_scale
len_58 = math.hypot(vx_58, vy_58) + 1e-8
seg_58 = (
    math.hypot(sh[6][0] - sh[5][0], sh[6][1] - sh[5][1])
    + math.hypot(sh[7][0] - sh[6][0], sh[7][1] - sh[6][1])
    + math.hypot(sh[8][0] - sh[7][0], sh[8][1] - sh[7][1])
) / palm_scale + 1e-8
manual_idx_straight = len_58 / seg_58

# Test what 0->8 idx_straight would be:
seg_08 = (
    math.hypot(sh[5][0] - sh[0][0], sh[5][1] - sh[0][1])
    + math.hypot(sh[6][0] - sh[5][0], sh[6][1] - sh[5][1])
    + math.hypot(sh[7][0] - sh[6][0], sh[7][1] - sh[6][1])
    + math.hypot(sh[8][0] - sh[7][0], sh[8][1] - sh[7][1])
) / palm_scale + 1e-8
len_08 = math.hypot(sh[8][0] / palm_scale, sh[8][1] / palm_scale) + 1e-8
ratio_08 = len_08 / seg_08

# 3. idx_horiz_ratio
manual_idx_horiz_ratio = abs(vx_58) / len_58
manual_idx_horiz_dy = abs(vx_58) / (abs(vy_58) + 1e-8)

# 4. mid_straight (9->12)
vx_912 = (sh[12][0] - sh[9][0]) / palm_scale
vy_912 = (sh[12][1] - sh[9][1]) / palm_scale
len_912 = math.hypot(vx_912, vy_912) + 1e-8
seg_912 = (
    math.hypot(sh[10][0] - sh[9][0], sh[10][1] - sh[9][1])
    + math.hypot(sh[11][0] - sh[10][0], sh[11][1] - sh[10][1])
    + math.hypot(sh[12][0] - sh[11][0], sh[12][1] - sh[11][1])
) / palm_scale + 1e-8
manual_mid_straight = len_912 / seg_912

# 5. ang_IM_dir
cos_im = max(-1.0, min(1.0, (vx_58 * vx_912 + vy_58 * vy_912) / (len_58 * len_912)))
manual_ang_im_dir = math.degrees(math.acos(cos_im))

print(f"palm_scale: feat={feat['palm_scale']} vs manual={palm_scale:.4f} (diff={abs(feat['palm_scale'] - palm_scale):.6f})")
print(f"d_IW:       feat={feat['d_IW']} vs manual={manual_d_iw:.4f} (diff={abs(feat['d_IW'] - manual_d_iw):.6f})")
print(f"idx_straight (5->8): feat={feat['idx_straight']} vs manual={manual_idx_straight:.4f} (diff={abs(feat['idx_straight'] - manual_idx_straight):.6f})")
print(f"idx_straight (0->8): if using 0->8 it would be {ratio_08:.4f} (DISCREPANCY with feat={feat['idx_straight']})")
print(f"idx_horiz_ratio:     feat={feat['idx_horiz_ratio']} vs abs(dx)/len={manual_idx_horiz_ratio:.4f} vs abs(dx)/abs(dy)={manual_idx_horiz_dy:.4f}")
print(f"mid_straight (9->12): feat={feat['mid_straight']} vs manual={manual_mid_straight:.4f} (diff={abs(feat['mid_straight'] - manual_mid_straight):.6f})")
print(f"ang_IM_dir:          feat={feat['ang_IM_dir']} vs manual={manual_ang_im_dir:.2f} (diff={abs(feat['ang_IM_dir'] - manual_ang_im_dir):.6f})")

