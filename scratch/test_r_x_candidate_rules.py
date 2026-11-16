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

all_samples = []
for idx in range(len(y)):
    m = meta[idx]
    lm = lms[idx].tolist()
    feat = clf.extract_features(lm)
    if not feat:
        continue
    feat['label'] = m['label']
    feat['label_idx'] = m['label_idx']
    feat['frame_index'] = m['frame_index']
    feat['is_representative'] = m.get('is_representative', False)
    feat['landmarks_raw'] = lm
    all_samples.append(feat)

print("=" * 80)
print("TESTING CANDIDATE RULE 17 (R)")
print("=" * 80)

# Candidate conditions for R:
# 1. Index & Middle extended upright:
#    d_iw >= 1.10, d_mw >= 1.30, idx_straight >= 0.85, mid_straight >= 0.85
#    idx_tip_y <= -0.80, mid_tip_y <= -1.00
# 2. Ring & Pinky curled:
#    d_rw <= 0.65, d_pw <= 0.55
# 3. Crossing or touch proximity:
#    d_im <= 0.55
#    middle finger crosses over index:
#    (mcp9_x - mcp5_x) * (tip12_x - tip8_x) <= 0.00003
#    d_mw >= d_iw - 0.25 (middle extends past or level with index)
# 4. Exclude U, V, K:
#    In U: ang_im_dir <= 10.0 and fingers parallel without crossing (base_dx * tip_dx >= 0.00006)
#    In V: ang_im_dir >= 12.0 and d_im >= 0.60
#    In K: mid_horiz_ratio >= 0.50

r_matches = []
fp_r = []

for s in all_samples:
    lm = s['landmarks_raw']
    p5 = lm[5]
    p8 = lm[8]
    p9 = lm[9]
    p12 = lm[12]
    base_dx = p9[0] - p5[0]
    tip_dx = p12[0] - p8[0]
    cross_ind = base_dx * tip_dx
    
    d_iw = s['d_IW']
    d_mw = s['d_MW']
    d_rw = s['d_RW']
    d_pw = s['d_PW']
    d_im = s['d_IM']
    idx_straight = s['idx_straight']
    mid_straight = s['mid_straight']
    idx_y = s['idx_tip_y']
    mid_y = s['mid_tip_y']
    
    # Rule R condition
    is_r_fingers = (
        d_iw >= 1.10 and d_mw >= 1.30
        and d_rw <= 0.65 and d_pw <= 0.55
        and idx_straight >= 0.85 and mid_straight >= 0.85
    )
    is_r_upright = (
        idx_y <= -0.85 and mid_y <= -1.05
    )
    is_r_cross = (
        d_im <= 0.55
        and cross_ind <= 0.00003
        and d_mw >= d_iw - 0.25
    )
    
    if is_r_fingers and is_r_upright and is_r_cross:
        if s['label'] == 'R':
            r_matches.append(s)
        else:
            fp_r.append(s)

print(f"R True Positives: {len(r_matches)}/{len([s for s in all_samples if s['label'] == 'R'])}")
print(f"R False Positives: {len(fp_r)}")
if fp_r:
    for fp in fp_r:
        print(f"  FP: label={fp['label']}, frame={fp['frame_index']}, d_IW={fp['d_IW']:.2f}, d_MW={fp['d_MW']:.2f}, cross_ind={cross_ind}")

print("\n" + "=" * 80)
print("TESTING CANDIDATE RULE 18 (X)")
print("=" * 80)

# Candidate conditions for X (dictionary hold frames):
# 1. Index hooked touching thumb:
#    d_ti <= 0.32, d_iw <= 1.10, idx_straight <= 0.78, index_curl >= 0.18
# 2. Ring & Pinky curled:
#    d_rw <= 0.70, d_pw <= 0.60
# 3. Thumb extended / stable:
#    0.70 <= d_tw <= 0.95
# 4. Middle not extended:
#    d_mw <= 1.55
# 5. Not fist (A, S) and not O:
#    In O: d_rw >= 0.75, d_pw >= 0.70
#    In A/S: index is folded flat (idx_straight >= 0.85 or index_curl <= 0.15)
#    In X: index is bent in hook (0.50 <= idx_straight <= 0.75, index_curl >= 0.20)

x_matches = []
fp_x = []

for s in all_samples:
    d_ti = s['d_TI']
    d_iw = s['d_IW']
    d_mw = s['d_MW']
    d_rw = s['d_RW']
    d_pw = s['d_PW']
    d_tw = s['d_TW']
    idx_straight = s['idx_straight']
    idx_curl = s['index_curl']
    idx_y = s['idx_tip_y']
    mid_y = s['mid_tip_y']
    
    is_x_hook = (
        d_ti <= 0.25
        and d_iw <= 1.10
        and idx_curl >= 0.15
        and idx_straight <= 0.85
    )
    is_x_curled = (
        d_rw <= 0.70
        and d_pw <= 0.60
    )
    is_x_thumb = (
        0.70 <= d_tw <= 0.95
    )
    is_x_upright = (
        idx_y <= -0.20
    )
    
    if is_x_hook and is_x_curled and is_x_thumb and is_x_upright:
        if s['label'] == 'X':
            x_matches.append(s)
        else:
            fp_x.append(s)

print(f"X True Positives: {len(x_matches)}/{len([s for s in all_samples if s['label'] == 'X'])}")
print(f"X False Positives: {len(fp_x)}")
if fp_x:
    for fp in fp_x:
        print(f"  FP: label={fp['label']}, frame={fp['frame_index']}, d_TI={fp['d_TI']:.2f}, d_IW={fp['d_IW']:.2f}")

