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

# Extract all features
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
    all_samples.append(feat)

print(f"Total valid samples: {len(all_samples)}")

# Analyze R candidates
r_samples = [s for s in all_samples if s['label'] == 'R']
non_r_samples = [s for s in all_samples if s['label'] != 'R']
print(f"R samples: {len(r_samples)}, Non-R samples: {len(non_r_samples)}")

# Let's inspect the crossing finger feature for R:
# In R:
# 1. Index and Middle are extended:
#    d_IW >= 1.10, d_MW >= 1.30
#    idx_straight >= 0.85, mid_straight >= 0.85
# 2. Ring and Pinky are curled:
#    d_RW <= 0.70, d_PW <= 0.60
#    (in hold frames: d_RW is 0.41-0.57, d_PW is 0.30-0.48)
# 3. Crossing:
#    Tip of Middle crosses Tip of Index or is extremely close:
#    In MediaPipe hand coordinates:
#    For right hand: MCP index (5) has x < MCP middle (9)
#    Let's check dx_im = tip_idx_x - tip_mid_x
#    and d_IM <= 0.55
# 4. Finger orientation:
#    idx_tip_y <= -0.80, mid_tip_y <= -1.00 (upright)
#    idx_horiz_ratio <= 0.45, mid_horiz_ratio <= 0.45

for s in r_samples:
    print(f"R frame {s['frame_index']:2d} (rep={s['is_representative']}): "
          f"d_IW={s['d_IW']:.2f}, d_MW={s['d_MW']:.2f}, d_RW={s['d_RW']:.2f}, d_PW={s['d_PW']:.2f}, "
          f"d_IM={s['d_IM']:.2f}, diff_x_IM={s['diff_x_IM']:.2f}, ang_IM={s['ang_IM_dir']:.1f}, "
          f"idx_y={s['idx_tip_y']:.2f}, mid_y={s['mid_tip_y']:.2f}")


