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

r_indices = [i for i, m in enumerate(meta) if m['label'] == 'R']
print(f"=== LETTER R (D0511) SAMPLES: {len(r_indices)} ===")
for idx in r_indices:
    m = meta[idx]
    lm = lms[idx].tolist()
    feat = clf.extract_features(lm)
    fidx = m['frame_index']
    rep = m.get('is_representative')
    # Let's inspect landmarks 8 and 12 coordinates
    # MediaPipe: x increases to right, y increases downward
    pt8 = lm[8]
    pt12 = lm[12]
    pt5 = lm[5]
    pt9 = lm[9]
    print(f"frame {fidx:2d} (rep={rep}): "
          f"d_IW={feat['d_IW']:.2f}, d_MW={feat['d_MW']:.2f}, d_RW={feat['d_RW']:.2f}, d_PW={feat['d_PW']:.2f}, "
          f"d_IM={feat['d_IM']:.2f}, ang_IM={feat['ang_IM_dir']:.1f}, "
          f"scale={feat['palm_scale']:.3f}, idx_y={feat['idx_tip_y']:.2f}, mid_y={feat['mid_tip_y']:.2f}")


