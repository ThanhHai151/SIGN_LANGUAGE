import numpy as np
import json
import sys
sys.path.insert(0, '.')
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()
data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
meta = json.loads(str(data['metadata_json']))
lms = data['landmarks']

x_indices = [i for i, m in enumerate(meta) if m['label'] == 'X']
print(f"=== LETTER X (D0517B) SAMPLES: {len(x_indices)} ===")
for idx in x_indices:
    m = meta[idx]
    lm = lms[idx].tolist()
    feat = clf.extract_features(lm)
    fidx = m['frame_index']
    rep = m.get('is_representative')
    print(f"frame {fidx:3d} (rep={rep}): "
          f"d_TI={feat['d_TI']:.2f}, d_IW={feat['d_IW']:.2f}, d_MW={feat['d_MW']:.2f}, "
          f"d_RW={feat['d_RW']:.2f}, d_PW={feat['d_PW']:.2f}, d_TW={feat['d_TW']:.2f}, "
          f"idx_curl={feat['index_curl']:.2f}, idx_str={feat['idx_straight']:.2f}, "
          f"idx_y={feat['idx_tip_y']:.2f}, mid_y={feat['mid_tip_y']:.2f}")


