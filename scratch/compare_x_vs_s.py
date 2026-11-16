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

print("COMPARING X HOLD FRAMES VS S HOLD FRAMES AND O FRAME 50:")
for idx in range(len(y)):
    m = meta[idx]
    lbl = m['label']
    fidx = m['frame_index']
    if (lbl == 'X' and 36 <= fidx <= 50) or (lbl == 'S' and 26 <= fidx <= 36) or (lbl == 'O' and fidx == 50):
        lm = lms[idx].tolist()
        feat = clf.extract_features(lm)
        print(f"{lbl} f{fidx:2d}: d_TI={feat['d_TI']:.2f}, d_TM={feat['d_TM']:.2f}, d_TW={feat['d_TW']:.2f}, "
              f"d_IW={feat['d_IW']:.2f}, d_MW={feat['d_MW']:.2f}, d_RW={feat['d_RW']:.2f}, d_PW={feat['d_PW']:.2f}, "
              f"thumb_lat={feat['thumb_lat']:.2f}, idx_curl={feat['index_curl']:.2f}, mid_curl={feat['middle_curl']:.2f}, "
              f"idx_str={feat['idx_straight']:.2f}")


