import sys
sys.path.insert(0, '.')
import numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()
data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(data['classes'])
landmarks = data['landmarks']
y = data['y']
p_idx = classes.index('P')

print('Samples of Class P in static_v2:')
for i in range(len(y)):
    if y[i] == p_idx:
        lm = landmarks[i].tolist()
        feat = clf.extract_features(lm)
        print(f"P sample {i}: IW={feat['d_IW']:.2f}, MW={feat['d_MW']:.2f}, TI={feat['d_TI']:.2f}, TM={feat['d_TM']:.2f}, TW={feat['d_TW']:.2f}, IM={feat['d_IM']:.2f}, ang_IM={feat['ang_IM_dir']:.1f}, thumb_lat={feat['thumb_lat']:.2f}, idx_tip_y={feat['idx_tip_y']:.2f}, mid_tip_y={feat['mid_tip_y']:.2f}")

