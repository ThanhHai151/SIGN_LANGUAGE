import sys
sys.path.insert(0, '.')
import numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()
data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(data['classes'])
landmarks = data['landmarks']
y = data['y']
q_idx = classes.index('Q')

print('Samples of Class Q in static_v2:')
for i in range(len(y)):
    if y[i] == q_idx:
        lm = landmarks[i].tolist()
        feat = clf.extract_features(lm)
        scale = feat['palm_scale']
        # calculate dy and dx from MCP 5 to Tip 8
        dx = (lm[8][0] - lm[5][0]) / scale
        dy = (lm[8][1] - lm[5][1]) / scale
        print(f"Q sample {i}: IW={feat['d_IW']:.2f}, MW={feat['d_MW']:.2f}, TW={feat['d_TW']:.2f}, TI={feat['d_TI']:.2f}, idx_tip_y={feat['idx_tip_y']:.2f}, dx={dx:.2f}, dy={dy:.2f}, idx_straight={feat['idx_straight']:.2f}, mid_straight={feat['mid_straight']:.2f}")

