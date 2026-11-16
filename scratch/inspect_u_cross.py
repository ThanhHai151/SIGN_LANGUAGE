import sys, os
sys.path.insert(0, os.path.abspath("."))
import json
import numpy as np
from backend.vision.alphabet_classifier import AlphabetClassifier

clf = AlphabetClassifier()
npz = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = npz['classes']
y = npz['y']
lms = npz['landmarks']

u_lms = lms[y == np.where(classes == 'U')[0][0]]
v_lms = lms[y == np.where(classes == 'V')[0][0]]
r_lms = lms[y == np.where(classes == 'R')[0][0]]

print(f"Total U samples: {len(u_lms)}")
for i in range(len(u_lms)):
    f = clf.extract_features(u_lms[i].tolist())
    print(f"U[{i:2d}]: cross_ind={f['cross_indicator']:.5f}, d_IM={f['d_IM']:.3f}, diff_x_im={f['diff_x_IM']:.3f}, ang_im={f['ang_IM_dir']:.1f}")

