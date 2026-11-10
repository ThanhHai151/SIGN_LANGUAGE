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
    all_samples.append(feat)

print("=" * 80)
print("TESTING X CANDIDATES AGAINST ALL 551 SAMPLES")
print("=" * 80)

# What are the features of X in hold frames (36-50)?
# d_TI in [0.13, 0.28]
# d_IW in [0.81, 0.95]
# d_MW in [0.69, 1.49]
# d_RW in [0.43, 0.68]
# d_PW in [0.34, 0.55]
# d_TW in [0.78, 0.83]
# idx_straight in [0.55, 0.72] (bent hook!)
# index_curl in [0.25, 0.38]

# Let's see which other classes have small d_TI (index touching thumb):
small_d_ti = [s for s in all_samples if s['d_TI'] <= 0.35]
print(f"Total samples with d_TI <= 0.35: {len(small_d_ti)}")
by_label = {}
for s in small_d_ti:
    lbl = s['label']
    by_label[lbl] = by_label.get(lbl, 0) + 1
print("Distribution of d_TI <= 0.35 by label:", sorted(by_label.items(), key=lambda x: -x[1]))


