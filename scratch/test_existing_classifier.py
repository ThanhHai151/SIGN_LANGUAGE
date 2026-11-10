import json
import numpy as np
import sys, os

sys.path.insert(0, os.path.abspath('.'))
from backend.vision.feature_extractor import _shift_and_scale, calculate_distance, calculate_angle, compute_finger_states
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

ds = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
classes = list(ds['classes'])
y = ds['y']
lms = ds['landmarks']
meta = json.loads(str(ds['metadata_json']))

print("=== Testing existing classifier on all 551 samples ===")
correct = 0
total = len(y)
cm = {c: {c2: 0 for c2 in classes + ['UNKNOWN']} for c in classes}

for i in range(total):
    true_label = classes[y[i]]
    p = lms[i].tolist()
    res = clf.classify_detailed(p)
    pred_label = res.get('letter', 'UNKNOWN')
    if pred_label == true_label:
        correct += 1
    cm[true_label][pred_label] = cm[true_label].get(pred_label, 0) + 1

print(f"Current overall accuracy: {correct}/{total} ({correct/total*100:.1f}%)")
for let in ['T', 'U', 'S']:
    preds = cm[let]
    non_zero = {k: v for k, v in preds.items() if v > 0}
    print(f"Existing classifier for true '{let}': {non_zero}")

