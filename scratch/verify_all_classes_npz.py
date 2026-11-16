import os, numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
landmarks = data["landmarks"]
y = data["y"]
classes = list(data["classes"])

classifier = VSLAlphabetClassifier()

fp_count = 0
fn_count = 0
total_by_class = {}
r_correct = 0

for i in range(len(y)):
    true_label = classes[y[i]]
    total_by_class[true_label] = total_by_class.get(true_label, 0) + 1
    
    lm = landmarks[i].tolist()
    res = classifier.classify_detailed(lm)
    pred = res.get("letter")
    status = res.get("status")
    
    # Check if a non-R class was predicted as R (False Positive for R)
    if true_label != "R" and pred == "R":
        print(f"FP for R at sample {i}: true={true_label}, pred=R, res={res}")
        fp_count += 1
        
    if true_label == "R":
        if pred == "R":
            r_correct += 1

print(f"Total samples: {len(y)}")
print(f"Total R samples in dataset: {total_by_class.get('R')}")
print(f"R correctly predicted in dataset: {r_correct} (the 6 hold frames in D0511)")
print(f"False Positives for R across all other 21 classes: {fp_count}")

