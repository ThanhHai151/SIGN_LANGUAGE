import os
import sys
import json
import hashlib
import numpy as np
import torch

WORKSPACE = os.path.abspath(".")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.models.alphabet_static_head import (
    VSLStaticAlphabetHead,
    STATIC_CLASSES_22
)

CKPT_PATH = os.path.join(WORKSPACE, "backend", "models", "alphabet_static_head_v1.pt")
ORIG_NPZ_PATH = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2", "datasets", "vsl_alphabet_static_v2.npz")
CLEAN_PREDS_PATH = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2", "audit", "static_head_v1_clean_predictions.json")

# 1. Load data
orig_data = np.load(ORIG_NPZ_PATH, allow_pickle=True)
X = orig_data["X"]
y = orig_data["y"]
meta = json.loads(str(orig_data["metadata_json"]))

# 2. Load model
ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

model = VSLStaticAlphabetHead(input_dim=72, hidden_dim1=64, hidden_dim2=32, num_classes=22, dropout=0.15)
model.load_state_dict(state_dict)
model.eval()

# 3. Predict
with torch.no_grad():
    logits = model(torch.tensor(X, dtype=torch.float32))
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)

correct = (preds == y)
print(f"Total: {len(y)}, Correct: {np.sum(correct)}, Incorrect: {len(y) - np.sum(correct)}, Acc: {np.mean(correct)*100:.2f}%")

# 4. Check 309 clean samples
ELIGIBLE_INDEX_RANGES = [
    (0, 98),
    (146, 256),
    (277, 324),
    (345, 379),
    (405, 420),
]
clean_indices = []
for s, e in ELIGIBLE_INDEX_RANGES:
    clean_indices.extend(range(s, e + 1))

clean_correct = correct[clean_indices]
print(f"Clean 309 sub-slice from 551: Correct: {np.sum(clean_correct)}, Acc: {np.mean(clean_correct)*100:.2f}%")

# Compare with predictions from Task 10.1
with open(CLEAN_PREDS_PATH, "r", encoding="utf-8") as f:
    t10_1_preds = json.load(f)

mismatches = 0
for clean_idx, orig_idx in enumerate(clean_indices):
    pred_551 = preds[orig_idx]
    conf_551 = confs[orig_idx]
    pred_10_1 = t10_1_preds[clean_idx]["pred_idx"]
    conf_10_1 = t10_1_preds[clean_idx]["confidence"]
    if pred_551 != pred_10_1 or abs(conf_551 - conf_10_1) > 1e-4:
        mismatches += 1
print(f"Mismatches between 551 slice and Task 10.1 clean predictions: {mismatches}")

