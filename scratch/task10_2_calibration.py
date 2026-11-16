import os, sys
sys.path.insert(0, os.path.abspath('.'))
import numpy as np
import torch
import json

orig_data = np.load('backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz', allow_pickle=True)
X = orig_data['X']
y = orig_data['y']

from backend.models.alphabet_static_head import VSLStaticAlphabetHead, STATIC_CLASSES_22

ckpt = torch.load('backend/models/alphabet_static_head_v1.pt', map_location='cpu', weights_only=False)
model = VSLStaticAlphabetHead(input_dim=72, hidden_dim1=64, hidden_dim2=32, num_classes=22)
model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
model.eval()

with torch.no_grad():
    logits = model(torch.tensor(X, dtype=torch.float32))
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)

correct = (preds == y)
N = len(y)

# Multi-class Brier score
one_hot = np.zeros_like(probs)
one_hot[np.arange(N), y] = 1.0
brier_multiclass = float(np.mean(np.sum((probs - one_hot)**2, axis=1)))
brier_top1 = float(np.mean((confs - correct.astype(float))**2))

# 10 equal-width bins for ECE
n_bins = 10
bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
bin_details = []
ece = 0.0

for b in range(n_bins):
    lower = bin_edges[b]
    upper = bin_edges[b + 1]
    if b == n_bins - 1:
        mask = (confs >= lower) & (confs <= upper)
    else:
        mask = (confs >= lower) & (confs < upper)
    
    bin_size = int(np.sum(mask))
    if bin_size > 0:
        bin_acc = float(np.mean(correct[mask]))
        bin_conf = float(np.mean(confs[mask]))
        bin_weight = bin_size / N
        gap = abs(bin_acc - bin_conf)
        ece += bin_weight * gap
        bin_details.append({
            "bin": b + 1,
            "range": f"[{lower:.1f}, {upper:.1f}]",
            "count": bin_size,
            "acc": round(bin_acc, 4),
            "conf": round(bin_conf, 4),
            "gap": round(gap, 4)
        })
    else:
        bin_details.append({
            "bin": b + 1,
            "range": f"[{lower:.1f}, {upper:.1f}]",
            "count": 0,
            "acc": 0.0,
            "conf": 0.0,
            "gap": 0.0
        })

print(f"Multi-class Brier Score: {brier_multiclass:.4f}")
print(f"Top-1 Brier Score: {brier_top1:.4f}")
print(f"ECE (10 bins): {ece:.4f} ({ece*100:.2f}%)")
print("Bin details:")
for bd in bin_details:
    print(bd)

