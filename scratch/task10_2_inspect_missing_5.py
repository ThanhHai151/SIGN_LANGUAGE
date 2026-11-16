import json
import numpy as np
import torch
import os

WORKSPACE = os.path.abspath(".")
CKPT_PATH = os.path.join(WORKSPACE, "backend", "models", "alphabet_static_head_v1.pt")
ORIG_NPZ_PATH = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2", "datasets", "vsl_alphabet_static_v2.npz")

from backend.models.alphabet_static_head import VSLStaticAlphabetHead, STATIC_CLASSES_22

orig_data = np.load(ORIG_NPZ_PATH, allow_pickle=True)
X = orig_data["X"]
y = orig_data["y"]
meta = json.loads(str(orig_data["metadata_json"]))

ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
model = VSLStaticAlphabetHead(input_dim=72, hidden_dim1=64, hidden_dim2=32, num_classes=22)
model.load_state_dict(ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt)
model.eval()

with torch.no_grad():
    logits = model(torch.tensor(X, dtype=torch.float32))
    probs = torch.softmax(logits, dim=-1).numpy()
    preds = np.argmax(probs, axis=-1)
    confs = np.max(probs, axis=-1)

MISSING_5 = ["G", "H", "P", "T", "X"]
print("=== PERFORMANCE ON THE 5 MISSING CLASSES (G, H, P, T, X) ===")
for c_name in MISSING_5:
    c_idx = STATIC_CLASSES_22.index(c_name)
    sample_indices = np.where(y == c_idx)[0]
    c_preds = preds[sample_indices]
    c_confs = confs[sample_indices]
    c_correct = (c_preds == c_idx)
    
    n_samples = len(sample_indices)
    n_correct = int(np.sum(c_correct))
    n_incorrect = n_samples - n_correct
    acc = round(100.0 * n_correct / n_samples, 2)
    
    mean_correct_conf = round(float(np.mean(c_confs[c_correct])), 4) if n_correct > 0 else 0.0
    mean_incorrect_conf = round(float(np.mean(c_confs[~c_correct])), 4) if n_incorrect > 0 else 0.0
    
    vids = list(set(meta[i]["video_id"] for i in sample_indices))
    reg = "NORTH" if vids[0].endswith("B") else ("SOUTH" if vids[0].endswith("N") else "SHARED")
    
    print(f"\nLetter {c_name}: Video {vids[0]} ({reg}) | Support: {n_samples} | Correct: {n_correct} | Incorrect: {n_incorrect} | Acc: {acc}%")
    print(f"  Mean Conf Correct: {mean_correct_conf} | Mean Conf Incorrect: {mean_incorrect_conf}")
    
    # Check what it confused with
    for i, orig_idx in enumerate(sample_indices):
        if not c_correct[i]:
            pred_lbl = STATIC_CLASSES_22[c_preds[i]]
            print(f"  [ERROR] Index {orig_idx}: frame={meta[orig_idx]['frame_index']}, true={c_name}, pred={pred_lbl}, conf={c_confs[i]:.4f}")

