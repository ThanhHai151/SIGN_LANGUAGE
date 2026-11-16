import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()
print(f"Recognizer live threshold Q: {rec.supplemental_thresholds.get('Q')}")

# 1. Canonical Q (D0510)
q_file = "backend/data/alphabet_v2/static_landmarks/Q.json"
with open(q_file, "r", encoding="utf-8") as f:
    q_data = json.load(f)
valid_q = [s for s in q_data["samples"] if s.get("landmarks") and s.get("quality_status") == "ok"]

corr_q = 0
conf_q = {}
sources_q = {}
for s in valid_q:
    res = rec.predict_landmarks(s["landmarks"], mode="letters", session_id="replay_q")
    p_sign = res["sign"]
    src = res.get("source", "none")
    conf_q[p_sign] = conf_q.get(p_sign, 0) + 1
    sources_q[src] = sources_q.get(src, 0) + 1
    if p_sign == "Q":
        corr_q += 1

print(f"Canonical Q: {corr_q}/{len(valid_q)} ({corr_q/len(valid_q)*100:.1f}%) | Predictions: {conf_q} | Sources: {sources_q}")

# 2. Controls P, X, G
for let, fname in [("P", "P.json"), ("X", "X.json"), ("G", "G.json")]:
    fp = os.path.join("backend/data/alphabet_v2/static_landmarks", fname)
    with open(fp, "r", encoding="utf-8") as f:
        d = json.load(f)
    valid = [s for s in d["samples"] if s.get("landmarks") and s.get("quality_status") == "ok"]
    q_mis = 0
    let_corr = 0
    preds = {}
    for s in valid:
        res = rec.predict_landmarks(s["landmarks"], mode="letters", session_id=f"replay_{let}")
        p_sign = res["sign"]
        preds[p_sign] = preds.get(p_sign, 0) + 1
        if p_sign == "Q":
            q_mis += 1
        if p_sign == let:
            let_corr += 1
    print(f"Canonical {let}: correct={let_corr}/{len(valid)}, misclassified_as_Q={q_mis}/{len(valid)} ({q_mis/len(valid)*100:.1f}%) | Predictions: {preds}")

# 3. Legacy 551 Dataset
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
classes = list(data["classes"])
landmarks = data["landmarks"]
y = data["y"]

overall_corr = 0
per_class_corr = {c: 0 for c in classes}
per_class_total = {c: 0 for c in classes}
per_class_mis_to_q = {c: 0 for c in classes}
q_preds_legacy = {}

for i in range(len(y)):
    target = classes[y[i]]
    lm = landmarks[i].tolist()
    res = rec.predict_landmarks(lm, mode="letters", session_id=f"legacy_replay_{i}")
    p_sign = res["sign"]
    per_class_total[target] += 1
    if p_sign == target:
        overall_corr += 1
        per_class_corr[target] += 1
    elif p_sign == "Q":
        per_class_mis_to_q[target] += 1
    if target == "Q":
        q_preds_legacy[p_sign] = q_preds_legacy.get(p_sign, 0) + 1

print(f"\nLegacy Overall: {overall_corr}/{len(y)} ({overall_corr/len(y)*100:.2f}%)")
print(f"Legacy Q Recall: {per_class_corr['Q']}/{per_class_total['Q']} ({per_class_corr['Q']/per_class_total['Q']*100:.1f}%) | Preds: {q_preds_legacy}")
print(f"P -> Q: {per_class_mis_to_q['P']}/{per_class_total['P']}")
print(f"X -> Q: {per_class_mis_to_q['X']}/{per_class_total['X']}")
print(f"G -> Q: {per_class_mis_to_q['G']}/{per_class_total['G']}")
other_q = {c: per_class_mis_to_q[c] for c in classes if c != "Q" and per_class_mis_to_q[c] > 0}
print(f"Other letters -> Q: {other_q}")

# Detailed 22 letters comparison vs baseline
baseline_counts = {
    'A': 25, 'B': 25, 'C': 25, 'D': 25, 'E': 25, 'G': 24, 'H': 24, 'I': 25,
    'K': 23, 'L': 25, 'M': 22, 'N': 17, 'O': 13, 'P': 20, 'Q': 14, 'R': 11,
    'S': 25, 'T': 24, 'U': 23, 'V': 25, 'X': 22, 'Y': 24
}
print("\n--- Detailed Comparison of 22 Letters vs Baseline 451/551 ---")
for c in classes:
    b = baseline_counts.get(c, 0)
    cur = per_class_corr[c]
    tot = per_class_total[c]
    diff = cur - b
    diff_str = f"+{diff}" if diff > 0 else (f"{diff}" if diff < 0 else "0")
    print(f"Letter {c:<2}: {cur:>2}/{tot:>2} ({cur/tot*100:>5.1f}%) | Baseline: {b:>2}/{tot:>2} | Change: {diff_str}")

