import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from backend.vision.recognizer import VSLRecognizer

def run_legacy_baseline():
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    if not os.path.exists(npz_path):
        print("NPZ not found:", npz_path)
        return None
    data = np.load(npz_path, allow_pickle=True)
    classes = list(data["classes"])
    landmarks = data["landmarks"]
    y = data["y"]
    meta = json.loads(str(data["metadata_json"]))

    rec = VSLRecognizer()

    total = len(y)
    correct = 0
    per_class = {c: {"total": 0, "correct": 0, "sources": {}, "confusions": {}} for c in classes}

    for i in range(total):
        target = classes[y[i]]
        lm = landmarks[i].tolist()
        res = rec.predict_landmarks(lm, mode="letters", session_id=f"npz_eval_{target}_{i}")
        pred = res["sign"]
        src = res.get("source", "none")
        per_class[target]["total"] += 1
        per_class[target]["sources"][src] = per_class[target]["sources"].get(src, 0) + 1
        if pred == target:
            correct += 1
            per_class[target]["correct"] += 1
        else:
            per_class[target]["confusions"][pred] = per_class[target]["confusions"].get(pred, 0) + 1

    print("=== LEGACY BASELINE (vsl_alphabet_static_v2.npz, 551 samples) ===")
    print(f"Overall Accuracy: {correct}/{total} ({correct/total*100:.2f}%)")
    print(f"{'Class':<6} | {'Total':<6} | {'Correct':<8} | {'Recall':<8} | {'Sources':<25} | {'Top Confusions'}")
    print("-" * 80)
    for c in classes:
        st = per_class[c]
        t = st["total"]
        corr = st["correct"]
        rec_pct = (corr / t * 100) if t > 0 else 0.0
        src_str = ", ".join(f"{k}:{v}" for k, v in st["sources"].items())
        conf_str = ", ".join(f"{k}:{v}" for k, v in st["confusions"].items())
        print(f"{c:<6} | {t:<6} | {corr:<8} | {rec_pct:<7.1f}% | {src_str:<25} | {conf_str}")
    return per_class

def run_canonical_baseline():
    p = "backend/data/alphabet_v2/static_landmarks"
    canonical_files = {
        "H": ("D0500N.json", "D0500N (South backup)"),
        "O": ("O.json", "D0506 (Nationwide)"),
        "Q": ("Q.json", "D0510 (Nationwide)"),
        "R": ("R.json", "D0511 (Nationwide)"),
        "M": ("M.json", "D0504 (Nationwide)"),
        "N": ("N.json", "D0505 (Nationwide)"),
    }
    rec = VSLRecognizer()

    print("\n=== CANONICAL EVALUATION BASELINE (6 Target Letters) ===")
    print(f"{'Letter':<6} | {'Source / Video':<24} | {'Total':<6} | {'Valid':<6} | {'Correct':<8} | {'Accuracy':<8} | {'Predictions / Confusions'}")
    print("-" * 90)

    for target, (fname, desc) in canonical_files.items():
        fp = os.path.join(p, fname)
        if not os.path.exists(fp):
            print(f"{target:<6} | {desc:<24} | NOT FOUND")
            continue
        with open(fp, "r", encoding="utf-8") as jf:
            d = json.load(jf)
        samples = d.get("samples", [])
        total = len(samples)
        valid_samples = [s for s in samples if s.get("landmarks") is not None and s.get("quality_status") == "ok"]
        valid_count = len(valid_samples)
        correct = 0
        preds = {}
        sources = {}

        for s in valid_samples:
            lm = s["landmarks"]
            res = rec.predict_landmarks(lm, mode="letters", session_id=f"canon_eval_{target}")
            sign = res["sign"]
            src = res.get("source", "none")
            preds[sign] = preds.get(sign, 0) + 1
            sources[src] = sources.get(src, 0) + 1
            if sign == target:
                correct += 1

        acc_pct = (correct / valid_count * 100) if valid_count > 0 else 0.0
        preds_str = ", ".join(f"{k}:{v}" for k, v in preds.items())
        print(f"{target:<6} | {desc:<24} | {total:<6} | {valid_count:<6} | {correct:<8} | {acc_pct:<7.1f}% | {preds_str}")

if __name__ == "__main__":
    run_legacy_baseline()
    run_canonical_baseline()
