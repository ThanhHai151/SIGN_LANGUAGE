import os
import sys
import json
import time
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, ".")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

# ==============================================================================
# 1. EVALUATION ON 551 STATIC SAMPLES (vsl_alphabet_static_v2.npz)
# ==============================================================================
npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(npz_path, allow_pickle=True)
landmarks_all = data["landmarks"]
y_all = data["y"]
classes = [str(c) for c in data["classes"]]

total_samples = len(y_all)
static_correct = 0
static_by_letter = defaultdict(lambda: {"total": 0, "correct": 0, "neutral": 0, "wrong": 0, "wrong_preds": []})
static_cm = defaultdict(lambda: defaultdict(int))

print("=" * 80)
print(f"1. RUNNING STATIC EVALUATION ON {total_samples} SAMPLES...")
print("=" * 80)

for idx in range(total_samples):
    lm = landmarks_all[idx].tolist()
    gt_letter = classes[y_all[idx]]
    
    # Single frame prediction
    session_id = f"static_eval_{idx}"
    rec.reset_diacritic_fsm(session_id)
    rec.reset()
    res = rec.predict_landmarks(lm, mode="letters", session_id=session_id, timestamp=1.0)
    pred_sign = res.get("sign")
    
    static_by_letter[gt_letter]["total"] += 1
    static_cm[gt_letter][pred_sign] += 1
    
    if pred_sign == gt_letter:
        static_correct += 1
        static_by_letter[gt_letter]["correct"] += 1
    elif pred_sign == "NEUTRAL":
        static_by_letter[gt_letter]["neutral"] += 1
    else:
        static_by_letter[gt_letter]["wrong"] += 1
        static_by_letter[gt_letter]["wrong_preds"].append(pred_sign)

print(f"Overall Static Accuracy: {static_correct}/{total_samples} ({static_correct/total_samples*100:.2f}%)")

# ==============================================================================
# 2. STATIC HOLD SIMULATION (10 frames each) - CHECK FOR FALSE DIACRITIC TRIGGERS
# ==============================================================================
print("\n" + "=" * 80)
print("2. RUNNING 10-FRAME STATIC HOLD SIMULATION (CHECKING FALSE DIACRITIC TRIGGERS)...")
print("=" * 80)

false_diacritics = []
diacritic_targets = {"Ă", "Â", "Ê", "Ô", "Ơ", "Ư", "Đ"}

for idx in range(total_samples):
    lm = landmarks_all[idx].tolist()
    gt_letter = classes[y_all[idx]]
    session_id = f"hold_eval_{idx}"
    rec.reset_diacritic_fsm(session_id)
    rec.reset()
    
    t = 1.0
    for f in range(10):
        res = rec.predict_landmarks(lm, mode="letters", session_id=session_id, timestamp=t)
        pred = res.get("sign")
        if pred in diacritic_targets:
            false_diacritics.append({
                "sample_idx": idx,
                "gt_letter": gt_letter,
                "frame": f,
                "false_diacritic": pred,
                "source": res.get("source"),
                "details": res.get("classifier_details")
            })
            break
        t += 0.04

print(f"Total false diacritic triggers across {total_samples} held samples: {len(false_diacritics)}")
if false_diacritics:
    for fd in false_diacritics:
        print(f"  -> Sample {fd['sample_idx']} ({fd['gt_letter']}) triggered {fd['false_diacritic']} at frame {fd['frame']}")
else:
    print("  -> PERFECT ZERO FALSE POSITIVES (100% CLEAN STATIC HOLDS)!")

# ==============================================================================
# 3. REPLAY OF 7 DYNAMIC MOTION SEQUENCES
# ==============================================================================
print("\n" + "=" * 80)
print("3. REPLAYING 7 DYNAMIC DIACRITIC MOTION SEQUENCES...")
print("=" * 80)

motion_files = {
    "Ă": ("backend/data/alphabet_v2/motion_sequences/Ă.json", "A"),
    "Â": ("backend/data/alphabet_v2/motion_sequences/Â.json", "A"),
    "Ê": ("backend/data/alphabet_v2/motion_sequences/Ê.json", "E"),
    "Đ": ("backend/data/alphabet_v2/motion_sequences/Đ.json", "D"),
    "Ô": ("backend/data/alphabet_v2/motion_sequences/Ô.json", "O"),
    "Ơ": ("backend/data/alphabet_v2/motion_sequences/Ơ.json", "O"),
    "Ư": ("backend/data/alphabet_v2/motion_sequences/Ư.json", "U"),
}

dynamic_results = {}

for target, (fpath, exp_base) in motion_files.items():
    with open(fpath, "r", encoding="utf-8") as f:
        seq_data = json.load(f)
    
    seq = seq_data.get("sequence", [])
    vid_id = seq_data.get("video_id", "")
    session_id = f"dynamic_replay_{target}"
    rec.reset()
    rec.reset_diacritic_fsm(session_id)
    
    confirmed = False
    confirmed_frame = None
    confirmed_time = None
    confirmed_source = None
    base_stable_frame = None
    base_stable_sign = None
    all_preds = []
    tokens = []
    
    for item in seq:
        lms = item.get("landmarks")
        t = item.get("timestamp_sec", 0.0)
        idx = item.get("frame_index", 0)
        
        res = rec.predict_landmarks(lms, mode="letters", session_id=session_id, timestamp=t)
        sign = res.get("sign")
        src = res.get("source")
        tok = res.get("new_token")
        if tok:
            tokens.append((idx, tok))
            
        fsm = rec.diacritic_fsms.get(session_id)
        if fsm is not None:
            if base_stable_frame is None and fsm.state in ("BASE_STABLE", "DIACRITIC_CANDIDATE", "COOLDOWN"):
                base_stable_frame = idx
                base_stable_sign = fsm.current_base
            if not confirmed and (fsm.state == "COOLDOWN" or sign == target):
                confirmed = True
                confirmed_frame = idx
                confirmed_time = t
                confirmed_source = src
                
        all_preds.append((idx, sign, src))
        
    dynamic_results[target] = {
        "video_id": vid_id,
        "expected_base": exp_base,
        "total_frames": len(seq),
        "base_stable_frame": base_stable_frame,
        "base_stable_sign": base_stable_sign,
        "confirmed": confirmed,
        "confirmed_frame": confirmed_frame,
        "confirmed_time": confirmed_time,
        "confirmed_source": confirmed_source,
        "tokens": tokens,
        "pred_counts": dict(Counter([s for _, s, _ in all_preds if s != "NEUTRAL"]))
    }

print(f"{'Target':<6} | {'Video':<8} | {'Base':<6} | {'Base Stable':<14} | {'Confirmed':<10} | {'Frame':<8} | {'Status'}")
print("-" * 75)
for target, r in dynamic_results.items():
    st = "PASS" if r["confirmed"] else "FAIL (BLOCKED)"
    bs_str = f"f{r['base_stable_frame']} ({r['base_stable_sign']})" if r['base_stable_frame'] is not None else "None"
    c_str = f"f{r['confirmed_frame']}" if r['confirmed'] else "None"
    print(f"{target:<6} | {r['video_id']:<8} | {r['expected_base']:<6} | {bs_str:<14} | {str(r['confirmed']):<10} | {c_str:<8} | {st}")

# Save full results to json
full_report = {
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "static_summary": {
        "total": total_samples,
        "correct": static_correct,
        "accuracy_pct": round(static_correct / total_samples * 100, 2),
        "false_diacritic_triggers": len(false_diacritics),
        "per_letter": {k: {"total": v["total"], "correct": v["correct"], "acc_pct": round(v["correct"]/v["total"]*100, 1), "neutral": v["neutral"], "wrong": v["wrong"]} for k, v in static_by_letter.items()}
    },
    "dynamic_summary": dynamic_results
}

with open("scratch/full_batch_regression_report.json", "w", encoding="utf-8") as f:
    json.dump(full_report, f, indent=2, ensure_ascii=False)

print("\nSaved full batch regression report to scratch/full_batch_regression_report.json")

