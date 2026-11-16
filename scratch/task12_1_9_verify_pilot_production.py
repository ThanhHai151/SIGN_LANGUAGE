import json
import os
import sys

sys.path.insert(0, ".")

from backend.vision.alphabet_classifier import VSLAlphabetClassifier

classifier = VSLAlphabetClassifier()
jsonl_path = "backend/data/diagnostics/pilot_pilot_20261107_2141.jsonl"

total_lines = 0
active_records = []

with open(jsonl_path, "r", encoding="utf-8") as f:
    for line in f:
        total_lines += 1
        r = json.loads(line)
        tid = r.get("trial_id")
        exp = r.get("expected_sign")
        if tid is not None and exp is not None:
            active_records.append(r)

print(f"Total lines: {total_lines}, Active trial records: {len(active_records)}")

# Replay pipeline
# For each record:
# - Run classifier.evaluate_rules(feat)
# - If matches non-empty: geom_pred = matches[0][0]
# - If matches empty:
#     if original prediction_source was geometric_rule:
#         fell back (no geometric rule matched) -> fallback/supplemental rejected -> NEUTRAL
#     else:
#         use original final_sign
#
# Also check base (Post-Task-12.1.4):
# In Post-Task-12.1.4:
# If original was geometric_rule 'V' but mid_horiz > 0.50 or ang_im > 38.0 -> NEUTRAL

base_correct = 0
replay_correct = 0

target_stats = {}
all_classes = sorted(list(set(r["expected_sign"] for r in active_records)))

for c in all_classes:
    target_stats[c] = {
        "total": 0,
        "base_preds": {},
        "replay_preds": {},
        "base_correct": 0,
        "replay_correct": 0,
    }

for r in active_records:
    exp = r["expected_sign"]
    orig_final = r.get("final_sign")
    orig_source = r.get("prediction_source")
    feat = r.get("geometric", {}).get("features")
    
    # 1. Base (Post Task 12.1.4)
    # Check if original was Rule V that should be blocked
    base_pred = orig_final
    if orig_source == "geometric_rule" and orig_final == "V":
        # Check if Rule V was blocked by 12.1.4 guards
        if feat:
            mid_h = feat.get("mid_horiz_ratio", 0)
            ang_im = feat.get("ang_IM_dir", 0)
            if mid_h > 0.50 or ang_im > 38.0:
                base_pred = "NEUTRAL"
    
    # 2. Replay with current production VSLAlphabetClassifier
    matches = classifier.evaluate_rules(feat) if feat else []
    if matches:
        replay_pred = matches[0][0]
    else:
        if orig_source == "geometric_rule":
            replay_pred = "NEUTRAL"
        else:
            replay_pred = orig_final

    target_stats[exp]["total"] += 1
    target_stats[exp]["base_preds"][base_pred] = target_stats[exp]["base_preds"].get(base_pred, 0) + 1
    target_stats[exp]["replay_preds"][replay_pred] = target_stats[exp]["replay_preds"].get(replay_pred, 0) + 1
    
    if base_pred == exp:
        base_correct += 1
        target_stats[exp]["base_correct"] += 1
    if replay_pred == exp:
        replay_correct += 1
        target_stats[exp]["replay_correct"] += 1

print(f"\n--- OVERALL RESULTS ---")
print(f"Base Correct:   {base_correct} / {len(active_records)} ({base_correct/len(active_records)*100:.2f}%)")
print(f"Replay Correct: {replay_correct} / {len(active_records)} ({replay_correct/len(active_records)*100:.2f}%)")
print(f"Delta:          {replay_correct - base_correct:+d}")

print(f"\n{'Target':<6} | {'Total':<6} | {'Base Correct':<12} | {'Replay Correct':<14} | {'Delta':<8} | {'Replay Dist'}")
print("-" * 80)
for c in all_classes:
    st = target_stats[c]
    delta = st["replay_correct"] - st["base_correct"]
    print(f"{c:<6} | {st['total']:<6} | {st['base_correct']:<12} | {st['replay_correct']:<14} | {delta:<8} | {st['replay_preds']}")

print("\n--- TARGET H DETAILED DISTRIBUTION ---")
print("Base H dist:  ", target_stats["H"]["base_preds"])
print("Replay H dist:", target_stats["H"]["replay_preds"])

print("\n--- TARGET V DETAILED DISTRIBUTION ---")
print("Base V dist:  ", target_stats["V"]["base_preds"])
print("Replay V dist:", target_stats["V"]["replay_preds"])

print("\n--- TARGET K DETAILED DISTRIBUTION ---")
print("Base K dist:  ", target_stats["K"]["base_preds"])
print("Replay K dist:", target_stats["K"]["replay_preds"])

