import json
import os
import sys

sys.path.insert(0, ".")
from backend.vision.alphabet_classifier import alphabet_classifier

files = [
    "backend/data/diagnostics/pilot_pilot_20261108_193712_20261108_211742_1.jsonl",
    "backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl",
    "backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl",
    "backend/data/diagnostics/pilot_pilot_20261109_013304.jsonl",
    "backend/data/diagnostics/pilot_pilot_20261109_022547.jsonl"
]

print(f"{'Filename':<35} | {'Active O':<8} | {'Final O':<8} | {'NEUTRAL':<8} | {'Final A':<8} | {'d_TI FAIL':<10} | {'idx_str FAIL':<12}")
print("-" * 105)

for pf in files:
    fname = os.path.basename(pf)
    if not os.path.exists(pf): continue
    active = []
    with open(pf, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("trial_id") and rec.get("expected_sign") == "O":
                active.append(rec)
    o_cnt = sum(1 for f in active if f.get("final_sign") == "O")
    neu_cnt = sum(1 for f in active if f.get("final_sign") == "NEUTRAL")
    a_cnt = sum(1 for f in active if f.get("final_sign") == "A")
    
    d_ti_fail = 0
    idx_str_fail = 0
    for f in active:
        feat = f.get("geometric", {}).get("features", {})
        if feat:
            if feat.get("d_TI", 0.0) > 0.22: d_ti_fail += 1
            if feat.get("idx_straight", 0.0) > 0.48: idx_str_fail += 1
            
    print(f"{fname:<35} | {len(active):<8} | {o_cnt:<8} | {neu_cnt:<8} | {a_cnt:<8} | {d_ti_fail:<10} | {idx_str_fail:<12}")

