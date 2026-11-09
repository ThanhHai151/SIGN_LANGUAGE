import json
import numpy as np
from collections import defaultdict

log_path = "backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl"
records = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

print(f"Total lines in session 004312: {len(records)}")

active_records = [r for r in records if not r.get("idle", False)]
idle_records = [r for r in records if r.get("idle", False)]
print(f"Active frames: {len(active_records)}, Idle frames: {len(idle_records)}")

matched_records = []
unmatched_records = []

for r in active_records:
    geom = r.get("geometric", {})
    # Check if matched O
    # In task 3.7: matched means geom.get("letter") == "O" and "O" in [m[0] for m in geom.get("matched_candidates", [])]
    # Let's check how task 3.7 counted matched
    o_diag = geom.get("o_rule_diagnostics", {})
    if geom.get("letter") == "O" or (o_diag and o_diag.get("matched") is True):
        matched_records.append(r)
    else:
        unmatched_records.append(r)

print(f"Matched records: {len(matched_records)}, Unmatched records: {len(unmatched_records)}")

# Now examine features of all active frames
idx_straight_all = [r["geometric"]["features"]["idx_straight"] for r in active_records]
d_ti_all = [r["geometric"]["features"]["d_TI"] for r in active_records]

# Violations of thresholds in ALL active frames (N=349):
viol_idx_straight_all = [r for r in active_records if r["geometric"]["features"]["idx_straight"] > 0.48]
viol_d_ti_all = [r for r in active_records if r["geometric"]["features"]["d_TI"] > 0.22]
interval_idx_050_065_all = [r for r in active_records if 0.50 <= r["geometric"]["features"]["idx_straight"] <= 0.65]
interval_dti_023_030_all = [r for r in active_records if 0.23 <= r["geometric"]["features"]["d_TI"] <= 0.30]

print("\n--- ALL ACTIVE FRAMES (N=349) ---")
print(f"idx_straight > 0.48: {len(viol_idx_straight_all)} frames ({len(viol_idx_straight_all)/349*100:.2f}%)")
print(f"0.50 <= idx_straight <= 0.65: {len(interval_idx_050_065_all)} frames ({len(interval_idx_050_065_all)/349*100:.2f}%)")
print(f"d_TI > 0.22: {len(viol_d_ti_all)} frames ({len(viol_d_ti_all)/349*100:.2f}%)")
print(f"0.23 <= d_TI <= 0.30: {len(interval_dti_023_030_all)} frames ({len(interval_dti_023_030_all)/349*100:.2f}%)")

# Now examine features in UNMATCHED frames (N=296)
viol_idx_straight_unm = [r for r in unmatched_records if r["geometric"]["features"]["idx_straight"] > 0.48]
viol_d_ti_unm = [r for r in unmatched_records if r["geometric"]["features"]["d_TI"] > 0.22]
interval_idx_050_065_unm = [r for r in unmatched_records if 0.50 <= r["geometric"]["features"]["idx_straight"] <= 0.65]
interval_dti_023_030_unm = [r for r in unmatched_records if 0.23 <= r["geometric"]["features"]["d_TI"] <= 0.30]

print("\n--- UNMATCHED FRAMES (N=296) ---")
print(f"idx_straight > 0.48: {len(viol_idx_straight_unm)} frames ({len(viol_idx_straight_unm)/296*100:.2f}%)")
print(f"0.50 <= idx_straight <= 0.65: {len(interval_idx_050_065_unm)} frames ({len(interval_idx_050_065_unm)/296*100:.2f}%)")
print(f"d_TI > 0.22: {len(viol_d_ti_unm)} frames ({len(viol_d_ti_unm)/296*100:.2f}%)")
print(f"0.23 <= d_TI <= 0.30: {len(interval_dti_023_030_unm)} frames ({len(interval_dti_023_030_unm)/296*100:.2f}%)")

# Now examine condition failures in o_rule_diagnostics for unmatched frames
failed_cond_counts = defaultdict(int)
single_viol = defaultdict(int)
multi_viol_count = 0
only_idx_straight = 0
only_d_ti = 0
both_idx_and_d_ti = 0

for r in unmatched_records:
    o_diag = r.get("geometric", {}).get("o_rule_diagnostics", {})
    failed = o_diag.get("failed_conditions", [])
    for f in failed:
        failed_cond_counts[f] += 1
    if len(failed) == 1:
        single_viol[failed[0]] += 1
    elif len(failed) > 1:
        multi_viol_count += 1
    if failed == ["idx_straight_le_max"]:
        only_idx_straight += 1
    if failed == ["d_ti_le_max"]:
        only_d_ti += 1
    if "idx_straight_le_max" in failed and "d_ti_le_max" in failed:
        both_idx_and_d_ti += 1

print("\n--- CONDITION FAILURE COUNTS IN UNMATCHED (N=296) ---")
for k, v in sorted(failed_cond_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} frames ({v/296*100:.2f}%)")

print("\n--- SINGLE VS MULTI VIOLATIONS ---")
print(f"Total single-condition violations: {sum(single_viol.values())}")
for k, v in sorted(single_viol.items(), key=lambda x: -x[1]):
    print(f"  Only {k}: {v} frames")
print(f"Total multi-condition violations: {multi_viol_count} frames")
print(f"Frames violating BOTH idx_straight and d_ti: {both_idx_and_d_ti} frames")

# Check MECE partition cross-check
# In task 3.7.1, MECE partition was:
# 167 (only idx_straight), 93 (only d_ti), 24 (both), 8 (other), 3 (d_tm), 1 (d_tw)...

