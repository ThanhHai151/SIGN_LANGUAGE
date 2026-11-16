import json
import sys
import numpy as np
from collections import defaultdict

sys.path.insert(0, ".")
from backend.vision.alphabet_classifier import AlphabetClassifier
clf = AlphabetClassifier()

log_path = "backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl"
records = []
with open(log_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            records.append(json.loads(line))

active_records = [r for r in records if r.get("expected_sign") == "O"]
idle_records = [r for r in records if r.get("expected_sign") != "O"]

print(f"Total: {len(records)}, Active: {len(active_records)}, Idle: {len(idle_records)}")

matched_records = []
unmatched_records = []

for r in active_records:
    # Feature extraction
    feat = r.get("geometric", {}).get("features")
    if not feat and r.get("landmarks"):
        feat = clf.extract_features(r["landmarks"])
    rule_eval = clf.evaluate_rules(feat)
    is_o = any(m[0] == "O" for m in rule_eval)
    if is_o:
        matched_records.append(r)
    else:
        unmatched_records.append(r)

print(f"Active matched: {len(matched_records)}, Active unmatched: {len(unmatched_records)}")

# Exact feature counts for Active (N=349)
idx_act = [r["geometric"]["features"]["idx_straight"] for r in active_records]
dti_act = [r["geometric"]["features"]["d_TI"] for r in active_records]

# 1. idx_straight > 0.48
idx_gt_048_act = [x for x in idx_act if x > 0.48]
# 2. 0.50 <= idx_straight <= 0.65
idx_int_050_065_act = [x for x in idx_act if 0.50 <= x <= 0.65]
# 3. d_TI > 0.22
dti_gt_022_act = [x for x in dti_act if x > 0.22]
# 4. 0.23 <= d_TI <= 0.30
dti_int_023_030_act = [x for x in dti_act if 0.23 <= x <= 0.30]

print("\n--- EXACT FEATURE COUNTS FOR ALL ACTIVE FRAMES (N=349) ---")
print(f"idx_straight > 0.48:          {len(idx_gt_048_act)} frames ({len(idx_gt_048_act)/349*100:.2f}%)")
print(f"0.50 <= idx_straight <= 0.65: {len(idx_int_050_065_act)} frames ({len(idx_int_050_065_act)/349*100:.2f}%)")
print(f"d_TI > 0.22:                  {len(dti_gt_022_act)} frames ({len(dti_gt_022_act)/349*100:.2f}%)")
print(f"0.23 <= d_TI <= 0.30:         {len(dti_int_023_030_act)} frames ({len(dti_int_023_030_act)/349*100:.2f}%)")

# Exact feature counts for UNMATCHED / REJECTED Frames (N=296)
idx_unm = [r["geometric"]["features"]["idx_straight"] for r in unmatched_records]
dti_unm = [r["geometric"]["features"]["d_TI"] for r in unmatched_records]

idx_gt_048_unm = [x for x in idx_unm if x > 0.48]
idx_int_050_065_unm = [x for x in idx_unm if 0.50 <= x <= 0.65]
dti_gt_022_unm = [x for x in dti_unm if x > 0.22]
dti_int_023_030_unm = [x for x in dti_unm if 0.23 <= x <= 0.30]

print("\n--- EXACT FEATURE COUNTS FOR REJECTED FRAMES (N=296) ---")
print(f"idx_straight > 0.48:          {len(idx_gt_048_unm)} frames ({len(idx_gt_048_unm)/296*100:.2f}%)")
print(f"0.50 <= idx_straight <= 0.65: {len(idx_int_050_065_unm)} frames ({len(idx_int_050_065_unm)/296*100:.2f}%)")
print(f"d_TI > 0.22:                  {len(dti_gt_022_unm)} frames ({len(dti_gt_022_unm)/296*100:.2f}%)")
print(f"0.23 <= d_TI <= 0.30:         {len(dti_int_023_030_unm)} frames ({len(dti_int_023_030_unm)/296*100:.2f}%)")

# Check MATCHED frames (N=53) just to be 100% thorough
idx_mat = [r["geometric"]["features"]["idx_straight"] for r in matched_records]
dti_mat = [r["geometric"]["features"]["d_TI"] for r in matched_records]
print("\n--- MATCHED FRAMES (N=53) ---")
print(f"idx_straight > 0.48: {len([x for x in idx_mat if x > 0.48])} (must be 0)")
print(f"0.50 <= idx_straight <= 0.65: {len([x for x in idx_mat if 0.50 <= x <= 0.65])} (must be 0)")
print(f"d_TI > 0.22: {len([x for x in dti_mat if x > 0.22])} (must be 0)")
print(f"0.23 <= d_TI <= 0.30: {len([x for x in dti_mat if 0.23 <= x <= 0.30])} (must be 0)")

# Multi-condition breakdowns for the 296 rejected frames:
# Categorize every rejected frame by its violations:
# Violated conditions: idx_straight > 0.48, d_TI > 0.22, outer_gate
buckets = defaultdict(list)
for r in unmatched_records:
    f = r["geometric"]["features"]
    idx_v = f["idx_straight"] > 0.48
    dti_v = f["d_TI"] > 0.22
    # Outer gate:
    # check states or check if any finger extended
    # Let's extract states from landmarks
    lm = r["landmarks"]
    from backend.vision.feature_extractor import extract_dual_hand_features
    _, meta = extract_dual_hand_features(lm, None)
    st = meta.get("primary_states", {})
    og_fail = (st.get("index_extended", False) or st.get("middle_extended", False) or 
               st.get("ring_extended", False) or st.get("pinky_extended", False))
    
    b_name = []
    if idx_v: b_name.append("idx_straight > 0.48")
    if dti_v: b_name.append("d_TI > 0.22")
    if og_fail: b_name.append("OuterGate")
    key = " + ".join(b_name) if b_name else "Other"
    buckets[key].append(r)

print("\n--- REJECTED FRAMES MECE VERIFICATION ---")
for k, v in sorted(buckets.items(), key=lambda x: -len(x[1])):
    print(f"  {k}: {len(v)} frames ({len(v)/296*100:.2f}%)")
