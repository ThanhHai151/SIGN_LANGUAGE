"""
Generate complete structured data for TASK 2.7.1 Audit
"""

import json
from collections import Counter
import numpy as np

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Classify records into 5 cohorts:
# Cohort 1: Stable hold inside trial (88 frames)
# Cohort 2: Transition inside trial (20 frames)
# Cohort 3: Idle with X posture (29 frames)
# Cohort 4: Idle without X posture (5 frames)

cohorts = {
    "stable_in_trial": [],
    "trans_in_trial": [],
    "idle_with_x": [],
    "idle_without_x": []
}

# Determine trial boundaries
trials = {}
for i, r in enumerate(records):
    tid = r.get("trial_id")
    if tid:
        if tid not in trials:
            trials[tid] = []
        trials[tid].append(i)

trans_indices = set()
for tid, idxs in trials.items():
    trans_indices.add(idxs[0])
    trans_indices.add(idxs[-1])

for i, r in enumerate(records):
    tid = r.get("trial_id")
    if tid is not None:
        if i in trans_indices:
            cohorts["trans_in_trial"].append((i, r))
        else:
            cohorts["stable_in_trial"].append((i, r))
    else:
        # Idle
        # Check if posture is X or Neutral
        if r.get("final_sign") == "X":
            cohorts["idle_with_x"].append((i, r))
        else:
            cohorts["idle_without_x"].append((i, r))

print(f"Cohort counts:")
for k, v in cohorts.items():
    print(f"  {k}: {len(v)} frames")

print("\nMetrics per cohort:")
for k, v in cohorts.items():
    signs = Counter(r[1].get("final_sign") for r in v)
    confs = sum(1 for r in v if r[1].get("confirmed"))
    geom_matches = Counter(r[1].get("geometric", {}).get("status") for r in v)
    print(f"Cohort '{k}':")
    print(f"  Total frames: {len(v)}")
    print(f"  Final signs: {dict(signs)}")
    print(f"  Geom status: {dict(geom_matches)}")
    print(f"  Confirmed events: {confs}")

print("\nAudit of the 5 idle frames without X posture (lines 137..141):")
for idx, r in cohorts["idle_without_x"]:
    print(f"Line {idx}: ts={r['timestamp']}, final_sign={r['final_sign']}, conf={r['final_confidence']}, confirmed={r['confirmed']}, geom_status={r['geometric']['status']}, d_TI={r['geometric']['features'].get('d_TI')}, tip_y={r['geometric']['features'].get('idx_tip_y')}")

