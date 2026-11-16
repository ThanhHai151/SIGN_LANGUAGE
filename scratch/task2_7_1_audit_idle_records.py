"""
Audit script for the 34 idle records in pilot_pilot_20261108_193712.jsonl
"""

import json
import numpy as np

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"

with open(filepath, "r", encoding="utf-8") as f:
    all_lines = [json.loads(line) for line in f if line.strip()]

idle_records = []
for i, r in enumerate(all_lines):
    if r.get("trial_id") is None:
        idle_records.append((i, r))

print(f"Total idle records: {len(idle_records)} (out of {len(all_lines)} total records)")

print("\nDetailed breakdown of all 34 idle records:")
print(f"{'Line':<5} | {'Timestamp':<14} | {'FrameIdx':<8} | {'FinalSign':<10} | {'Conf':<6} | {'Confirmed':<10} | {'GeomStatus':<10} | {'d_IW':<6} | {'d_TI':<6} | {'idx_str':<8} | {'tip_y':<7} | {'Context'}")
print("-" * 125)

for line_no, r in idle_records:
    ts = r.get("timestamp")
    f_idx = r.get("frame_idx")
    sign = r.get("final_sign")
    conf = r.get("final_confidence")
    confirmed = r.get("confirmed")
    geom = r.get("geometric", {})
    geom_status = geom.get("status")
    feat = geom.get("features", {})
    d_iw = feat.get("d_IW")
    d_ti = feat.get("d_TI")
    idx_str = feat.get("idx_straight")
    tip_y = feat.get("idx_tip_y")
    
    # Determine context relative to trial lines
    if line_no < 1:
        ctx = "Pre-session (before rep01)"
    elif line_no < 19:
        ctx = "Inter-trial (rep01 -> rep02)"
    elif line_no < 30:
        ctx = "Inter-trial (rep02 -> rep03)"
    elif line_no < 43:
        ctx = "Inter-trial (rep03 -> rep04)"
    elif line_no < 55:
        ctx = "Inter-trial (rep04 -> rep05)"
    elif line_no < 67:
        ctx = "Inter-trial (rep05 -> rep06)"
    elif line_no < 80:
        ctx = "Inter-trial (rep06 -> rep07)"
    elif line_no < 94:
        ctx = "Inter-trial (rep07 -> rep08)"
    elif line_no < 110:
        ctx = "Inter-trial (rep08 -> rep09)"
    elif line_no < 123:
        ctx = "Inter-trial (rep09 -> rep10)"
    else:
        ctx = "Post-session (after rep10)"

    d_iw_s = f"{d_iw:.3f}" if d_iw is not None else "None"
    d_ti_s = f"{d_ti:.3f}" if d_ti is not None else "None"
    idx_s = f"{idx_str:.3f}" if idx_str is not None else "None"
    tip_s = f"{tip_y:.3f}" if tip_y is not None else "None"

    print(f"{line_no:<5} | {ts:<14.4f} | {f_idx:<8} | {sign:<10} | {conf:<6.2f} | {str(confirmed):<10} | {geom_status:<10} | {d_iw_s:<6} | {d_ti_s:<6} | {idx_s:<8} | {tip_s:<7} | {ctx}")

