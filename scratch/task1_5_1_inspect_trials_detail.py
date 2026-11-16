import sys
import os
import json
import collections

filepath = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"

with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

# Split by trial
from collections import defaultdict
trial_segments = defaultdict(list)
current_trial = None
seg_id = 0

for r in lines:
    t_id = r.get("trial_id")
    if t_id is not None:
        if t_id != current_trial:
            current_trial = t_id
            seg_id += 1
        trial_segments[f"run_{seg_id:02d}_{current_trial}"].append(r)

print(f"Total trial runs: {len(trial_segments)}")

for run_name, frames in sorted(trial_segments.items()):
    n = len(frames)
    p_frames = [i for i, f in enumerate(frames) if f.get("final_sign") == "P"]
    neu_frames = [i for i, f in enumerate(frames) if f.get("final_sign") == "NEUTRAL"]
    
    p_cnt = len(p_frames)
    neu_cnt = len(neu_frames)
    
    # Check positions of neutral frames
    neu_positions = neu_frames
    
    # Identify entry, hold, exit
    # First contiguous block of P
    first_p = p_frames[0] if p_frames else -1
    last_p = p_frames[-1] if p_frames else -1
    
    entry_neutrals = [i for i in neu_frames if i < first_p] if first_p != -1 else neu_frames
    exit_neutrals = [i for i in neu_frames if i > last_p] if last_p != -1 else []
    interior_neutrals = [i for i in neu_frames if first_p < i < last_p]
    
    confirmed_p_frames = [i for i, f in enumerate(frames) if f.get("confirmed") and f.get("final_sign") == "P"]
    
    print(f"\n{run_name} (Total: {n} frames):")
    print(f"  P frames        : {p_cnt}/{n} ({p_cnt/n*100:.1f}%) | range=[f{first_p}..f{last_p}]")
    print(f"  NEUTRAL frames  : {neu_cnt}/{n} (entry={len(entry_neutrals)}, exit={len(exit_neutrals)}, interior={len(interior_neutrals)})")
    if interior_neutrals:
        print(f"    Interior neutral frame indices: {interior_neutrals}")
        for idx in interior_neutrals:
            f = frames[idx]
            diag = f.get("geometric", {}).get("p_rule_diagnostics", {})
            fails = diag.get("failed_conditions", [])
            print(f"      f{idx} (t={f.get('timestamp')}): failed_conditions = {fails}")
    print(f"  Confirmed P events: at frame indices {confirmed_p_frames}")

