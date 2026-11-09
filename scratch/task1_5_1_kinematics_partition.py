import sys
sys.path.insert(0, '.')
import os
import json
import numpy as np
import collections

filepath = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"

with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

# Group active trial frames by trial
trials_raw = collections.defaultdict(list)
current_trial = None
trial_run_id = 0
trial_runs = []

for r in lines:
    t_id = r.get("trial_id")
    if t_id is not None:
        if t_id != current_trial:
            current_trial = t_id
            trial_run_id += 1
            trial_runs.append((f"run_{trial_run_id:02d}_{current_trial}", []))
        trial_runs[-1][1].append(r)

print(f"Total trial runs to analyze: {len(trial_runs)}")

# Inspect motion speed (landmark Euclidean displacement between consecutive frames)
all_analyzed_frames = []

for run_name, frames in trial_runs:
    n = len(frames)
    t_start = frames[0].get("timestamp")
    t_end = frames[-1].get("timestamp")
    duration = t_end - t_start
    
    # Calculate wrist velocity (landmark 0 displacement per second)
    wrist_speeds = []
    for i in range(n):
        lms_curr = frames[i].get("geometric", {}).get("features", {})
        # Use wrist coordinates if available or feat
        if i == 0:
            wrist_speeds.append(0.0)
        else:
            dt = frames[i].get("timestamp") - frames[i-1].get("timestamp")
            # We can check stability of idx_tip_y and d_IW
            curr_y = float(frames[i].get("geometric", {}).get("features", {}).get("idx_tip_y", 0.0))
            prev_y = float(frames[i-1].get("geometric", {}).get("features", {}).get("idx_tip_y", 0.0))
            dy_speed = abs(curr_y - prev_y) / max(0.01, dt)
            wrist_speeds.append(dy_speed)
            
    # Print trial timeline
    p_indices = [i for i, f in enumerate(frames) if f.get("final_sign") == "P"]
    print(f"\n{run_name}: {n} frames, {duration:.2f}s, P count = {len(p_indices)}")
    
    # Detailed per-frame kinematic view
    for i, f in enumerate(frames):
        ts = f.get("timestamp") - t_start
        sign = f.get("final_sign")
        conf = f.get("final_confidence")
        feat = f.get("geometric", {}).get("features", {})
        tip_y = float(feat.get("idx_tip_y", 0.0))
        d_iw = float(feat.get("d_IW", 0.0))
        ang = float(feat.get("ang_IM_dir", 0.0))
        horiz = float(feat.get("idx_horiz_ratio", 0.0))
        spd = wrist_speeds[i]
        
        # Determine kinematic phase:
        # Stable hold: hand is in posture, small kinematic motion, tip_y pointing down >= 0.50
        # Entry: before stable hold is established
        # Exit: when hand lowers (e.g. rep09 f6..f28 when signer lowers hand)
        is_p_match = (sign == "P")
        
        all_analyzed_frames.append({
            "run_name": run_name,
            "trial_id": f.get("trial_id"),
            "frame_idx_in_trial": i,
            "frame_idx_global": f.get("frame_idx"),
            "rel_time": ts,
            "sign": sign,
            "confidence": conf,
            "confirmed": f.get("confirmed"),
            "new_token": f.get("new_token"),
            "tip_y": tip_y,
            "ang": ang,
            "horiz": horiz,
            "speed": spd,
            "raw_frame": f
        })

print(f"\nTotal frames analyzed across all runs: {len(all_analyzed_frames)}")

