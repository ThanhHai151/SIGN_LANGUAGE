import sys
sys.path.insert(0, '.')
import os
import json
import hashlib
import collections

filepath = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"

# Compute SHA256
h = hashlib.sha256()
with open(filepath, "rb") as f:
    while chunk := f.read(65536):
        h.update(chunk)
sha256 = h.hexdigest()

# Load all lines
lines = []
with open(filepath, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            lines.append(json.loads(line))

total_lines = len(lines)
active_frames = [r for r in lines if r.get("expected_sign") == "P"]
idle_frames = [r for r in lines if r.get("expected_sign") is None]

# Identify distinct trial runs
from collections import defaultdict
trial_runs = []
curr_trial = None
curr_frames = []

for r in active_frames:
    t_id = r.get("trial_id")
    # Detect gap in frame_idx or timestamp if same trial_id repeated
    if curr_trial is None:
        curr_trial = t_id
        curr_frames = [r]
    elif t_id == curr_trial:
        # Check if there is an idle gap between them
        prev_f = curr_frames[-1]
        dt = r.get("timestamp") - prev_f.get("timestamp")
        if dt > 2.0:
            trial_runs.append((f"{curr_trial}_part1", curr_frames))
            curr_frames = [r]
            curr_trial = t_id
        else:
            curr_frames.append(r)
    else:
        trial_runs.append((curr_trial, curr_frames))
        curr_trial = t_id
        curr_frames = [r]

if curr_frames:
    name = f"{curr_trial}_part2" if any(tr[0].startswith(str(curr_trial)) for tr in trial_runs) else curr_trial
    trial_runs.append((name, curr_frames))

print(f"Total lines: {total_lines}")
print(f"Active frames: {len(active_frames)}")
print(f"Idle frames: {len(idle_frames)}")
print(f"Total trial executions detected: {len(trial_runs)}")

# Partition each trial execution into Stable Hold vs Transition/Exit
stable_frames = []
transition_frames = []

trial_summaries = []

for run_name, frames in trial_runs:
    n = len(frames)
    p_indices = [i for i, f in enumerate(frames) if f.get("final_sign") == "P"]
    neu_indices = [i for i, f in enumerate(frames) if f.get("final_sign") == "NEUTRAL"]
    
    # Kinematic partition for this trial
    # In run_01: f0..f5 hold (6), f6..f8 adjust (3), f9..f14 hold (6), f15..f26 exit (12)
    # In run_09: f0..f5 hold (6), f6..f28 exit (23)
    # In run_10 part1: f0..f8 entry/adjust (9), f9..f28 hold (20)
    # In others: 100% hold
    if "rep01" in run_name:
        for i, f in enumerate(frames):
            if (0 <= i <= 5) or (9 <= i <= 14):
                stable_frames.append(f)
            else:
                transition_frames.append(f)
    elif "rep09" in run_name:
        for i, f in enumerate(frames):
            if 0 <= i <= 5:
                stable_frames.append(f)
            else:
                transition_frames.append(f)
    elif "rep10_part1" in run_name:
        for i, f in enumerate(frames):
            if 9 <= i <= 28:
                stable_frames.append(f)
            else:
                transition_frames.append(f)
    else:
        # All frames in normal full trials are stable hold
        for f in frames:
            stable_frames.append(f)
            
    p_cnt = len(p_indices)
    neu_cnt = len(neu_indices)
    other_cnt = n - p_cnt - neu_cnt
    confirmed_p_cnt = sum(1 for f in frames if f.get("confirmed") and f.get("final_sign") == "P")
    
    trial_summaries.append({
        "run_name": run_name,
        "total_frames": n,
        "duration_sec": round(frames[-1].get("timestamp") - frames[0].get("timestamp"), 2),
        "p_count": p_cnt,
        "neutral_count": neu_cnt,
        "other_count": other_cnt,
        "p_ratio": round(p_cnt / n * 100, 1),
        "confirmed_p_events": confirmed_p_cnt,
        "dominant_letter": "P" if p_cnt > 0 else "NONE",
        "has_p_confirmed": confirmed_p_cnt > 0,
        "has_false_positive": other_cnt > 0
    })

print(f"\nStable hold frames total: {len(stable_frames)}")
print(f"Transition/exit frames total: {len(transition_frames)}")

# Stable metrics
s_p_cnt = sum(1 for f in stable_frames if f.get("final_sign") == "P")
s_neu_cnt = sum(1 for f in stable_frames if f.get("final_sign") == "NEUTRAL")
s_oth_cnt = sum(1 for f in stable_frames if f.get("final_sign") not in ("P", "NEUTRAL"))

# Transition metrics
t_p_cnt = sum(1 for f in transition_frames if f.get("final_sign") == "P")
t_neu_cnt = sum(1 for f in transition_frames if f.get("final_sign") == "NEUTRAL")
t_oth_cnt = sum(1 for f in transition_frames if f.get("final_sign") not in ("P", "NEUTRAL"))
t_conf_p = sum(1 for f in transition_frames if f.get("confirmed") and f.get("final_sign") == "P")
t_conf_oth = sum(1 for f in transition_frames if f.get("confirmed") and f.get("final_sign") not in ("P", "NEUTRAL"))

report_data = {
    "file_path": filepath,
    "file_size_bytes": os.path.getsize(filepath),
    "sha256": sha256,
    "session_id": "pilot_20261108_183703",
    "total_lines": total_lines,
    "active_frames_count": len(active_frames),
    "idle_frames_count": len(idle_frames),
    "model_status": {
        "has_static_head": False,
        "load_status": "failed",
        "pytorch_version": lines[0].get("static_head", {}).get("pytorch_version")
    },
    "stable_hold": {
        "total_frames": len(stable_frames),
        "geometric_p_match": s_p_cnt,
        "final_p": s_p_cnt,
        "neutral": s_neu_cnt,
        "misclassified": s_oth_cnt,
        "frame_accuracy_pct": round(s_p_cnt / max(1, len(stable_frames)) * 100, 2),
        "coverage_pct": round(s_p_cnt / max(1, len(stable_frames)) * 100, 2)
    },
    "transition_exit": {
        "total_frames": len(transition_frames),
        "geometric_p_match": t_p_cnt,
        "instantaneous_p": t_p_cnt,
        "confirmed_p": t_conf_p,
        "neutral": t_neu_cnt,
        "misclassified": t_oth_cnt,
        "confirmed_other_letters": t_conf_oth,
        "false_tokens_added": "NOT AVAILABLE"
    },
    "trial_level": {
        "total_trial_runs": len(trial_runs),
        "planned_trials": 10,
        "p_dominant_trials": sum(1 for ts in trial_summaries if ts["dominant_letter"] == "P"),
        "p_confirmed_trials": sum(1 for ts in trial_summaries if ts["has_p_confirmed"]),
        "false_positive_trials": sum(1 for ts in trial_summaries if ts["has_false_positive"]),
        "trial_accuracy_pct": round(sum(1 for ts in trial_summaries if ts["has_p_confirmed"]) / len(trial_summaries) * 100, 1),
        "details": trial_summaries
    }
}

with open("scratch/task1_5_1_report_data.json", "w", encoding="utf-8") as f:
    json.dump(report_data, f, indent=2, ensure_ascii=False)

print("\nSaved report data to scratch/task1_5_1_report_data.json")

