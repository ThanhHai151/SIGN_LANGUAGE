import sys
import os
import json
import collections

filepath = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"

with open(filepath, "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

print(f"Total lines: {len(lines)}")

# Group by session_id and trial_id sequence
segments = []
current_trial = lines[0].get("trial_id")
count = 0
start_idx = lines[0].get("frame_idx")
start_time = lines[0].get("timestamp")

for idx, r in enumerate(lines):
    t_id = r.get("trial_id")
    if t_id != current_trial:
        segments.append({
            "trial_id": current_trial,
            "session_id": lines[idx-1].get("session_id"),
            "expected_sign": lines[idx-1].get("expected_sign"),
            "count": count,
            "start_time": start_time,
            "end_time": lines[idx-1].get("timestamp"),
            "duration": (lines[idx-1].get("timestamp") or 0) - (start_time or 0)
        })
        current_trial = t_id
        count = 1
        start_idx = r.get("frame_idx")
        start_time = r.get("timestamp")
    else:
        count += 1

segments.append({
    "trial_id": current_trial,
    "session_id": lines[-1].get("session_id"),
    "expected_sign": lines[-1].get("expected_sign"),
    "count": count,
    "start_time": start_time,
    "end_time": lines[-1].get("timestamp"),
    "duration": (lines[-1].get("timestamp") or 0) - (start_time or 0)
})

print("\n=== TIMELINE OF SEGMENTS IN FILE ===")
for i, seg in enumerate(segments):
    t_str = f"{seg['trial_id']}" if seg['trial_id'] is not None else "[IDLE / NO TRIAL]"
    exp_str = f"{seg['expected_sign']}" if seg['expected_sign'] is not None else "[NONE]"
    print(f"Seg {i:2d}: {t_str:40s} | expected: {exp_str:6s} | frames: {seg['count']:3d} | duration: {seg['duration']:.2f}s | t: {seg['start_time']:.2f} -> {seg['end_time']:.2f}")

print("\n=== DISTINCT NAMED TRIALS ===")
named_trials = [s for s in segments if s['trial_id'] is not None]
for i, seg in enumerate(named_trials, 1):
    print(f"Trial {i:2d}: {seg['trial_id']} ({seg['count']} frames, {seg['duration']:.2f}s)")


