"""
Detailed audit of all confirmed token events across pilot_pilot_20261108_193712.jsonl
"""

import json

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

confirmed_events = []
for i, r in enumerate(records):
    if r.get("confirmed"):
        confirmed_events.append((i, r))

print(f"Total confirmed events in file: {len(confirmed_events)} (out of {len(records)} records)")

active_conf = [e for e in confirmed_events if e[1].get("trial_id") is not None]
idle_conf = [e for e in confirmed_events if e[1].get("trial_id") is None]
print(f"- Inside Active Trials: {len(active_conf)} events")
print(f"- Inside Idle Frames: {len(idle_conf)} events")

print("\nList of all confirmed events:")
print(f"{'Line':<5} | {'Timestamp':<14} | {'TrialID':<30} | {'Frame':<6} | {'Sign':<8} | {'Conf':<6} | {'Delta_t (s)':<12} | {'Context'}")
print("-" * 115)

prev_ts = None
for line_no, r in confirmed_events:
    ts = r.get("timestamp")
    tid = r.get("trial_id") or "IDLE"
    f_idx = r.get("frame_idx")
    sign = r.get("final_sign")
    conf = r.get("final_confidence")
    dt = (ts - prev_ts) if prev_ts is not None else 0.0
    prev_ts = ts
    
    if line_no < 1:
        ctx = "Pre-session"
    elif line_no > 134:
        ctx = "Post-session"
    elif r.get("trial_id") is None:
        ctx = "Inter-trial idle"
    else:
        ctx = "In-trial active"
        
    print(f"{line_no:<5} | {ts:<14.4f} | {tid:<30} | {f_idx:<6} | {sign:<8} | {conf:<6.2f} | {dt:<12.4f} | {ctx}")

