"""
Audit script for Task 2.7.2 Part C: Trial Timer and Logged Frame Boundaries
"""

import json
from collections import defaultdict

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Group records
trials = defaultdict(list)
for i, r in enumerate(records):
    tid = r.get("trial_id")
    if tid:
        trials[tid].append((i, r))

trial_keys = sorted(trials.keys())

print(f"{'Trial ID':<30} | {'Cfg(s)':<6} | {'t_prev_idle':<12} | {'t_first':<12} | {'t_last':<12} | {'t_next_idle':<12} | {'Span f-to-l':<11} | {'Max span (idle-to-idle)'}")
print("-" * 135)

for k, tid in enumerate(trial_keys):
    tf = trials[tid]
    first_idx = tf[0][0]
    last_idx = tf[-1][0]
    t_first = tf[0][1]["timestamp"]
    t_last = tf[-1][1]["timestamp"]
    span = t_last - t_first
    
    # prev idle
    t_prev_idle = records[first_idx - 1]["timestamp"] if first_idx > 0 else None
    # next idle
    t_next_idle = records[last_idx + 1]["timestamp"] if last_idx < len(records) - 1 else None
    
    t_prev_s = f"{t_prev_idle:.4f}" if t_prev_idle is not None else "None"
    t_next_s = f"{t_next_idle:.4f}" if t_next_idle is not None else "None"
    
    max_span_s = f"{t_next_idle - t_prev_idle:.2f}s" if (t_next_idle is not None and t_prev_idle is not None) else "N/A"
    
    print(f"{tid:<30} | 5.0s   | {t_prev_s:<12} | {t_first:<12.4f} | {t_last:<12.4f} | {t_next_s:<12} | {span:<10.2f}s | {max_span_s}")

