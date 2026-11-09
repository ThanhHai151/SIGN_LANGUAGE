"""
Analyze frame sampling rates, intervals, and trial durations in pilot_pilot_20261108_193712.jsonl
"""

import json
from collections import defaultdict
import numpy as np

filepath = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
with open(filepath, "r", encoding="utf-8") as f:
    records = [json.loads(line) for line in f if line.strip()]

# Total session
t_start = records[0]["timestamp"]
t_end = records[-1]["timestamp"]
total_duration = t_end - t_start

# Inter-frame intervals
all_dts = [records[i+1]["timestamp"] - records[i]["timestamp"] for i in range(len(records)-1)]

# Group by trial
trials = defaultdict(list)
for i, r in enumerate(records):
    tid = r.get("trial_id")
    if tid:
        trials[tid].append((i, r))

print(f"Total session duration: {total_duration:.2f}s ({len(records)} records)")
print(f"Global dt: min={min(all_dts):.4f}s, max={max(all_dts):.4f}s, mean={np.mean(all_dts):.4f}s, median={np.median(all_dts):.4f}s")
print(f"Global effective sample rate: {len(records)/total_duration:.2f} records/second (~1 record every {np.mean(all_dts):.2f}s)")

print("\nPer-trial sampling analysis:")
print(f"{'Trial ID':<30} | {'Frames':<6} | {'t_start':<12} | {'t_end':<12} | {'Dur (s)':<8} | {'Min dt':<8} | {'Max dt':<8} | {'Mean dt':<8} | {'Med dt':<8}")
print("-" * 115)

trial_dts_all = []
for tid in sorted(trials.keys()):
    tf = trials[tid]
    n = len(tf)
    t0 = tf[0][1]["timestamp"]
    t1 = tf[-1][1]["timestamp"]
    dur = t1 - t0
    dts = [tf[k+1][1]["timestamp"] - tf[k][1]["timestamp"] for k in range(n-1)]
    trial_dts_all.extend(dts)
    min_dt = min(dts) if dts else 0.0
    max_dt = max(dts) if dts else 0.0
    mean_dt = np.mean(dts) if dts else 0.0
    med_dt = np.median(dts) if dts else 0.0
    print(f"{tid:<30} | {n:<6} | {t0:<12.4f} | {t1:<12.4f} | {dur:<8.2f} | {min_dt:<8.4f} | {max_dt:<8.4f} | {mean_dt:<8.4f} | {med_dt:<8.4f}")

print(f"\nWithin-trial dt across all 10 trials: min={min(trial_dts_all):.4f}s, max={max(trial_dts_all):.4f}s, mean={np.mean(trial_dts_all):.4f}s, median={np.median(trial_dts_all):.4f}s")
print(f"Effective within-trial frame rate: ~{1.0/np.mean(trial_dts_all):.2f} FPS (approx 2.4 - 2.5 HTTP inferences/second)")

# Inter-trial gap analysis
print("\nInter-trial pauses (gap between end of rep_k and start of rep_k+1):")
trial_list = sorted(trials.keys())
for k in range(len(trial_list)-1):
    prev_end = trials[trial_list[k]][-1][1]["timestamp"]
    next_start = trials[trial_list[k+1]][0][1]["timestamp"]
    gap = next_start - prev_end
    # count idle frames in this gap
    prev_line = trials[trial_list[k]][-1][0]
    next_line = trials[trial_list[k+1]][0][0]
    idle_count = next_line - prev_line - 1
    print(f"  Gap between {trial_list[k][-5:]} and {trial_list[k+1][-5:]}: {gap:.2f}s ({idle_count} idle frames)")

