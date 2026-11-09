import json
import numpy as np

records = [json.loads(line) for line in open("backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl", encoding="utf-8") if json.loads(line).get("expected_sign") == "O"]

# Group by trial
trials = {}
for r in records:
    t_id = r.get("trial_id")
    trials.setdefault(t_id, []).append(r)

jitter_d_ti = []
jitter_idx_straight = []

for t_id, t_recs in trials.items():
    d_ti_vals = [r["geometric"]["features"]["d_TI"] for r in t_recs]
    idx_vals = [r["geometric"]["features"]["idx_straight"] for r in t_recs]
    
    diff_d_ti = np.abs(np.diff(d_ti_vals))
    diff_idx = np.abs(np.diff(idx_vals))
    
    jitter_d_ti.extend(diff_d_ti)
    jitter_idx_straight.extend(diff_idx)

print(f"Jitter analysis (frame-to-frame delta) across {len(jitter_d_ti)} frame pairs:")
print(f"  d_TI delta: mean = {np.mean(jitter_d_ti):.4f}, median = {np.median(jitter_d_ti):.4f}, max = {np.max(jitter_d_ti):.4f}, p95 = {np.percentile(jitter_d_ti, 95):.4f}")
print(f"  idx_straight delta: mean = {np.mean(jitter_idx_straight):.4f}, median = {np.median(jitter_idx_straight):.4f}, max = {np.max(jitter_idx_straight):.4f}, p95 = {np.percentile(jitter_idx_straight, 95):.4f}")

