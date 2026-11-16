import json
import os
import sys
import numpy as np

def calc_stats(values):
    if not values: return {}
    arr = np.array(values, dtype=float)
    return {
        "count": len(arr),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "p5": round(float(np.percentile(arr, 5)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p95": round(float(np.percentile(arr, 95)), 4),
    }

pf = "backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl"
total_lines = 0
active_frames = []
idle_frames = []

with open(pf, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line: continue
        total_lines += 1
        rec = json.loads(line)
        t_id = rec.get("trial_id")
        exp = rec.get("expected_sign")
        if t_id and exp == "O":
            active_frames.append(rec)
        else:
            idle_frames.append(rec)

print(f"File: {os.path.basename(pf)}")
print(f"Total lines: {total_lines}")
print(f"Active frames (expected_sign == 'O'): {len(active_frames)}")
print(f"Idle/non-trial frames: {len(idle_frames)}")

# Final sign breakdown
final_signs = [f.get("final_sign") for f in active_frames]
final_sign_counts = {s: final_signs.count(s) for s in set(final_signs)}
print("\nFinal Signs Breakdown on Active Frames:")
for s, cnt in sorted(final_sign_counts.items(), key=lambda x: -x[1]):
    print(f"  {s:<10}: {cnt:3d} / {len(active_frames)} ({cnt/len(active_frames)*100:.2f}%)")

# Prediction source breakdown
sources = [f.get("prediction_source") for f in active_frames]
source_counts = {s: sources.count(s) for s in set(sources)}
print("\nPrediction Source Breakdown:")
for s, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
    print(f"  {str(s):<25}: {cnt:3d} / {len(active_frames)} ({cnt/len(active_frames)*100:.2f}%)")

# Gating status breakdown
gating = [f.get("gating_status") for f in active_frames]
gating_counts = {g: gating.count(g) for g in set(gating)}
print("\nGating Status Breakdown:")
for g, cnt in sorted(gating_counts.items(), key=lambda x: -x[1]):
    print(f"  {str(g):<25}: {cnt:3d} / {len(active_frames)} ({cnt/len(active_frames)*100:.2f}%)")

failed_cond_tallies = {
    "outer_gate": 0,
    "d_ti_le_max": 0,
    "d_tm_le_max": 0,
    "d_iw_ge_min": 0,
    "d_mw_ge_min": 0,
    "d_rw_ge_min": 0,
    "d_pw_ge_min": 0,
    "idx_straight_le_max": 0,
    "d_tw_ge_min": 0
}

features_collected = {fn: [] for fn in ["d_TI", "d_TM", "d_IW", "d_MW", "d_RW", "d_PW", "idx_straight", "d_TW"]}

for f in active_frames:
    geom = f.get("geometric", {})
    feat = geom.get("features", {})
    if not feat:
        continue

    i_ext = feat.get("index_ext", False)
    m_ext = feat.get("middle_ext", False)
    r_ext = feat.get("ring_ext", False)
    p_ext = feat.get("pinky_ext", False)
    outer_gate = (not i_ext) and (not m_ext) and (not r_ext) and (not p_ext)

    if not outer_gate:
        failed_cond_tallies["outer_gate"] += 1

    d_ti = feat.get("d_TI", 0.0)
    d_tm = feat.get("d_TM", 0.0)
    d_iw = feat.get("d_IW", 0.0)
    d_mw = feat.get("d_MW", 0.0)
    d_rw = feat.get("d_RW", 0.0)
    d_pw = feat.get("d_PW", 0.0)
    idx_s = feat.get("idx_straight", 0.0)
    d_tw = feat.get("d_TW", 0.0)

    for fn, val in [("d_TI", d_ti), ("d_TM", d_tm), ("d_IW", d_iw), ("d_MW", d_mw), ("d_RW", d_rw), ("d_PW", d_pw), ("idx_straight", idx_s), ("d_TW", d_tw)]:
        features_collected[fn].append(val)

    if not (d_ti <= 0.22): failed_cond_tallies["d_ti_le_max"] += 1
    if not (d_tm <= 0.35): failed_cond_tallies["d_tm_le_max"] += 1
    if not (d_iw >= 0.95): failed_cond_tallies["d_iw_ge_min"] += 1
    if not (d_mw >= 0.90): failed_cond_tallies["d_mw_ge_min"] += 1
    if not (d_rw >= 0.80): failed_cond_tallies["d_rw_ge_min"] += 1
    if not (d_pw >= 0.75): failed_cond_tallies["d_pw_ge_min"] += 1
    if not (idx_s <= 0.48): failed_cond_tallies["idx_straight_le_max"] += 1
    if not (d_tw >= 0.85): failed_cond_tallies["d_tw_ge_min"] += 1

print("\nNumber of Active Frames FAILING Each Rule O Condition:")
for cond, cnt in failed_cond_tallies.items():
    print(f"  {cond:<20}: {cnt:3d} / {len(active_frames)} FAIL ({cnt/len(active_frames)*100:.2f}%) | PASS: {len(active_frames)-cnt} ({(len(active_frames)-cnt)/len(active_frames)*100:.2f}%)")

print("\nActive Frames Feature Distribution on Webcam Session 004312:")
for fn, vals in features_collected.items():
    st = calc_stats(vals)
    print(f"  {fn:<14}: min={st['min']:.3f}, max={st['max']:.3f}, mean={st['mean']:.3f}, median={st['median']:.3f}, p5={st['p5']:.3f}, p25={st['p25']:.3f}, p75={st['p75']:.3f}, p95={st['p95']:.3f}")
