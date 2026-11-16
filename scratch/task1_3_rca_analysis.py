import json
import math
import numpy as np
from pathlib import Path
import sys

sys.path.append(".")
from backend.vision.alphabet_classifier import AlphabetClassifier

# Paths
o_json_path = Path("backend/data/alphabet_v2/static_landmarks/O.json")
a_json_path = Path("backend/data/alphabet_v2/static_landmarks/A.json")
s_json_path = Path("backend/data/alphabet_v2/static_landmarks/S.json")
c_json_path = Path("backend/data/alphabet_v2/static_landmarks/C.json")
pilot_path = Path("backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl")

# Helper for percentile dict
def get_stats(arr):
    if len(arr) == 0:
        return {}
    a = np.array(arr, dtype=float)
    return {
        "count": len(a),
        "min": round(float(np.min(a)), 4),
        "max": round(float(np.max(a)), 4),
        "mean": round(float(np.mean(a)), 4),
        "median": round(float(np.median(a)), 4),
        "p5": round(float(np.percentile(a, 5)), 4),
        "p25": round(float(np.percentile(a, 25)), 4),
        "p75": round(float(np.percentile(a, 75)), 4),
        "p95": round(float(np.percentile(a, 95)), 4),
        "std": round(float(np.std(a)), 4)
    }

# Thresholds from Rule 6
THRESHOLDS = {
    "d_ti_le_max": 0.22,
    "d_tm_le_max": 0.40,
    "d_iw_ge_min": 0.85,
    "d_mw_ge_min": 0.90,
    "d_rw_ge_min": 0.80,
    "d_pw_ge_min": 0.75,
    "idx_straight_le_max": 0.48,
    "d_tw_ge_min": 0.85
}

# 1. Parse pilot_pilot_20261109_004312.jsonl
pilot_records = []
with open(pilot_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            pilot_records.append(json.loads(line))

active_records = [r for r in pilot_records if r.get("trial_id") and r.get("expected_sign") == "O"]
print(f"Total pilot records: {len(pilot_records)}, Active frames: {len(active_records)}")

# Evaluate each active frame
frame_evaluations = []
for idx, r in enumerate(active_records):
    geom = r.get("geometric", {})
    feat = geom.get("features", {})
    
    # Outer gate flags: not i_ext and not m_ext and not r_ext and not p_ext
    i_ext = feat.get("index_ext", False)
    m_ext = feat.get("middle_ext", False)
    r_ext = feat.get("ring_ext", False)
    p_ext = feat.get("pinky_ext", False)
    outer_gate_pass = not (i_ext or m_ext or r_ext or p_ext)
    
    # 8 inner conditions
    d_ti = feat.get("d_TI", 999.0)
    d_tm = feat.get("d_TM", 999.0)
    d_iw = feat.get("d_IW", 0.0)
    d_mw = feat.get("d_MW", 0.0)
    d_rw = feat.get("d_RW", 0.0)
    d_pw = feat.get("d_PW", 0.0)
    idx_straight = feat.get("idx_straight", 999.0)
    d_tw = feat.get("d_TW", 0.0)
    
    conds = {
        "outer_gate": outer_gate_pass,
        "d_ti_le_max": d_ti <= THRESHOLDS["d_ti_le_max"],
        "d_tm_le_max": d_tm <= THRESHOLDS["d_tm_le_max"],
        "d_iw_ge_min": d_iw >= THRESHOLDS["d_iw_ge_min"],
        "d_mw_ge_min": d_mw >= THRESHOLDS["d_mw_ge_min"],
        "d_rw_ge_min": d_rw >= THRESHOLDS["d_rw_ge_min"],
        "d_pw_ge_min": d_pw >= THRESHOLDS["d_pw_ge_min"],
        "idx_straight_le_max": idx_straight <= THRESHOLDS["idx_straight_le_max"],
        "d_tw_ge_min": d_tw >= THRESHOLDS["d_tw_ge_min"]
    }
    
    # Check failed conditions
    failed = [k for k, v in conds.items() if not v]
    
    frame_evaluations.append({
        "frame_idx": idx,
        "trial_id": r.get("trial_id", "unknown"),
        "timestamp": r.get("timestamp"),
        "final_sign": r.get("final_sign"),
        "prediction_source": r.get("prediction_source"),
        "rule_matches": r.get("rule_matches", []),
        "conds": conds,
        "failed": failed,
        "feat": feat,
        "outer_flags": (i_ext, m_ext, r_ext, p_ext)
    })

# Combination Failure Statistics
n_total = len(frame_evaluations)
pass_all = [f for f in frame_evaluations if len(f["failed"]) == 0]

only_outer = [f for f in frame_evaluations if f["failed"] == ["outer_gate"]]
only_d_ti = [f for f in frame_evaluations if f["failed"] == ["d_ti_le_max"]]
only_idx_straight = [f for f in frame_evaluations if f["failed"] == ["idx_straight_le_max"]]

fail_2_or_more = [f for f in frame_evaluations if len(f["failed"]) >= 2]

# Detailed breakdown of combinations
comb_counts = {}
for f in frame_evaluations:
    key = " + ".join(sorted(f["failed"])) if f["failed"] else "PASS_ALL"
    comb_counts[key] = comb_counts.get(key, 0) + 1

# Marginal unique contribution (frames where condition X is the SOLE failure)
marginal_unique = {
    "outer_gate": len(only_outer),
    "d_ti_le_max": len(only_d_ti),
    "idx_straight_le_max": len(only_idx_straight),
}
for k in THRESHOLDS.keys():
    if k not in marginal_unique:
        marginal_unique[k] = len([f for f in frame_evaluations if f["failed"] == [k]])

print("\n=== COMBINATION FAILURE ANALYSIS (N = 349 active frames) ===")
print(f"Pass all (Rule O Match): {len(pass_all)} ({len(pass_all)/n_total*100:.2f}%)")
print(f"Fail ONLY outer_gate: {len(only_outer)} ({len(only_outer)/n_total*100:.2f}%)")
print(f"Fail ONLY d_ti_le_max: {len(only_d_ti)} ({len(only_d_ti)/n_total*100:.2f}%)")
print(f"Fail ONLY idx_straight_le_max: {len(only_idx_straight)} ({len(only_idx_straight)/n_total*100:.2f}%)")
print(f"Fail 2 or more conditions: {len(fail_2_or_more)} ({len(fail_2_or_more)/n_total*100:.2f}%)")

print("\n--- All Failure Combination Breakdown ---")
for k, v in sorted(comb_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {k}: {v} frames ({v/n_total*100:.2f}%)")

print("\n--- Unique Marginal Elimination (Sole Condition Blocking Match) ---")
for k, v in marginal_unique.items():
    print(f"  {k}: {v} frames ({v/n_total*100:.2f}%)")

# Per condition individual fail count
indiv_fails = {}
for cond_name in ["outer_gate"] + list(THRESHOLDS.keys()):
    c = sum(1 for f in frame_evaluations if not f["conds"][cond_name])
    indiv_fails[cond_name] = c
print("\n--- Individual Condition Failure Rates ---")
for k, v in indiv_fails.items():
    print(f"  {k}: {v} frames ({v/n_total*100:.2f}%)")

# Breakdown of outer gate fails
outer_sub_fails = {"index_ext": 0, "middle_ext": 0, "ring_ext": 0, "pinky_ext": 0}
for f in frame_evaluations:
    flags = f["outer_flags"]
    if flags[0]: outer_sub_fails["index_ext"] += 1
    if flags[1]: outer_sub_fails["middle_ext"] += 1
    if flags[2]: outer_sub_fails["ring_ext"] += 1
    if flags[3]: outer_sub_fails["pinky_ext"] += 1
print("\n--- Outer Gate Flags Breakdown ---")
for k, v in outer_sub_fails.items():
    print(f"  {k}: {v} frames ({v/n_total*100:.2f}%)")

# 2. Distributions on Webcam O (349 active frames)
webcam_features = {}
for feat_name in ["d_TI", "d_TM", "d_IW", "d_MW", "d_RW", "d_PW", "idx_straight", "d_TW"]:
    vals = [f["feat"].get(feat_name) for f in frame_evaluations if f["feat"].get(feat_name) is not None]
    webcam_features[feat_name] = get_stats(vals)

# Distributions on O.json (stable frames with valid landmarks)
with open(o_json_path, "r", encoding="utf-8") as f:
    o_json_raw = json.load(f)

clf = AlphabetClassifier()

o_samples = o_json_raw.get("samples", [])
o_valid_stable_feats = []
for s in o_samples:
    if s.get("phase") == "stable_pose" and s.get("landmarks"):
        lms = s["landmarks"]
        feat_dict = clf.extract_features(lms)
        o_valid_stable_feats.append(feat_dict)

print(f"\nO.json valid stable frames: {len(o_valid_stable_feats)}")
fixture_o_features = {}
for feat_name in ["d_TI", "d_TM", "d_IW", "d_MW", "d_RW", "d_PW", "idx_straight", "d_TW"]:
    vals = [f.get(feat_name) for f in o_valid_stable_feats if f.get(feat_name) is not None]
    fixture_o_features[feat_name] = get_stats(vals)

print("\n=== FEATURE COMPARISON: FIXTURE O (n=20) vs WEBCAM O (n=349) ===")
for feat_name in ["idx_straight", "d_TI", "d_TM", "d_IW", "d_MW", "d_RW", "d_PW", "d_TW"]:
    f_stat = fixture_o_features.get(feat_name, {})
    w_stat = webcam_features.get(feat_name, {})
    print(f"\nFeature: {feat_name}")
    print(f"  FIXTURE : min={f_stat.get('min')} median={f_stat.get('median')} max={f_stat.get('max')} mean={f_stat.get('mean')} p5={f_stat.get('p5')} p25={f_stat.get('p25')} p75={f_stat.get('p75')} p95={f_stat.get('p95')}")
    print(f"  WEBCAM  : min={w_stat.get('min')} median={w_stat.get('median')} max={w_stat.get('max')} mean={w_stat.get('mean')} p5={w_stat.get('p5')} p25={w_stat.get('p25')} p75={w_stat.get('p75')} p95={w_stat.get('p95')}")

# Exceedance rates
print("\n--- Current Threshold Exceedance Rates ---")
print(f"idx_straight > 0.48 on Webcam: {indiv_fails['idx_straight_le_max']}/349 ({indiv_fails['idx_straight_le_max']/349*100:.2f}%)")
print(f"idx_straight > 0.48 on Fixture: {sum(1 for f in o_valid_stable_feats if f.get('idx_straight', 0) > 0.48)}/20 (0.00%)")
print(f"d_TI > 0.22 on Webcam: {indiv_fails['d_ti_le_max']}/349 ({indiv_fails['d_ti_le_max']/349*100:.2f}%)")
print(f"d_TI > 0.22 on Fixture: {sum(1 for f in o_valid_stable_feats if f.get('d_TI', 0) > 0.22)}/20 (0.00%)")

# 3. Per-trial Breakdown on Session 004312 (10 trials)
trial_ids = sorted(list(set(f["trial_id"] for f in frame_evaluations)))
per_trial_stats = {}
print("\n=== PER-TRIAL BREAKDOWN (SESSION 004312) ===")
for tid in trial_ids:
    t_frames = [f for f in frame_evaluations if f["trial_id"] == tid]
    n_tf = len(t_frames)
    t_pass = sum(1 for f in t_frames if len(f["failed"]) == 0)
    t_fail_outer = sum(1 for f in t_frames if not f["conds"]["outer_gate"])
    t_fail_d_ti = sum(1 for f in t_frames if not f["conds"]["d_ti_le_max"])
    t_fail_idx = sum(1 for f in t_frames if not f["conds"]["idx_straight_le_max"])
    
    t_idx_vals = [f["feat"].get("idx_straight") for f in t_frames if f["feat"].get("idx_straight") is not None]
    t_d_ti_vals = [f["feat"].get("d_TI") for f in t_frames if f["feat"].get("d_TI") is not None]
    
    signs = [f["final_sign"] for f in t_frames]
    sign_counts = {s: signs.count(s) for s in set(signs)}
    
    per_trial_stats[tid] = {
        "n_frames": n_tf,
        "pass_rule_o": t_pass,
        "pass_pct": round(t_pass / n_tf * 100, 2) if n_tf > 0 else 0.0,
        "fail_outer": t_fail_outer,
        "fail_d_ti": t_fail_d_ti,
        "fail_idx_straight": t_fail_idx,
        "idx_straight_stats": get_stats(t_idx_vals),
        "d_ti_stats": get_stats(t_d_ti_vals),
        "sign_counts": sign_counts
    }
    
    print(f"Trial {tid}: {n_tf} frames | Match O: {t_pass} ({t_pass/n_tf*100:.1f}%) | Fail Outer: {t_fail_outer} | Fail d_TI: {t_fail_d_ti} | Fail idx: {t_fail_idx} | idx median: {get_stats(t_idx_vals).get('median')} | d_TI median: {get_stats(t_d_ti_vals).get('median')} | Signs: {sign_counts}")

# 4. Antagonistic Letters: A, S, C Fixtures Evaluation
antagonistic_stats = {}
for let, p in [("A", a_json_path), ("S", s_json_path), ("C", c_json_path)]:
    if not p.exists():
        continue
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    samples = data.get("samples", [])
    valid_feats = []
    for s in samples:
        if s.get("phase") == "stable_pose" and s.get("landmarks"):
            fdict = clf.extract_features(s["landmarks"])
            valid_feats.append(fdict)
    
    fails_per_cond = {cond: 0 for cond in ["outer_gate"] + list(THRESHOLDS.keys())}
    o_matches = 0
    feature_dists = {}
    for feat_name in ["d_TI", "d_TM", "d_IW", "d_MW", "d_RW", "d_PW", "idx_straight", "d_TW"]:
        vals = [f.get(feat_name) for f in valid_feats if f.get(feat_name) is not None]
        feature_dists[feat_name] = get_stats(vals)
    
    for f in valid_feats:
        i_ext = f.get("index_ext", False)
        m_ext = f.get("middle_ext", False)
        r_ext = f.get("ring_ext", False)
        p_ext = f.get("pinky_ext", False)
        og = not (i_ext or m_ext or r_ext or p_ext)
        if not og: fails_per_cond["outer_gate"] += 1
        
        m = og
        for cname, th in THRESHOLDS.items():
            fname = cname.replace("_le_max", "").replace("_ge_min", "").upper() if "d_" in cname else "idx_straight"
            val = f.get(fname)
            if "le_max" in cname:
                passed = val <= th if val is not None else False
            else:
                passed = val >= th if val is not None else False
            if not passed:
                fails_per_cond[cname] += 1
                m = False
        if m:
            o_matches += 1
            
    antagonistic_stats[let] = {
        "n_frames": len(valid_feats),
        "rule_o_matches": o_matches,
        "fails_per_cond": fails_per_cond,
        "feature_dists": feature_dists
    }

print("\n=== ANTAGONISTIC FIXTURES EVALUATION (A, S, C) ===")
for let, stat in antagonistic_stats.items():
    print(f"\nLetter {let} (n={stat['n_frames']}): Rule O Matches = {stat['rule_o_matches']}")
    print(f"  Fails per condition: {stat['fails_per_cond']}")
    print(f"  d_TI median: {stat['feature_dists'].get('d_TI', {}).get('median')} (O max: 0.22)")
    print(f"  idx_straight median: {stat['feature_dists'].get('idx_straight', {}).get('median')} (O max: 0.48)")
    print(f"  d_IW median: {stat['feature_dists'].get('d_IW', {}).get('median')} (O min: 0.85)")
    print(f"  d_MW median: {stat['feature_dists'].get('d_MW', {}).get('median')} (O min: 0.90)")

# Save all results to json
full_results = {
    "combination_results": {
        "total_active_frames": n_total,
        "pass_all_count": len(pass_all),
        "pass_all_pct": round(len(pass_all)/n_total*100, 2),
        "only_outer_count": len(only_outer),
        "only_d_ti_count": len(only_d_ti),
        "only_idx_straight_count": len(only_idx_straight),
        "fail_2_or_more_count": len(fail_2_or_more),
        "combinations": comb_counts,
        "marginal_unique": marginal_unique,
        "individual_fails": indiv_fails,
        "outer_sub_fails": outer_sub_fails
    },
    "feature_comparison": {
        "fixture_o": fixture_o_features,
        "webcam_o": webcam_features
    },
    "per_trial_breakdown": per_trial_stats,
    "antagonistic_stats": antagonistic_stats
}

with open("scratch/task1_3_full_rca_metrics.json", "w", encoding="utf-8") as f:
    json.dump(full_results, f, indent=2)

print("\nWrote full metrics to scratch/task1_3_full_rca_metrics.json")
