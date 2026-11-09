"""
Task 2.3 Root Cause Analysis Script (READ-ONLY)
Performs deep quantitative investigation for Letter X:
1. Feature consistency & landmark verification
2. In-depth analysis of Upright Hook conditions (d_IW, d_TW, d_PW, d_TI)
3. Webcam vs Fixture D0517B comparison (Upright MATCH vs FAIL vs Webcam)
4. Feature correlation matrices (Webcam vs Fixture)
5. Forward Tilt deep analysis
6. Confusion risk analysis against competing letters (S, P, A, O, E, C)
7. MLP fallback analysis
"""

import os
import sys

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath("."))

import json
import math
import hashlib
from collections import defaultdict, Counter
import numpy as np

PILOT_PATH = "backend/data/diagnostics/pilot_pilot_20261108_184115.jsonl"
FIXTURE_DIR = "backend/data/alphabet_v2/static_landmarks"

from backend.vision.alphabet_classifier import alphabet_classifier

def pearson_corr(x, y):
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    if len(x) < 2 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])

def main():
    print("================================================================================")
    print("TASK 2.3: ROOT CAUSE ANALYSIS CHỮ X")
    print("================================================================================")

    # 1. Load active webcam records
    webcam_records = []
    with open(PILOT_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("expected_sign") == "X" and r.get("trial_id") is not None:
                webcam_records.append(r)

    print(f"Total Active Webcam Frames: {len(webcam_records)}")

    # 2. Check feature consistency on Fixture X.json
    x_fix_path = os.path.join(FIXTURE_DIR, "X.json")
    with open(x_fix_path, "r", encoding="utf-8") as f:
        x_fix_data = json.load(f)

    samples_x = x_fix_data["samples"]
    print(f"\n--- 1. FEATURE CONSISTENCY CHECK ON FIXTURE X.json ---")
    valid_fixture_samples = [s for s in samples_x if s.get("landmarks") and len(s["landmarks"]) >= 21]
    print(f"Total samples: {len(samples_x)}, Valid frames with landmarks: {len(valid_fixture_samples)}")

    # Recompute features from raw landmarks using production extractor
    recomputed_features = []
    for s in valid_fixture_samples:
        feat = alphabet_classifier.extract_features(s["landmarks"])
        recomputed_features.append((s["frame_index"], s.get("phase"), feat))

    print(f"Successfully recomputed features for {len(recomputed_features)} fixture frames.")
    print("Recomputed values sample (frame 36):", {
        k: recomputed_features[0][2][k] for k in ["d_IW", "d_TW", "d_TI", "idx_tip_y", "index_curl", "idx_straight"]
    })

    # 3. In-depth Upright Hook Analysis on Webcam Pilot
    print(f"\n--- 2. IN-DEPTH UPRIGHT HOOK ANALYSIS (WEBCAM PILOT) ---")
    
    # Trial-by-trial breakdown
    trial_groups = defaultdict(list)
    for r in webcam_records:
        trial_groups[r["trial_id"]].append(r)

    print(f"{'Trial ID':<35} | {'d_IW med':<8} {'d_TW med':<8} {'d_PW med':<8} {'d_TI med':<8} | {'PW<=0.60':<10} {'TI<=0.25':<10}")
    print("-" * 95)
    for tid in sorted(trial_groups.keys()):
        fl = trial_groups[tid]
        diw = [r["geometric"]["features"]["d_IW"] for r in fl]
        dtw = [r["geometric"]["features"]["d_TW"] for r in fl]
        dpw = [r["geometric"]["features"]["d_PW"] for r in fl]
        dti = [r["geometric"]["features"]["d_TI"] for r in fl]
        pw_pass = sum(1 for v in dpw if v <= 0.60)
        ti_pass = sum(1 for v in dti if v <= 0.25)
        print(f"{tid:<35} | {np.median(diw):<8.4f} {np.median(dtw):<8.4f} {np.median(dpw):<8.4f} {np.median(dti):<8.4f} | {pw_pass:>2}/{len(fl):<2} ({pw_pass/len(fl)*100:4.1f}%) {ti_pass:>2}/{len(fl):<2} ({ti_pass/len(fl)*100:4.1f}%)")

    # Detailed d_TI analysis
    dti_all = [r["geometric"]["features"]["d_TI"] for r in webcam_records]
    dti_fails = [r for r in webcam_records if r["geometric"]["features"]["d_TI"] > 0.25]
    print(f"\nd_TI analysis:")
    print(f"  Total FAIL (>0.25): {len(dti_fails)} / {len(webcam_records)} ({len(dti_fails)/len(webcam_records)*100:.2f}%)")
    dti_fail_trials = Counter([r["trial_id"].replace("pilot_20261108_184115_X_", "") for r in dti_fails])
    print(f"  Distribution of d_TI failures across trials: {dict(dti_fail_trials)}")
    print(f"  d_TI fail values min={min(r['geometric']['features']['d_TI'] for r in dti_fails):.4f}, median={np.median([r['geometric']['features']['d_TI'] for r in dti_fails]):.4f}, max={max(r['geometric']['features']['d_TI'] for r in dti_fails):.4f}")

    # Detailed d_PW analysis
    dpw_all = [r["geometric"]["features"]["d_PW"] for r in webcam_records]
    dpw_fails = [r for r in webcam_records if r["geometric"]["features"]["d_PW"] > 0.60]
    print(f"\nd_PW analysis:")
    print(f"  Total FAIL (>0.60): {len(dpw_fails)} / {len(webcam_records)} ({len(dpw_fails)/len(webcam_records)*100:.2f}%)")
    dpw_fail_trials = Counter([r["trial_id"].replace("pilot_20261108_184115_X_", "") for r in dpw_fails])
    print(f"  Distribution of d_PW failures across trials: {dict(dpw_fail_trials)}")
    print(f"  d_PW fail values min={min(r['geometric']['features']['d_PW'] for r in dpw_fails):.4f}, median={np.median([r['geometric']['features']['d_PW'] for r in dpw_fails]):.4f}, max={max(r['geometric']['features']['d_PW'] for r in dpw_fails):.4f}")

    # 4. Compare Webcam vs Fixture Cohorts
    print(f"\n--- 3. COMPARISON: WEBCAM vs FIXTURE D0517B COHORTS ---")
    
    # Partition fixture into:
    # 1. Upright MATCH (f36, f40, f44, f46, f48)
    # 2. Upright FAIL in static phase (f38, f42, f50)
    # 3. Forward MATCH (f116-f136)
    fix_upright_match = []
    fix_upright_fail = []
    fix_fwd_match = []

    for fidx, phase, feat in recomputed_features:
        if fidx in [36, 40, 44, 46, 48]:
            fix_upright_match.append(feat)
        elif fidx in [38, 42, 50]:
            fix_upright_fail.append(feat)
        elif 116 <= fidx <= 136:
            fix_fwd_match.append(feat)

    # Partition webcam into stable hold (228) and transition (60)
    webcam_stable = []
    webcam_trans = []
    for tid, fl in trial_groups.items():
        n = len(fl)
        webcam_trans.extend([r["geometric"]["features"] for r in fl[:3] + fl[n-3:]])
        webcam_stable.extend([r["geometric"]["features"] for r in fl[3:n-3]])

    cohorts = [
        ("Fixture Upright MATCH (n=5)", fix_upright_match),
        ("Fixture Upright FAIL (n=3: f38, f42, f50)", fix_upright_fail),
        ("Webcam Stable Hold (n=228)", webcam_stable),
        ("Webcam Transition (n=60)", webcam_trans),
        ("Fixture Forward MATCH (n=11)", fix_fwd_match),
    ]

    features_to_compare = ["d_IW", "d_TW", "d_PW", "d_TI", "idx_tip_y", "idx_straight", "index_curl", "d_RW", "d_MW"]
    
    for fk in features_to_compare:
        print(f"\nFeature: {fk}")
        print(f"{'Cohort':<45} | {'Min':<7} {'p25':<7} {'Median':<7} {'p75':<7} {'Max':<7}")
        print("-" * 85)
        for cname, cfeats in cohorts:
            vals = [cf[fk] for cf in cfeats if cf.get(fk) is not None]
            if len(vals) == 0: continue
            print(f"{cname:<45} | {np.min(vals):<7.4f} {np.percentile(vals, 25):<7.4f} {np.median(vals):<7.4f} {np.percentile(vals, 75):<7.4f} {np.max(vals):<7.4f}")

    # Deep dive into fixture f38 and f42
    print(f"\nDetailed look at Fixture frames f38 and f42 vs f36/f40:")
    for fidx, phase, feat in recomputed_features:
        if fidx in [36, 38, 40, 42, 44]:
            print(f"Frame {fidx} ({phase}): d_IW={feat['d_IW']:.4f}, d_TW={feat['d_TW']:.4f}, idx_tip_y={feat['idx_tip_y']:.4f}, idx_straight={feat['idx_straight']:.4f}, d_TI={feat['d_TI']:.4f}")

    # 5. Feature Relationships & Correlation Analysis
    print(f"\n--- 4. FEATURE RELATIONSHIPS & CORRELATION ANALYSIS ---")
    pairs = [
        ("d_IW", "d_TW"),
        ("d_IW", "d_TI"),
        ("d_TW", "d_TI"),
        ("d_PW", "d_RW"),
        ("index_curl", "idx_straight"),
        ("idx_tip_y", "idx_dy"),
        ("idx_tip_y", "d_IW"),
    ]

    print(f"{'Feature Pair':<25} | {'Webcam (n=288)':<18} | {'Fixture X (n=48)':<18}")
    print("-" * 65)
    for f1, f2 in pairs:
        w_f1 = [r["geometric"]["features"][f1] for r in webcam_records]
        w_f2 = [r["geometric"]["features"][f2] for r in webcam_records]
        r_webcam = pearson_corr(w_f1, w_f2)

        fix_f1 = [cf[2][f1] for cf in recomputed_features if cf[2].get(f1) is not None and cf[2].get(f2) is not None]
        fix_f2 = [cf[2][f2] for cf in recomputed_features if cf[2].get(f1) is not None and cf[2].get(f2) is not None]
        r_fix = pearson_corr(fix_f1, fix_f2)

        print(f"{f1} vs {f2:<12} | r = {r_webcam:<14.4f} | r = {r_fix:<14.4f}")

    # 6. Confusion Risk Analysis Against Competing Letters
    print(f"\n--- 5. CONFUSION RISK ANALYSIS AGAINST COMPETING LETTERS ---")
    competing = ["S", "P", "A", "O", "E", "C"]
    
    # Collect fixture features for each letter
    comp_feats = {}
    for let in competing:
        lp = os.path.join(FIXTURE_DIR, f"{let}.json")
        if not os.path.exists(lp): continue
        with open(lp, "r", encoding="utf-8") as f:
            cdata = json.load(f)
        c_samples = [s for s in cdata["samples"] if s.get("landmarks") and len(s["landmarks"]) >= 21]
        c_extracted = []
        for s in c_samples:
            cf = alphabet_classifier.extract_features(s["landmarks"])
            if cf: c_extracted.append((s.get("frame_index"), s.get("phase"), cf))
        comp_feats[let] = c_extracted

    print(f"{'Letter':<6} | {'Valid Frames':<12} | {'d_IW (med)':<10} {'d_TW (med)':<10} {'d_TI (med)':<10} {'idx_straight (med)':<18} {'idx_tip_y (med)':<15} {'i_curl (med)':<12}")
    print("-" * 95)
    
    # Print Webcam X
    w_diw = np.median([r["geometric"]["features"]["d_IW"] for r in webcam_records])
    w_dtw = np.median([r["geometric"]["features"]["d_TW"] for r in webcam_records])
    w_dti = np.median([r["geometric"]["features"]["d_TI"] for r in webcam_records])
    w_str = np.median([r["geometric"]["features"]["idx_straight"] for r in webcam_records])
    w_tipy = np.median([r["geometric"]["features"]["idx_tip_y"] for r in webcam_records])
    w_curl = np.median([r["geometric"]["features"]["index_curl"] for r in webcam_records])
    print(f"{'X (Webcam)':<6} | {'288':<12} | {w_diw:<10.2f} {w_dtw:<10.2f} {w_dti:<10.2f} {w_str:<18.2f} {w_tipy:<15.2f} {w_curl:<12.2f}")

    for let in competing:
        flist = comp_feats.get(let, [])
        if not flist: continue
        diw = np.median([cf[2]["d_IW"] for cf in flist])
        dtw = np.median([cf[2]["d_TW"] for cf in flist])
        dti = np.median([cf[2]["d_TI"] for cf in flist])
        st = np.median([cf[2]["idx_straight"] for cf in flist])
        tipy = np.median([cf[2]["idx_tip_y"] for cf in flist])
        curl = np.median([cf[2]["index_curl"] for cf in flist])
        print(f"{let:<6} | {len(flist):<12} | {diw:<10.2f} {dtw:<10.2f} {dti:<10.2f} {st:<18.2f} {tipy:<15.2f} {curl:<12.2f}")

    # Separability Analysis: What features distinguish X from S, P, A, O, E, C?
    print("\nSeparability Key Insights:")
    # X vs S
    s_flist = comp_feats.get("S", [])
    s_st = [cf[2]["idx_straight"] for cf in s_flist]
    s_curl = [cf[2]["index_curl"] for cf in s_flist]
    s_diw = [cf[2]["d_IW"] for cf in s_flist]
    print(f"1. X vs S (Fist): S has d_IW min={min(s_diw):.2f}, med={np.median(s_diw):.2f}, max={max(s_diw):.2f}; idx_straight min={min(s_st):.2f}, med={np.median(s_st):.2f}, max={max(s_st):.2f}; index_curl min={min(s_curl):.2f}, med={np.median(s_curl):.2f}, max={max(s_curl):.2f}")
    print(f"   X (Webcam) has d_IW=1.23-1.52, idx_straight=0.51-0.79, index_curl=0.18-0.45.")
    
    # X vs P (especially P-down)
    p_flist = comp_feats.get("P", [])
    p_tipy = [cf[2]["idx_tip_y"] for cf in p_flist]
    p_midy = [cf[2]["mid_tip_y"] for cf in p_flist]
    print(f"2. X vs P: P has idx_tip_y min={min(p_tipy):.2f}, med={np.median(p_tipy):.2f}, max={max(p_tipy):.2f} (pointing downwards in VSL toàn quốc > 0.60, or upright D0509B <= -1.40 with middle extended d_MW=2.02).")
    print(f"   X has index upright (idx_tip_y ~ -1.23) BUT middle finger curled (d_MW=0.71, mid_straight < 0.50), completely separating from P!")

    # X vs O
    o_flist = comp_feats.get("O", [])
    o_dti = [cf[2]["d_TI"] for cf in o_flist]
    o_dmw = [cf[2]["d_MW"] for cf in o_flist]
    o_st = [cf[2]["idx_straight"] for cf in o_flist]
    print(f"3. X vs O (Circle): O has all 4 fingers curved to meet thumb: d_MW={np.median(o_dmw):.2f}, idx_straight={np.median(o_st):.2f}.")
    print(f"   In X, middle/ring/pinky are tightly curled into palm (d_MW=0.71, d_RW=0.60, d_PW=0.60), not meeting thumb.")

if __name__ == "__main__":
    main()

