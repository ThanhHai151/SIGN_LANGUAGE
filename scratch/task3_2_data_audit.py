import json
import os
import sys
import hashlib
import numpy as np
from typing import Dict, Any, List

sys.path.insert(0, os.path.abspath("."))

def compute_sha256(filepath: str) -> str:
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def main():
    filepath = os.path.abspath("backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl")
    print(f"Filepath: {filepath}")
    if not os.path.exists(filepath):
        print("ERROR: File does not exist!")
        return

    sha256 = compute_sha256(filepath)
    size_bytes = os.path.getsize(filepath)
    print(f"File size: {size_bytes} bytes")
    print(f"SHA256: {sha256}")
    
    x_holdout_sha256 = "bc58a5fa8a1b2986e7fd3613f88ea46f889e8db54407f3f9d81ad627bcbb610b"
    if sha256 == x_holdout_sha256:
        print("ERROR: SHA256 MATCHES X HOLDOUT! THIS IS NOT O DATA!")
        return

    # Parse JSONL records
    records = []
    line_errors = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                data['_line_no'] = idx
                records.append(data)
            except Exception as e:
                line_errors.append((idx, str(e)))

    print(f"Total lines read: {len(records) + len(line_errors)}")
    print(f"Parse errors: {len(line_errors)}")

    # Extract session IDs, expected_signs, trials
    session_ids = set()
    expected_signs = {}
    trials = {}
    idle_count = 0
    active_count = 0

    for r in records:
        sess = r.get("session_id")
        if sess:
            session_ids.add(sess)
        exp = r.get("expected_sign")
        expected_signs[exp] = expected_signs.get(exp, 0) + 1
        
        trial = r.get("trial_id")
        if trial:
            active_count += 1
            if trial not in trials:
                trials[trial] = []
            trials[trial].append(r)
        else:
            idle_count += 1

    print("\n--- STEP 1 SUMMARY ---")
    print(f"Session IDs in records: {session_ids}")
    print(f"Expected signs distribution: {expected_signs}")
    print(f"Active frames (with trial_id): {active_count}")
    print(f"Idle frames (without trial_id): {idle_count}")
    print(f"Number of trials: {len(trials)}")
    for t_id, t_recs in sorted(trials.items()):
        print(f"  {t_id}: {len(t_recs)} frames")

    # Inspect current AlphabetClassifier Rule O constants
    from backend.vision.alphabet_classifier import (
        O_D_TI_MAX, O_D_TM_MAX, O_D_IW_MIN, O_D_MW_MIN,
        O_D_RW_MIN, O_D_PW_MIN, O_IDX_STRAIGHT_MAX,
        AlphabetClassifier
    )
    clf = AlphabetClassifier()
    print("\n--- CURRENT RULE 6 O CONSTANTS IN SOURCE ---")
    print(f"O_D_TI_MAX = {O_D_TI_MAX}")
    print(f"O_D_TM_MAX = {O_D_TM_MAX}")
    print(f"O_D_IW_MIN = {O_D_IW_MIN}")
    print(f"O_D_MW_MIN = {O_D_MW_MIN}")
    print(f"O_D_RW_MIN = {O_D_RW_MIN}")
    print(f"O_D_PW_MIN = {O_D_PW_MIN}")
    print(f"O_IDX_STRAIGHT_MAX = {O_IDX_STRAIGHT_MAX}")

    # Step 2: Quality & Prediction Breakdown by Trial
    print("\n--- STEP 2: PREDICTION BREAKDOWN BY TRIAL ---")
    trial_stats = {}
    all_active_records = []
    for t_id, t_recs in sorted(trials.items()):
        total = len(t_recs)
        valid = 0
        pred_o = 0
        pred_other = {}
        pred_neutral = 0
        confirmed_o = 0
        
        for r in t_recs:
            geom = r.get("geometric", {})
            feat = geom.get("features") if geom else None
            if feat is not None:
                valid += 1
                all_active_records.append(r)
            
            final_sign = r.get("final_sign")
            if final_sign == "O":
                pred_o += 1
            elif final_sign == "NEUTRAL" or final_sign is None:
                pred_neutral += 1
            else:
                pred_other[final_sign] = pred_other.get(final_sign, 0) + 1
            
            if r.get("confirmed") and final_sign == "O":
                confirmed_o += 1

        trial_stats[t_id] = {
            "total": total,
            "valid": valid,
            "pred_o": pred_o,
            "pred_neutral": pred_neutral,
            "pred_other": pred_other,
            "confirmed_o": confirmed_o
        }
        print(f"Trial {t_id}: Total={total}, Valid={valid}, O={pred_o}, NEUTRAL={pred_neutral}, Other={pred_other}, Confirmed_O={confirmed_o}")

    # Step 3: Rule 6 O Frame-by-Frame Detailed Analysis
    print(f"\n--- STEP 3: RULE 6 ANALYSIS ON {len(all_active_records)} VALID ACTIVE FRAMES ---")

    outer_gate_fails = 0
    i_ext_fails = 0
    m_ext_fails = 0
    r_ext_fails = 0
    p_ext_fails = 0

    cond_stats = {
        "d_ti_le_max": [],
        "d_tm_le_max": [],
        "d_iw_ge_min": [],
        "d_mw_ge_min": [],
        "d_rw_ge_min": [],
        "d_pw_ge_min": [],
        "idx_straight_le_max": []
    }

    cond_pass_counts = {k: 0 for k in cond_stats}
    both_outer_and_7_pass = 0
    diag_matched_count = 0
    production_matched_count = 0

    features_list = {
        "d_TI": [],
        "d_TM": [],
        "d_IW": [],
        "d_MW": [],
        "d_RW": [],
        "d_PW": [],
        "idx_straight": []
    }

    trial_outer_fails = {t_id: 0 for t_id in trials}
    trial_geom_fails = {t_id: 0 for t_id in trials}
    trial_full_matches = {t_id: 0 for t_id in trials}

    for r in all_active_records:
        t_id = r.get("trial_id")
        geom = r.get("geometric", {})
        feat = geom.get("features", {})
        
        i_ext = feat.get("index_ext", False)
        m_ext = feat.get("middle_ext", False)
        r_ext = feat.get("ring_ext", False)
        p_ext = feat.get("pinky_ext", False)

        outer_fail = (i_ext or m_ext or r_ext or p_ext)
        if outer_fail:
            outer_gate_fails += 1
            trial_outer_fails[t_id] += 1
            if i_ext: i_ext_fails += 1
            if m_ext: m_ext_fails += 1
            if r_ext: r_ext_fails += 1
            if p_ext: p_ext_fails += 1

        d_ti = feat.get("d_TI", 99.0)
        d_tm = feat.get("d_TM", 99.0)
        d_iw = feat.get("d_IW", 0.0)
        d_mw = feat.get("d_MW", 0.0)
        d_rw = feat.get("d_RW", 0.0)
        d_pw = feat.get("d_PW", 0.0)
        idx_straight = feat.get("idx_straight", 1.0)

        features_list["d_TI"].append(d_ti)
        features_list["d_TM"].append(d_tm)
        features_list["d_IW"].append(d_iw)
        features_list["d_MW"].append(d_mw)
        features_list["d_RW"].append(d_rw)
        features_list["d_PW"].append(d_pw)
        features_list["idx_straight"].append(idx_straight)

        c_d_ti = (d_ti <= O_D_TI_MAX)
        c_d_tm = (d_tm <= O_D_TM_MAX)
        c_d_iw = (d_iw >= O_D_IW_MIN)
        c_d_mw = (d_mw >= O_D_MW_MIN)
        c_d_rw = (d_rw >= O_D_RW_MIN)
        c_d_pw = (d_pw >= O_D_PW_MIN)
        c_idx_str = (idx_straight <= O_IDX_STRAIGHT_MAX)

        if c_d_ti: cond_pass_counts["d_ti_le_max"] += 1
        if c_d_tm: cond_pass_counts["d_tm_le_max"] += 1
        if c_d_iw: cond_pass_counts["d_iw_ge_min"] += 1
        if c_d_mw: cond_pass_counts["d_mw_ge_min"] += 1
        if c_d_rw: cond_pass_counts["d_rw_ge_min"] += 1
        if c_d_pw: cond_pass_counts["d_pw_ge_min"] += 1
        if c_idx_str: cond_pass_counts["idx_straight_le_max"] += 1

        all_7_pass = (c_d_ti and c_d_tm and c_d_iw and c_d_mw and c_d_rw and c_d_pw and c_idx_str)
        if not all_7_pass:
            trial_geom_fails[t_id] += 1

        # Check diagnostics in log
        o_diag = geom.get("o_rule_diagnostics", {})
        if o_diag.get("matched", False):
            diag_matched_count += 1

        # Production rule result
        prod_match = (not outer_fail) and all_7_pass
        if prod_match:
            production_matched_count += 1
            trial_full_matches[t_id] += 1

    N_valid = len(all_active_records)
    print(f"\nA. OUTER GATE ANALYSIS (N={N_valid}):")
    print(f"Outer Gate Fails: {outer_gate_fails}/{N_valid} ({outer_gate_fails/N_valid*100:.2f}%)")
    print(f"Outer Gate Passes: {N_valid - outer_gate_fails}/{N_valid} ({(N_valid - outer_gate_fails)/N_valid*100:.2f}%)")
    print(f"Failures by finger:")
    print(f"  - index_ext: {i_ext_fails}/{N_valid} ({i_ext_fails/N_valid*100:.2f}%)")
    print(f"  - middle_ext: {m_ext_fails}/{N_valid} ({m_ext_fails/N_valid*100:.2f}%)")
    print(f"  - ring_ext: {r_ext_fails}/{N_valid} ({r_ext_fails/N_valid*100:.2f}%)")
    print(f"  - pinky_ext: {p_ext_fails}/{N_valid} ({p_ext_fails/N_valid*100:.2f}%)")

    print(f"\nOuter Gate Fails by Trial:")
    for t_id, cnt in sorted(trial_outer_fails.items()):
        t_total = trial_stats[t_id]["valid"]
        print(f"  {t_id}: {cnt}/{t_total} ({cnt/t_total*100:.1f}%)")

    print("\nB. 7 GEOMETRIC CONDITIONS ANALYSIS:")
    thresholds = {
        "d_TI": ("<=", O_D_TI_MAX),
        "d_TM": ("<=", O_D_TM_MAX),
        "d_IW": (">=", O_D_IW_MIN),
        "d_MW": (">=", O_D_MW_MIN),
        "d_RW": (">=", O_D_RW_MIN),
        "d_PW": (">=", O_D_PW_MIN),
        "idx_straight": ("<=", O_IDX_STRAIGHT_MAX),
    }

    cond_key_map = {
        "d_TI": "d_ti_le_max",
        "d_TM": "d_tm_le_max",
        "d_IW": "d_iw_ge_min",
        "d_MW": "d_mw_ge_min",
        "d_RW": "d_rw_ge_min",
        "d_PW": "d_pw_ge_min",
        "idx_straight": "idx_straight_le_max"
    }

    for feat_name, (op, thresh) in thresholds.items():
        vals = np.array(features_list[feat_name])
        c_key = cond_key_map[feat_name]
        passed = cond_pass_counts[c_key]
        failed = N_valid - passed
        pass_pct = passed / N_valid * 100.0
        
        p10 = np.percentile(vals, 10)
        p25 = np.percentile(vals, 25)
        median = np.percentile(vals, 50)
        p75 = np.percentile(vals, 75)
        p90 = np.percentile(vals, 90)
        v_min = np.min(vals)
        v_max = np.max(vals)
        
        print(f"\nCondition: {feat_name} {op} {thresh}")
        print(f"  Passed: {passed}/{N_valid} ({pass_pct:.2f}%) | Failed: {failed}/{N_valid} ({100-pass_pct:.2f}%)")
        print(f"  Min: {v_min:.4f}, P10: {p10:.4f}, P25: {p25:.4f}, Median: {median:.4f}, P75: {p75:.4f}, P90: {p90:.4f}, Max: {v_max:.4f}")
        if op == "<=":
            print(f"  Distance from Median to Threshold: {median - thresh:+.4f} (positive means violates <=)")
        else:
            print(f"  Distance from Median to Threshold: {thresh - median:+.4f} (positive means violates >=)")

    print(f"\nC. PRODUCTION RULE 6 O MATCH:")
    print(f"Production O MATCH: {production_matched_count}/{N_valid} ({production_matched_count/N_valid*100:.2f}%)")
    print(f"Diagnostics reported MATCH: {diag_matched_count}/{N_valid} ({diag_matched_count/N_valid*100:.2f}%)")
    print(f"Discrepancy (Diag Match but Outer Gate Fail): {diag_matched_count - production_matched_count} frames")

    print("\nProduction MATCH by Trial:")
    for t_id, cnt in sorted(trial_full_matches.items()):
        t_total = trial_stats[t_id]["valid"]
        print(f"  {t_id}: {cnt}/{t_total} ({cnt/t_total*100:.1f}%)")

    # Step 4: Compare with Fixture backend/data/alphabet_v2/static_landmarks/O.json
    print("\n--- STEP 4: FIXTURE COMPARISON ---")
    fixture_path = "backend/data/alphabet_v2/static_landmarks/O.json"
    with open(fixture_path, 'r', encoding='utf-8') as f:
        fix_data = json.load(f)
    print(f"Fixture letter: {fix_data.get('letter')}, video_id: {fix_data.get('video_id')}")
    print(f"Total frames: {fix_data.get('total_frames')}, valid frames: {fix_data.get('valid_frames')}")

    fix_samples = fix_data.get("samples", [])
    fix_valid_samples = [s for s in fix_samples if s.get("landmarks") is not None]
    print(f"Loaded {len(fix_valid_samples)} valid landmark frames from fixture O.")

    fix_feat_list = {k: [] for k in features_list}
    fix_outer_fails = 0
    fix_cond_pass = {k: 0 for k in cond_stats}
    fix_prod_matches = 0

    for s in fix_valid_samples:
        feat = clf.extract_features(s["landmarks"])
        if feat is None:
            continue
        
        i_ext = feat.get("index_ext", False)
        m_ext = feat.get("middle_ext", False)
        r_ext = feat.get("ring_ext", False)
        p_ext = feat.get("pinky_ext", False)

        outer_fail = (i_ext or m_ext or r_ext or p_ext)
        if outer_fail:
            fix_outer_fails += 1

        d_ti = feat.get("d_TI", 99.0)
        d_tm = feat.get("d_TM", 99.0)
        d_iw = feat.get("d_IW", 0.0)
        d_mw = feat.get("d_MW", 0.0)
        d_rw = feat.get("d_RW", 0.0)
        d_pw = feat.get("d_PW", 0.0)
        idx_straight = feat.get("idx_straight", 1.0)

        fix_feat_list["d_TI"].append(d_ti)
        fix_feat_list["d_TM"].append(d_tm)
        fix_feat_list["d_IW"].append(d_iw)
        fix_feat_list["d_MW"].append(d_mw)
        fix_feat_list["d_RW"].append(d_rw)
        fix_feat_list["d_PW"].append(d_pw)
        fix_feat_list["idx_straight"].append(idx_straight)

        c_d_ti = (d_ti <= O_D_TI_MAX)
        c_d_tm = (d_tm <= O_D_TM_MAX)
        c_d_iw = (d_iw >= O_D_IW_MIN)
        c_d_mw = (d_mw >= O_D_MW_MIN)
        c_d_rw = (d_rw >= O_D_RW_MIN)
        c_d_pw = (d_pw >= O_D_PW_MIN)
        c_idx_str = (idx_straight <= O_IDX_STRAIGHT_MAX)

        if c_d_ti: fix_cond_pass["d_ti_le_max"] += 1
        if c_d_tm: fix_cond_pass["d_tm_le_max"] += 1
        if c_d_iw: fix_cond_pass["d_iw_ge_min"] += 1
        if c_d_mw: fix_cond_pass["d_mw_ge_min"] += 1
        if c_d_rw: fix_cond_pass["d_rw_ge_min"] += 1
        if c_d_pw: fix_cond_pass["d_pw_ge_min"] += 1
        if c_idx_str: fix_cond_pass["idx_straight_le_max"] += 1

        all_7 = (c_d_ti and c_d_tm and c_d_iw and c_d_mw and c_d_rw and c_d_pw and c_idx_str)
        if (not outer_fail) and all_7:
            fix_prod_matches += 1

    N_fix = len(fix_valid_samples)
    print(f"\nFixture Outer Gate Fails: {fix_outer_fails}/{N_fix} ({fix_outer_fails/N_fix*100:.1f}%)")
    print(f"Fixture Production MATCH: {fix_prod_matches}/{N_fix} ({fix_prod_matches/N_fix*100:.1f}%)")
    for feat_name, (op, thresh) in thresholds.items():
        c_key = cond_key_map[feat_name]
        f_pass = fix_cond_pass[c_key]
        vals = np.array(fix_feat_list[feat_name])
        print(f"  {feat_name} ({op} {thresh}): Pass={f_pass}/{N_fix} ({f_pass/N_fix*100:.1f}%) | Median={np.median(vals):.4f} [Pilot Median: {np.median(features_list[feat_name]):.4f}]")

    # Step 5: MLP Fallback & Recognizer Pipeline
    print("\n--- STEP 5: MLP FALLBACK & RECOGNIZER STATUS ---")
    mlp_loads = {}
    final_signs = {}
    for r in all_active_records:
        fs = r.get("final_sign")
        final_signs[fs] = final_signs.get(fs, 0) + 1
        sh = r.get("static_head", {})
        lst = sh.get("load_status")
        mlp_loads[lst] = mlp_loads.get(lst, 0) + 1
    
    print(f"Static Head Load Status: {mlp_loads}")
    print(f"Final Signs across all valid active records: {final_signs}")

    # Check Idle records
    idle_records = [r for r in records if r.get("trial_id") is None]
    print(f"\n--- IDLE RECORDS ANALYSIS (N={len(idle_records)}) ---")
    idle_final_signs = {}
    idle_prod_matches = 0
    for r in idle_records:
        fs = r.get("final_sign")
        idle_final_signs[fs] = idle_final_signs.get(fs, 0) + 1
        geom = r.get("geometric", {})
        feat = geom.get("features") if geom else None
        if feat:
            i_ext = feat.get("index_ext", False)
            m_ext = feat.get("middle_ext", False)
            r_ext = feat.get("ring_ext", False)
            p_ext = feat.get("pinky_ext", False)
            outer_ok = not (i_ext or m_ext or r_ext or p_ext)
            
            c_all = (
                feat.get("d_TI", 99) <= O_D_TI_MAX and
                feat.get("d_TM", 99) <= O_D_TM_MAX and
                feat.get("d_IW", 0) >= O_D_IW_MIN and
                feat.get("d_MW", 0) >= O_D_MW_MIN and
                feat.get("d_RW", 0) >= O_D_RW_MIN and
                feat.get("d_PW", 0) >= O_D_PW_MIN and
                feat.get("idx_straight", 1) <= O_IDX_STRAIGHT_MAX
            )
            if outer_ok and c_all:
                idle_prod_matches += 1
    print(f"Idle Final Signs: {idle_final_signs}")
    print(f"Idle Production O Matches: {idle_prod_matches}/{len(idle_records)}")

if __name__ == "__main__":
    main()
