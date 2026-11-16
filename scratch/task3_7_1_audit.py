"""
TASK 3.7.1 - Deep Mathematical Audit of TASK 3.7 Holdout Metrics
File: backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl
"""

import json
import numpy as np
from collections import Counter, defaultdict

def audit():
    path = "backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl"
    with open(path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    total_records = len(records)
    active_records = [r for r in records if r.get("expected_sign") == "O"]
    idle_records = [r for r in records if r.get("expected_sign") is None]

    print(f"Total records: {total_records}")
    print(f"Active records: {len(active_records)}")
    print(f"Idle records: {len(idle_records)}")

    # 1. Active Frame Geometry
    matched = [r for r in active_records if r.get("geometric", {}).get("letter") == "O"]
    unmatched = [r for r in active_records if r.get("geometric", {}).get("letter") != "O"]

    print(f"Matched active frames: {len(matched)}")
    print(f"Unmatched active frames: {len(unmatched)}")

    # 2. Condition violations across all 296 unmatched frames
    # Let's inspect each condition
    conditions = [
        "d_ti_le_max", "d_tm_le_max", "d_iw_ge_min", "d_mw_ge_min",
        "d_rw_ge_min", "d_pw_ge_min", "idx_straight_le_max", "d_tw_ge_min"
    ]

    cond_counts = Counter()
    outer_gate_counts = 0

    for r in unmatched:
        feat = r.get("geometric", {}).get("features", {})
        diag = r.get("geometric", {}).get("o_rule_diagnostics", {})
        c_dict = diag.get("conditions", {})
        for c in conditions:
            if c_dict.get(c) is False:
                cond_counts[c] += 1
        
        # Check outer gate
        i_ext = feat.get("index_ext", False)
        m_ext = feat.get("middle_ext", False)
        r_ext = feat.get("ring_ext", False)
        p_ext = feat.get("pinky_ext", False)
        if i_ext or m_ext or r_ext or p_ext:
            outer_gate_counts += 1

    print("\n--- VIOLATION COUNTS (Non-exclusive, sum > 296) ---")
    print(f"Outer gate fails: {outer_gate_counts}")
    for c, cnt in cond_counts.items():
        print(f"  {c}: {cnt}")

    # 3. Investigation of d_TI statistics and denominator reconciliation
    print("\n--- d_TI & idx_straight STATISTICAL DISTRIBUTIONS ---")
    # Groups:
    # A. All active (N=349)
    # B. Matched active (N=53)
    # C. Unmatched active (N=296)
    # D. Subset of unmatched with d_TI > 0.22 (N=102)
    # E. Subset of unmatched with d_TI <= 0.22 (N=194)
    # F. Subset of unmatched with idx_straight > 0.48 (N=200)

    def stats_for(vals, label):
        arr = np.array(vals)
        return {
            "group": label,
            "N": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "median": float(np.median(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr))
        }

    dtis_all = [r["geometric"]["features"]["d_TI"] for r in active_records]
    dtis_matched = [r["geometric"]["features"]["d_TI"] for r in matched]
    dtis_unmatched = [r["geometric"]["features"]["d_TI"] for r in unmatched]
    dtis_violating = [r["geometric"]["features"]["d_TI"] for r in unmatched if r["geometric"]["features"]["d_TI"] > 0.22]
    dtis_non_violating_unmatched = [r["geometric"]["features"]["d_TI"] for r in unmatched if r["geometric"]["features"]["d_TI"] <= 0.22]

    for s in [
        stats_for(dtis_all, "d_TI on All Active (N=349)"),
        stats_for(dtis_matched, "d_TI on Matched Active (N=53)"),
        stats_for(dtis_unmatched, "d_TI on All Unmatched Active (N=296)"),
        stats_for(dtis_violating, "d_TI on d_TI Violating Subset (N=102)"),
        stats_for(dtis_non_violating_unmatched, "d_TI on d_TI Non-Violating Unmatched (N=194)"),
    ]:
        print(f"{s['group']:<48}: mean={s['mean']:.4f}, median={s['median']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}")

    idx_all = [r["geometric"]["features"]["idx_straight"] for r in active_records]
    idx_matched = [r["geometric"]["features"]["idx_straight"] for r in matched]
    idx_unmatched = [r["geometric"]["features"]["idx_straight"] for r in unmatched]
    idx_violating = [r["geometric"]["features"]["idx_straight"] for r in unmatched if r["geometric"]["features"]["idx_straight"] > 0.48]
    idx_non_violating_unmatched = [r["geometric"]["features"]["idx_straight"] for r in unmatched if r["geometric"]["features"]["idx_straight"] <= 0.48]

    print()
    for s in [
        stats_for(idx_all, "idx_straight on All Active (N=349)"),
        stats_for(idx_matched, "idx_straight on Matched Active (N=53)"),
        stats_for(idx_unmatched, "idx_straight on All Unmatched Active (N=296)"),
        stats_for(idx_violating, "idx_straight on idx_straight Violating Subset (N=200)"),
        stats_for(idx_non_violating_unmatched, "idx_straight on idx_straight Non-Violating Unmatched (N=96)"),
    ]:
        print(f"{s['group']:<48}: mean={s['mean']:.4f}, median={s['median']:.4f}, min={s['min']:.4f}, max={s['max']:.4f}")

    # 4. Mutually Exclusive & Collectively Exhaustive (MECE) Partition of 296 frames
    # Let's test combinations of 3 orthogonal violation flags:
    # Flag A: outer_gate_fail (True/False)
    # Flag B: idx_straight_fail (idx_straight > 0.48)
    # Flag C: d_ti_fail (d_TI > 0.22)
    print("\n--- MECE PARTITION OF 296 UNMATCHED FRAMES ---")
    mece = defaultdict(list)
    for r in unmatched:
        feat = r["geometric"]["features"]
        i_ext = feat.get("index_ext", False) or feat.get("middle_ext", False) or feat.get("ring_ext", False) or feat.get("pinky_ext", False)
        idx_fail = feat.get("idx_straight", 0.0) > 0.48
        dti_fail = feat.get("d_TI", 0.0) > 0.22
        
        category = []
        if i_ext: category.append("OuterGate")
        if idx_fail: category.append("idx_straight>0.48")
        if dti_fail: category.append("d_TI>0.22")
        
        cat_key = " + ".join(category) if category else "None"
        mece[cat_key].append(r)

    total_mece = 0
    for k in sorted(mece.keys(), key=lambda x: len(mece[x]), reverse=True):
        cnt = len(mece[k])
        total_mece += cnt
        pct = cnt / len(unmatched) * 100
        print(f"  {k:<40}: {cnt:>3} frames ({pct:>5.1f}%)")
    print(f"Total MECE frames: {total_mece} / {len(unmatched)}")

    # 5. Trial by Trial Re-computation
    trials = defaultdict(list)
    for r in active_records:
        trials[r.get("trial_id")].append(r)

    print("\n--- TRIAL BY TRIAL RE-COMPUTATION ---")
    for tid in sorted(trials.keys()):
        recs = trials[tid]
        n_total = len(recs)
        n_match = sum(1 for r in recs if r.get("geometric", {}).get("letter") == "O")
        rate = n_match / n_total if n_total > 0 else 0.0
        pass_70 = (rate >= 0.70)
        
        # Check confirmations in raw stream
        confirmed_records = [r for r in recs if r.get("confirmed") and r.get("final_sign") == "O"]
        has_token = len(confirmed_records) > 0
        
        # Trace consecutive runs of matches
        match_runs = []
        curr_run = 0
        for r in recs:
            if r.get("geometric", {}).get("letter") == "O":
                curr_run += 1
            else:
                if curr_run > 0:
                    match_runs.append(curr_run)
                curr_run = 0
        if curr_run > 0:
            match_runs.append(curr_run)

        max_consecutive = max(match_runs) if match_runs else 0
        print(f"{tid[-5:]}: N={n_total:>2}, Match={n_match:>2} ({rate*100:>5.1f}%), MaxConsecutive={max_consecutive:>2}, ConfirmedToken={str(has_token):<5}, Runs={match_runs}")

    # 6. Detailed Inspection of Temporal Smoothing for rep04, rep08, rep09
    print("\n--- TEMPORAL BUFFER TIMELINE FOR rep04, rep08, rep09 ---")
    for target_trial in ["pilot_20261109_004312_O_rep04", "pilot_20261109_004312_O_rep08", "pilot_20261109_004312_O_rep09"]:
        recs = trials[target_trial]
        print(f"\nTimeline for {target_trial[-5:]} (N={len(recs)}):")
        for r in recs:
            fidx = r.get("frame_idx")
            geom_letter = r.get("geometric", {}).get("letter")
            final_sign = r.get("final_sign")
            conf = r.get("final_confidence")
            confirmed = r.get("confirmed")
            idx_s = r["geometric"]["features"].get("idx_straight")
            d_ti = r["geometric"]["features"].get("d_TI")
            print(f"  fidx={fidx:>2}: geom={str(geom_letter):<4}, final={final_sign:<7}, conf={conf:.2f}, confirmed={str(confirmed):<5}, idx_straight={idx_s:.2f}, d_TI={d_ti:.2f}")

    # 7. Idle Records Analysis
    print("\n--- IDLE RECORDS ANALYSIS (N=87) ---")
    idle_signs = Counter(r.get("final_sign") for r in idle_records)
    print(f"Final sign distribution in idle: {dict(idle_signs)}")

    # Let's group idle records into contiguous segments between trials
    segments = []
    curr_seg = []
    for r in records:
        if r.get("expected_sign") is None:
            curr_seg.append(r)
        else:
            if curr_seg:
                segments.append(curr_seg)
                curr_seg = []
    if curr_seg:
        segments.append(curr_seg)

    print(f"Total idle segments: {len(segments)}")
    for s_idx, seg in enumerate(segments):
        signs = Counter(r.get("final_sign") for r in seg)
        start_f = seg[0].get("frame_idx")
        end_f = seg[-1].get("frame_idx")
        t_start = seg[0].get("timestamp")
        t_end = seg[-1].get("timestamp")
        dur = t_end - t_start if t_end and t_start else 0.0
        print(f"  Segment {s_idx+1}: {len(seg):>2} frames (fidx {start_f} -> {end_f}, dur={dur:.2f}s). Signs: {dict(signs)}")
        for r in seg:
            if r.get("final_sign") != "NEUTRAL":
                print(f"    -> fidx={r.get('frame_idx')}, sign={r.get('final_sign')}, geom_letter={r.get('geometric', {}).get('letter')}, reason={r.get('geometric', {}).get('reason')}")

if __name__ == "__main__":
    audit()

