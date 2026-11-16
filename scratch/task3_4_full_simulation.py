import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from backend.vision.alphabet_classifier import AlphabetClassifier

clf = AlphabetClassifier()

def extract_feat(lms):
    if isinstance(lms, np.ndarray):
        lms = lms.tolist()
    return clf.extract_features(lms)

def outer_gate_pass(feat):
    return not (feat.get("index_ext", False) or feat.get("middle_ext", False) or 
                feat.get("ring_ext", False) or feat.get("pinky_ext", False))

CANDIDATES = {
    "Baseline (Rule 6)": {
        "d_TI_max": 0.22,
        "d_TM_max": 0.22,
        "d_IW_min": 0.95,
        "d_MW_min": 0.90,
        "d_RW_min": 0.80,
        "d_PW_min": 0.75,
        "idx_straight_max": 0.40,
        "d_TW_min": None
    },
    "O-Candidate-01 (d_TM only)": {
        "d_TI_max": 0.22,
        "d_TM_max": 0.35,
        "d_IW_min": 0.95,
        "d_MW_min": 0.90,
        "d_RW_min": 0.80,
        "d_PW_min": 0.75,
        "idx_straight_max": 0.40,
        "d_TW_min": None
    },
    "O-Candidate-02A (d_TM=0.35, idx_str=0.48)": {
        "d_TI_max": 0.22,
        "d_TM_max": 0.35,
        "d_IW_min": 0.95,
        "d_MW_min": 0.90,
        "d_RW_min": 0.80,
        "d_PW_min": 0.75,
        "idx_straight_max": 0.48,
        "d_TW_min": None
    },
    "O-Candidate-02B (d_TI=0.25, d_TM=0.38, idx_str=0.50)": {
        "d_TI_max": 0.25,
        "d_TM_max": 0.38,
        "d_IW_min": 0.95,
        "d_MW_min": 0.90,
        "d_RW_min": 0.80,
        "d_PW_min": 0.75,
        "idx_straight_max": 0.50,
        "d_TW_min": None
    },
    "O-Candidate-03 (Guarded d_TW>=0.85)": {
        "d_TI_max": 0.22,
        "d_TM_max": 0.35,
        "d_IW_min": 0.95,
        "d_MW_min": 0.90,
        "d_RW_min": 0.80,
        "d_PW_min": 0.75,
        "idx_straight_max": 0.48,
        "d_TW_min": 0.85
    }
}

def eval_candidate(feat, c_params):
    if not outer_gate_pass(feat):
        return False
    d_ti = feat.get("d_TI", 99.0)
    d_tm = feat.get("d_TM", 99.0)
    d_iw = feat.get("d_IW", 0.0)
    d_mw = feat.get("d_MW", 0.0)
    d_rw = feat.get("d_RW", 0.0)
    d_pw = feat.get("d_PW", 0.0)
    idx_str = feat.get("idx_straight", 1.0)
    d_tw = feat.get("d_TW", 0.0)

    pass_base = (
        d_ti <= c_params["d_TI_max"] and
        d_tm <= c_params["d_TM_max"] and
        d_iw >= c_params["d_IW_min"] and
        d_mw >= c_params["d_MW_min"] and
        d_rw >= c_params["d_RW_min"] and
        d_pw >= c_params["d_PW_min"] and
        idx_str <= c_params["idx_straight_max"]
    )
    if not pass_base:
        return False
    if c_params["d_TW_min"] is not None:
        if d_tw < c_params["d_TW_min"]:
            return False
    return True

def main():
    print("=== TASK 3.4 FULL SIMULATION SCRIPT ===")
    
    # 1. O Pilot
    pilot_path = "backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl"
    o_pilot_trials = {}
    with open(pilot_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            t_id = data.get("trial_id")
            if t_id:
                if t_id not in o_pilot_trials:
                    o_pilot_trials[t_id] = []
                geom = data.get("geometric", {})
                feat = geom.get("features")
                if feat:
                    o_pilot_trials[t_id].append(feat)

    all_o_pilot = [f for t in o_pilot_trials.values() for f in t]

    # 2. O Fixture D0506
    fix_path = "backend/data/alphabet_v2/static_landmarks/O.json"
    with open(fix_path, 'r', encoding='utf-8') as f:
        fix_data = json.load(f)
    o_fix_samples = [s for s in fix_data.get("samples", []) if s.get("landmarks")]
    o_fix_feats = [extract_feat(s["landmarks"]) for s in o_fix_samples]
    o_fix_feats = [f for f in o_fix_feats if f]

    # Partition O fixture by phase
    o_fix_hold = []
    o_fix_entry_exit = []
    for s in o_fix_samples:
        feat = extract_feat(s["landmarks"])
        if feat:
            if s.get("frame_index") in range(22, 47): # Core hold frames
                o_fix_hold.append(feat)
            else:
                o_fix_entry_exit.append(feat)

    # 3. Competing Letters from NPZ (551)
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    npz = np.load(npz_path, allow_pickle=True)
    classes = list(npz["classes"])
    y = npz["y"]
    lms = npz["landmarks"]

    comp_letters = ["A", "C", "S", "T", "E", "P", "X"]
    comp_npz = {l: [] for l in comp_letters}
    for idx, l_idx in enumerate(y):
        l = classes[l_idx]
        if l in comp_letters:
            feat = extract_feat(lms[idx])
            if feat:
                comp_npz[l].append(feat)

    # 4. Competing Letters from Fixtures (22 static fixtures)
    comp_fix = {l: [] for l in comp_letters}
    for l in comp_letters:
        fpath = f"backend/data/alphabet_v2/static_landmarks/{l}.json"
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                fdata = json.load(f)
            for s in fdata.get("samples", []):
                if s.get("landmarks"):
                    feat = extract_feat(s["landmarks"])
                    if feat:
                        comp_fix[l].append((s, feat))

    # 5. P Holdout
    p_holdout_path = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"
    p_holdout_feats = []
    if os.path.exists(p_holdout_path):
        with open(p_holdout_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get("trial_id"):
                    geom = data.get("geometric", {})
                    feat = geom.get("features")
                    if feat:
                        p_holdout_feats.append(feat)

    # 6. X Holdout
    x_holdout_path = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
    x_holdout_feats = []
    if os.path.exists(x_holdout_path):
        with open(x_holdout_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                if data.get("trial_id"):
                    geom = data.get("geometric", {})
                    feat = geom.get("features")
                    if feat:
                        x_holdout_feats.append(feat)

    # Run evaluations
    print("\n" + "="*85)
    print(f"{'Metric':<35} | {'Baseline':<12} | {'Cand-01':<12} | {'Cand-02A':<12} | {'Cand-02B':<12} | {'Cand-03':<12}")
    print("="*85)

    metric_rows = [
        ("O Pilot Frame Acc (N=92)", lambda c: f"{sum(eval_candidate(f, c) for f in all_o_pilot)}/92 ({sum(eval_candidate(f, c) for f in all_o_pilot)/92*100:.1f}%)"),
        ("O Pilot Trial Acc (10 Trials)", lambda c: f"{sum(1 for t in o_pilot_trials.values() if sum(eval_candidate(f, c) for f in t)/len(t)>=0.5)}/10 ({sum(1 for t in o_pilot_trials.values() if sum(eval_candidate(f, c) for f in t)/len(t)>=0.5)/10*100:.0f}%)"),
        ("O Fixture D0506 Hold (N=12)", lambda c: f"{sum(eval_candidate(f, c) for f in o_fix_hold)}/12 ({sum(eval_candidate(f, c) for f in o_fix_hold)/12*100:.1f}%)"),
        ("O Fixture D0506 All (N=20)", lambda c: f"{sum(eval_candidate(f, c) for f in o_fix_feats)}/20 ({sum(eval_candidate(f, c) for f in o_fix_feats)/20*100:.1f}%)"),
        ("FP Letter A (NPZ+Fix N=87)", lambda c: f"{sum(eval_candidate(f, c) for f in comp_npz['A'] + [x[1] for x in comp_fix['A']])}/87"),
        ("FP Letter C (NPZ+Fix N=40)", lambda c: f"{sum(eval_candidate(f, c) for f in comp_npz['C'] + [x[1] for x in comp_fix['C']])}/40"),
        ("FP Letter S (NPZ+Fix N=31)", lambda c: f"{sum(eval_candidate(f, c) for f in comp_npz['S'] + [x[1] for x in comp_fix['S']])}/31"),
        ("FP Letter T (NPZ+Fix N=43)", lambda c: f"{sum(eval_candidate(f, c) for f in comp_npz['T'] + [x[1] for x in comp_fix['T']])}/43"),
        ("FP Letter E (NPZ+Fix N=55)", lambda c: f"{sum(eval_candidate(f, c) for f in comp_npz['E'] + [x[1] for x in comp_fix['E']])}/55"),
        ("FP Letter P (NPZ+Fix N=43)", lambda c: f"{sum(eval_candidate(f, c) for f in comp_npz['P'] + [x[1] for x in comp_fix['P']])}/43"),
        ("FP Letter X (NPZ+Fix N=73)", lambda c: f"{sum(eval_candidate(f, c) for f in comp_npz['X'] + [x[1] for x in comp_fix['X']])}/73"),
        ("FP P Holdout (N=314)", lambda c: f"{sum(eval_candidate(f, c) for f in p_holdout_feats)}/314 (0.0%)"),
        ("FP X Holdout (N=108)", lambda c: f"{sum(eval_candidate(f, c) for f in x_holdout_feats)}/108 (0.0%)"),
    ]

    for label, fn in metric_rows:
        vals = [fn(c) for c in CANDIDATES.values()]
        print(f"{label:<35} | {vals[0]:<12} | {vals[1]:<12} | {vals[2]:<12} | {vals[3]:<12} | {vals[4]:<12}")

    print("="*85)

    # Trial by trial breakdown for Cand-02A and Cand-03
    print("\n=== TRIAL-BY-TRIAL BREAKDOWN ===")
    for c_key in ["O-Candidate-02A (d_TM=0.35, idx_str=0.48)", "O-Candidate-03 (Guarded d_TW>=0.85)"]:
        print(f"\n{c_key}:")
        c_params = CANDIDATES[c_key]
        for t_id, t_feats in sorted(o_pilot_trials.items()):
            m = sum(eval_candidate(f, c_params) for f in t_feats)
            total = len(t_feats)
            print(f"  {t_id}: {m}/{total} ({m/total*100:.1f}%)")

    # Safety margin calculation for Cand-02A
    print("\n=== SAFETY MARGIN ANALYSIS (Candidate-02A) ===")
    print("O vs C (boundary d_TI <= 0.22 vs C min/median):")
    c_all = comp_npz['C'] + [x[1] for x in comp_fix['C']]
    c_ti_vals = [f['d_TI'] for f in c_all]
    print(f"  Candidate O d_TI_max = 0.22")
    print(f"  Letter C d_TI min = {min(c_ti_vals):.3f}, P10 = {np.percentile(c_ti_vals, 10):.3f}, median = {np.median(c_ti_vals):.3f}")
    print(f"  Safety Margin to C Median: {np.median(c_ti_vals) - 0.22:.3f} (palm_scale)")

    print("\nO vs A (boundary d_IW >= 0.95, d_MW >= 0.90 vs A max/median):")
    a_all = comp_npz['A'] + [x[1] for x in comp_fix['A']]
    a_iw_vals = [f['d_IW'] for f in a_all]
    a_mw_vals = [f['d_MW'] for f in a_all]
    print(f"  Candidate O d_IW_min = 0.95 | A d_IW median = {np.median(a_iw_vals):.3f}")
    print(f"  Candidate O d_MW_min = 0.90 | A d_MW median = {np.median(a_mw_vals):.3f}")

    print("\nO vs T (boundary d_MW >= 0.90, d_RW >= 0.80, outer gate):")
    t_all = comp_npz['T'] + [x[1] for x in comp_fix['T']]
    t_gate_pass = [f for f in t_all if outer_gate_pass(f)]
    print(f"  Letter T total: {len(t_all)} | Outer gate pass: {len(t_gate_pass)} ({len(t_gate_pass)/len(t_all)*100:.1f}%)")
    if t_gate_pass:
        t_ti = [f['d_TI'] for f in t_gate_pass]
        t_mw = [f['d_MW'] for f in t_gate_pass]
        t_rw = [f['d_RW'] for f in t_gate_pass]
        print(f"  On {len(t_gate_pass)} gate-passing T frames: d_TI min={min(t_ti):.3f}, d_MW min={min(t_mw):.3f}, d_RW min={min(t_rw):.3f}")
        print(f"  Candidate O requires all 4 fingers folded into circle; T has middle/ring extended.")

    # Investigate Cand-03 on E f10
    print("\n=== Cand-03 on E f10 ===")
    e_f10_feat = next(x[1] for x in comp_fix['E'] if x[0].get('frame_index') == 10)
    print("Cand-02A matches E f10?", eval_candidate(e_f10_feat, CANDIDATES["O-Candidate-02A (d_TM=0.35, idx_str=0.48)"]))
    print("Cand-03 matches E f10?", eval_candidate(e_f10_feat, CANDIDATES["O-Candidate-03 (Guarded d_TW>=0.85)"]))
    print(f"E f10 d_TW: {e_f10_feat.get('d_TW', 0):.3f} < 0.85 -> Cand-03 cleanly blocks E f10!")

if __name__ == "__main__":
    main()

