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
        "idx_straight_max": 0.40
    },
    "O-Candidate-01 (d_TM only)": {
        "d_TI_max": 0.22,
        "d_TM_max": 0.35,
        "d_IW_min": 0.95,
        "d_MW_min": 0.90,
        "d_RW_min": 0.80,
        "d_PW_min": 0.75,
        "idx_straight_max": 0.40
    },
    "O-Candidate-02A (d_TM=0.35, idx_str=0.48)": {
        "d_TI_max": 0.22,
        "d_TM_max": 0.35,
        "d_IW_min": 0.95,
        "d_MW_min": 0.90,
        "d_RW_min": 0.80,
        "d_PW_min": 0.75,
        "idx_straight_max": 0.48
    },
    "O-Candidate-02B (d_TI=0.25, d_TM=0.40, idx_str=0.48)": {
        "d_TI_max": 0.25,
        "d_TM_max": 0.40,
        "d_IW_min": 0.95,
        "d_MW_min": 0.90,
        "d_RW_min": 0.80,
        "d_PW_min": 0.75,
        "idx_straight_max": 0.48
    },
    "O-Candidate-02C (d_TI=0.22, d_TM=0.38, idx_str=0.48)": {
        "d_TI_max": 0.22,
        "d_TM_max": 0.38,
        "d_IW_min": 0.95,
        "d_MW_min": 0.90,
        "d_RW_min": 0.80,
        "d_PW_min": 0.75,
        "idx_straight_max": 0.48
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

    return (
        d_ti <= c_params["d_TI_max"] and
        d_tm <= c_params["d_TM_max"] and
        d_iw >= c_params["d_IW_min"] and
        d_mw >= c_params["d_MW_min"] and
        d_rw >= c_params["d_RW_min"] and
        d_pw >= c_params["d_PW_min"] and
        idx_str <= c_params["idx_straight_max"]
    )

def main():
    print("=== TASK 3.4 COMPREHENSIVE CANDIDATE EVALUATION ===")

    # 1. Load O Pilot
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
    print(f"O Pilot: {len(all_o_pilot)} frames across {len(o_pilot_trials)} trials.")

    # 2. Load O Fixture D0506
    fix_path = "backend/data/alphabet_v2/static_landmarks/O.json"
    with open(fix_path, 'r', encoding='utf-8') as f:
        fix_data = json.load(f)
    o_fix_feats = []
    for s in fix_data.get("samples", []):
        if s.get("landmarks"):
            feat = extract_feat(s["landmarks"])
            if feat:
                o_fix_feats.append(feat)
    print(f"O Fixture D0506: {len(o_fix_feats)} valid frames.")

    # 3. Load NPZ Competing Samples (551 samples)
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

    # 4. Load Fixtures for Competing Letters
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
                        comp_fix[l].append(feat)

    # 5. Load P Holdout (pilot_pilot_20261108_183703.jsonl)
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
    print(f"P Holdout (20261108_183703): {len(p_holdout_feats)} valid active frames.")

    # 6. Load X Holdout (pilot_pilot_20261108_193712.jsonl)
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
    print(f"X Holdout (20261108_193712): {len(x_holdout_feats)} valid active frames.")

    # Evaluate each Candidate
    results = {}
    for c_name, c_params in CANDIDATES.items():
        print(f"\n================ EVALUATING: {c_name} ================")
        print("Parameters:", c_params)

        # O Pilot Evaluation
        o_pilot_matches = sum(eval_candidate(f, c_params) for f in all_o_pilot)
        o_pilot_acc = o_pilot_matches / len(all_o_pilot) * 100.0

        # O Pilot Trial Accuracy (Majority vote per trial >= 50%)
        trial_passes = 0
        trial_details = {}
        for t_id, t_feats in sorted(o_pilot_trials.items()):
            m = sum(eval_candidate(f, c_params) for f in t_feats)
            rate = m / len(t_feats)
            trial_details[t_id] = f"{m}/{len(t_feats)} ({rate*100:.1f}%)"
            if rate >= 0.50:
                trial_passes += 1

        # O Fixture Evaluation
        o_fix_matches = sum(eval_candidate(f, c_params) for f in o_fix_feats)
        o_fix_acc = o_fix_matches / len(o_fix_feats) * 100.0

        # False Positives on Competing Letters (NPZ + Fixtures)
        fp_stats = {}
        for l in comp_letters:
            all_l_feats = comp_npz[l] + comp_fix[l]
            fps = sum(eval_candidate(f, c_params) for f in all_l_feats)
            fp_stats[l] = {
                "total": len(all_l_feats),
                "fps": fps,
                "fp_rate": fps / len(all_l_feats) * 100.0 if all_l_feats else 0.0
            }

        # False Positives on P Holdout & X Holdout
        p_holdout_fps = sum(eval_candidate(f, c_params) for f in p_holdout_feats)
        x_holdout_fps = sum(eval_candidate(f, c_params) for f in x_holdout_feats)

        print(f"O Pilot Frame Acc: {o_pilot_matches}/{len(all_o_pilot)} ({o_pilot_acc:.2f}%)")
        print(f"O Pilot Trial Acc: {trial_passes}/10 ({trial_passes/10*100:.1f}%)")
        print(f"O Fixture D0506 Frame Acc: {o_fix_matches}/{len(o_fix_feats)} ({o_fix_acc:.2f}%)")
        print("False Positives across Competing Letters:")
        for l, stat in fp_stats.items():
            print(f"  {l}: {stat['fps']}/{stat['total']} ({stat['fp_rate']:.2f}%)")
        print(f"P Holdout FPs: {p_holdout_fps}/{len(p_holdout_feats)} ({p_holdout_fps/len(p_holdout_feats)*100:.2f}%)")
        print(f"X Holdout FPs: {x_holdout_fps}/{len(x_holdout_feats)} ({x_holdout_fps/len(x_holdout_feats)*100:.2f}%)")

        results[c_name] = {
            "o_pilot_acc": o_pilot_acc,
            "o_pilot_matches": o_pilot_matches,
            "o_pilot_total": len(all_o_pilot),
            "trial_passes": trial_passes,
            "trial_details": trial_details,
            "o_fix_acc": o_fix_acc,
            "o_fix_matches": o_fix_matches,
            "o_fix_total": len(o_fix_feats),
            "fp_stats": fp_stats,
            "p_holdout_fps": p_holdout_fps,
            "x_holdout_fps": x_holdout_fps,
        }

    # Save results to scratch
    with open("scratch/task3_4_candidate_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("\nSaved full candidate evaluation results to scratch/task3_4_candidate_results.json")

if __name__ == "__main__":
    main()
