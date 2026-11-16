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

def make_evaluator(params):
    def eval_fn(feat):
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

        if "d_TI_max" in params and d_ti > params["d_TI_max"]: return False
        if "d_TM_max" in params and d_tm > params["d_TM_max"]: return False
        if "d_IW_min" in params and d_iw < params["d_IW_min"]: return False
        if "d_MW_min" in params and d_mw < params["d_MW_min"]: return False
        if "d_RW_min" in params and d_rw < params["d_RW_min"]: return False
        if "d_PW_min" in params and d_pw < params["d_PW_min"]: return False
        if "idx_straight_max" in params and idx_str > params["idx_straight_max"]: return False
        if "d_TW_min" in params and params["d_TW_min"] is not None and d_tw < params["d_TW_min"]: return False
        return True
    return eval_fn

# Define all configurations that have appeared across all scripts and texts:
CONFIGS = {
    # 1. Baseline
    "Config-Baseline (Production Rule 6)": {
        "d_TI_max": 0.22, "d_TM_max": 0.22, "d_IW_min": 0.95, "d_MW_min": 0.90,
        "d_RW_min": 0.80, "d_PW_min": 0.75, "idx_straight_max": 0.40, "d_TW_min": None
    },
    # 2. Candidate-01 in task3_4_candidate_validation.py & full_simulation.py & task3_4_8_audit.py
    "Config-C1-Actual (d_TM=0.35, idx=0.40)": {
        "d_TI_max": 0.22, "d_TM_max": 0.35, "d_IW_min": 0.95, "d_MW_min": 0.90,
        "d_RW_min": 0.80, "d_PW_min": 0.75, "idx_straight_max": 0.40, "d_TW_min": None
    },
    # 3. Candidate-02A in all scripts & TASK 3.4 report
    "Config-C2A-Actual (d_TM=0.35, idx=0.48)": {
        "d_TI_max": 0.22, "d_TM_max": 0.35, "d_IW_min": 0.95, "d_MW_min": 0.90,
        "d_RW_min": 0.80, "d_PW_min": 0.75, "idx_straight_max": 0.48, "d_TW_min": None
    },
    # 4. Candidate-02B in task3_4_candidate_validation.py (d_TI=0.25, d_TM=0.40, idx=0.48)
    "Config-C2B-Script1 (d_TI=0.25, d_TM=0.40, idx=0.48)": {
        "d_TI_max": 0.25, "d_TM_max": 0.40, "d_IW_min": 0.95, "d_MW_min": 0.90,
        "d_RW_min": 0.80, "d_PW_min": 0.75, "idx_straight_max": 0.48, "d_TW_min": None
    },
    # 5. Candidate-02C in task3_4_candidate_validation.py (d_TI=0.22, d_TM=0.38, idx=0.48)
    "Config-C2C-Script1 (d_TI=0.22, d_TM=0.38, idx=0.48)": {
        "d_TI_max": 0.22, "d_TM_max": 0.38, "d_IW_min": 0.95, "d_MW_min": 0.90,
        "d_RW_min": 0.80, "d_PW_min": 0.75, "idx_straight_max": 0.48, "d_TW_min": None
    },
    # 6. Candidate-02B in task3_4_full_simulation.py & task3_4_8_audit.py & TASK 3.4 report (d_TI=0.25, d_TM=0.38, idx=0.50)
    "Config-C2B-FullSim (d_TI=0.25, d_TM=0.38, idx=0.50)": {
        "d_TI_max": 0.25, "d_TM_max": 0.38, "d_IW_min": 0.95, "d_MW_min": 0.90,
        "d_RW_min": 0.80, "d_PW_min": 0.75, "idx_straight_max": 0.50, "d_TW_min": None
    },
    # 7. Candidate-02B described in table of TASK 3.4.8 text (d_TM=0.35, idx=0.44)
    "Config-C2B-TextMismatch (d_TM=0.35, idx=0.44)": {
        "d_TI_max": 0.22, "d_TM_max": 0.35, "d_IW_min": 0.95, "d_MW_min": 0.90,
        "d_RW_min": 0.80, "d_PW_min": 0.75, "idx_straight_max": 0.44, "d_TW_min": None
    },
    # 8. Candidate-03 in full_simulation.py, task3_4_8_audit.py, TASK 3.4 report, TASK 3.4.8 report
    "Config-C3-Actual (d_TM=0.35, idx=0.48, d_TW>=0.85)": {
        "d_TI_max": 0.22, "d_TM_max": 0.35, "d_IW_min": 0.95, "d_MW_min": 0.90,
        "d_RW_min": 0.80, "d_PW_min": 0.75, "idx_straight_max": 0.48, "d_TW_min": 0.85
    }
}

def main():
    print("=== TASK 3.4.9 CANDIDATE RECONCILIATION & AUDIT ===")

    # 1. Load Data
    # O Pilot
    pilot_path = "backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl"
    o_pilot_feats = []
    with open(pilot_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get("trial_id"):
                geom = data.get("geometric", {})
                feat = geom.get("features")
                if feat:
                    o_pilot_feats.append(feat)

    # O Fixture D0506
    fix_path = "backend/data/alphabet_v2/static_landmarks/O.json"
    with open(fix_path, 'r', encoding='utf-8') as f:
        fix_data = json.load(f)
    o_fix_hold = []
    for s in fix_data.get("samples", []):
        if s.get("phase") == "stable_pose" and s.get("landmarks"):
            feat = extract_feat(s["landmarks"])
            if feat:
                o_fix_hold.append(feat)

    # NPZ Competing
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    npz = np.load(npz_path, allow_pickle=True)
    classes = list(npz["classes"])
    y = npz["y"]
    lms = npz["landmarks"]
    meta = json.loads(npz["metadata_json"].item() if npz["metadata_json"].shape == () else npz["metadata_json"][0])

    comp_letters = ["A", "C", "S", "T", "E", "P", "X"]
    comp_npz = {l: [] for l in comp_letters}
    for idx, l_idx in enumerate(y):
        l = classes[l_idx]
        if l in comp_letters:
            feat = extract_feat(lms[idx])
            if feat:
                comp_npz[l].append((meta[idx], feat))

    # Fixtures Competing
    comp_fix = {l: [] for l in comp_letters}
    for l in comp_letters:
        fpath = f"backend/data/alphabet_v2/static_landmarks/{l}.json"
        with open(fpath, 'r', encoding='utf-8') as f:
            fdata = json.load(f)
        for s in fdata.get("samples", []):
            if s.get("landmarks"):
                feat = extract_feat(s["landmarks"])
                if feat:
                    comp_fix[l].append((s, feat))

    # P Holdout
    p_holdout_path = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"
    p_holdout_feats = []
    with open(p_holdout_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get("trial_id"):
                geom = data.get("geometric", {})
                feat = geom.get("features")
                if feat:
                    p_holdout_feats.append(feat)

    # X Holdout
    x_holdout_path = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
    x_holdout_feats = []
    with open(x_holdout_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            if data.get("trial_id"):
                geom = data.get("geometric", {})
                feat = geom.get("features")
                if feat:
                    x_holdout_feats.append(feat)

    total_obs = sum(len(comp_npz[l]) for l in comp_letters) + sum(len(comp_fix[l]) for l in comp_letters) + len(p_holdout_feats) + len(x_holdout_feats)
    print(f"Total competing observations = {total_obs}")

    print("\n" + "="*110)
    print(f"{'Config Name':<38} | {'O Pilot (92)':<12} | {'O D0506 (12)':<12} | {'NPZ FP (199)':<12} | {'Fix FP (173)':<12} | {'P/X (422)':<10} | {'Total FP (794)':<14}")
    print("="*110)

    eval_results = {}
    for c_name, params in CONFIGS.items():
        fn = make_evaluator(params)
        o_pilot_m = sum(fn(f) for f in o_pilot_feats)
        o_fix_m = sum(fn(f) for f in o_fix_hold)
        
        npz_fp = sum(sum(fn(f[1]) for f in comp_npz[l]) for l in comp_letters)
        fix_fp = sum(sum(fn(f[1]) for f in comp_fix[l]) for l in comp_letters)
        p_fp = sum(fn(f) for f in p_holdout_feats)
        x_fp = sum(fn(f) for f in x_holdout_feats)
        total_fp = npz_fp + fix_fp + p_fp + x_fp

        # Find which letters had FPs
        fp_details = []
        for l in comp_letters:
            l_npz = sum(fn(f[1]) for f in comp_npz[l])
            l_fix = sum(fn(f[1]) for f in comp_fix[l])
            if l_npz + l_fix > 0:
                fp_details.append(f"{l}: npz={l_npz}, fix={l_fix}")

        eval_results[c_name] = {
            "o_pilot": f"{o_pilot_m}/92 ({o_pilot_m/92*100:.1f}%)",
            "o_fix": f"{o_fix_m}/12 ({o_fix_m/12*100:.1f}%)",
            "npz_fp": npz_fp,
            "fix_fp": fix_fp,
            "px_fp": p_fp + x_fp,
            "total_fp": total_fp,
            "fp_details": fp_details
        }

        print(f"{c_name:<38} | {eval_results[c_name]['o_pilot']:<12} | {eval_results[c_name]['o_fix']:<12} | {npz_fp:<12} | {fix_fp:<12} | {p_fp+x_fp:<10} | {total_fp:<14} {fp_details}")

    print("="*110)

    # Now let's analyze unique samples vs observations
    print("\n=== SAMPLE UNIQUENESS & METADATA AUDIT ===")
    unique_studio_keys = set()
    for l in comp_letters:
        for m, f in comp_npz[l]:
            unique_studio_keys.add((m.get("video_id"), m.get("frame_index")))
        for s, f in comp_fix[l]:
            unique_studio_keys.add((s.get("video_id"), s.get("frame_index")))
    print(f"Unique (video_id, frame_index) keys in studio dataset across 7 letters: {len(unique_studio_keys)}")
    print(f"NPZ samples count: {sum(len(comp_npz[l]) for l in comp_letters)}")
    print(f"Fixture samples count: {sum(len(comp_fix[l]) for l in comp_letters)}")
    print(f"Overlap count: {sum(len(comp_npz[l]) for l in comp_letters) + sum(len(comp_fix[l]) for l in comp_letters) - len(unique_studio_keys)}")

    # Video clips count
    all_video_ids = set()
    for l in comp_letters:
        for m, f in comp_npz[l]:
            all_video_ids.add(m.get("video_id"))
        for s, f in comp_fix[l]:
            all_video_ids.add(s.get("video_id"))
    print(f"Unique studio video IDs across 7 letters: {len(all_video_ids)} -> {sorted(list(all_video_ids))}")

    # Save summary
    with open("scratch/task3_4_9_audit_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "eval_results": eval_results,
            "unique_studio_samples": len(unique_studio_keys),
            "total_observations": total_obs,
            "webcam_p_holdout_frames": len(p_holdout_feats),
            "webcam_x_holdout_frames": len(x_holdout_feats),
            "unique_samples_metadata": len(unique_studio_keys) + len(p_holdout_feats) + len(x_holdout_feats),
            "studio_video_ids": sorted(list(all_video_ids))
        }, f, indent=2, ensure_ascii=False)
    print("\nSaved summary to scratch/task3_4_9_audit_summary.json")

if __name__ == "__main__":
    main()

