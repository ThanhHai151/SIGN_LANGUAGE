import json
import os
import sys
import numpy as np
import hashlib

sys.path.insert(0, os.path.abspath("."))
from backend.vision.alphabet_classifier import (
    AlphabetClassifier,
    O_D_TI_MAX, O_D_TM_MAX, O_D_IW_MIN, O_D_MW_MIN,
    O_D_RW_MIN, O_D_PW_MIN, O_IDX_STRAIGHT_MAX
)

clf = AlphabetClassifier()

def extract_feat(lms):
    if isinstance(lms, np.ndarray):
        lms = lms.tolist()
    return clf.extract_features(lms)

def main():
    print("=== TASK 3.4 FEATURE DISTRIBUTION & SEPARATION ANALYSIS ===")
    
    # 1. Load O Pilot Data
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
    print(f"Loaded {len(o_pilot_feats)} valid active frames from O pilot.")

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
    print(f"Loaded {len(o_fix_feats)} valid frames from O fixture D0506.")

    # 3. Load NPZ Dataset (551 samples)
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    npz = np.load(npz_path, allow_pickle=True)
    classes = list(npz["classes"])
    y = npz["y"]
    lms = npz["landmarks"]
    meta = json.loads(npz["metadata_json"].item() if npz["metadata_json"].shape == () else npz["metadata_json"][0])

    competing_letters = ["A", "C", "S", "T", "E", "P", "X"]
    comp_feats_npz = {letter: [] for letter in competing_letters}
    for idx, label_idx in enumerate(y):
        letter = classes[label_idx]
        if letter in competing_letters:
            feat = extract_feat(lms[idx])
            if feat:
                comp_feats_npz[letter].append(feat)

    print("\nNPZ Dataset Competing Letters sample counts:")
    for l, feats in comp_feats_npz.items():
        print(f"  {l}: {len(feats)} samples")

    # 4. Load Fixtures for Competing Letters
    comp_feats_fix = {letter: [] for letter in competing_letters}
    for letter in competing_letters:
        fpath = f"backend/data/alphabet_v2/static_landmarks/{letter}.json"
        if os.path.exists(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                fdata = json.load(f)
            for s in fdata.get("samples", []):
                if s.get("landmarks"):
                    feat = extract_feat(s["landmarks"])
                    if feat:
                        comp_feats_fix[letter].append(feat)
        print(f"Fixture {letter}: {len(comp_feats_fix[letter])} valid frames")

    # Features to compare:
    keys = ["d_TI", "d_TM", "d_IW", "d_MW", "d_RW", "d_PW", "idx_straight"]

    def get_stats(feats, k):
        vals = [f[k] for f in feats if k in f]
        if not vals:
            return "N/A"
        return f"[{min(vals):.3f}, {np.median(vals):.3f}, {max(vals):.3f}]"

    print("\n=== FEATURE DISTRIBUTIONS: [Min, Median, Max] ===")
    print(f"{'Letter/Source':<18} | {'d_TI':<16} | {'d_TM':<16} | {'idx_straight':<16} | {'d_IW':<16} | {'d_MW':<16}")
    print("-" * 95)
    print(f"{'O (Pilot N=92)':<18} | {get_stats(o_pilot_feats, 'd_TI'):<16} | {get_stats(o_pilot_feats, 'd_TM'):<16} | {get_stats(o_pilot_feats, 'idx_straight'):<16} | {get_stats(o_pilot_feats, 'd_IW'):<16} | {get_stats(o_pilot_feats, 'd_MW'):<16}")
    print(f"{'O (Fix D0506 N=20)':<18} | {get_stats(o_fix_feats, 'd_TI'):<16} | {get_stats(o_fix_feats, 'd_TM'):<16} | {get_stats(o_fix_feats, 'idx_straight'):<16} | {get_stats(o_fix_feats, 'd_IW'):<16} | {get_stats(o_fix_feats, 'd_MW'):<16}")
    
    for l in competing_letters:
        all_comp = comp_feats_npz[l] + comp_feats_fix[l]
        print(f"{l + ' (NPZ+Fix N=' + str(len(all_comp)) + ')':<18} | {get_stats(all_comp, 'd_TI'):<16} | {get_stats(all_comp, 'd_TM'):<16} | {get_stats(all_comp, 'idx_straight'):<16} | {get_stats(all_comp, 'd_IW'):<16} | {get_stats(all_comp, 'd_MW'):<16}")

    # Outer Gate Pass Rates:
    def outer_gate_pass(feat):
        return not (feat.get("index_ext", False) or feat.get("middle_ext", False) or 
                    feat.get("ring_ext", False) or feat.get("pinky_ext", False))

    print("\n=== OUTER GATE PASS RATE (not i_ext and not m_ext and not r_ext and not p_ext) ===")
    print(f"O Pilot: {sum(outer_gate_pass(f) for f in o_pilot_feats)}/{len(o_pilot_feats)} ({sum(outer_gate_pass(f) for f in o_pilot_feats)/len(o_pilot_feats)*100:.1f}%)")
    print(f"O Fixture: {sum(outer_gate_pass(f) for f in o_fix_feats)}/{len(o_fix_feats)} ({sum(outer_gate_pass(f) for f in o_fix_feats)/len(o_fix_feats)*100:.1f}%)")
    for l in competing_letters:
        all_comp = comp_feats_npz[l] + comp_feats_fix[l]
        n_pass = sum(outer_gate_pass(f) for f in all_comp)
        print(f"Letter {l}: {n_pass}/{len(all_comp)} ({n_pass/len(all_comp)*100:.1f}%) pass outer gate")

if __name__ == "__main__":
    main()
