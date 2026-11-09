import json
import os
import sys
import numpy as np
import hashlib

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

def eval_cand03(feat):
    if not outer_gate_pass(feat):
        return False
    return (
        feat.get("d_TI", 99.0) <= 0.22 and
        feat.get("d_TM", 99.0) <= 0.35 and
        feat.get("d_IW", 0.0) >= 0.95 and
        feat.get("d_MW", 0.0) >= 0.90 and
        feat.get("d_RW", 0.0) >= 0.80 and
        feat.get("d_PW", 0.0) >= 0.75 and
        feat.get("idx_straight", 1.0) <= 0.48 and
        feat.get("d_TW", 0.0) >= 0.85
    )

def eval_cand02a(feat):
    if not outer_gate_pass(feat):
        return False
    return (
        feat.get("d_TI", 99.0) <= 0.22 and
        feat.get("d_TM", 99.0) <= 0.35 and
        feat.get("d_IW", 0.0) >= 0.95 and
        feat.get("d_MW", 0.0) >= 0.90 and
        feat.get("d_RW", 0.0) >= 0.80 and
        feat.get("d_PW", 0.0) >= 0.75 and
        feat.get("idx_straight", 1.0) <= 0.48
    )

def eval_cand01(feat):
    if not outer_gate_pass(feat):
        return False
    return (
        feat.get("d_TI", 99.0) <= 0.22 and
        feat.get("d_TM", 99.0) <= 0.35 and
        feat.get("d_IW", 0.0) >= 0.95 and
        feat.get("d_MW", 0.0) >= 0.90 and
        feat.get("d_RW", 0.0) >= 0.80 and
        feat.get("d_PW", 0.0) >= 0.75 and
        feat.get("idx_straight", 1.0) <= 0.40
    )

def eval_cand02b(feat):
    if not outer_gate_pass(feat):
        return False
    return (
        feat.get("d_TI", 99.0) <= 0.25 and
        feat.get("d_TM", 99.0) <= 0.38 and
        feat.get("d_IW", 0.0) >= 0.95 and
        feat.get("d_MW", 0.0) >= 0.90 and
        feat.get("d_RW", 0.0) >= 0.80 and
        feat.get("d_PW", 0.0) >= 0.75 and
        feat.get("idx_straight", 1.0) <= 0.50
    )

def main():
    print("=== TASK 3.4.8 AUDIT & REPRODUCIBILITY VERIFICATION ===")

    # 1. Audit NPZ Data
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    npz = np.load(npz_path, allow_pickle=True)
    classes = list(npz["classes"])
    y = npz["y"]
    lms = npz["landmarks"]
    meta = json.loads(npz["metadata_json"].item() if npz["metadata_json"].shape == () else npz["metadata_json"][0])

    comp_letters = ["A", "C", "S", "T", "E", "P", "X"]
    npz_samples = {l: [] for l in comp_letters}
    npz_metadata = {l: [] for l in comp_letters}
    for idx, l_idx in enumerate(y):
        l = classes[l_idx]
        if l in comp_letters:
            feat = extract_feat(lms[idx])
            if feat:
                npz_samples[l].append((lms[idx], feat))
                npz_metadata[l].append(meta[idx])

    # 2. Audit Fixture Data
    fix_samples = {l: [] for l in comp_letters}
    for l in comp_letters:
        fpath = f"backend/data/alphabet_v2/static_landmarks/{l}.json"
        with open(fpath, "r", encoding="utf-8") as f:
            fdata = json.load(f)
        for s in fdata.get("samples", []):
            if s.get("landmarks"):
                feat = extract_feat(s["landmarks"])
                if feat:
                    fix_samples[l].append((s, feat))

    # 3. Audit Webcam Holdouts
    p_holdout_path = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"
    p_holdout = []
    with open(p_holdout_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("trial_id"):
                geom = data.get("geometric", {})
                feat = geom.get("features")
                if feat:
                    p_holdout.append(feat)

    x_holdout_path = "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl"
    x_holdout = []
    with open(x_holdout_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            if data.get("trial_id"):
                geom = data.get("geometric", {})
                feat = geom.get("features")
                if feat:
                    x_holdout.append(feat)

    # -------------------------------------------------------------
    # TASK 3.4.8.1: Đối chiếu tổng số mẫu Cross-Letter
    # -------------------------------------------------------------
    print("\n--- 3.4.8.1: CROSS-LETTER SAMPLE COUNT AUDIT ---")
    print(f"{'Chữ':<5} | {'NPZ':<6} | {'Fixture':<8} | {'Webcam':<8} | {'Tổng tính lại':<14} | {'Báo cáo cũ':<12}")
    print("-" * 65)
    
    old_reported = {
        "A": 87, "C": 40, "S": 31, "T": 43, "E": 55, "P": 357, "X": 181
    }

    total_npz = 0
    total_fix = 0
    total_webcam = 0
    total_recalc = 0
    total_old = 0

    for l in comp_letters:
        n_npz = len(npz_samples[l])
        n_fix = len(fix_samples[l])
        n_wc = len(p_holdout) if l == "P" else (len(x_holdout) if l == "X" else 0)
        recalc = n_npz + n_fix + n_wc
        old = old_reported[l]

        total_npz += n_npz
        total_fix += n_fix
        total_webcam += n_wc
        total_recalc += recalc
        total_old += old

        print(f"{l:<5} | {n_npz:<6} | {n_fix:<8} | {n_wc:<8} | {recalc:<14} | {old:<12}")

    print("-" * 65)
    print(f"{'TỔNG':<5} | {total_npz:<6} | {total_fix:<8} | {total_webcam:<8} | {total_recalc:<14} | {total_old:<12}")
    print(f"Tổng thực tế của 7 chữ = {total_recalc}")
    print(f"Tổng theo bảng Section 2 của báo cáo TASK 3.4 = {total_old}")
    print(f"Tại sao báo cáo cũ ghi 714? Kiểm tra: {total_recalc} - 714 = {total_recalc - 714}")

    # -------------------------------------------------------------
    # TASK 3.4.8.3: Kiểm tra Trùng lặp giữa NPZ và Fixture
    # -------------------------------------------------------------
    print("\n--- 3.4.8.3: DUPLICATE & INDEPENDENCE CHECK (NPZ vs FIXTURE) ---")
    duplicate_counts = {}
    for l in comp_letters:
        # Check video IDs and frame indices
        fix_keys = set((s[0].get("video_id"), s[0].get("frame_index")) for s in fix_samples[l])
        npz_keys = set((m.get("video_id"), m.get("frame_index")) for m in npz_metadata[l])
        overlap = fix_keys.intersection(npz_keys)
        duplicate_counts[l] = len(overlap)
        print(f"Letter {l}:")
        print(f"  Fixture video_ids: {set(k[0] for k in fix_keys)}")
        print(f"  NPZ video_ids:     {set(k[0] for k in npz_keys)}")
        print(f"  Exact (video_id, frame_index) overlap: {len(overlap)} / {len(fix_keys)} fixture frames")
        if overlap:
            print(f"  Sample overlapping frames: {list(overlap)[:3]}")

    # -------------------------------------------------------------
    # TASK 3.4.8.2: Xác minh Mẫu số False Positive và Tái lập
    # -------------------------------------------------------------
    print("\n--- 3.4.8.2 & 3.4.8.5: RECALCULATING FALSE POSITIVES ---")
    cands = {
        "Baseline": clf.evaluate_rules, # Note: using classifier's O rule
        "Cand-01": eval_cand01,
        "Cand-02A": eval_cand02a,
        "Cand-02B": eval_cand02b,
        "Cand-03": eval_cand03
    }

    for c_name, c_fn in cands.items():
        print(f"\nCandidate: {c_name}")
        fp_fix_npz = 0
        total_fix_npz = 0
        fp_details = []
        for l in comp_letters:
            all_l = [x[1] for x in npz_samples[l]] + [x[1] for x in fix_samples[l]]
            total_fix_npz += len(all_l)
            l_fps = 0
            for idx, feat in enumerate(all_l):
                if c_name == "Baseline":
                    # Evaluate Rule 6
                    matches = clf.evaluate_rules(feat)
                    is_match = any(m[0] == "O" for m in matches)
                else:
                    is_match = c_fn(feat)
                if is_match:
                    l_fps += 1
                    fp_fix_npz += 1
                    fp_details.append((l, idx))
            print(f"  FP on {l}: {l_fps}/{len(all_l)}")
        
        # Check P Holdout
        p_fps = 0
        for feat in p_holdout:
            if c_name == "Baseline":
                m = clf.evaluate_rules(feat)
                is_match = any(x[0] == "O" for x in m)
            else:
                is_match = c_fn(feat)
            if is_match: p_fps += 1

        # Check X Holdout
        x_fps = 0
        for feat in x_holdout:
            if c_name == "Baseline":
                m = clf.evaluate_rules(feat)
                is_match = any(x[0] == "O" for x in m)
            else:
                is_match = c_fn(feat)
            if is_match: x_fps += 1

        print(f"  FP on P Holdout: {p_fps}/{len(p_holdout)}")
        print(f"  FP on X Holdout: {x_fps}/{len(x_holdout)}")
        print(f"  TOTAL FP (on {total_recalc} samples): {fp_fix_npz + p_fps + x_fps}/{total_recalc}")
        if fp_details:
            print(f"  Details of FPs: {fp_details}")

    # -------------------------------------------------------------
    # TASK 3.4.8.4: Xác minh Tiêu chí Trial PASS trên O Pilot
    # -------------------------------------------------------------
    print("\n--- 3.4.8.4: O PILOT TRIAL-BY-TRIAL AUDIT (Cand-03) ---")
    pilot_path = "backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl"
    trials = {}
    with open(pilot_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            t_id = data.get("trial_id")
            if t_id:
                if t_id not in trials:
                    trials[t_id] = []
                trials[t_id].append(data)

    print(f"{'Trial ID':<32} | {'Total':<6} | {'Correct O':<10} | {'Rejected':<10} | {'Wrong':<6} | {'Accuracy':<10} | {'Status':<15}")
    print("-" * 95)
    for t_id, recs in sorted(trials.items()):
        total = len(recs)
        correct_o = 0
        rejected = 0
        wrong = 0
        for r in recs:
            feat = r.get("geometric", {}).get("features")
            if feat and eval_cand03(feat):
                correct_o += 1
            else:
                rejected += 1
        acc = correct_o / total * 100.0
        # Check trial pass
        status = "PASS (>=50%)" if acc >= 50.0 else "FAIL (<50%)"
        print(f"{t_id:<32} | {total:<6} | {correct_o:<10} | {rejected:<10} | {wrong:<6} | {acc:.1f}%{'':<5} | {status:<15}")

if __name__ == "__main__":
    main()

