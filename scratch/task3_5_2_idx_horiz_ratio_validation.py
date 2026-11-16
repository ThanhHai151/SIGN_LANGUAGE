import json
import os
import sys
import numpy as np
import hashlib

sys.path.insert(0, os.path.abspath("."))
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

def get_stats(vals):
    if not vals:
        return {"count": 0, "min": None, "max": None, "median": None, "p10": None, "p90": None}
    arr = np.array(vals)
    return {
        "count": len(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)),
        "p90": float(np.percentile(arr, 90)),
        "lt_020": int(np.sum(arr < 0.20)),
        "eq_020": int(np.sum(arr == 0.20)),
        "ge_020": int(np.sum(arr >= 0.20))
    }

def eval_cand03_base(feat):
    # Candidate-03 outer gate
    if feat.get("index_ext", False) or feat.get("middle_ext", False) or feat.get("ring_ext", False) or feat.get("pinky_ext", False):
        return False
    d_ti = feat.get("d_TI", 99.0)
    d_tm = feat.get("d_TM", 99.0)
    d_iw = feat.get("d_IW", 0.0)
    d_mw = feat.get("d_MW", 0.0)
    d_rw = feat.get("d_RW", 0.0)
    d_pw = feat.get("d_PW", 0.0)
    idx_str = feat.get("idx_straight", 1.0)
    d_tw = feat.get("d_TW", 0.0)

    return (
        d_ti <= 0.22 and
        d_tm <= 0.35 and
        idx_str <= 0.48 and
        d_iw >= 0.95 and
        d_mw >= 0.90 and
        d_rw >= 0.80 and
        d_pw >= 0.75 and
        d_tw >= 0.85
    )

def main():
    print("=== TASK 3.5.2 READ-ONLY VALIDATION OF idx_horiz_ratio ===")

    # 1. Dataset SHA-256 Hashes
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    with open(npz_path, "rb") as f:
        npz_sha256 = hashlib.sha256(f.read()).hexdigest()
    print(f"NPZ Path: {npz_path}")
    print(f"NPZ SHA256: {npz_sha256}")

    # Load NPZ
    npz = np.load(npz_path, allow_pickle=True)
    classes = list(npz["classes"])
    y = npz["y"]
    landmarks = npz["landmarks"]
    metadata_json = npz["metadata_json"]
    meta = json.loads(metadata_json.item() if metadata_json.shape == () else metadata_json[0])

    # -------------------------------------------------------------
    # SECTION B: AUDIT ALL O SAMPLES ACROSS SOURCES
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("--- SECTION B: AUDIT ALL O SAMPLES ACROSS INDEPENDENT SOURCES ---")
    print("="*80)

    # Nguồn B.1: O trong NPZ
    o_indices_npz = [i for i, label_idx in enumerate(y) if classes[label_idx] == "O"]
    o_npz_by_video = {}
    o_npz_all_ratios = []
    o_npz_details = []
    for idx in o_indices_npz:
        m = meta[idx]
        vid = m.get("video_id", "unknown")
        lm = landmarks[idx].tolist()
        feat = clf.extract_features(lm)
        ratio = feat.get("idx_horiz_ratio", 0.0)
        o_npz_all_ratios.append(ratio)
        if vid not in o_npz_by_video:
            o_npz_by_video[vid] = []
        o_npz_by_video[vid].append((idx, m.get("frame_index"), ratio, feat))
        o_npz_details.append((idx, vid, m.get("frame_index"), ratio, feat))

    print(f"\n1. O trong NPZ: Tổng cộng {len(o_indices_npz)} mẫu across {len(o_npz_by_video)} videos:")
    for vid, items in o_npz_by_video.items():
        v_ratios = [x[2] for x in items]
        st = get_stats(v_ratios)
        print(f"   Video {vid:<10} (N={len(items):>2}): min={st['min']:.4f}, med={st['median']:.4f}, max={st['max']:.4f}, P10={st['p10']:.4f}, P90={st['p90']:.4f} | <0.20: {st['lt_020']}, >=0.20: {st['ge_020']}")

    st_npz_all = get_stats(o_npz_all_ratios)
    print(f"   --> TỔNG NPZ O (N={len(o_npz_all_ratios)}): min={st_npz_all['min']:.4f}, med={st_npz_all['median']:.4f}, max={st_npz_all['max']:.4f}, P10={st_npz_all['p10']:.4f}, P90={st_npz_all['p90']:.4f}")
    print(f"       Số mẫu < 0.20: {st_npz_all['lt_020']}, == 0.20: {st_npz_all['eq_020']}, >= 0.20: {st_npz_all['ge_020']}")
    if st_npz_all['lt_020'] > 0:
        print("       CÁC MẪU NPZ BỊ LOẠI:")
        for idx, vid, fidx, r, f in o_npz_details:
            if r < 0.20:
                print(f"         Idx {idx}: video={vid}, frame={fidx}, ratio={r:.4f}")

    # Nguồn B.2: O trong Fixture O.json (D0506)
    fix_path = "backend/data/alphabet_v2/static_landmarks/O.json"
    with open(fix_path, "r", encoding="utf-8") as f:
        o_fix_data = json.load(f)
    o_fix_samples = o_fix_data.get("samples", [])
    o_fix_hold_ratios = []
    o_fix_all_ratios = []
    for s in o_fix_samples:
        if s.get("landmarks"):
            feat = clf.extract_features(s["landmarks"])
            r = feat.get("idx_horiz_ratio", 0.0)
            o_fix_all_ratios.append((s.get("frame_index"), s.get("phase"), r, feat))
            if s.get("phase") == "stable_pose":
                o_fix_hold_ratios.append(r)

    st_fix_hold = get_stats(o_fix_hold_ratios)
    st_fix_all = get_stats([x[2] for x in o_fix_all_ratios])
    print(f"\n2. O trong Fixture O.json (Video D0506, Signer qipedc_signer_01):")
    print(f"   Hold Frames (stable_pose, N={len(o_fix_hold_ratios)}): min={st_fix_hold['min']:.4f}, med={st_fix_hold['median']:.4f}, max={st_fix_hold['max']:.4f}, P10={st_fix_hold['p10']:.4f}, P90={st_fix_hold['p90']:.4f}")
    print(f"       Số mẫu < 0.20: {st_fix_hold['lt_020']}, == 0.20: {st_fix_hold['eq_020']}, >= 0.20: {st_fix_hold['ge_020']}")
    print(f"   All Valid Frames (N={len(o_fix_all_ratios)}): min={st_fix_all['min']:.4f}, med={st_fix_all['median']:.4f}, max={st_fix_all['max']:.4f}, P10={st_fix_all['p10']:.4f}, P90={st_fix_all['p90']:.4f}")
    print(f"       Số mẫu < 0.20: {st_fix_all['lt_020']}, == 0.20: {st_fix_all['eq_020']}, >= 0.20: {st_fix_all['ge_020']}")
    if st_fix_all['lt_020'] > 0:
        print("       CÁC MẪU FIXTURE BỊ LOẠI:")
        for fidx, ph, r, f in o_fix_all_ratios:
            if r < 0.20:
                print(f"         Frame {fidx} ({ph}): ratio={r:.4f}")

    # Nguồn B.3: O Webcam Pilot 213800 (pilot_pilot_20261108_213800.jsonl)
    pilot_path_2138 = "backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl"
    pilot_2138_ratios = []
    pilot_2138_trials = {}
    with open(pilot_path_2138, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t_id = r.get("trial_id")
            if t_id and r.get("geometric", {}).get("features"):
                feat = r["geometric"]["features"]
                ratio = feat.get("idx_horiz_ratio", 0.0)
                pilot_2138_ratios.append((r.get("frame_idx"), t_id, ratio, feat))
                if t_id not in pilot_2138_trials:
                    pilot_2138_trials[t_id] = []
                pilot_2138_trials[t_id].append(ratio)

    st_p2138 = get_stats([x[2] for x in pilot_2138_ratios])
    print(f"\n3. O Webcam Pilot 213800 (Exploratory Dataset, N={len(pilot_2138_ratios)} active frames across 10 trials):")
    print(f"   Toàn bộ active frames: min={st_p2138['min']:.4f}, med={st_p2138['median']:.4f}, max={st_p2138['max']:.4f}, P10={st_p2138['p10']:.4f}, P90={st_p2138['p90']:.4f}")
    print(f"   Số mẫu < 0.20: {st_p2138['lt_020']}, == 0.20: {st_p2138['eq_020']}, >= 0.20: {st_p2138['ge_020']}")

    # Nguồn B.4: O Webcam Pilot 211742 (pilot_pilot_20261108_193712_20261108_211742_1.jsonl)
    pilot_path_2117 = "backend/data/diagnostics/pilot_pilot_20261108_193712_20261108_211742_1.jsonl"
    pilot_2117_ratios = []
    with open(pilot_path_2117, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            t_id = r.get("trial_id")
            if t_id and r.get("geometric", {}).get("features"):
                feat = r["geometric"]["features"]
                ratio = feat.get("idx_horiz_ratio", 0.0)
                pilot_2117_ratios.append((r.get("frame_idx"), t_id, ratio, feat))

    st_p2117 = get_stats([x[2] for x in pilot_2117_ratios])
    print(f"\n4. O Webcam Pilot 211742 (Phụ trợ trước đó, N={len(pilot_2117_ratios)} active frames across 10 trials):")
    print(f"   Toàn bộ active frames: min={st_p2117['min']:.4f}, med={st_p2117['median']:.4f}, max={st_p2117['max']:.4f}, P10={st_p2117['p10']:.4f}, P90={st_p2117['p90']:.4f}")
    print(f"   Số mẫu < 0.20: {st_p2117['lt_020']}, == 0.20: {st_p2117['eq_020']}, >= 0.20: {st_p2117['ge_020']}")

    # -------------------------------------------------------------
    # SECTION C: AUDIT ALL 21 NON-O CLASSES IN NPZ
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("--- SECTION C: AUDIT ALL 21 NON-O CLASSES IN NPZ STATIC V2 (N=503) ---")
    print("="*80)

    non_o_classes = [c for c in classes if c != "O"]
    non_o_samples = {}
    for idx in range(len(y)):
        cls = classes[y[idx]]
        if cls != "O":
            if cls not in non_o_samples:
                non_o_samples[cls] = []
            lm = landmarks[idx].tolist()
            feat = clf.extract_features(lm)
            ratio = feat.get("idx_horiz_ratio", 0.0)
            non_o_samples[cls].append((idx, meta[idx], ratio, feat))

    print(f"{'Class':<6} | {'Total':<6} | {'idx_horiz >= 0.20':<20} | {'Cand-03 Pass':<15} | {'Cand-03 + horiz>=0.20 Pass':<28}")
    print("-" * 85)

    non_o_cand03_fps = {}
    non_o_new_fps = {}

    for cls in non_o_classes:
        items = non_o_samples[cls]
        total = len(items)
        ge_020 = sum(1 for x in items if x[2] >= 0.20)
        c3_pass = sum(1 for x in items if eval_cand03_base(x[3]))
        new_pass = sum(1 for x in items if eval_cand03_base(x[3]) and x[2] >= 0.20)
        
        non_o_cand03_fps[cls] = c3_pass
        non_o_new_fps[cls] = new_pass
        
        c3_str = f"{c3_pass}/{total}"
        new_str = f"{new_pass}/{total}"
        print(f"{cls:<6} | {total:<6} | {ge_020:>3}/{total:<16} | {c3_str:<15} | {new_str:<28}")

    # Đặc biệt khảo sát chữ B
    print("\nChi tiết chữ B (D0492) trong NPZ:")
    b_items = non_o_samples["B"]
    for idx, m, r, f in b_items:
        fidx = m.get("frame_index")
        c3_match = eval_cand03_base(f)
        new_match = c3_match and (r >= 0.20)
        print(f"  Idx {idx:<2}: frame={fidx:<2}, horiz_ratio={r:.4f}, Cand03={c3_match}, Cand03+horiz={new_match}")

    # -------------------------------------------------------------
    # SECTION D: OFFLINE SIMULATION OF 3 CONFIGS & SENSITIVITY
    # -------------------------------------------------------------
    print("\n" + "="*80)
    print("--- SECTION D: OFFLINE SIMULATION OF 3 CONFIGS & SENSITIVITY ---")
    print("="*80)

    # Threshold sweeps for sensitivity: None (Cand03), 0.10, 0.15, 0.20, 0.25, 0.30
    thresholds = [None, 0.10, 0.15, 0.20, 0.25, 0.30]

    # Evaluated sets:
    # 1. O Pilot 2138 (N=92)
    # 2. O Pilot 2117 (N=68)
    # 3. O Fixture D0506 Hold (N=12)
    # 4. O NPZ (N=48)
    # 5. Non-O NPZ (N=503)
    # 6. P Holdout (N=314)
    # 7. X Holdout (N=108)

    # Load P & X holdouts
    p_holdout = []
    with open("backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("trial_id") and r.get("geometric", {}).get("features"):
                p_holdout.append(r["geometric"]["features"])

    x_holdout = []
    with open("backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r.get("trial_id") and r.get("geometric", {}).get("features"):
                x_holdout.append(r["geometric"]["features"])

    print(f"\nSimulation across thresholds: [None (Cand03), 0.10, 0.15, 0.20, 0.25, 0.30]")
    header = f"{'Threshold':<16} | {'O Pilot 2138':<14} | {'O Pilot 2117':<14} | {'O Fix D0506':<13} | {'O NPZ (48)':<12} | {'B f54':<8} | {'NPZ Non-O FP':<14} | {'P Holdout':<11} | {'X Holdout':<11}"
    print(header)
    print("-" * 125)

    sim_results = {}
    for th in thresholds:
        th_label = "Cand-03 (None)" if th is None else f"+ horiz>={th:.2f}"
        
        def test_cond(f):
            if not eval_cand03_base(f): return False
            if th is not None:
                if f.get("idx_horiz_ratio", 0.0) < th: return False
            return True

        m_p2138 = sum(1 for x in pilot_2138_ratios if test_cond(x[3]))
        m_p2117 = sum(1 for x in pilot_2117_ratios if test_cond(x[3]))
        m_fix = sum(1 for x in o_fix_all_ratios if x[1] == "stable_pose" and test_cond(x[3]))
        m_onpz = sum(1 for x in o_npz_details if test_cond(x[4]))
        
        # B f54 specifically
        b_f54_feat = next(x[3] for x in b_items if x[1].get("frame_index") == 54)
        b_f54_match = test_cond(b_f54_feat)
        b_f54_str = "FP" if b_f54_match else "BLOCKED"

        # Non-O NPZ (503 samples)
        non_o_fps = sum(sum(1 for x in non_o_samples[cls] if test_cond(x[3])) for cls in non_o_classes)
        
        # P & X Holdouts
        p_fps = sum(1 for f in p_holdout if test_cond(f))
        x_fps = sum(1 for f in x_holdout if test_cond(f))

        sim_results[th_label] = {
            "o_pilot_2138": f"{m_p2138}/92 ({m_p2138/92*100:.1f}%)",
            "o_pilot_2117": f"{m_p2117}/68 ({m_p2117/68*100:.1f}%)",
            "o_fix_d0506": f"{m_fix}/12 ({m_fix/12*100:.1f}%)",
            "o_npz": f"{m_onpz}/48 ({m_onpz/48*100:.1f}%)",
            "b_f54": b_f54_str,
            "non_o_npz_fp": f"{non_o_fps}/503",
            "p_holdout_fp": f"{p_fps}/314",
            "x_holdout_fp": f"{x_fps}/108"
        }

        print(f"{th_label:<16} | {m_p2138:>2}/92 ({m_p2138/92*100:.1f}%) | {m_p2117:>2}/68 ({m_p2117/68*100:.1f}%) | {m_fix:>2}/12 ({m_fix/12*100:.1f}%) | {m_onpz:>2}/48 ({m_onpz/48*100:.1f}%) | {b_f54_str:<8} | {non_o_fps:>2}/503 ({non_o_fps/503*100:.2f}%) | {p_fps:>2}/314      | {x_fps:>2}/108")

    # Save detailed JSON summary
    out_artifact = "scratch/task3_5_2_validation_summary.json"
    with open(out_artifact, "w", encoding="utf-8") as f:
        json.dump({
            "st_npz_all": st_npz_all,
            "st_fix_hold": st_fix_hold,
            "st_fix_all": st_fix_all,
            "st_p2138": st_p2138,
            "st_p2117": st_p2117,
            "sim_results": sim_results
        }, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full artifact to {out_artifact}")

if __name__ == "__main__":
    main()

