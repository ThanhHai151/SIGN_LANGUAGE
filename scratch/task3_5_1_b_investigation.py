import json
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

clf = VSLAlphabetClassifier()

def main():
    print("=== TASK 3.5.1 INVESTIGATION: B -> O FALSE POSITIVE ===")

    # 1. Target Sample 39 in NPZ
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    npz = np.load(npz_path, allow_pickle=True)
    classes = list(npz["classes"])
    y = npz["y"]
    landmarks = npz["landmarks"]
    metadata_json = npz["metadata_json"]
    meta = json.loads(metadata_json.item() if metadata_json.shape == () else metadata_json[0])

    s39_meta = meta[39]
    s39_label = classes[y[39]]
    s39_lms = landmarks[39].tolist()
    s39_feat = clf.extract_features(s39_lms)
    s39_rules = clf.evaluate_rules(s39_feat)
    s39_diag = clf.evaluate_rule_o_diagnostics(s39_feat)
    s39_detailed = clf.classify_detailed(s39_lms)

    print("\n--- A. METADATA & IDENTITY OF SAMPLE 39 IN NPZ ---")
    print(f"NPZ Index: 39")
    print(f"True Label: {s39_label} (label_idx = {y[39]})")
    print(f"Metadata: {json.dumps(s39_meta, indent=2)}")
    print(f"\nPredicted Letter: {s39_detailed['letter']} (confidence = {s39_detailed['confidence']})")
    print(f"Status: {s39_detailed['status']}")
    print(f"Reason: {s39_detailed['reason']}")
    print(f"Rules Matched: {s39_rules}")
    print(f"Rule O Diagnostics: {s39_diag}")

    # 2. 8 Geometric Conditions & Outer Gate on Sample 39
    print("\n--- B. 8 GEOMETRIC CONDITIONS & OUTER GATE FOR SAMPLE 39 ---")
    outer_gate = not (s39_feat["index_ext"] or s39_feat["middle_ext"] or s39_feat["ring_ext"] or s39_feat["pinky_ext"])
    print(f"Outer Gate (not i_ext and not m_ext and not r_ext and not p_ext): {outer_gate}")
    print(f"  index_ext:  {s39_feat['index_ext']}")
    print(f"  middle_ext: {s39_feat['middle_ext']}")
    print(f"  ring_ext:   {s39_feat['ring_ext']}")
    print(f"  pinky_ext:  {s39_feat['pinky_ext']}")

    cond_checks = [
        ("d_TI <= 0.22", s39_feat["d_TI"], s39_feat["d_TI"] <= 0.22),
        ("d_TM <= 0.35", s39_feat["d_TM"], s39_feat["d_TM"] <= 0.35),
        ("idx_straight <= 0.48", s39_feat["idx_straight"], s39_feat["idx_straight"] <= 0.48),
        ("d_IW >= 0.95", s39_feat["d_IW"], s39_feat["d_IW"] >= 0.95),
        ("d_MW >= 0.90", s39_feat["d_MW"], s39_feat["d_MW"] >= 0.90),
        ("d_RW >= 0.80", s39_feat["d_RW"], s39_feat["d_RW"] >= 0.80),
        ("d_PW >= 0.75", s39_feat["d_PW"], s39_feat["d_PW"] >= 0.75),
        ("d_TW >= 0.85", s39_feat["d_TW"], s39_feat["d_TW"] >= 0.85),
    ]
    for c_name, val, is_pass in cond_checks:
        status_str = "PASS" if is_pass else "FAIL"
        print(f"  {c_name:<25}: actual = {val:.4f} -> {status_str}")

    # 3. Trace entire video D0492 in B.json
    print("\n--- C. KINEMATIC TIMELINE OF VIDEO D0492 (B.json) ---")
    b_path = "backend/data/alphabet_v2/static_landmarks/B.json"
    with open(b_path, "r", encoding="utf-8") as f:
        b_fixture = json.load(f)

    print(f"{'Frame':<6} | {'Phase':<12} | {'Qual':<6} | {'Ext [i,m,r,p]':<16} | {'d_TI':<7} | {'d_TM':<7} | {'idx_str':<8} | {'d_TW':<7} | {'Wrist Y':<8} | {'Rule O Match':<12}")
    print("-" * 105)

    all_b_samples_status = []
    for s in b_fixture.get("samples", []):
        fidx = s.get("frame_index")
        ph = s.get("phase")
        qual = s.get("quality_status")
        lms = s.get("landmarks")
        if lms:
            feat = clf.extract_features(lms)
            exts = f"[{int(feat['index_ext'])},{int(feat['middle_ext'])},{int(feat['ring_ext'])},{int(feat['pinky_ext'])}]"
            d_ti = f"{feat['d_TI']:.3f}"
            d_tm = f"{feat['d_TM']:.3f}"
            idx_str = f"{feat['idx_straight']:.3f}"
            d_tw = f"{feat['d_TW']:.3f}"
            wrist_y = f"{lms[0][1]:.3f}"
            rules = clf.evaluate_rules(feat)
            o_match = any(m[0] == "O" for m in rules)
            o_str = "YES (O MATCH)" if o_match else "no"
            print(f"{fidx:<6} | {ph:<12} | {qual:<6} | {exts:<16} | {d_ti:<7} | {d_tm:<7} | {idx_str:<8} | {d_tw:<7} | {wrist_y:<8} | {o_str:<12}")
            all_b_samples_status.append((fidx, ph, qual, o_match))
        else:
            print(f"{fidx:<6} | {ph:<12} | {qual:<6} | NO HAND DETECTED")

    # 4. Check ALL B samples in NPZ
    print("\n--- D. AUDIT ALL 22 B SAMPLES IN NPZ ---")
    b_indices = [i for i, label_idx in enumerate(y) if classes[label_idx] == "B"]
    print(f"Total B samples in NPZ: {len(b_indices)}")
    b_npz_fps = []
    for idx in b_indices:
        lm = landmarks[idx].tolist()
        feat = clf.extract_features(lm)
        rules = clf.evaluate_rules(feat)
        o_match = any(m[0] == "O" for m in rules)
        m = meta[idx]
        if o_match:
            b_npz_fps.append((idx, m))
            print(f"  [FP MATCH] NPZ idx {idx}: video_id={m.get('video_id')}, frame={m.get('frame_index')}, disp={m.get('displacement_to_prev')}")
    print(f"Total B samples in NPZ matching O: {len(b_npz_fps)} / {len(b_indices)}")

    # 5. Check ALL 551 samples across ALL 22 classes in NPZ
    print("\n--- E. AUDIT ALL 551 SAMPLES ACROSS ALL 22 CLASSES IN NPZ ---")
    fp_by_class = {}
    for idx in range(len(y)):
        true_cls = classes[y[idx]]
        lm = landmarks[idx].tolist()
        feat = clf.extract_features(lm)
        rules = clf.evaluate_rules(feat)
        pred_letter, _ = clf.classify(lm)
        
        # Check if Rule 6 matched
        rule_6_o = any(m[0] == "O" for m in rules)
        # Check if final prediction was O
        final_o = (pred_letter == "O")
        
        if (rule_6_o or final_o) and true_cls != "O":
            if true_cls not in fp_by_class:
                fp_by_class[true_cls] = []
            fp_by_class[true_cls].append((idx, meta[idx], rule_6_o, final_o))

    print(f"Non-O classes with False Positives to O: {list(fp_by_class.keys())}")
    for cls_name, items in fp_by_class.items():
        print(f"  Class {cls_name}: {len(items)} False Positives")
        for idx, m, r6, fin in items:
            print(f"    Index {idx}: video={m.get('video_id')}, frame={m.get('frame_index')}, Rule6_O={r6}, Final_O={fin}")

    # 6. Save detailed artifact summary
    summary = {
        "s39_metadata": s39_meta,
        "s39_features": s39_feat,
        "s39_conditions": {c[0]: (c[1], c[2]) for c in cond_checks},
        "all_b_npz_count": len(b_indices),
        "all_b_npz_fps": len(b_npz_fps),
        "all_551_npz_fps_to_o": {k: len(v) for k, v in fp_by_class.items()}
    }
    with open("scratch/task3_5_1_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("\nSaved summary to scratch/task3_5_1_summary.json")

if __name__ == "__main__":
    main()

