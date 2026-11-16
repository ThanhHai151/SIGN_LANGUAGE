import os
import sys
import json
import numpy as np

sys.path.insert(0, os.path.abspath("."))
from backend.vision.recognizer import VSLRecognizer

def run_analysis():
    rec = VSLRecognizer()

    print("================================================================================")
    print("PHASE B & C: INVESTIGATION & OFFLINE THRESHOLD SWEEP FOR CHỮ Q")
    print("================================================================================")

    # 1. Inspect Canonical Q (D0510)
    q_file = "backend/data/alphabet_v2/static_landmarks/Q.json"
    with open(q_file, "r", encoding="utf-8") as f:
        q_data = json.load(f)
    q_samples = q_data.get("samples", [])
    valid_q = [s for s in q_samples if s.get("landmarks") and s.get("quality_status") == "ok"]

    print(f"\n--- CANONICAL Q (D0510.mp4) ---")
    print(f"Total samples: {len(q_samples)}, Valid samples (quality_status='ok'): {len(valid_q)}")
    
    # Detailed inspection of valid Q frames under Static Head & Geometric Rule
    print("\nValid Q frames breakdown:")
    for idx, s in enumerate(valid_q):
        lm = s["landmarks"]
        # evaluate geometric rules
        alpha_res = rec.alphabet_classifier.classify_detailed(lm)
        geom_st = alpha_res.get("status")
        geom_let = alpha_res.get("letter")
        geom_reason = alpha_res.get("reason", "")
        geom_candidates = alpha_res.get("matched_candidates", [])

        # evaluate static head
        from backend.vision.feature_extractor import normalize_hand_landmarks
        feat_72, _ = normalize_hand_landmarks(lm)
        head_res = rec.static_alphabet_head.predict_feature(feat_72[:72], confidence_threshold=0.0)
        h_let = head_res.get("raw_predicted_letter")
        h_conf = head_res.get("confidence", 0.0)
        top3 = [(t["letter"], round(t["confidence"], 3)) for t in head_res.get("top_k", [])[:3]]

        print(f"  Frame {idx:02d} (frame_index={s.get('frame_index')}, phase={s.get('phase')}): "
              f"Geom=[{geom_st}, letter={geom_let}] | "
              f"StaticHead=[pred={h_let}, conf={h_conf:.4f}, top3={top3}]")

    # 2. Control groups: Canonical P (D0509B), X (D0517B), G (D0498B)
    control_files = {
        "P": ("P.json", "D0509B"),
        "X": ("X.json", "D0517B"),
        "G": ("G.json", "D0498B"),
    }
    control_samples = {}
    for letter, (fname, vdesc) in control_files.items():
        fp = os.path.join("backend/data/alphabet_v2/static_landmarks", fname)
        with open(fp, "r", encoding="utf-8") as f:
            d = json.load(f)
        samples = d.get("samples", [])
        valid = [s for s in samples if s.get("landmarks") and s.get("quality_status") == "ok"]
        control_samples[letter] = valid
        print(f"\n--- CONTROL GROUP {letter} ({vdesc}) ---")
        print(f"Total samples: {len(samples)}, Valid: {len(valid)}")
        # Check if static head predicts Q for any frame
        q_conf_list = []
        for s in valid:
            feat_72, _ = normalize_hand_landmarks(s["landmarks"])
            head_res = rec.static_alphabet_head.predict_feature(feat_72[:72], confidence_threshold=0.0)
            # Find Q probability in top_k
            q_p = 0.0
            for t in head_res.get("top_k", []):
                if t["letter"] == "Q":
                    q_p = t["confidence"]
                    break
            q_conf_list.append(q_p)
            if head_res.get("raw_predicted_letter") == "Q":
                print(f"  ALERT: {letter} frame {s.get('frame_index')} predicted as Q by Static Head with conf={head_res.get('confidence'):.4f}!")
        max_q_p = max(q_conf_list) if q_conf_list else 0.0
        avg_q_p = sum(q_conf_list) / len(q_conf_list) if q_conf_list else 0.0
        print(f"  Q confidence in {letter} samples: max={max_q_p:.4f}, avg={avg_q_p:.4f}")

    # 3. Simulate Recognizer at different thresholds for Q
    candidates = [0.85, 0.80, 0.75, 0.70]
    
    print("\n================================================================================")
    print("SIMULATION RESULTS: RECOGNIZER PREDICTIONS ON CANONICAL DATA")
    print("================================================================================")

    for thresh in candidates:
        rec.supplemental_thresholds["Q"] = thresh
        
        # Test Canonical Q
        corr_q = 0
        conf_q = {}
        for s in valid_q:
            res = rec.predict_landmarks(s["landmarks"], mode="letters", session_id=f"sim_q_{thresh}")
            p_sign = res["sign"]
            conf_q[p_sign] = conf_q.get(p_sign, 0) + 1
            if p_sign == "Q":
                corr_q += 1

        # Test Control P
        mis_p_to_q = 0
        corr_p = 0
        for s in control_samples["P"]:
            res = rec.predict_landmarks(s["landmarks"], mode="letters", session_id=f"sim_p_{thresh}")
            if res["sign"] == "Q":
                mis_p_to_q += 1
            if res["sign"] == "P":
                corr_p += 1

        # Test Control X
        mis_x_to_q = 0
        corr_x = 0
        for s in control_samples["X"]:
            res = rec.predict_landmarks(s["landmarks"], mode="letters", session_id=f"sim_x_{thresh}")
            if res["sign"] == "Q":
                mis_x_to_q += 1
            if res["sign"] == "X":
                corr_x += 1

        # Test Control G
        mis_g_to_q = 0
        corr_g = 0
        for s in control_samples["G"]:
            res = rec.predict_landmarks(s["landmarks"], mode="letters", session_id=f"sim_g_{thresh}")
            if res["sign"] == "Q":
                mis_g_to_q += 1
            if res["sign"] == "G":
                corr_g += 1

        total_q = len(valid_q)
        total_p = len(control_samples["P"])
        total_x = len(control_samples["X"])
        total_g = len(control_samples["G"])
        fn_q = total_q - corr_q

        print(f"\n--- Candidate Threshold Q = {thresh:.2f} ---")
        print(f"  Q True Positive: {corr_q}/{total_q} ({corr_q/total_q*100:.1f}%)")
        print(f"  Q False Negative: {fn_q}/{total_q} ({fn_q/total_q*100:.1f}%)")
        print(f"  Predictions for Q: {conf_q}")
        print(f"  P misclassified as Q: {mis_p_to_q}/{total_p} ({mis_p_to_q/total_p*100:.1f}%) [P correct: {corr_p}/{total_p}]")
        print(f"  X misclassified as Q: {mis_x_to_q}/{total_x} ({mis_x_to_q/total_x*100:.1f}%) [X correct: {corr_x}/{total_x}]")
        print(f"  G misclassified as Q: {mis_g_to_q}/{total_g} ({mis_g_to_q/total_g*100:.1f}%) [G correct: {corr_g}/{total_g}]")

    # 4. Now run Legacy Dataset sweep (vsl_alphabet_static_v2.npz, 551 samples)
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    if os.path.exists(npz_path):
        data = np.load(npz_path, allow_pickle=True)
        classes = list(data["classes"])
        landmarks = data["landmarks"]
        y = data["y"]

        print("\n================================================================================")
        print("SIMULATION RESULTS: RECOGNIZER PREDICTIONS ON LEGACY DATASET (551 samples)")
        print("================================================================================")

        for thresh in candidates:
            rec.supplemental_thresholds["Q"] = thresh
            total_samples = len(y)
            overall_correct = 0
            per_class_correct = {c: 0 for c in classes}
            per_class_total = {c: 0 for c in classes}
            per_class_mis_to_q = {c: 0 for c in classes}
            q_confusions = {}

            for i in range(total_samples):
                target = classes[y[i]]
                lm = landmarks[i].tolist()
                res = rec.predict_landmarks(lm, mode="letters", session_id=f"npz_sim_{thresh}_{i}")
                pred = res["sign"]
                per_class_total[target] += 1
                if pred == target:
                    overall_correct += 1
                    per_class_correct[target] += 1
                elif pred == "Q":
                    per_class_mis_to_q[target] += 1
                
                if target == "Q":
                    q_confusions[pred] = q_confusions.get(pred, 0) + 1

            q_corr = per_class_correct["Q"]
            q_tot = per_class_total["Q"]
            p_mis = per_class_mis_to_q["P"]
            p_tot = per_class_total["P"]
            x_mis = per_class_mis_to_q["X"]
            x_tot = per_class_total["X"]
            g_mis = per_class_mis_to_q["G"]
            g_tot = per_class_total["G"]

            other_mis_to_q = {c: per_class_mis_to_q[c] for c in classes if c != "Q" and per_class_mis_to_q[c] > 0}

            print(f"\n--- Candidate Threshold Q = {thresh:.2f} (Legacy 551) ---")
            print(f"  Overall Accuracy: {overall_correct}/{total_samples} ({overall_correct/total_samples*100:.2f}%)")
            print(f"  Q Recall: {q_corr}/{q_tot} ({q_corr/q_tot*100:.1f}%) | False Neg: {q_tot - q_corr}/{q_tot} | Predictions: {q_confusions}")
            print(f"  P -> Q False Positive: {p_mis}/{p_tot} ({p_mis/p_tot*100:.1f}%)")
            print(f"  X -> Q False Positive: {x_mis}/{x_tot} ({x_mis/x_tot*100:.1f}%)")
            print(f"  G -> Q False Positive: {g_mis}/{g_tot} ({g_mis/g_tot*100:.1f}%)")
            print(f"  Other letters misclassified as Q: {other_mis_to_q}")

            # Check regression on 21 letters
            # compare per_class_correct against baseline
            if thresh == 0.85:
                baseline_per_class = dict(per_class_correct)
            else:
                regressed = []
                for c in classes:
                    if c != "Q" and per_class_correct[c] < baseline_per_class[c]:
                        regressed.append((c, baseline_per_class[c], per_class_correct[c]))
                if regressed:
                    print(f"  REGRESSION DETECTED on letters: {regressed}")
                else:
                    print(f"  Regression on 21 other letters: ZERO REGRESSION (all match baseline exactly)")

    # Reset back to production threshold 0.85
    rec.supplemental_thresholds["Q"] = 0.85
    print("\nReset production threshold Q = 0.85 complete.")

if __name__ == "__main__":
    run_analysis()
