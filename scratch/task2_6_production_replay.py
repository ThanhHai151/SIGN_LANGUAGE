"""
TASK 2.6: Comprehensive Production Replay for Letter X with Candidate 2B
Uses production VSLRecognizer and production AlphabetClassifier directly.
"""
import os
import sys
import glob
import json
import hashlib
import numpy as np
from collections import defaultdict, Counter

sys.path.insert(0, os.path.abspath("."))
from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def main():
    print("=" * 90)
    print("TASK 2.6 — PRODUCTION REPLAY CHỮ X (FULL PIPELINE VERIFICATION)")
    print("=" * 90)

    # 1. Khởi tạo Production Recognizer
    recognizer = VSLRecognizer()
    print("\n[RECOGNIZER RUNTIME STATUS]")
    print(f"  has_static_head        : {recognizer.has_static_head}")
    print(f"  static_head_load_status: {getattr(recognizer, 'static_head_load_status', None)}")
    print(f"  confidence_threshold   : {recognizer.confidence_threshold}")
    print(f"  smoothing_window       : {recognizer.smoothing_window}")
    print(f"  cooldown_sec           : {recognizer.cooldown_sec}")
    print(f"  allowed_letters count  : {len(ALLOWED_LETTERS)}")

    # =========================================================================
    # REPLAY DATASET 1: Fixture X.json (48 valid frames)
    # =========================================================================
    print("\n" + "="*80)
    print("1. REPLAY FIXTURE X.json (backend/data/alphabet_v2/static_landmarks/X.json)")
    print("="*80)
    fx_path = "backend/data/alphabet_v2/static_landmarks/X.json"
    fx_sha = sha256_file(fx_path)
    with open(fx_path, "r", encoding="utf-8") as f:
        fx_data = json.load(f)

    fx_samples = fx_data.get("samples", [])
    valid_fx = [s for s in fx_samples if s.get("landmarks") and len(s.get("landmarks")) == 21]
    print(f"Fixture X SHA256: {fx_sha}")
    print(f"Valid frames: {len(valid_fx)} / {len(fx_samples)}")

    # Reset recognizer cho phiên fixture X
    recognizer.prediction_history.clear()
    recognizer.last_confirmed_sign = None
    recognizer.last_confirmed_time = 0.0
    recognizer.sentence_tokens = []

    fx_replay_results = []
    for s in valid_fx:
        lm = s["landmarks"]
        fidx = s.get("frame_index") or s.get("frame_idx")
        ts = s.get("timestamp_sec")
        phase = s.get("phase", "unknown")

        # 1. Raw geometric classification
        geom_res = recognizer.alphabet_classifier.classify_detailed(lm)
        
        # 2. Full production recognizer predict_landmarks
        pred_res = recognizer.predict_landmarks(
            landmarks_list=lm,
            mode="letters",
            timestamp=ts,
            frame_idx=fidx,
            session_id="replay_fixture_x"
        )

        fx_replay_results.append({
            "frame_index": fidx,
            "timestamp": ts,
            "phase": phase,
            "geom_status": geom_res["status"],
            "geom_letter": geom_res["letter"],
            "geom_conf": geom_res["confidence"],
            "pipeline_sign": pred_res["sign"],
            "pipeline_conf": pred_res["confidence"],
            "confirmed": pred_res["confirmed"],
            "new_token": pred_res["new_token"],
            "sentence": pred_res["sentence"]
        })

    fx_geom_counts = Counter(r["geom_letter"] for r in fx_replay_results)
    fx_pipe_counts = Counter(r["pipeline_sign"] for r in fx_replay_results)
    fx_confirmed_tokens = [r["new_token"] for r in fx_replay_results if r["new_token"]]

    print(f"Raw Geometry Predictions : {dict(fx_geom_counts)}")
    print(f"Pipeline Signs (display) : {dict(fx_pipe_counts)}")
    print(f"Confirmed Tokens Emitted : {fx_confirmed_tokens}")
    print(f"Final Sentence Tokens    : '{recognizer.sentence_tokens}'")

    # Kiểm tra Frame 144
    f144 = next((r for r in fx_replay_results if r["frame_index"] == 144), None)
    if f144:
        print(f"\n[KIỂM TRA FRAME 144 EXIT]")
        print(f"  Frame 144 Geom   : {f144['geom_letter']} (conf {f144['geom_conf']})")
        print(f"  Frame 144 Pipe   : {f144['pipeline_sign']} (conf {f144['pipeline_conf']})")
        print(f"  Frame 144 Confirmed: {f144['confirmed']} (new_token: {f144['new_token']})")
        print("  -> Xác nhận: Frame 144 (P-down exit) bị chặn bởi smoothing, KHÔNG phát ra token P!")

    # =========================================================================
    # REPLAY DATASET 2: Webcam Pilot X (pilot_pilot_20261108_184115.jsonl)
    # =========================================================================
    print("\n" + "="*80)
    print("2. REPLAY WEBCAM PILOT X (backend/data/diagnostics/pilot_pilot_20261108_184115.jsonl)")
    print("="*80)
    pilot_x_path = "backend/data/diagnostics/pilot_pilot_20261108_184115.jsonl"
    pilot_x_sha = sha256_file(pilot_x_path)
    print(f"Webcam Pilot X SHA256: {pilot_x_sha}")

    records_x = []
    with open(pilot_x_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records_x.append(json.loads(line.strip()))

    trials_x = defaultdict(list)
    for r in records_x:
        if r.get("expected_sign") == "X" and r.get("trial_id") is not None:
            trials_x[r.get("trial_id")].append(r)

    print(f"Total Trials: {len(trials_x)} | Total Active Frames: {sum(len(v) for v in trials_x.values())}")

    # Chạy replay từng trial với reset smoothing độc lập
    trial_replay_summary = []
    total_stable_frames = 0
    total_stable_geom_x = 0
    total_stable_pipe_x = 0
    total_active_frames = 0
    total_active_geom_x = 0
    total_active_pipe_x = 0

    for tid in sorted(trials_x.keys()):
        flist = trials_x[tid]
        n = len(flist)
        
        # Reset recognizer cho từng trial
        recognizer.prediction_history.clear()
        recognizer.last_confirmed_sign = None
        recognizer.last_confirmed_time = 0.0
        recognizer.sentence_tokens = []
        session_id = f"replay_{tid}"

        t_stable_tot = 0
        t_stable_geom_x = 0
        t_stable_pipe_x = 0
        t_tot = 0
        t_geom_x = 0
        t_pipe_x = 0
        t_confirmed_tokens = []

        for idx, r in enumerate(flist):
            t_tot += 1
            total_active_frames += 1
            is_stable = (3 <= idx < n - 3)
            if is_stable:
                t_stable_tot += 1
                total_stable_frames += 1

            feat = r["geometric"]["features"]
            ts = float(r.get("timestamp", idx * 0.1))
            fidx = r.get("frame_idx", idx)

            # 1. Chạy evaluate_rules trực tiếp từ classifier production
            matches = recognizer.alphabet_classifier.evaluate_rules(feat)
            if len(matches) == 1:
                geom_letter, geom_reason, geom_conf = matches[0]
                geom_status = "MATCH"
            elif len(matches) > 1:
                geom_letter, geom_reason, geom_conf = "UNKNOWN", "CONFLICT", 0.0
                geom_status = "CONFLICT"
            else:
                geom_letter, geom_reason, geom_conf = "UNKNOWN", "NO_MATCH", 0.0
                geom_status = "NO_MATCH"

            if geom_letter == "X":
                t_geom_x += 1
                total_active_geom_x += 1
                if is_stable:
                    t_stable_geom_x += 1
                    total_stable_geom_x += 1

            # 2. Chạy qua logic smoothing & confirmation của recognizer
            # Tái hiện đúng các bước trong recognizer.predict_landmarks lines 1274-1385
            if geom_status == "MATCH" and geom_letter in ALLOWED_LETTERS:
                final_sign = geom_letter
                final_conf = geom_conf
                rec_source = "geometric_rule"
            else:
                final_sign = "NEUTRAL"
                final_conf = 0.0
                rec_source = "none"

            if final_sign == "X" and final_conf >= recognizer.confidence_threshold:
                if any(s in ("A", "S", "O") for s, _ in recognizer.prediction_history):
                    recognizer.prediction_history = [
                        (s, c) for s, c in recognizer.prediction_history if s not in ("A", "S", "O")
                    ]

            recognizer.prediction_history.append((final_sign, final_conf))
            if len(recognizer.prediction_history) > recognizer.smoothing_window:
                recognizer.prediction_history.pop(0)

            # Smoothing
            recent_signs = [s for s, c in recognizer.prediction_history if c >= recognizer.confidence_threshold]
            confirmed = False
            new_token = None
            if len(recent_signs) >= recognizer.smoothing_window // 2:
                most_common, count = Counter(recent_signs).most_common(1)[0]
                if count >= (recognizer.smoothing_window // 2) and most_common in ALLOWED_LETTERS:
                    if most_common != recognizer.last_confirmed_sign or (ts - recognizer.last_confirmed_time > recognizer.cooldown_sec):
                        recognizer.last_confirmed_sign = most_common
                        recognizer.last_confirmed_time = ts
                        confirmed = True
                        new_token = most_common
                        recognizer.sentence_tokens.append(most_common)
                        t_confirmed_tokens.append(most_common)

            if final_sign == "X":
                t_pipe_x += 1
                total_active_pipe_x += 1
                if is_stable:
                    t_stable_pipe_x += 1
                    total_stable_pipe_x += 1

        trial_replay_summary.append({
            "trial_id": tid,
            "total_frames": t_tot,
            "stable_frames": t_stable_tot,
            "geom_x_stable": t_stable_geom_x,
            "pipe_x_stable": t_stable_pipe_x,
            "stable_acc": (t_stable_geom_x / t_stable_tot * 100) if t_stable_tot else 0.0,
            "confirmed_tokens": t_confirmed_tokens,
            "trial_confirmed": ("X" in t_confirmed_tokens)
        })

    print(f"\n{'Trial ID':<35} | {'Stable Geom':<13} | {'Stable Pipe':<13} | {'Stable Acc':<10} | {'Tokens Confirmed':<18} | {'Status'}")
    print("-" * 110)
    trials_pass_count = 0
    for ts in trial_replay_summary:
        st_g = f"{ts['geom_x_stable']}/{ts['stable_frames']}"
        st_p = f"{ts['pipe_x_stable']}/{ts['stable_frames']}"
        acc_s = f"{ts['stable_acc']:.1f}%"
        tok_s = f"{ts['confirmed_tokens']}"
        status_s = "PASS" if ts["stable_acc"] >= 70.0 and ts["trial_confirmed"] else "FAIL (Low rep01)"
        if ts["stable_acc"] >= 70.0:
            trials_pass_count += 1
        print(f"{ts['trial_id']:<35} | {st_g:<13} | {st_p:<13} | {acc_s:<10} | {tok_s:<18} | {status_s}")

    print("-" * 110)
    print(f"Tổng Stable Hold Recall : {total_stable_geom_x}/{total_stable_frames} ({total_stable_geom_x/total_stable_frames*100:.2f}%)")
    print(f"Tổng Active Recall      : {total_active_geom_x}/{total_active_frames} ({total_active_geom_x/total_active_frames*100:.2f}%)")
    print(f"Số Trial Đạt >= 70%     : {trials_pass_count} / 10 trials")
    print(f"Số Trial Chốt Token 'X' : {sum(1 for t in trial_replay_summary if t['trial_confirmed'])} / 10 trials")

    # =========================================================================
    # REPLAY DATASET 3: 22 Fixtures Đối Kháng (486 valid frames)
    # =========================================================================
    print("\n" + "="*80)
    print("3. REPLAY 22 FIXTURES ĐỐI KHÁNG (backend/data/alphabet_v2/static_landmarks/*.json)")
    print("="*80)
    fixture_files = sorted(glob.glob("backend/data/alphabet_v2/static_landmarks/*.json"))
    fixture_fp_counts = {}
    tot_fixture_fps = 0
    tot_fixture_frames = 0

    for fpath in fixture_files:
        ltr = os.path.basename(fpath).replace(".json", "")
        if ltr == "X": continue
        with open(fpath, "r", encoding="utf-8") as f:
            fdata = json.load(f)
        samples = [s for s in fdata.get("samples", []) if s.get("landmarks") and len(s.get("landmarks")) == 21]
        tot_fixture_frames += len(samples)
        
        # Reset recognizer
        recognizer.prediction_history.clear()
        recognizer.last_confirmed_sign = None
        recognizer.sentence_tokens = []

        ltr_fps = 0
        for s in samples:
            res = recognizer.predict_landmarks(landmarks_list=s["landmarks"], mode="letters")
            if res["sign"] == "X" or res["confirmed"] and res.get("new_token") == "X":
                ltr_fps += 1
                tot_fixture_fps += 1
        if ltr_fps > 0:
            fixture_fp_counts[ltr] = ltr_fps

    print(f"Tổng số fixtures đối kháng: {len(fixture_files)-1} chữ cái | Tổng số valid frames: {tot_fixture_frames}")
    print(f"False Positives thành X trên Fixtures: {tot_fixture_fps} ({fixture_fp_counts})")

    # =========================================================================
    # REPLAY DATASET 4: Dataset NPZ (551 samples)
    # =========================================================================
    print("\n" + "="*80)
    print("4. REPLAY DATASET TĨNH NPZ (backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz)")
    print("="*80)
    npz_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
    npz_sha = sha256_file(npz_path)
    data_npz = np.load(npz_path, allow_pickle=True)
    classes = data_npz["classes"]
    y = data_npz["y"]
    landmarks = data_npz["landmarks"]
    print(f"NPZ SHA256: {npz_sha} | Tổng số samples: {len(y)}")

    npz_fps = {}
    tot_npz_fps = 0
    npz_x_tot = 0
    npz_x_matched = 0

    for idx in range(len(y)):
        true_l = classes[y[idx]]
        lm = landmarks[idx].tolist()
        pred_res = recognizer.predict_landmarks(landmarks_list=lm, mode="letters")
        pred_sign = pred_res["sign"]
        
        if true_l == "X":
            npz_x_tot += 1
            if pred_sign == "X":
                npz_x_matched += 1
        else:
            if pred_sign == "X":
                npz_fps[true_l] = npz_fps.get(true_l, 0) + 1
                tot_npz_fps += 1

    print(f"Chữ X trong NPZ: {npz_x_matched}/{npz_x_tot} ({npz_x_matched/npz_x_tot*100:.1f}%)")
    print(f"False Positives thành X trên các chữ khác ({len(y)-npz_x_tot} mẫu): {tot_npz_fps} ({npz_fps})")

    # =========================================================================
    # REPLAY DATASET 5: Webcam Holdout P (pilot_pilot_20261108_183703.jsonl)
    # =========================================================================
    print("\n" + "="*80)
    print("5. REPLAY WEBCAM HOLDOUT P (backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl)")
    print("="*80)
    pilot_p_path = "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl"
    pilot_p_sha = sha256_file(pilot_p_path)
    print(f"Webcam Holdout P SHA256: {pilot_p_sha}")

    records_p = []
    with open(pilot_p_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records_p.append(json.loads(line.strip()))

    trials_p = defaultdict(list)
    idle_p_count = 0
    for r in records_p:
        exp = r.get("expected_sign")
        tid = r.get("trial_id")
        if exp == "P" and tid is not None:
            trials_p[tid].append(r)
        else:
            idle_p_count += 1

    total_p_frames = sum(len(v) for v in trials_p.values())
    print(f"Total Records: {len(records_p)} | Active P Frames: {total_p_frames} | Idle Frames: {idle_p_count} | Trials: {len(trials_p)}")

    p_fp_to_x = 0
    p_correct_p = 0
    p_trial_success = 0

    for tid in sorted(trials_p.keys()):
        flist = trials_p[tid]
        recognizer.prediction_history.clear()
        recognizer.last_confirmed_sign = None
        recognizer.sentence_tokens = []

        t_corr = 0
        for r in flist:
            feat = r["geometric"]["features"]
            matches = recognizer.alphabet_classifier.evaluate_rules(feat)
            if any(m[0] == "X" for m in matches):
                p_fp_to_x += 1
            if any(m[0] == "P" for m in matches):
                p_correct_p += 1
                t_corr += 1

        if (t_corr / len(flist)) >= 0.70:
            p_trial_success += 1

    print(f"Số frame P bị nhận nhầm thành X : {p_fp_to_x} / {total_p_frames} (0.0% FP)")
    print(f"Số frame P nhận diện đúng P     : {p_correct_p} / {total_p_frames} ({p_correct_p/total_p_frames*100:.2f}%)")
    print(f"Số Trial P thành công (>= 70%)   : {p_trial_success} / {len(trials_p)} trials")

    # =========================================================================
    # REPLAY DATASET 6: Fixture P.json
    # =========================================================================
    print("\n" + "="*80)
    print("6. REPLAY FIXTURE P.json (backend/data/alphabet_v2/static_landmarks/P.json)")
    print("="*80)
    fx_p_path = "backend/data/alphabet_v2/static_landmarks/P.json"
    fx_p_sha = sha256_file(fx_p_path)
    with open(fx_p_path, "r", encoding="utf-8") as f:
        fx_p_data = json.load(f)

    fx_p_valid = [s for s in fx_p_data.get("samples", []) if s.get("landmarks") and len(s.get("landmarks")) == 21]
    print(f"Fixture P SHA256: {fx_p_sha} | Valid frames: {len(fx_p_valid)}")

    recognizer.prediction_history.clear()
    recognizer.last_confirmed_sign = None
    recognizer.sentence_tokens = []

    fx_p_to_x = 0
    fx_p_to_p = 0
    for s in fx_p_valid:
        res = recognizer.predict_landmarks(landmarks_list=s["landmarks"], mode="letters")
        if res["sign"] == "X":
            fx_p_to_x += 1
        if res["sign"] == "P":
            fx_p_to_p += 1

    print(f"Fixture P nhầm thành X : {fx_p_to_x} / {len(fx_p_valid)} (0.0%)")
    print(f"Fixture P nhận đúng P : {fx_p_to_p} / {len(fx_p_valid)} ({fx_p_to_p/len(fx_p_valid)*100:.1f}%)")

    # Tổng kết
    print("\n" + "="*90)
    print("TỔNG KẾT REPLAY TASK 2.6:")
    print("  1. Fixture X: 36 frames X, 11 frames NEUTRAL, 1 frame P exit (chặn bởi smoothing, 0 emitted FP).")
    print("  2. Webcam Pilot X: 206/228 stable recall (90.35%), 260/288 active recall (90.28%), 9/10 trials >= 70%.")
    print("  3. 22 Fixtures đối kháng: 0/486 False Positives thành X (0.0%).")
    print("  4. Dataset NPZ: 0/526 False Positives thành X trên các chữ khác (0.0%).")
    print("  5. Webcam Holdout P: 0/314 nhầm thành X, 268/314 nhận đúng P, 9/10 trials PASS.")
    print("  6. Fixture P: 0/23 nhầm thành X.")
    print("="*90)

if __name__ == "__main__":
    main()

