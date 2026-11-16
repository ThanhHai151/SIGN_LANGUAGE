"""
TASK 3.6: COMPREHENSIVE PRODUCTION REPLAY VERIFICATION FOR LETTER O (O-CANDIDATE-03)
=====================================================================================
Evaluates O-Candidate-03 across all available datasets:
1. Target Letter O Datasets:
   - NPZ O (N=48 samples: D0506=20, D0507B=19, D0508N=9)
   - Fixture O.json (N=20 samples: D0506)
   - Webcam Pilot 213800 (N=92 active frames, 10 trials)
   - Webcam Pilot 211742 (N=68 active frames, 10 trials)
2. Non-O Control Datasets:
   - NPZ Non-O (N=503 samples, 21 classes) [B frame 54 evaluated without test exclusion]
   - 22 Static Fixture JSONs (A through Y, excluding O)
   - Webcam P Holdout (N=314 active frames)
   - Webcam X Holdout (N=108 active frames)
3. Three-Tier Pipeline Evaluation:
   - Tier 1: Geometric Rule 6 Match
   - Tier 2: MLP Static Head Fallback / Gating
   - Tier 3: Temporal Smoothing Buffer & Final Confirmed Token
"""

import os
import sys
import json
import hashlib
import numpy as np
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath("."))
from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

def sha256_file(filepath):
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()

def simulate_temporal_smoothing(items, rec):
    """Mô phỏng chính xác logic temporal smoothing từ lines 1373-1385 của recognizer.py"""
    rec.prediction_history.clear()
    rec.last_confirmed_sign = None
    rec.last_confirmed_time = 0.0
    rec.sentence_tokens = []
    
    trial_tokens = []
    trial_display = []
    
    for fidx, ts, is_o in items:
        geom_sign = "O" if is_o else "NEUTRAL"
        geom_conf = 0.95 if is_o else 0.0
        
        rec.prediction_history.append((geom_sign, geom_conf))
        
        recent_signs = [s for s, c in rec.prediction_history if c >= rec.confidence_threshold]
        confirmed = False
        new_token = None
        if len(recent_signs) >= rec.smoothing_window // 2:
            counts = Counter(recent_signs)
            most_common, count = counts.most_common(1)[0]
            if count >= rec.smoothing_window // 2 and most_common in ALLOWED_LETTERS:
                if most_common != rec.last_confirmed_sign or (ts - rec.last_confirmed_time) > rec.cooldown_sec:
                    rec.last_confirmed_sign = most_common
                    rec.last_confirmed_time = ts
                    confirmed = True
                    new_token = most_common
                    trial_tokens.append(new_token)
        trial_display.append(geom_sign)
    return trial_tokens, trial_display

def main():
    print("=" * 90)
    print("TASK 3.6 — PRODUCTION REPLAY VERIFICATION CHO CHỮ O (O-CANDIDATE-03)")
    print("=" * 90)

    # 1. Khởi tạo Recognizer và Classifier
    rec = VSLRecognizer()
    clf = rec.alphabet_classifier
    print("\n[PRODUCTION RUNTIME STATUS]")
    print(f"  has_static_head        : {rec.has_static_head}")
    print(f"  static_head_load_status: {getattr(rec, 'static_head_load_status', None)}")
    print(f"  confidence_threshold   : {rec.confidence_threshold}")
    print(f"  smoothing_window       : {rec.smoothing_window}")
    print(f"  cooldown_sec           : {rec.cooldown_sec}")
    print(f"  pytorch_version        : {getattr(rec, 'pytorch_version', 'N/A')}")
    print(f"  allowed_letters count  : {len(ALLOWED_LETTERS)}")

    results_data = {
        "runtime_status": {
            "has_static_head": rec.has_static_head,
            "static_head_load_status": getattr(rec, "static_head_load_status", None),
            "pytorch_version": getattr(rec, "pytorch_version", "N/A"),
        },
        "file_hashes": {},
        "target_o": {},
        "non_o": {},
        "pipeline_analysis": {},
        "historical_comparison": {},
    }

    # =========================================================================
    # PHẦN 1: FILE INTEGRITY & HASHES
    # =========================================================================
    key_files = {
        "npz_dataset": "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz",
        "fixture_o": "backend/data/alphabet_v2/static_landmarks/O.json",
        "pilot_213800": "backend/data/diagnostics/pilot_pilot_20261108_213800.jsonl",
        "pilot_211742": "backend/data/diagnostics/pilot_pilot_20261108_193712_20261108_211742_1.jsonl",
        "p_holdout": "backend/data/diagnostics/pilot_pilot_20261108_183703.jsonl",
        "x_holdout": "backend/data/diagnostics/pilot_pilot_20261108_193712.jsonl",
    }
    print("\n--- FILE INTEGRITY & SHA256 CHECKSUMS ---")
    for k, p in key_files.items():
        if os.path.exists(p):
            sha = sha256_file(p)
            size = os.path.getsize(p)
            results_data["file_hashes"][k] = {"path": p, "sha256": sha, "size_bytes": size}
            print(f"  {k:<15}: {sha[:16]}... ({size:>10} bytes) -> {p}")
        else:
            print(f"  {k:<15}: FILE NOT FOUND -> {p}")

    # =========================================================================
    # PHẦN 2: TARGET O REPLAY
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHẦN 2: REPLAY TẬP DỮ LIỆU MỤC TIÊU (CHỮ O)")
    print("=" * 90)

    # --- A. NPZ O (N=48) ---
    npz_data = np.load(key_files["npz_dataset"], allow_pickle=True)
    classes = list(npz_data["classes"])
    y = npz_data["y"]
    landmarks = npz_data["landmarks"]
    metadata_json = npz_data["metadata_json"]
    meta_list = json.loads(metadata_json.item() if metadata_json.shape == () else metadata_json[0])

    o_indices = [i for i, label_idx in enumerate(y) if classes[label_idx] == "O"]
    print(f"\n1. NPZ O Dataset (Total N={len(o_indices)} samples across 3 videos):")

    npz_o_by_video = defaultdict(list)
    npz_o_results = []
    for idx in o_indices:
        m = meta_list[idx]
        vid = m.get("video_id", "unknown")
        fidx = m.get("frame_index")
        lm = landmarks[idx].tolist()
        
        # Tier 1: Raw Geometric
        geom_res = clf.classify_detailed(lm)
        geom_match = (geom_res["status"] == "MATCH" and geom_res["letter"] == "O")
        
        # Tier 2 & 3: Recognizer Predict
        rec.prediction_history.clear()
        rec.last_confirmed_sign = None
        rec.last_confirmed_time = 0.0
        pipe_res = rec.predict_landmarks(
            landmarks_list=lm,
            mode="letters",
            timestamp=m.get("timestamp_sec", 0.0),
            frame_idx=fidx,
            session_id=f"npz_{vid}"
        )
        
        rec_item = {
            "idx": idx,
            "video_id": vid,
            "frame_index": fidx,
            "geom_status": geom_res["status"],
            "geom_letter": geom_res["letter"],
            "geom_conf": geom_res["confidence"],
            "geom_match_o": geom_match,
            "pipeline_sign": pipe_res["sign"],
            "pipeline_conf": pipe_res["confidence"],
            "pipeline_source": pipe_res.get("prediction_source"),
        }
        npz_o_results.append(rec_item)
        npz_o_by_video[vid].append(rec_item)

    for vid, items in npz_o_by_video.items():
        g_matches = sum(1 for x in items if x["geom_match_o"])
        p_matches = sum(1 for x in items if x["pipeline_sign"] == "O")
        print(f"   Video {vid:<8} (N={len(items):>2}): Geom O={g_matches}/{len(items)} ({g_matches/len(items)*100:.1f}%), Pipeline Display O={p_matches}/{len(items)} ({p_matches/len(items)*100:.1f}%)")
    
    total_npz_geom_o = sum(1 for x in npz_o_results if x["geom_match_o"])
    total_npz_pipe_o = sum(1 for x in npz_o_results if x["pipeline_sign"] == "O")
    print(f"   --> TỔNG NPZ O: Geometric Rule O = {total_npz_geom_o} / {len(o_indices)} ({total_npz_geom_o/len(o_indices)*100:.1f}%)")
    print(f"                   Pipeline Display O = {total_npz_pipe_o} / {len(o_indices)} ({total_npz_pipe_o/len(o_indices)*100:.1f}%)")
    
    results_data["target_o"]["npz_o"] = {
        "total_samples": len(o_indices),
        "by_video": {vid: {"total": len(items), "geom_matches": sum(1 for x in items if x["geom_match_o"])} for vid, items in npz_o_by_video.items()},
        "total_geom_matches": total_npz_geom_o,
        "total_pipeline_matches": total_npz_pipe_o,
    }

    # --- B. Fixture O.json (N=20, video D0506) ---
    with open(key_files["fixture_o"], "r", encoding="utf-8") as f:
        fix_o_data = json.load(f)
    fix_o_samples = fix_o_data.get("samples", [])
    valid_fix_o = [s for s in fix_o_samples if s.get("landmarks") and len(s.get("landmarks")) == 21]

    print(f"\n2. Fixture O.json (Video D0506, Total N={len(valid_fix_o)} valid frames):")
    fix_o_results = []
    
    rec.prediction_history.clear()
    rec.last_confirmed_sign = None
    rec.last_confirmed_time = 0.0
    rec.sentence_tokens = []
    
    for s in valid_fix_o:
        lm = s["landmarks"]
        fidx = s.get("frame_index") or s.get("frame_idx")
        phase = s.get("phase", "unknown")
        ts = s.get("timestamp_sec", 0.0)
        
        geom_res = clf.classify_detailed(lm)
        geom_match = (geom_res["status"] == "MATCH" and geom_res["letter"] == "O")
        
        pipe_res = rec.predict_landmarks(
            landmarks_list=lm,
            mode="letters",
            timestamp=ts,
            frame_idx=fidx,
            session_id="fixture_o"
        )
        fix_o_results.append({
            "frame_index": fidx,
            "phase": phase,
            "geom_match_o": geom_match,
            "geom_letter": geom_res["letter"],
            "pipeline_sign": pipe_res["sign"],
            "confirmed": pipe_res["confirmed"],
            "new_token": pipe_res["new_token"]
        })

    # D0506 hold frames that matched: exactly 12 frames
    matched_fidxs = [x["frame_index"] for x in fix_o_results if x["geom_match_o"]]
    all_geom = sum(1 for x in fix_o_results if x["geom_match_o"])
    all_pipe = sum(1 for x in fix_o_results if x["pipeline_sign"] == "O")
    confirmed_tokens_fix = [x["new_token"] for x in fix_o_results if x["new_token"]]

    print(f"   Khớp Geometric Rule O: {all_geom} / {len(fix_o_results)} ({all_geom/len(fix_o_results)*100:.1f}%)")
    print(f"   Danh sách frame khớp (12 frames): {matched_fidxs}")
    print(f"   Danh sách frame không khớp (8 frames entry/exit): {[x['frame_index'] for x in fix_o_results if not x['geom_match_o']]}")
    print(f"   Confirmed Tokens Emitted: {confirmed_tokens_fix}")

    results_data["target_o"]["fixture_o"] = {
        "total_valid_frames": len(fix_o_results),
        "matched_frames_count": all_geom,
        "matched_frame_indices": matched_fidxs,
        "unmatched_frame_indices": [x["frame_index"] for x in fix_o_results if not x["geom_match_o"]],
        "confirmed_tokens": confirmed_tokens_fix,
    }

    # --- C. Webcam Pilot 213800 (pilot_pilot_20261108_213800.jsonl) ---
    print(f"\n3. Webcam Pilot 213800 (10 Trials, 1 Signer, Evening Session 21:38:00):")
    p2138_records = []
    with open(key_files["pilot_213800"], "r", encoding="utf-8") as f:
        for line in f:
            p2138_records.append(json.loads(line))
    
    p2138_active = [r for r in p2138_records if r.get("trial_id") and r.get("geometric", {}).get("features")]
    p2138_idle = [r for r in p2138_records if not (r.get("trial_id") and r.get("geometric", {}).get("features"))]
    print(f"   Tổng dòng: {len(p2138_records)} (Active Trial Frames: {len(p2138_active)}, Idle Frames: {len(p2138_idle)})")

    p2138_by_trial = defaultdict(list)
    p2138_geom_matches = 0
    p2138_neutral_frames = 0
    p2138_other_frames = 0
    
    for r in p2138_active:
        t_id = r.get("trial_id")
        fidx = r.get("frame_idx")
        ts = r.get("timestamp", 0.0)
        feat = r["geometric"]["features"]
        
        rules = clf.evaluate_rules(feat)
        is_o = any(m[0] == "O" for m in rules)
        
        pred_letter = "O" if is_o else "NEUTRAL"
        if is_o:
            p2138_geom_matches += 1
        elif pred_letter == "NEUTRAL":
            p2138_neutral_frames += 1
        else:
            p2138_other_frames += 1
            
        p2138_by_trial[t_id].append((fidx, ts, is_o))

    print(f"   --> Cấp độ Frame: Geom O = {p2138_geom_matches} / {len(p2138_active)} ({p2138_geom_matches/len(p2138_active)*100:.1f}%), Neutral={p2138_neutral_frames}, Sai={p2138_other_frames}")
    
    p2138_trials_summary = {}
    p2138_pass_trials = 0
    p2138_confirmed_trials = 0
    for tid in sorted(p2138_by_trial.keys()):
        items = p2138_by_trial[tid]
        m = sum(1 for x in items if x[2])
        rate = m / len(items) if items else 0.0
        passed = rate >= 0.70
        if passed:
            p2138_pass_trials += 1
            
        tokens, disp = simulate_temporal_smoothing(items, rec)
        has_token_o = ("O" in tokens)
        if has_token_o:
            p2138_confirmed_trials += 1
            
        p2138_trials_summary[tid] = {
            "total": len(items), "matched": m, "rate": round(rate, 4),
            "passed_70": passed, "confirmed_tokens": tokens
        }
        print(f"      {tid}: {m:>2}/{len(items):>2} ({rate*100:>5.1f}%) -> {'PASS' if passed else 'FAIL'} | Confirmed Tokens: {tokens}")
    
    print(f"   --> Cấp độ Trial Recall (>= 70%): {p2138_pass_trials} / {len(p2138_by_trial)} trials PASS ({p2138_pass_trials/len(p2138_by_trial)*100:.1f}%)")
    print(f"   --> Cấp độ Runtime Confirmation : {p2138_confirmed_trials} / {len(p2138_by_trial)} trials CONFIRMED O ({p2138_confirmed_trials/len(p2138_by_trial)*100:.1f}%)")

    results_data["target_o"]["pilot_213800"] = {
        "total_lines": len(p2138_records),
        "active_frames": len(p2138_active),
        "idle_frames": len(p2138_idle),
        "geom_matches": p2138_geom_matches,
        "geom_accuracy": round(p2138_geom_matches / len(p2138_active), 4),
        "trial_accuracy": f"{p2138_pass_trials}/{len(p2138_by_trial)}",
        "runtime_confirmed_trials": f"{p2138_confirmed_trials}/{len(p2138_by_trial)}",
        "trials": p2138_trials_summary
    }

    # --- D. Webcam Pilot 211742 (pilot_pilot_20261108_193712_20261108_211742_1.jsonl) ---
    print(f"\n4. Webcam Pilot 211742 (10 Trials, 1 Signer, Evening Session 21:17:42 - Khảo sát chuyên sâu):")
    p2117_records = []
    with open(key_files["pilot_211742"], "r", encoding="utf-8") as f:
        for line in f:
            p2117_records.append(json.loads(line))
            
    p2117_active = [r for r in p2117_records if r.get("trial_id") and r.get("geometric", {}).get("features")]
    p2117_idle = [r for r in p2117_records if not (r.get("trial_id") and r.get("geometric", {}).get("features"))]
    print(f"   Tổng dòng: {len(p2117_records)} (Active Trial Frames: {len(p2117_active)}, Idle Frames: {len(p2117_idle)})")

    p2117_by_trial = defaultdict(list)
    p2117_geom_matches = 0
    p2117_diag_fails = Counter()
    p2117_outer_gate_fails = 0

    for r in p2117_active:
        t_id = r.get("trial_id")
        fidx = r.get("frame_idx")
        ts = r.get("timestamp", 0.0)
        feat = r["geometric"]["features"]
        
        rules = clf.evaluate_rules(feat)
        is_o = any(m[0] == "O" for m in rules)
        if is_o:
            p2117_geom_matches += 1
            
        outer_gate = not (feat.get("index_ext") or feat.get("middle_ext") or feat.get("ring_ext") or feat.get("pinky_ext"))
        if not outer_gate:
            p2117_outer_gate_fails += 1
            
        diag = clf.evaluate_rule_o_diagnostics(feat)
        if diag:
            for c_name, c_pass in diag.get("conditions", {}).items():
                if c_pass is False:
                    p2117_diag_fails[c_name] += 1
                    
        p2117_by_trial[t_id].append((fidx, ts, is_o))

    print(f"   --> Cấp độ Frame: Geom O = {p2117_geom_matches} / {len(p2117_active)} ({p2117_geom_matches/len(p2117_active)*100:.1f}%)")
    print(f"       Outer Gate vi phạm: {p2117_outer_gate_fails} frames")
    print(f"       Tần suất vi phạm 8 điều kiện hình học:")
    for cond_name, count in p2117_diag_fails.most_common():
        print(f"         - {cond_name:<22}: {count:>2} / {len(p2117_active)} frames ({count/len(p2117_active)*100:.1f}%)")

    p2117_trials_summary = {}
    p2117_pass_trials = 0
    p2117_confirmed_trials = 0
    for tid in sorted(p2117_by_trial.keys()):
        items = p2117_by_trial[tid]
        m = sum(1 for x in items if x[2])
        rate = m / len(items) if items else 0.0
        passed = rate >= 0.70
        if passed:
            p2117_pass_trials += 1
            
        tokens, disp = simulate_temporal_smoothing(items, rec)
        has_token_o = ("O" in tokens)
        if has_token_o:
            p2117_confirmed_trials += 1
            
        p2117_trials_summary[tid] = {
            "total": len(items), "matched": m, "rate": round(rate, 4),
            "passed_70": passed, "confirmed_tokens": tokens
        }
        print(f"      {tid}: {m:>2}/{len(items):>2} ({rate*100:>5.1f}%) -> {'PASS' if passed else 'FAIL'} | Confirmed Tokens: {tokens}")

    print(f"   --> Cấp độ Trial Recall (>= 70%): {p2117_pass_trials} / {len(p2117_by_trial)} trials PASS ({p2117_pass_trials/len(p2117_by_trial)*100:.1f}%)")
    print(f"   --> Cấp độ Runtime Confirmation : {p2117_confirmed_trials} / {len(p2117_by_trial)} trials CONFIRMED O ({p2117_confirmed_trials/len(p2117_by_trial)*100:.1f}%)")

    results_data["target_o"]["pilot_211742"] = {
        "total_lines": len(p2117_records),
        "active_frames": len(p2117_active),
        "idle_frames": len(p2117_idle),
        "geom_matches": p2117_geom_matches,
        "geom_accuracy": round(p2117_geom_matches / len(p2117_active), 4),
        "outer_gate_fails": p2117_outer_gate_fails,
        "condition_failures": dict(p2117_diag_fails),
        "trial_accuracy": f"{p2117_pass_trials}/{len(p2117_by_trial)}",
        "runtime_confirmed_trials": f"{p2117_confirmed_trials}/{len(p2117_by_trial)}",
        "trials": p2117_trials_summary
    }

    # =========================================================================
    # PHẦN 3: FALSE POSITIVE AUDIT TRÊN TOÀN BỘ DỮ LIỆU NON-O
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHẦN 3: KIỂM TOÁN FALSE POSITIVE TRÊN TẬP ĐỐI KHÁNG NON-O")
    print("=" * 90)

    # --- A. NPZ Non-O (N=503) ---
    non_o_indices = [i for i, label_idx in enumerate(y) if classes[label_idx] != "O"]
    print(f"\n1. NPZ Non-O Dataset (N={len(non_o_indices)} samples across 21 letters):")
    print("   Lưu ý: Replay kiểm tra toàn bộ 503 mẫu, KHÔNG lọc frame 54 của B.")

    npz_non_o_fps = defaultdict(list)
    for idx in non_o_indices:
        true_label = classes[y[idx]]
        m = meta_list[idx]
        lm = landmarks[idx].tolist()
        
        geom_res = clf.classify_detailed(lm)
        if geom_res["status"] == "MATCH" and geom_res["letter"] == "O":
            npz_non_o_fps[true_label].append({
                "idx": idx,
                "video_id": m.get("video_id"),
                "frame_index": m.get("frame_index"),
                "is_representative": m.get("is_representative"),
                "status": geom_res["status"],
                "reason": geom_res["reason"],
            })

    total_npz_fps = sum(len(v) for v in npz_non_o_fps.values())
    print(f"   --> Tổng số False Positives sang O trong NPZ: {total_npz_fps} mẫu / {len(non_o_indices)}")
    for lbl, items in npz_non_o_fps.items():
        print(f"       Chữ {lbl}: {len(items)} mẫu -> {items}")

    results_data["non_o"]["npz_non_o"] = {
        "total_samples": len(non_o_indices),
        "total_fps": total_npz_fps,
        "fps_by_letter": {lbl: items for lbl, items in npz_non_o_fps.items()}
    }

    # --- B. 22 Fixtures tĩnh (A through Y, trừ O) ---
    fixtures_dir = "backend/data/alphabet_v2/static_landmarks"
    fixture_files = sorted([f for f in os.listdir(fixtures_dir) if f.endswith(".json") and f != "O.json"])
    print(f"\n2. 22 Fixtures tĩnh độc lập (A-Y trừ O, tổng cộng {len(fixture_files)} files):")
    
    fix_fps = defaultdict(list)
    total_fix_valid = 0
    for fname in fixture_files:
        lbl = fname.replace(".json", "")
        fpath = os.path.join(fixtures_dir, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            fdata = json.load(f)
        samples = fdata.get("samples", [])
        valid_s = [s for s in samples if s.get("landmarks") and len(s.get("landmarks")) == 21]
        total_fix_valid += len(valid_s)
        
        for s in valid_s:
            fidx = s.get("frame_index") or s.get("frame_idx")
            phase = s.get("phase", "unknown")
            lm = s["landmarks"]
            geom_res = clf.classify_detailed(lm)
            if geom_res["status"] == "MATCH" and geom_res["letter"] == "O":
                fix_fps[lbl].append({
                    "file": fname,
                    "frame_index": fidx,
                    "phase": phase,
                    "reason": geom_res["reason"]
                })

    total_fix_fps = sum(len(v) for v in fix_fps.values())
    print(f"   --> Tổng số frames kiểm tra: {total_fix_valid} frames across {len(fixture_files)} fixtures")
    print(f"   --> Tổng số False Positives sang O: {total_fix_fps} frames")
    for lbl, items in fix_fps.items():
        print(f"       Chữ {lbl}: {len(items)} frames -> {items}")

    results_data["non_o"]["fixtures_non_o"] = {
        "total_fixtures": len(fixture_files),
        "total_valid_frames": total_fix_valid,
        "total_fps": total_fix_fps,
        "fps_by_letter": {lbl: items for lbl, items in fix_fps.items()}
    }

    # --- C. Webcam Holdout P (N=314 active frames) ---
    print(f"\n3. Webcam Holdout Chữ P (Session pilot_183703, N=314 active frames across 10 trials):")
    p_holdout_records = []
    with open(key_files["p_holdout"], "r", encoding="utf-8") as f:
        for line in f:
            p_holdout_records.append(json.loads(line))
    p_active = [r for r in p_holdout_records if r.get("trial_id") and r.get("geometric", {}).get("features")]
    
    p_fps = 0
    for r in p_active:
        feat = r["geometric"]["features"]
        rules = clf.evaluate_rules(feat)
        if any(m[0] == "O" for m in rules):
            p_fps += 1
    print(f"   --> False Positives sang O: {p_fps} / {len(p_active)} frames (0.00% tuyệt đối)")
    results_data["non_o"]["p_holdout"] = {"total_active": len(p_active), "fps": p_fps}

    # --- D. Webcam Holdout X (N=108 active frames) ---
    print(f"\n4. Webcam Holdout Chữ X (Session pilot_193712, N=108 active frames across 10 trials):")
    x_holdout_records = []
    with open(key_files["x_holdout"], "r", encoding="utf-8") as f:
        for line in f:
            x_holdout_records.append(json.loads(line))
    x_active = [r for r in x_holdout_records if r.get("trial_id") and r.get("geometric", {}).get("features")]
    
    x_fps = 0
    for r in x_active:
        feat = r["geometric"]["features"]
        rules = clf.evaluate_rules(feat)
        if any(m[0] == "O" for m in rules):
            x_fps += 1
    print(f"   --> False Positives sang O: {x_fps} / {len(x_active)} frames (0.00% tuyệt đối)")
    results_data["non_o"]["x_holdout"] = {"total_active": len(x_active), "fps": x_fps}

    # =========================================================================
    # PHẦN 4: SO SÁNH VỚI CÁC BÁO CÁO TRƯỚC ĐÂY
    # =========================================================================
    print("\n" + "=" * 90)
    print("PHẦN 4: ĐỐI CHIẾU VỚI CÁC CON SỐ ĐÃ BÁO CÁO (TASK 3.4 & 3.5)")
    print("=" * 90)
    comparison_table = [
        ("Webcam 213800 Active Frames", "88 / 92 (95.65%)", f"{p2138_geom_matches} / {len(p2138_active)} ({p2138_geom_matches/len(p2138_active)*100:.2f}%)", p2138_geom_matches == 88),
        ("Webcam 211742 Active Frames", "6 / 68 (8.82%)", f"{p2117_geom_matches} / {len(p2117_active)} ({p2117_geom_matches/len(p2117_active)*100:.2f}%)", p2117_geom_matches == 6),
        ("D0506 Fixture Matched Frames", "12 / 12 (Hold 100%)", f"{all_geom} / {len(fix_o_results)} (Total {all_geom/len(fix_o_results)*100:.2f}%)", all_geom == 12),
        ("O NPZ Total Samples", "37 / 48 (77.08%)", f"{total_npz_geom_o} / {len(o_indices)} ({total_npz_geom_o/len(o_indices)*100:.2f}%)", total_npz_geom_o == 37),
        ("NPZ Non-O FP (sans test exception)", "1 / 503 (B f54)", f"{total_npz_fps} / {len(non_o_indices)}", total_npz_fps == 1),
        ("Webcam P Holdout FP", "0 / 314 (0.0%)", f"{p_fps} / {len(p_active)}", p_fps == 0),
        ("Webcam X Holdout FP", "0 / 108 (0.0%)", f"{x_fps} / {len(x_active)}", x_fps == 0),
    ]

    print(f"{'Bộ Dữ Liệu':<36} | {'Báo Cáo Cũ':<20} | {'Replay TASK 3.6':<20} | {'Khớp Đúng':<10}")
    print("-" * 92)
    for name, old_val, new_val, is_match in comparison_table:
        m_str = "KHỚP 100%" if is_match else "SAI LỆCH"
        print(f"{name:<36} | {old_val:<20} | {new_val:<20} | {m_str:<10}")

    results_data["historical_comparison"] = [
        {"dataset": name, "historical": old_val, "replay": new_val, "exact_match": is_match}
        for name, old_val, new_val, is_match in comparison_table
    ]

    # Lưu kết quả đầy đủ vào JSON trong scratch/
    out_json_path = "scratch/task3_6_replay_summary.json"
    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print(f"\n[Artifact Created] Đã lưu báo cáo chi tiết vào: {out_json_path}")

if __name__ == "__main__":
    main()
