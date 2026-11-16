import os
import sys
import json
import numpy as np
import torch
from collections import Counter, defaultdict

sys.path.insert(0, ".")
sys.stdout.reconfigure(encoding='utf-8')

from backend.vision.alphabet_classifier import VSLAlphabetClassifier
from backend.models.alphabet_static_head import (
    load_static_alphabet_head,
    STATIC_CLASSES_22,
    CLASS_TO_IDX_22,
    IDX_TO_CLASS_22
)
from backend.vision.feature_extractor import normalize_hand_landmarks
from backend.vision.recognizer import VSLRecognizer

SEED = 42

def compute_percentiles(arr):
    if len(arr) == 0:
        return {}
    return {
        "min": round(float(np.min(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "median": round(float(np.median(arr)), 4),
        "mean": round(float(np.mean(arr)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
        "max": round(float(np.max(arr)), 4),
        "std": round(float(np.std(arr)), 4)
    }

print("================================================================================")
print("TASK 7E.5A: KIỂM TOÁN CHỮ NỀN TĨNH A, E, D (READ-ONLY AUDIT)")
print("================================================================================")

# 1. Tải Dataset V2 và Checkpoint Static Head
dataset_path = "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"
data = np.load(dataset_path, allow_pickle=True)
X_all = data["X"]
y_all = data["y"]
classes_all = list(data["classes"])
lms_all = data["landmarks"]
metadata_all = json.loads(str(data["metadata_json"]))

head, head_meta = load_static_alphabet_head()
clf = VSLAlphabetClassifier()

# 2. Tái tạo chính xác Split Train / Holdout (Seed 42 + c_idx)
split_membership = {}
train_indices_by_class = {}
val_indices_by_class = {}
all_train_indices = []
all_val_indices = []

for c_idx in range(len(STATIC_CLASSES_22)):
    c_samples = np.where(y_all == c_idx)[0]
    rng = np.random.RandomState(SEED + c_idx)
    shuffled = rng.permutation(c_samples)
    n_val = max(1, int(len(shuffled) * 0.20))
    val_idx = shuffled[:n_val]
    train_idx = shuffled[n_val:]
    
    train_indices_by_class[STATIC_CLASSES_22[c_idx]] = train_idx.tolist()
    val_indices_by_class[STATIC_CLASSES_22[c_idx]] = val_idx.tolist()
    all_train_indices.extend(train_idx)
    all_val_indices.extend(val_idx)
    
    for idx in train_idx:
        split_membership[idx] = "train"
    for idx in val_idx:
        split_membership[idx] = "holdout"

print(f"[Split Reproduction] Total Train: {len(all_train_indices)}, Total Holdout: {len(all_val_indices)}")

# -----------------------------------------------------------------------------------
# GIAI ĐOẠN A: KIỂM TOÁN GEOMETRIC RULES TRÊN A, E, D
# -----------------------------------------------------------------------------------
print("\n--- GIAI ĐOẠN A: KIỂM TOÁN GEOMETRIC RULES ---")

target_classes = ["A", "D", "E"]
geom_audit = {}

for target in target_classes:
    c_idx = CLASS_TO_IDX_22[target]
    c_indices = np.where(y_all == c_idx)[0]
    c_lms = lms_all[c_indices]
    
    total_samples = len(c_indices)
    extracted_features = []
    rule_results = []
    
    # Conditions tracker
    cond_counts = defaultdict(int)
    cond_failed_samples = defaultdict(list)
    
    for idx_in_c, global_idx in enumerate(c_indices):
        lm = c_lms[idx_in_c].tolist()
        feat = clf.extract_features(lm)
        res = clf.classify_detailed(lm)
        
        extracted_features.append(feat)
        rule_results.append({
            "global_idx": int(global_idx),
            "split": split_membership[global_idx],
            "video_id": metadata_all[global_idx]["video_id"],
            "status": res["status"],
            "letter": res["letter"],
            "confidence": res["confidence"],
            "reason": res["reason"]
        })
        
        if feat is None:
            continue
            
        t_ext = feat["thumb_ext"]
        i_ext = feat["index_ext"]
        m_ext = feat["middle_ext"]
        r_ext = feat["ring_ext"]
        p_ext = feat["pinky_ext"]
        d_ti = feat["d_TI"]
        d_tm = feat["d_TM"]
        d_tw = feat["d_TW"]
        d_im = feat["d_IM"]
        d_iw = feat["d_IW"]
        d_mw = feat["d_MW"]
        d_pw = feat["d_PW"]
        
        if target == "A":
            c1 = (not i_ext and not m_ext and not r_ext and not p_ext)
            c2 = (d_tw >= 1.32)
            c3 = (d_tm >= 0.75)
            c4 = (d_iw <= 0.90)
            c5 = (d_mw <= 0.75)
            
            if c1: cond_counts["fingers_curled"] += 1
            else: cond_failed_samples["fingers_curled"].append(int(global_idx))
            if c2: cond_counts["d_TW >= 1.32"] += 1
            else: cond_failed_samples["d_TW >= 1.32"].append(int(global_idx))
            if c3: cond_counts["d_TM >= 0.75"] += 1
            else: cond_failed_samples["d_TM >= 0.75"].append(int(global_idx))
            if c4: cond_counts["d_IW <= 0.90"] += 1
            else: cond_failed_samples["d_IW <= 0.90"].append(int(global_idx))
            if c5: cond_counts["d_MW <= 0.75"] += 1
            else: cond_failed_samples["d_MW <= 0.75"].append(int(global_idx))
            
            if c1 and c2 and c3 and c4 and c5:
                cond_counts["all_simultaneous"] += 1
                
        elif target == "D":
            c1 = (i_ext and not m_ext and not r_ext and not p_ext)
            c2 = (d_iw >= 1.70)
            c3 = (d_ti <= 1.28)
            c4 = (d_tm <= 0.30)
            
            if c1: cond_counts["index_only_ext"] += 1
            else: cond_failed_samples["index_only_ext"].append(int(global_idx))
            if c2: cond_counts["d_IW >= 1.70"] += 1
            else: cond_failed_samples["d_IW >= 1.70"].append(int(global_idx))
            if c3: cond_counts["d_TI <= 1.28"] += 1
            else: cond_failed_samples["d_TI <= 1.28"].append(int(global_idx))
            if c4: cond_counts["d_TM <= 0.30"] += 1
            else: cond_failed_samples["d_TM <= 0.30"].append(int(global_idx))
            
            if c1 and c2 and c3 and c4:
                cond_counts["all_simultaneous"] += 1
                
        elif target == "E":
            c1 = (not i_ext and not m_ext and not r_ext and not p_ext)
            c2 = (d_ti <= 0.35)
            c3 = (d_tm <= 0.38)
            c4 = (d_tw <= 0.95)
            c5 = (d_iw >= 1.05)
            c6 = (d_mw >= 1.05)
            
            if c1: cond_counts["fingers_curled"] += 1
            else: cond_failed_samples["fingers_curled"].append(int(global_idx))
            if c2: cond_counts["d_TI <= 0.35"] += 1
            else: cond_failed_samples["d_TI <= 0.35"].append(int(global_idx))
            if c3: cond_counts["d_TM <= 0.38"] += 1
            else: cond_failed_samples["d_TM <= 0.38"].append(int(global_idx))
            if c4: cond_counts["d_TW <= 0.95"] += 1
            else: cond_failed_samples["d_TW <= 0.95"].append(int(global_idx))
            if c5: cond_counts["d_IW >= 1.05"] += 1
            else: cond_failed_samples["d_IW >= 1.05"].append(int(global_idx))
            if c6: cond_counts["d_MW >= 1.05"] += 1
            else: cond_failed_samples["d_MW >= 1.05"].append(int(global_idx))
            
            if c1 and c2 and c3 and c4 and c5 and c6:
                cond_counts["all_simultaneous"] += 1

    # Thống kê phân phối feature
    feat_dists = {}
    key_features = ["d_TI", "d_TM", "d_TW", "d_IW", "d_MW", "d_PW", "ang_TI", "ang_TP", "thumb_lat", "index_curl", "middle_curl"]
    for kf in key_features:
        vals = [f[kf] for f in extracted_features if f is not None and kf in f]
        feat_dists[kf] = compute_percentiles(vals)

    status_counter = Counter(r["status"] for r in rule_results)
    letter_counter = Counter(r["letter"] for r in rule_results)
    
    geom_audit[target] = {
        "total_samples": total_samples,
        "matched_correct": letter_counter.get(target, 0),
        "matched_wrong": {k: v for k, v in letter_counter.items() if k not in (target, "UNKNOWN")},
        "no_match_count": status_counter.get("NO_MATCH", 0),
        "unknown_count": status_counter.get("UNKNOWN", 0),
        "condition_satisfaction": dict(cond_counts),
        "rejection_by_condition_pct": {
            k: round((1.0 - v / total_samples) * 100, 2)
            for k, v in cond_counts.items() if k != "all_simultaneous"
        },
        "feature_distributions": feat_dists,
        "sample_rule_results": rule_results
    }
    
    print(f"[{target}] Total: {total_samples} | MATCH {target}: {letter_counter.get(target, 0)} ({letter_counter.get(target, 0)/total_samples*100:.1f}%) | "
          f"NO_MATCH: {status_counter.get('NO_MATCH', 0)} | Wrong MATCH: {dict(geom_audit[target]['matched_wrong'])}")
    print(f"    Điều kiện trượt nhiều nhất: {sorted(geom_audit[target]['rejection_by_condition_pct'].items(), key=lambda x: x[1], reverse=True)[:3]}")

# -----------------------------------------------------------------------------------
# GIAI ĐOẠN B: ĐÁNH GIÁ STATIC HEAD TRÊN A, E, D VÀ TOÀN BỘ DATASET V2
# -----------------------------------------------------------------------------------
print("\n--- GIAI ĐOẠN B: ĐÁNH GIÁ STATIC HEAD ---")

# Đánh giá toàn bộ 551 mẫu
head_preds_all = []
for i in range(len(X_all)):
    res = head.predict_feature(X_all[i], confidence_threshold=0.0)
    top_k = res.get("top_k", [])
    top1_letter = res.get("raw_predicted_letter", "UNKNOWN")
    top1_conf = float(res.get("confidence", 0.0))
    top2_letter = top_k[1]["letter"] if len(top_k) > 1 else None
    top2_conf = top_k[1]["confidence"] if len(top_k) > 1 else 0.0
    
    gt_label = STATIC_CLASSES_22[y_all[i]]
    head_preds_all.append({
        "global_idx": i,
        "split": split_membership[i],
        "video_id": metadata_all[i]["video_id"],
        "gt": gt_label,
        "top1": top1_letter,
        "top1_conf": top1_conf,
        "top2": top2_letter,
        "top2_conf": top2_conf,
        "correct": (top1_letter == gt_label)
    })

# Confusion matrix 22x22
conf_matrix_22 = {c1: {c2: 0 for c2 in STATIC_CLASSES_22} for c1 in STATIC_CLASSES_22}
conf_matrix_train = {c1: {c2: 0 for c2 in STATIC_CLASSES_22} for c1 in STATIC_CLASSES_22}
conf_matrix_holdout = {c1: {c2: 0 for c2 in STATIC_CLASSES_22} for c1 in STATIC_CLASSES_22}

for p in head_preds_all:
    gt = p["gt"]
    top1 = p["top1"]
    if top1 in conf_matrix_22[gt]:
        conf_matrix_22[gt][top1] += 1
        if p["split"] == "train":
            conf_matrix_train[gt][top1] += 1
        elif p["split"] == "holdout":
            conf_matrix_holdout[gt][top1] += 1

static_head_eval = {}
for target in target_classes:
    preds_target = [p for p in head_preds_all if p["gt"] == target]
    preds_train = [p for p in preds_target if p["split"] == "train"]
    preds_holdout = [p for p in preds_target if p["split"] == "holdout"]
    
    acc_all = np.mean([p["correct"] for p in preds_target])
    acc_train = np.mean([p["correct"] for p in preds_train]) if preds_train else 0.0
    acc_holdout = np.mean([p["correct"] for p in preds_holdout]) if preds_holdout else 0.0
    
    confs_correct = [p["top1_conf"] for p in preds_target if p["correct"]]
    confs_incorrect = [p["top1_conf"] for p in preds_target if not p["correct"]]
    
    # False positive: Các mẫu lớp khác bị dự đoán nhầm thành target
    fp_samples = [p for p in head_preds_all if p["gt"] != target and p["top1"] == target]
    fp_by_gt = Counter(p["gt"] for p in fp_samples)
    
    static_head_eval[target] = {
        "total_samples": len(preds_target),
        "train_samples": len(preds_train),
        "holdout_samples": len(preds_holdout),
        "accuracy_all_pct": round(acc_all * 100, 2),
        "accuracy_train_pct": round(acc_train * 100, 2),
        "accuracy_holdout_pct": round(acc_holdout * 100, 2),
        "conf_distribution_correct": compute_percentiles(confs_correct),
        "conf_distribution_incorrect": compute_percentiles(confs_incorrect),
        "confusion_targets": Counter(p["top1"] for p in preds_target if not p["correct"]),
        "false_positive_count": len(fp_samples),
        "false_positive_sources": dict(fp_by_gt),
        "samples": preds_target
    }
    
    print(f"[{target} Static Head] All: {acc_all*100:.1f}% ({len(preds_target)} mẫu) | "
          f"Train: {acc_train*100:.1f}% ({len(preds_train)} mẫu) | "
          f"Holdout: {acc_holdout*100:.1f}% ({len(preds_holdout)} mẫu) | "
          f"Conf Đúng: {static_head_eval[target]['conf_distribution_correct'].get('median', 0.0)} | "
          f"FP count: {len(fp_samples)} từ {dict(fp_by_gt)}")

# Đánh giá leakage: Kiểm tra video ID overlap
leakage_audit = {}
for target in target_classes:
    c_idx = CLASS_TO_IDX_22[target]
    tr_idx = train_indices_by_class[target]
    val_idx = val_indices_by_class[target]
    tr_vids = set(metadata_all[i]["video_id"] for i in tr_idx)
    val_vids = set(metadata_all[i]["video_id"] for i in val_idx)
    overlap = tr_vids.intersection(val_vids)
    
    leakage_audit[target] = {
        "train_video_ids": list(tr_vids),
        "holdout_video_ids": list(val_vids),
        "overlapping_video_ids": list(overlap),
        "leakage_level": "COMPLETE_VIDEO_OVERLAP" if len(overlap) == len(val_vids) else "PARTIAL_OVERLAP",
        "description": "Holdout frames come from the exact same videos as training frames (temporal/consecutive frame split leakage)."
    }

print("\n[Leakage Audit Summary]")
for t, la in leakage_audit.items():
    print(f"  {t}: {la['leakage_level']} (Overlap: {la['overlapping_video_ids']})")

# -----------------------------------------------------------------------------------
# GIAI ĐOẠN C: REPLAY BỐN VIDEO ĐỘNG (Ă, Â, Ê, Đ)
# -----------------------------------------------------------------------------------
print("\n--- GIAI ĐOẠN C: REPLAY 4 VIDEO ĐỘNG ---")

# Verified Phase Ranges từ phase_review_report.json (Task 5A/5B verified)
VERIFIED_PHASES = {
    "Ă": {"entry": (0, 6), "base_pose": (7, 25), "diacritic_motion": (26, 48), "exit": (49, 88)},
    "Â": {"entry": (0, 26), "base_pose": (27, 49), "diacritic_motion": (50, 74), "exit": (75, 107)},
    "Ê": {"entry": (0, 7), "base_pose": (8, 27), "diacritic_motion": (28, 47), "exit": (48, 75)},
    "Đ": {"entry": (0, 37), "base_pose": (38, 63), "diacritic_motion": (64, 72), "exit": (73, 104)}
}

def get_verified_phase(letter: str, frame_idx: int) -> str:
    phases = VERIFIED_PHASES.get(letter, {})
    for p_name, (start, end) in phases.items():
        if start <= frame_idx <= end:
            return "BASE_POSE" if p_name == "base_pose" else ("DIACRITIC" if p_name == "diacritic_motion" else p_name.upper())
    return "UNKNOWN"

dynamic_video_files = {
    "Ă": ("backend/data/alphabet_v2/motion_sequences/Ă.json", "A"),
    "Â": ("backend/data/alphabet_v2/motion_sequences/Â.json", "A"),
    "Ê": ("backend/data/alphabet_v2/motion_sequences/Ê.json", "E"),
    "Đ": ("backend/data/alphabet_v2/motion_sequences/Đ.json", "D")
}

dynamic_replay_results = {}
rec = VSLRecognizer()

for target, (fpath, expected_base) in dynamic_video_files.items():
    with open(fpath, "r", encoding="utf-8") as f:
        seq_data = json.load(f)
    seq = seq_data.get("sequence", [])
    vid_id = seq_data.get("video_id", "")
    
    session_id = f"audit_7e5a_{target}"
    rec.reset_diacritic_fsm(session_id=session_id)
    
    frame_timeline = []
    base_pose_head_predictions = []
    diacritic_head_predictions = []
    
    # Bộ đếm chuỗi liên tiếp của Static Head cho expected_base
    head_consecutive_base_streaks = []
    current_streak = 0
    current_streak_start_frame = None
    
    for item in seq:
        lms = item.get("landmarks")
        t = item.get("timestamp_sec", 0.0)
        idx = item.get("frame_index", 0)
        phase = get_verified_phase(target, idx)
        is_valid = (lms is not None and len(lms) == 21)
        
        # 1. Geometric Rule kết quả
        geom_res = clf.classify_detailed(lms)
        geom_letter = geom_res.get("letter", "UNKNOWN")
        geom_status = geom_res.get("status", "INVALID_INPUT")
        geom_conf = geom_res.get("confidence", 0.0)
        
        # 2. Static Head inference
        head_letter = "UNKNOWN"
        head_conf = 0.0
        head_top2_letter = None
        head_top2_conf = 0.0
        
        if is_valid:
            feat_72, _ = normalize_hand_landmarks(lms)
            h_res = head.predict_feature(feat_72[:72], confidence_threshold=0.0)
            head_letter = h_res.get("raw_predicted_letter", "UNKNOWN")
            head_conf = float(h_res.get("confidence", 0.0))
            top_k = h_res.get("top_k", [])
            if len(top_k) > 1:
                head_top2_letter = top_k[1]["letter"]
                head_top2_conf = float(top_k[1]["confidence"])
        
        # 3. Production recognizer run
        rec_res = rec.predict_landmarks(lms, mode="letters", session_id=session_id, timestamp=t)
        fsm_curr = rec.diacritic_fsms.get(session_id)
        fsm_state = fsm_curr.state if fsm_curr else "IDLE"
        fsm_base = fsm_curr.current_base if fsm_curr else None
        has_baseline = (fsm_curr.baseline is not None) if fsm_curr else False
        
        # Tracking streak của Head (với ngưỡng conf >= 0.60)
        if head_letter == expected_base and head_conf >= 0.60:
            if current_streak == 0:
                current_streak_start_frame = idx
            current_streak += 1
        else:
            if current_streak > 0:
                head_consecutive_base_streaks.append({
                    "start_frame": current_streak_start_frame,
                    "end_frame": idx - 1,
                    "streak_len": current_streak,
                    "phase": phase
                })
            current_streak = 0
            current_streak_start_frame = None
            
        frame_entry = {
            "frame_index": idx,
            "timestamp": t,
            "phase": phase,
            "landmarks_valid": is_valid,
            "geom_status": geom_status,
            "geom_letter": geom_letter,
            "geom_conf": geom_conf,
            "head_letter": head_letter,
            "head_conf": head_conf,
            "head_top2_letter": head_top2_letter,
            "head_top2_conf": head_top2_conf,
            "expected_base": expected_base,
            "prod_sign": rec_res.get("sign"),
            "fsm_state": fsm_state,
            "fsm_base": fsm_base,
            "has_baseline": has_baseline,
            "geom_matched_base": (geom_letter == expected_base and geom_status == "MATCH"),
            "head_matched_base": (head_letter == expected_base)
        }
        frame_timeline.append(frame_entry)
        
        if phase == "BASE_POSE" and is_valid:
            base_pose_head_predictions.append(head_letter)
        elif phase == "DIACRITIC" and is_valid:
            diacritic_head_predictions.append(head_letter)

    if current_streak > 0:
        head_consecutive_base_streaks.append({
            "start_frame": current_streak_start_frame,
            "end_frame": len(seq) - 1,
            "streak_len": current_streak,
            "phase": "END"
        })

    # Phân tích BASE_POSE
    base_pose_frames = [f for f in frame_timeline if f["phase"] == "BASE_POSE"]
    bp_geom_matches = sum(1 for f in base_pose_frames if f["geom_matched_base"])
    bp_head_matches = sum(1 for f in base_pose_frames if f["head_matched_base"])
    bp_head_matches_conf70 = sum(1 for f in base_pose_frames if f["head_matched_base"] and f["head_conf"] >= 0.70)
    
    # Phân tích DIACRITIC
    diac_frames = [f for f in frame_timeline if f["phase"] == "DIACRITIC"]
    diac_head_predicting_base = sum(1 for f in diac_frames if f["head_matched_base"])
    
    # Kiểm tra FSM có đạt BASE_STABLE không
    reached_stable = any(f["has_baseline"] for f in frame_timeline)
    
    dynamic_replay_results[target] = {
        "video_id": vid_id,
        "total_frames": len(seq),
        "expected_base": expected_base,
        "base_pose_total_frames": len(base_pose_frames),
        "base_pose_geom_match_count": bp_geom_matches,
        "base_pose_head_match_count": bp_head_matches,
        "base_pose_head_match_conf70_count": bp_head_matches_conf70,
        "diacritic_total_frames": len(diac_frames),
        "diacritic_head_predicting_base_count": diac_head_predicting_base,
        "prod_reached_base_stable": reached_stable,
        "head_streaks_ge_4_frames": [s for s in head_consecutive_base_streaks if s["streak_len"] >= 4],
        "all_head_streaks": head_consecutive_base_streaks,
        "timeline": frame_timeline
    }
    
    print(f"[{target} - Vid {vid_id}] BASE_POSE frames: {len(base_pose_frames)} | "
          f"Geom match {expected_base}: {bp_geom_matches} | "
          f"Head match {expected_base}: {bp_head_matches} (Conf >= 0.70: {bp_head_matches_conf70}) | "
          f"Prod reached BASE_STABLE: {reached_stable} | "
          f"Head streaks >= 4: {len(dynamic_replay_results[target]['head_streaks_ge_4_frames'])}")

# -----------------------------------------------------------------------------------
# GIAI ĐOẠN D: SO SÁNH HAI PHƯƠNG ÁN THIẾT KẾ
# -----------------------------------------------------------------------------------
print("\n--- GIAI ĐOẠN D: MÔ PHỎNG & SO SÁNH HAI PHƯƠNG ÁN ---")

# PHƯƠNG ÁN 1: Static Head Fallback (Thử nghiệm các ngưỡng confidence)
thresholds_to_test = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
fallback_simulation = {}

for target in target_classes:
    fallback_simulation[target] = {}
    c_idx = CLASS_TO_IDX_22[target]
    c_indices = np.where(y_all == c_idx)[0]
    total_samples = len(c_indices)
    
    # Mẫu lớp khác (để tính False Positive)
    other_indices = np.where(y_all != c_idx)[0]
    
    for thresh in thresholds_to_test:
        rescued_count = 0
        correct_count = 0
        wrong_count = 0
        
        for g_idx in c_indices:
            lm = lms_all[g_idx].tolist()
            geom_res = clf.classify_detailed(lm)
            
            if geom_res["status"] == "MATCH" and geom_res["letter"] == target:
                # Đã được geometric rule nhận diện
                correct_count += 1
            else:
                # Geom NO_MATCH -> kích hoạt fallback
                feat = X_all[g_idx]
                h_res = head.predict_feature(feat, confidence_threshold=0.0)
                h_letter = h_res["raw_predicted_letter"]
                h_conf = h_res["confidence"]
                
                if h_letter == target and h_conf >= thresh:
                    rescued_count += 1
                    correct_count += 1
                elif h_conf >= thresh:
                    wrong_count += 1
                    
        # Tính False Positives từ các lớp khác
        fp_count = 0
        for o_idx in other_indices:
            lm = lms_all[o_idx].tolist()
            geom_res = clf.classify_detailed(lm)
            
            # Chỉ fallback khi geom NO_MATCH
            if geom_res["status"] != "MATCH":
                feat = X_all[o_idx]
                h_res = head.predict_feature(feat, confidence_threshold=0.0)
                if h_res["raw_predicted_letter"] == target and h_res["confidence"] >= thresh:
                    fp_count += 1
                    
        fallback_simulation[target][f"thresh_{thresh:.2f}"] = {
            "threshold": thresh,
            "geom_only_correct": int(geom_audit[target]["matched_correct"]),
            "rescued_by_fallback": rescued_count,
            "total_correct": correct_count,
            "recall_pct": round(correct_count / total_samples * 100, 2),
            "false_positive_count": fp_count,
            "fp_rate_pct": round(fp_count / len(other_indices) * 100, 2)
        }

print("\n[Mô phỏng Phương án 1 - Static Head Fallback]")
for target in target_classes:
    print(f"[{target}] Geom Only: {geom_audit[target]['matched_correct']}/{geom_audit[target]['total_samples']} ({geom_audit[target]['matched_correct']/geom_audit[target]['total_samples']*100:.1f}%)")
    for thresh in [0.55, 0.70, 0.80]:
        res = fallback_simulation[target][f"thresh_{thresh:.2f}"]
        print(f"   Thresh {thresh:.2f} -> Cứu thêm: {res['rescued_by_fallback']} | Recall mới: {res['recall_pct']}% | FP: {res['false_positive_count']} ({res['fp_rate_pct']}%)")

# PHƯƠNG ÁN 2: Controlled Geometric Tolerance
# Phân tích tác động của việc nới lỏng từng điều kiện hình học
geom_tolerance_simulation = {}

# Thử nghiệm quy tắc nới lỏng cho A:
# A hiện tại: d_TW >= 1.32, d_TM >= 0.75, d_IW <= 0.90, d_MW <= 0.75
# Thử nới: d_IW <= 1.05, d_TM >= 0.65, d_TW >= 1.20
a_relaxed_configs = [
    ("A_original", 1.32, 0.75, 0.90, 0.75),
    ("A_relax_IW_1.00", 1.32, 0.75, 1.00, 0.75),
    ("A_relax_IW_1.10", 1.32, 0.75, 1.10, 0.75),
    ("A_relax_TM_0.65", 1.32, 0.65, 0.90, 0.75),
    ("A_relax_combo_1", 1.25, 0.68, 1.05, 0.80),
]

geom_tolerance_simulation["A"] = []
c_a_idx = CLASS_TO_IDX_22["A"]
a_indices = np.where(y_all == c_a_idx)[0]
non_a_indices = np.where(y_all != c_a_idx)[0]

for name, min_tw, min_tm, max_iw, max_mw in a_relaxed_configs:
    correct = 0
    fp = 0
    for idx in a_indices:
        lm = lms_all[idx].tolist()
        feat = clf.extract_features(lm)
        if not feat: continue
        i_ext, m_ext, r_ext, p_ext = feat["index_ext"], feat["middle_ext"], feat["ring_ext"], feat["pinky_ext"]
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if feat["d_TW"] >= min_tw and feat["d_TM"] >= min_tm and feat["d_IW"] <= max_iw and feat["d_MW"] <= max_mw:
                correct += 1
                
    for idx in non_a_indices:
        lm = lms_all[idx].tolist()
        feat = clf.extract_features(lm)
        if not feat: continue
        i_ext, m_ext, r_ext, p_ext = feat["index_ext"], feat["middle_ext"], feat["ring_ext"], feat["pinky_ext"]
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if feat["d_TW"] >= min_tw and feat["d_TM"] >= min_tm and feat["d_IW"] <= max_iw and feat["d_MW"] <= max_mw:
                fp += 1
                
    geom_tolerance_simulation["A"].append({
        "config_name": name,
        "correct_a": correct,
        "recall_pct": round(correct / len(a_indices) * 100, 2),
        "false_positive": fp,
        "fp_pct": round(fp / len(non_a_indices) * 100, 2)
    })

# Thử nghiệm quy tắc nới lỏng cho D:
# D hiện tại: d_IW >= 1.70, d_TI <= 1.28, d_TM <= 0.30
d_relaxed_configs = [
    ("D_original", 1.70, 1.28, 0.30),
    ("D_relax_IW_1.55", 1.55, 1.28, 0.30),
    ("D_relax_IW_1.45", 1.45, 1.28, 0.30),
    ("D_relax_TM_0.38", 1.70, 1.28, 0.38),
    ("D_relax_combo_1", 1.50, 1.35, 0.38),
]

geom_tolerance_simulation["D"] = []
c_d_idx = CLASS_TO_IDX_22["D"]
d_indices = np.where(y_all == c_d_idx)[0]
non_d_indices = np.where(y_all != c_d_idx)[0]

for name, min_iw, max_ti, max_tm in d_relaxed_configs:
    correct = 0
    fp = 0
    for idx in d_indices:
        lm = lms_all[idx].tolist()
        feat = clf.extract_features(lm)
        if not feat: continue
        i_ext, m_ext, r_ext, p_ext = feat["index_ext"], feat["middle_ext"], feat["ring_ext"], feat["pinky_ext"]
        if i_ext and not m_ext and not r_ext and not p_ext:
            if feat["d_IW"] >= min_iw and feat["d_TI"] <= max_ti and feat["d_TM"] <= max_tm:
                correct += 1
                
    for idx in non_d_indices:
        lm = lms_all[idx].tolist()
        feat = clf.extract_features(lm)
        if not feat: continue
        i_ext, m_ext, r_ext, p_ext = feat["index_ext"], feat["middle_ext"], feat["ring_ext"], feat["pinky_ext"]
        if i_ext and not m_ext and not r_ext and not p_ext:
            if feat["d_IW"] >= min_iw and feat["d_TI"] <= max_ti and feat["d_TM"] <= max_tm:
                fp += 1
                
    geom_tolerance_simulation["D"].append({
        "config_name": name,
        "correct_d": correct,
        "recall_pct": round(correct / len(d_indices) * 100, 2),
        "false_positive": fp,
        "fp_pct": round(fp / len(non_d_indices) * 100, 2)
    })

# Thử nghiệm quy tắc nới lỏng cho E:
# E hiện tại: d_TI <= 0.35, d_TM <= 0.38, d_TW <= 0.95, d_IW >= 1.05, d_MW >= 1.05
e_relaxed_configs = [
    ("E_original", 0.35, 0.38, 0.95, 1.05, 1.05),
    ("E_relax_distances", 0.45, 0.48, 1.05, 0.95, 0.95),
    ("E_relax_combo_1", 0.42, 0.45, 1.00, 0.98, 0.98),
]

geom_tolerance_simulation["E"] = []
c_e_idx = CLASS_TO_IDX_22["E"]
e_indices = np.where(y_all == c_e_idx)[0]
non_e_indices = np.where(y_all != c_e_idx)[0]

for name, max_ti, max_tm, max_tw, min_iw, min_mw in e_relaxed_configs:
    correct = 0
    fp = 0
    for idx in e_indices:
        lm = lms_all[idx].tolist()
        feat = clf.extract_features(lm)
        if not feat: continue
        i_ext, m_ext, r_ext, p_ext = feat["index_ext"], feat["middle_ext"], feat["ring_ext"], feat["pinky_ext"]
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if feat["d_TI"] <= max_ti and feat["d_TM"] <= max_tm and feat["d_TW"] <= max_tw and feat["d_IW"] >= min_iw and feat["d_MW"] >= min_mw:
                correct += 1
                
    for idx in non_e_indices:
        lm = lms_all[idx].tolist()
        feat = clf.extract_features(lm)
        if not feat: continue
        i_ext, m_ext, r_ext, p_ext = feat["index_ext"], feat["middle_ext"], feat["ring_ext"], feat["pinky_ext"]
        if not i_ext and not m_ext and not r_ext and not p_ext:
            if feat["d_TI"] <= max_ti and feat["d_TM"] <= max_tm and feat["d_TW"] <= max_tw and feat["d_IW"] >= min_iw and feat["d_MW"] >= min_mw:
                fp += 1
                
    geom_tolerance_simulation["E"].append({
        "config_name": name,
        "correct_e": correct,
        "recall_pct": round(correct / len(e_indices) * 100, 2),
        "false_positive": fp,
        "fp_pct": round(fp / len(non_e_indices) * 100, 2)
    })

print("\n[Mô phỏng Phương án 2 - Geometric Tolerance]")
for target in target_classes:
    print(f"[{target} Geometric Tolerance Simulation]")
    for cfg in geom_tolerance_simulation[target]:
        print(f"   Config: {cfg['config_name']} -> Recall: {cfg['recall_pct']}% | FP: {cfg['false_positive']} ({cfg['fp_pct']}%)")

# -----------------------------------------------------------------------------------
# LƯU KẾT QUẢ VÀO FILE JSON VÀ CSV
# -----------------------------------------------------------------------------------
results_json_path = "backend/data/alphabet_v2/audit/task_7e5a_results.json"

final_results = {
    "metadata": {
        "task": "TASK 7E.5A",
        "description": "Read-only Audit of Static Base Letters A, E, D",
        "timestamp": "2026-10-31T17:25:00+0700",
        "checkpoint_evaluated": "alphabet_static_head_v1.pt",
        "dataset_evaluated": dataset_path
    },
    "geometric_rules_audit": geom_audit,
    "static_head_evaluation": static_head_eval,
    "leakage_audit": leakage_audit,
    "dynamic_video_replays": dynamic_replay_results,
    "design_option_1_fallback_simulation": fallback_simulation,
    "design_option_2_geometric_tolerance_simulation": geom_tolerance_simulation,
    "confusion_matrices": {
        "all_22x22": conf_matrix_22,
        "train_22x22": conf_matrix_train,
        "holdout_22x22": conf_matrix_holdout
    }
}

with open(results_json_path, "w", encoding="utf-8") as f:
    json.dump(final_results, f, indent=2, ensure_ascii=False)

print(f"\n[Xong] Kết quả kiểm toán chi tiết đã được lưu tại: {results_json_path}")

# Xuất CSV confusion matrix
csv_conf_matrix_path = "backend/data/alphabet_v2/audit/task_7e5a_confusion_matrix_22.csv"
with open(csv_conf_matrix_path, "w", encoding="utf-8") as f:
    f.write("GT_Letter," + ",".join(STATIC_CLASSES_22) + "\n")
    for gt in STATIC_CLASSES_22:
        row = [gt] + [str(conf_matrix_22[gt][pred]) for pred in STATIC_CLASSES_22]
        f.write(",".join(row) + "\n")

print(f"[Xong] File CSV Confusion Matrix lưu tại: {csv_conf_matrix_path}")

# Xuất CSV frame audit cho 4 video động
csv_frame_audit_path = "backend/data/alphabet_v2/audit/task_7e5a_dynamic_frames_audit.csv"
with open(csv_frame_audit_path, "w", encoding="utf-8") as f:
    f.write("video_target,video_id,frame_index,timestamp,phase,landmarks_valid,geom_status,geom_letter,geom_conf,head_letter,head_conf,head_top2_letter,head_top2_conf,expected_base,prod_sign,fsm_state,fsm_base,has_baseline,geom_matched_base,head_matched_base\n")
    for target, res in dynamic_replay_results.items():
        vid_id = res["video_id"]
        for row in res["timeline"]:
            f.write(f"{target},{vid_id},{row['frame_index']},{row['timestamp']:.3f},{row['phase']},{row['landmarks_valid']},"
                    f"{row['geom_status']},{row['geom_letter']},{row['geom_conf']:.2f},{row['head_letter']},{row['head_conf']:.2f},"
                    f"{row['head_top2_letter']},{row['head_top2_conf']:.2f},{row['expected_base']},{row['prod_sign']},"
                    f"{row['fsm_state']},{row['fsm_base']},{row['has_baseline']},{row['geom_matched_base']},{row['head_matched_base']}\n")

print(f"[Xong] File CSV Frame Audit lưu tại: {csv_frame_audit_path}")
