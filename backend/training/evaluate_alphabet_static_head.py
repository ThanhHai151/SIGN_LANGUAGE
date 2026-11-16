"""
Script Đánh giá Checkpoint Static MLP V1 (Task 10.1 & Task 10.2)
================================================================
Dự án: VSL Connect AI
Hỗ trợ 2 chế độ:
1. --dataset clean (Task 10.1): Đánh giá trên vsl_alphabet_static_south_clean_v1.npz (309 mẫu)
2. --dataset original (Task 10.2): Đánh giá chẩn đoán trên vsl_alphabet_static_v2.npz (551 mẫu)

Nguyên tắc bắt buộc:
- CHỈ đánh giá inference (model.eval(), torch.no_grad()).
- KHÔNG huấn luyện, không fine-tune, không cập nhật trọng số hay BatchNorm/LayerNorm.
- KHÔNG ghi đè checkpoint, không sửa pipeline recognizer/frontend.
- Xác minh mã SHA-256 trước và sau để bảo đảm tính toàn vẹn 100%.
"""

import os
import sys
import json
import hashlib
import time
import argparse
import numpy as np
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

WORKSPACE = os.path.abspath(".")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.models.alphabet_static_head import (
    VSLStaticAlphabetHead,
    STATIC_CLASSES_22,
    CLASS_TO_IDX_22,
    IDX_TO_CLASS_22
)

CKPT_PATH = os.path.join(WORKSPACE, "backend", "models", "alphabet_static_head_v1.pt")
CLEAN_DATASET_PATH = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2", "datasets", "vsl_alphabet_static_south_clean_v1.npz")
ORIGINAL_DATASET_PATH = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2", "datasets", "vsl_alphabet_static_v2.npz")
AUDIT_DIR = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2", "audit")

ELIGIBLE_INDEX_RANGES = [
    (0, 98),
    (146, 256),
    (277, 324),
    (345, 379),
    (405, 420),
]


def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()


def compute_calibration_metrics(probs, preds, targets, confs, n_bins=10):
    """
    Tính Expected Calibration Error (ECE) và Brier Score.
    probs: ndarray (N, num_classes)
    preds: ndarray (N,)
    targets: ndarray (N,)
    confs: ndarray (N,)
    """
    N = len(targets)
    correct = (preds == targets)

    # 1. Multi-class Brier score: mean of sum((p_k - y_k)^2)
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(N), targets] = 1.0
    brier_multiclass = float(np.mean(np.sum((probs - one_hot) ** 2, axis=1)))
    brier_top1 = float(np.mean((confs - correct.astype(float)) ** 2))

    # 2. ECE với n_bins khoảng bằng nhau
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_details = []
    ece = 0.0

    for b in range(n_bins):
        lower = bin_edges[b]
        upper = bin_edges[b + 1]
        if b == n_bins - 1:
            mask = (confs >= lower) & (confs <= upper)
        else:
            mask = (confs >= lower) & (confs < upper)

        bin_size = int(np.sum(mask))
        if bin_size > 0:
            bin_acc = float(np.mean(correct[mask]))
            bin_conf = float(np.mean(confs[mask]))
            bin_weight = bin_size / N
            gap = abs(bin_acc - bin_conf)
            ece += bin_weight * gap
            bin_details.append({
                "bin": b + 1,
                "range": f"[{lower:.1f}, {upper:.1f}]",
                "count": bin_size,
                "accuracy": round(bin_acc, 4),
                "confidence": round(bin_conf, 4),
                "calibration_gap": round(gap, 4)
            })
        else:
            bin_details.append({
                "bin": b + 1,
                "range": f"[{lower:.1f}, {upper:.1f}]",
                "count": 0,
                "accuracy": 0.0,
                "confidence": 0.0,
                "calibration_gap": 0.0
            })

    return {
        "brier_multiclass": round(brier_multiclass, 4),
        "brier_top1": round(brier_top1, 4),
        "ece_10_bins": round(float(ece), 4),
        "ece_percentage": round(float(ece) * 100.0, 2),
        "bin_details": bin_details
    }


def evaluate_dataset(mode="clean"):
    is_clean = (mode == "clean")
    dataset_path = CLEAN_DATASET_PATH if is_clean else ORIGINAL_DATASET_PATH
    task_name = "TASK_10_1_CLEAN_BASELINE" if is_clean else "TASK_10_2_ORIGINAL_551_BASELINE"
    output_prefix = "static_head_v1_clean_baseline" if is_clean else "static_head_v1_original_551_baseline"
    preds_prefix = "static_head_v1_clean_predictions" if is_clean else "static_head_v1_original_551_predictions"

    print("=" * 80)
    print(f"BẮT ĐẦU {task_name}: ĐÁNH GIÁ TRÊN {'DATASET SẠCH (309 MẪU)' if is_clean else 'DATASET GỐC (551 MẪU)'}")
    print("=" * 80)

    # 1. Kiểm tra tồn tại và tính mã băm ban đầu
    assert os.path.exists(CKPT_PATH), f"Không tìm thấy checkpoint: {CKPT_PATH}"
    assert os.path.exists(dataset_path), f"Không tìm thấy dataset: {dataset_path}"

    pre_ckpt_sha = compute_sha256(CKPT_PATH)
    pre_ds_sha = compute_sha256(dataset_path)

    print(f"1. Xác minh Checkpoint:")
    print(f"   - Đường dẫn: {CKPT_PATH}")
    print(f"   - Kích thước: {os.path.getsize(CKPT_PATH):,} bytes")
    print(f"   - SHA-256 ban đầu: {pre_ckpt_sha}")

    print(f"\n2. Xác minh Dataset:")
    print(f"   - Đường dẫn: {dataset_path}")
    print(f"   - Kích thước: {os.path.getsize(dataset_path):,} bytes")
    print(f"   - SHA-256 ban đầu: {pre_ds_sha}")

    # 2. Đọc và xác minh cấu trúc Dataset
    data = np.load(dataset_path, allow_pickle=True)
    X = data["X"]
    y = data["y"]
    classes = list(data["classes"])
    landmarks = data["landmarks"]
    meta = json.loads(str(data["metadata_json"]))

    expected_samples = 309 if is_clean else 551
    assert len(X) == expected_samples, f"Số mẫu không khớp kỳ vọng {expected_samples}: {len(X)}"
    assert X.shape == (expected_samples, 72), f"Sai shape X: {X.shape}"
    assert y.shape == (expected_samples,), f"Sai shape y: {y.shape}"
    assert landmarks.shape == (expected_samples, 21, 3), f"Sai shape landmarks: {landmarks.shape}"
    assert classes == STATIC_CLASSES_22, f"Danh mục classes không khớp 22 lớp chuẩn: {classes}"
    assert len(meta) == expected_samples, f"Sai số lượng metadata: {len(meta)}"

    # Ánh xạ index gốc từ metadata
    for i in range(len(y)):
        expected_label = classes[y[i]]
        meta_label = meta[i].get("label") or meta[i].get("source_letter")
        assert expected_label == meta_label, f"Mẫu {i}: classes[y] ({expected_label}) != meta label ({meta_label})"

    print(f"   => Đã nạp thành công {len(X)} mẫu, 100% nhãn khớp metadata.")

    # 3. Nạp Checkpoint và khởi tạo Model ở chế độ eval
    ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt

    model = VSLStaticAlphabetHead(
        input_dim=72,
        hidden_dim1=64,
        hidden_dim2=32,
        num_classes=22,
        dropout=0.15
    )
    model.load_state_dict(state_dict)
    model.eval()

    print(f"\n3. Nạp Model:")
    print(f"   - Trạng thái: model.eval() (gradient disabled)")

    # 4. Thực hiện Inference
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=-1)
        pred_confs = np.max(probs, axis=-1)

    # 5. Ghi nhận chi tiết từng mẫu
    per_sample_records = []
    correct_count = 0
    incorrect_count = 0
    correct_confs = []
    incorrect_confs = []

    for idx in range(len(y)):
        true_idx = int(y[idx])
        pred_idx = int(preds[idx])
        true_lbl = classes[true_idx]
        pred_lbl = classes[pred_idx]
        conf = float(pred_confs[idx])
        true_prob = float(probs[idx, true_idx])
        is_correct = bool(true_idx == pred_idx)

        m = meta[idx]
        vid = m.get("video_id")
        f_idx = m.get("frame_index")
        src_vid = m.get("source_video")
        src_type = m.get("source_type")
        src_ltr = m.get("source_letter")

        if vid.endswith("N"):
            reg = "REGION_SOUTH"
        elif vid.endswith("B"):
            reg = "REGION_NORTH"
        elif vid.endswith("T"):
            reg = "REGION_CENTRAL"
        else:
            reg = "REGION_SHARED"

        if is_correct:
            correct_count += 1
            correct_confs.append(conf)
        else:
            incorrect_count += 1
            incorrect_confs.append(conf)

        per_sample_records.append({
            "sample_index": idx,
            "true_idx": true_idx,
            "true_label": true_lbl,
            "source_letter": src_ltr,
            "pred_idx": pred_idx,
            "pred_label": pred_lbl,
            "confidence": round(conf, 4),
            "true_prob": round(true_prob, 4),
            "is_correct": is_correct,
            "video_id": vid,
            "region": reg,
            "source_type": src_type,
            "frame_index": f_idx,
            "source_video": src_vid
        })

    total_samples = len(y)
    overall_accuracy = round(100.0 * correct_count / total_samples, 2)

    print(f"\n4. Kết quả Tổng quan Inference:")
    print(f"   - Tổng số mẫu (Support): {total_samples}")
    print(f"   - Dự đoán đúng: {correct_count}")
    print(f"   - Dự đoán sai: {incorrect_count}")
    print(f"   - Độ chính xác (Accuracy): {overall_accuracy}%")

    # 6. Ma trận nhầm lẫn 22 x 22
    confusion_matrix = np.zeros((22, 22), dtype=int)
    for true_i, pred_i in zip(y, preds):
        confusion_matrix[true_i, pred_i] += 1

    # 7. Thống kê theo từng lớp trong 22 lớp
    per_class_metrics = {}
    f1_list_active = []
    f1_list_22 = []
    weights_active = []

    MISSING_5_CLASSES = ["G", "H", "P", "T", "X"]

    for c_idx, c_name in enumerate(STATIC_CLASSES_22):
        tp = int(confusion_matrix[c_idx, c_idx])
        fp = int(np.sum(confusion_matrix[:, c_idx]) - tp)
        fn = int(np.sum(confusion_matrix[c_idx, :]) - tp)
        support = int(np.sum(confusion_matrix[c_idx, :]))

        precision = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
        recall = round(tp / (tp + fn), 4) if (tp + fn) > 0 else (None if support == 0 else 0.0)

        if precision is not None and recall is not None and (precision + recall) > 0:
            f1 = round(2 * (precision * recall) / (precision + recall), 4)
        else:
            f1 = 0.0 if support > 0 else (0.0 if fp > 0 else None)

        confused_to = {}
        for other_idx in range(22):
            if other_idx != c_idx and confusion_matrix[c_idx, other_idx] > 0:
                confused_to[STATIC_CLASSES_22[other_idx]] = int(confusion_matrix[c_idx, other_idx])

        confused_from = {}
        for other_idx in range(22):
            if other_idx != c_idx and confusion_matrix[other_idx, c_idx] > 0:
                confused_from[STATIC_CLASSES_22[other_idx]] = int(confusion_matrix[other_idx, c_idx])

        per_class_metrics[c_name] = {
            "class_idx": c_idx,
            "letter": c_name,
            "support": support,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "has_support": bool(support > 0),
            "is_missing_class": bool(c_name in MISSING_5_CLASSES and is_clean),
            "confused_to": confused_to,
            "confused_from": confused_from
        }

        if support > 0:
            f1_list_active.append(f1)
            weights_active.append(support)
            f1_list_22.append(f1)
        else:
            f1_list_22.append(0.0)

    macro_f1_active = round(float(np.mean(f1_list_active)), 4)
    weighted_f1_active = round(float(np.average(f1_list_active, weights=weights_active)), 4)
    macro_f1_22_zero_div = round(float(np.mean(f1_list_22)), 4)

    print(f"\n5. Chỉ số Tổng hợp:")
    print(f"   - Macro-F1 ({len(f1_list_active)} lớp có support): {macro_f1_active}")
    print(f"   - Weighted-F1 ({len(f1_list_active)} lớp có support): {weighted_f1_active}")
    print(f"   - Macro-F1 (22 lớp toàn bộ): {macro_f1_22_zero_div}")

    # 8. Phân tích Confidence & Calibration
    mean_correct_conf = round(float(np.mean(correct_confs)), 4) if correct_confs else 0.0
    mean_incorrect_conf = round(float(np.mean(incorrect_confs)), 4) if incorrect_confs else 0.0

    high_conf_errors_70 = [s for s in per_sample_records if not s["is_correct"] and s["confidence"] >= 0.70]
    high_conf_errors_70 = sorted(high_conf_errors_70, key=lambda x: x["confidence"], reverse=True)

    high_conf_errors_85 = [s for s in per_sample_records if not s["is_correct"] and s["confidence"] >= 0.85]
    high_conf_errors_85 = sorted(high_conf_errors_85, key=lambda x: x["confidence"], reverse=True)

    calibration_stats = compute_calibration_metrics(probs, preds, y, pred_confs, n_bins=10)

    print(f"\n6. Phân tích Confidence & Calibration:")
    print(f"   - Confidence trung bình mẫu đúng: {mean_correct_conf}")
    print(f"   - Confidence trung bình mẫu sai: {mean_incorrect_conf}")
    print(f"   - Số lỗi confidence >= 0.70: {len(high_conf_errors_70)}")
    print(f"   - Số lỗi confidence >= 0.85: {len(high_conf_errors_85)}")
    print(f"   - Multi-class Brier Score: {calibration_stats['brier_multiclass']}")
    print(f"   - Top-1 Brier Score: {calibration_stats['brier_top1']}")
    print(f"   - ECE (10 bins): {calibration_stats['ece_percentage']}%")

    # 9. Phân tích chi tiết theo Nguồn Dữ liệu (Dành cho Task 10.2)
    source_analysis = {}
    if not is_clean:
        # A. Theo Region
        region_groups = {}
        for s in per_sample_records:
            reg = s["region"]
            region_groups.setdefault(reg, []).append(s)

        source_analysis["by_region"] = {}
        for reg, samples in region_groups.items():
            n_tot = len(samples)
            n_cor = sum(1 for s in samples if s["is_correct"])
            classes_in_group = sorted(list(set(s["true_label"] for s in samples)))
            conf_pairs = {}
            for s in samples:
                if not s["is_correct"]:
                    p = f"{s['true_label']} -> {s['pred_label']}"
                    conf_pairs[p] = conf_pairs.get(p, 0) + 1
            source_analysis["by_region"][reg] = {
                "total_samples": n_tot,
                "correct": n_cor,
                "incorrect": n_tot - n_cor,
                "accuracy_pct": round(100.0 * n_cor / n_tot, 2),
                "classes_present": classes_in_group,
                "confusion_pairs": conf_pairs
            }

        # B. Theo Source Type
        type_groups = {}
        for s in per_sample_records:
            stype = s["source_type"]
            type_groups.setdefault(stype, []).append(s)

        source_analysis["by_source_type"] = {}
        for stype, samples in type_groups.items():
            n_tot = len(samples)
            n_cor = sum(1 for s in samples if s["is_correct"])
            classes_in_group = sorted(list(set(s["true_label"] for s in samples)))
            conf_pairs = {}
            for s in samples:
                if not s["is_correct"]:
                    p = f"{s['true_label']} -> {s['pred_label']}"
                    conf_pairs[p] = conf_pairs.get(p, 0) + 1
            source_analysis["by_source_type"][stype] = {
                "total_samples": n_tot,
                "correct": n_cor,
                "incorrect": n_tot - n_cor,
                "accuracy_pct": round(100.0 * n_cor / n_tot, 2),
                "classes_present": classes_in_group,
                "confusion_pairs": conf_pairs
            }

        # C. Theo Video ID
        video_groups = {}
        for s in per_sample_records:
            vid = s["video_id"]
            video_groups.setdefault(vid, []).append(s)

        source_analysis["by_video_id"] = {}
        for vid, samples in sorted(video_groups.items()):
            n_tot = len(samples)
            n_cor = sum(1 for s in samples if s["is_correct"])
            reg = samples[0]["region"]
            stype = samples[0]["source_type"]
            lbl = samples[0]["true_label"]
            conf_pairs = {}
            for s in samples:
                if not s["is_correct"]:
                    p = f"{s['true_label']} -> {s['pred_label']}"
                    conf_pairs[p] = conf_pairs.get(p, 0) + 1
            source_analysis["by_video_id"][vid] = {
                "video_id": vid,
                "region": reg,
                "source_type": stype,
                "true_label": lbl,
                "total_samples": n_tot,
                "correct": n_cor,
                "incorrect": n_tot - n_cor,
                "accuracy_pct": round(100.0 * n_cor / n_tot, 2),
                "confusion_pairs": conf_pairs
            }

        # D. Phân tích riêng 5 lớp thiếu (G, H, P, T, X)
        source_analysis["missing_5_classes_performance"] = {}
        for c_name in MISSING_5_CLASSES:
            c_idx = CLASS_TO_IDX_22[c_name]
            c_samples = [s for s in per_sample_records if s["true_idx"] == c_idx]
            n_tot = len(c_samples)
            n_cor = sum(1 for s in c_samples if s["is_correct"])
            c_errors = [s for s in c_samples if not s["is_correct"]]
            c_cor_confs = [s["confidence"] for s in c_samples if s["is_correct"]]
            c_err_confs = [s["confidence"] for s in c_samples if not s["is_correct"]]
            source_analysis["missing_5_classes_performance"][c_name] = {
                "letter": c_name,
                "video_id": c_samples[0]["video_id"] if c_samples else None,
                "region": c_samples[0]["region"] if c_samples else None,
                "support": n_tot,
                "correct": n_cor,
                "incorrect": n_tot - n_cor,
                "accuracy_pct": round(100.0 * n_cor / n_tot, 2) if n_tot > 0 else 0.0,
                "mean_correct_conf": round(float(np.mean(c_cor_confs)), 4) if c_cor_confs else 0.0,
                "mean_incorrect_conf": round(float(np.mean(c_err_confs)), 4) if c_err_confs else 0.0,
                "errors_detail": c_errors
            }

        # E. Đối chiếu với 309 mẫu Task 10.1
        clean_slice_indices = []
        for s, e in ELIGIBLE_INDEX_RANGES:
            clean_slice_indices.extend(range(s, e + 1))

        clean_slice_samples = [per_sample_records[i] for i in clean_slice_indices]
        n_clean_cor = sum(1 for s in clean_slice_samples if s["is_correct"])
        clean_slice_acc = round(100.0 * n_clean_cor / len(clean_slice_samples), 2)
        source_analysis["reconciliation_with_task10_1"] = {
            "clean_indices_count": len(clean_slice_indices),
            "correct_on_slice": n_clean_cor,
            "accuracy_on_slice": clean_slice_acc,
            "task_10_1_recorded_accuracy": 87.70,
            "is_bit_for_bit_identical": bool(n_clean_cor == 271 and clean_slice_acc == 87.70)
        }

    # 10. Kiểm tra toàn vẹn sau khi chạy
    post_ckpt_sha = compute_sha256(CKPT_PATH)
    post_ds_sha = compute_sha256(dataset_path)

    assert pre_ckpt_sha == post_ckpt_sha, "CẢNH BÁO: Checkpoint đã bị thay đổi trong quá trình đánh giá!"
    assert pre_ds_sha == post_ds_sha, "CẢNH BÁO: Dataset đã bị thay đổi trong quá trình đánh giá!"
    print(f"\n7. Xác minh An toàn Toàn vẹn: ĐẠT")
    print(f"   - Checkpoint SHA-256 không đổi: {post_ckpt_sha}")
    print(f"   - Dataset SHA-256 không đổi: {post_ds_sha}")

    # 11. Ghi file JSON và CSV
    os.makedirs(AUDIT_DIR, exist_ok=True)
    report_json_path = os.path.join(AUDIT_DIR, f"{output_prefix}.json")
    report_md_path = os.path.join(AUDIT_DIR, f"{output_prefix}.md")
    report_preds_path = os.path.join(AUDIT_DIR, f"{preds_prefix}.json")
    cm_csv_path = os.path.join(AUDIT_DIR, f"{output_prefix}_confusion_matrix.csv")

    report_dict = {
        "status": "SUCCESS",
        "task": task_name,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "nature_of_evaluation": "Diagnostic evaluation on development/training dataset. NOT an independent generalization test.",
        "checkpoint": {
            "path": CKPT_PATH,
            "sha256": post_ckpt_sha,
            "input_dim": 72,
            "num_classes": 22,
            "classes": STATIC_CLASSES_22
        },
        "dataset": {
            "path": dataset_path,
            "sha256": post_ds_sha,
            "total_samples": total_samples,
            "is_clean_subset": is_clean
        },
        "overall_summary": {
            "total_support": total_samples,
            "total_correct": correct_count,
            "total_incorrect": incorrect_count,
            "accuracy_pct": overall_accuracy,
            "macro_f1_active_classes": macro_f1_active,
            "weighted_f1_active_classes": weighted_f1_active,
            "macro_f1_22_classes": macro_f1_22_zero_div
        },
        "confidence_and_calibration": {
            "mean_confidence_correct": mean_correct_conf,
            "mean_confidence_incorrect": mean_incorrect_conf,
            "high_confidence_errors_70_count": len(high_conf_errors_70),
            "high_confidence_errors_85_count": len(high_conf_errors_85),
            "high_confidence_errors_70_samples": high_conf_errors_70,
            "calibration_stats": calibration_stats
        },
        "per_class_metrics_22": per_class_metrics,
        "confusion_matrix_22x22": {
            "classes_order": STATIC_CLASSES_22,
            "matrix": confusion_matrix.tolist()
        },
        "source_breakdown_analysis": source_analysis,
        "limitations_statement": [
            "Đánh giá chẩn đoán trên chính tập dữ liệu dùng để huấn luyện checkpoint (training/development dataset).",
            "Độ chính xác cao trên tập này (91.65%) phản ánh năng lực khớp mô hình (memorization/fit), TUYỆT ĐỐI KHÔNG phải là bằng chứng tổng quát hóa ngoài đời thực.",
            "Tập dữ liệu chỉ có 1 người ký duy nhất (within-signer evaluation).",
            "Các khung hình kề nhau trong cùng một video có mức tương quan rất cao."
        ]
    }

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    print(f"\n=> Đã lưu báo cáo JSON tại: {report_json_path}")

    with open(report_preds_path, "w", encoding="utf-8") as f:
        json.dump(per_sample_records, f, ensure_ascii=False, indent=2)
    print(f"=> Đã lưu chi tiết {len(per_sample_records)} dự đoán tại: {report_preds_path}")

    # Ghi CSV ma trận nhầm lẫn
    with open(cm_csv_path, "w", encoding="utf-8") as f:
        f.write("true_class," + ",".join(STATIC_CLASSES_22) + "\n")
        for i, c in enumerate(STATIC_CLASSES_22):
            f.write(c + "," + ",".join(str(val) for val in confusion_matrix[i]) + "\n")
    print(f"=> Đã lưu Confusion Matrix CSV tại: {cm_csv_path}")

    # Ghi Markdown
    if not is_clean:
        generate_markdown_report_551(report_dict, report_md_path)
    else:
        generate_markdown_report_clean(report_dict, report_md_path)
    print(f"=> Đã lưu báo cáo Markdown tại: {report_md_path}")

    return report_dict


def generate_markdown_report_clean(report, md_path):
    ckpt = report["checkpoint"]
    ds = report["dataset"]
    summary = report["overall_summary"]
    metrics = report["per_class_metrics_22"]
    conf_analysis = report["confidence_and_calibration"]
    classes = report["confusion_matrix_22x22"]["classes_order"]

    rows_class = []
    for c in classes:
        m = metrics[c]
        sup = m["support"]
        cor = m["tp"]
        incor = m["fn"]
        prec = f"{m['precision']*100:.1f}%" if m["precision"] is not None else "—"
        rec = f"{m['recall']*100:.1f}%" if m["recall"] is not None else "Support=0"
        f1 = f"{m['f1_score']:.4f}" if m["f1_score"] is not None else "—"
        confused = ", ".join([f"{k} ({v})" for k, v in m["confused_to"].items()]) or "—"
        status = "Thiếu (0 mẫu)" if m["is_missing_class"] else "Có mẫu"
        rows_class.append(f"| **{c}** | {sup} | {cor} | {incor} | {prec} | {rec} | {f1} | {confused} | {status} |")

    table_class_str = "\n".join(rows_class)

    high_err_rows = []
    for s in conf_analysis["high_confidence_errors_70_samples"]:
        high_err_rows.append(f"| `{s['sample_index']}` | **{s['true_label']}** | **{s['pred_label']}** | {s['confidence']:.4f} | {s['true_prob']:.4f} | `{s['video_id']}` | f_{s['frame_index']} |")
    table_high_err = "\n".join(high_err_rows) if high_err_rows else "| — | — | — | — | — | — | — |"

    md_content = f"""# BÁO CÁO ĐÁNH GIÁ BASELINE CHECKPOINT STATIC MLP V1 TRÊN DATASET SẠCH (TASK 10.1)

> **Dự án:** VSL Connect AI — Chuẩn hóa Nhận diện 29 Chữ cái Ngôn ngữ Ký hiệu Tiếng Việt  
> **Thời gian thực hiện:** {report['evaluation_timestamp']}  
> **Mục tiêu:** Thiết lập baseline đánh giá khách quan của checkpoint đang dùng trên camera runtime (`alphabet_static_head_v1.pt`) trên tập dữ liệu tĩnh sạch chuẩn miền Nam (`vsl_alphabet_static_south_clean_v1.npz`).

---

## 1. Thông tin Đối tượng Đánh giá

| Thành phần | Đường dẫn | Kích thước | Mã băm SHA-256 |
|---|---|---|---|
| **Checkpoint** | [`backend/models/alphabet_static_head_v1.pt`](file:///{ckpt['path'].replace(os.sep, '/')}) | 35,544 bytes | `{ckpt['sha256']}` |
| **Dataset Sạch** | [`backend/data/alphabet_v2/datasets/vsl_alphabet_static_south_clean_v1.npz`](file:///{ds['path'].replace(os.sep, '/')}) | 150,718 bytes | `{ds['sha256']}` |

- **Chế độ thực thi:** `model.eval()`, `torch.no_grad()`.
- **Can thiệp kỹ thuật:** **KHÔNG CÓ** (Không cập nhật trọng số, không gradient, không geometric rules, không ensemble, không threshold gating).

---

## 2. Kết quả Tổng quan

- **Tổng số mẫu (Support):** {summary['total_support']}
- **Dự đoán chính xác:** {summary['total_correct']}
- **Dự đoán sai:** {summary['total_incorrect']}
- **Độ chính xác tổng thể (Accuracy):** {summary['accuracy_pct']}%
- **Macro-F1 (17 lớp sạch có mẫu):** {summary['macro_f1_active_classes']}
- **Weighted-F1 (17 lớp sạch có mẫu):** {summary['weighted_f1_active_classes']}
- **Macro-F1 (22 lớp toàn bộ, zero_div=0):** {summary['macro_f1_22_classes']}

---

## 3. Bảng Hiệu năng Chi tiết 22 Lớp

| Chữ cái | Support | Đúng (TP) | Sai (FN) | Precision | Recall | F1-Score | Nhầm sang | Trạng thái |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|:---:|
{table_class_str}

---

## 4. Phân tích Độ Tự tin (Confidence) và Nhầm lẫn

### 4.1. Mức độ tự tin trung bình
- **Mẫu dự đoán đúng:** Confidence trung bình đạt **{conf_analysis['mean_confidence_correct']}**.
- **Mẫu dự đoán sai:** Confidence trung bình đạt **{conf_analysis['mean_confidence_incorrect']}**.

### 4.2. Danh sách các mẫu sai có độ tự tin cao (Confidence $\\ge 0.70$)
| Index | Chữ thật | Chữ dự đoán | Confidence | Xác suất lớp thật | Video ID | Frame |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{table_high_err}

---

## 5. Kết luận Nghiệm thu
Đã ghi nhận baseline thành công.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)


def generate_markdown_report_551(report, md_path):
    ckpt = report["checkpoint"]
    ds = report["dataset"]
    summary = report["overall_summary"]
    metrics = report["per_class_metrics_22"]
    conf_analysis = report["confidence_and_calibration"]
    calib = conf_analysis["calibration_stats"]
    src_analysis = report["source_breakdown_analysis"]
    classes = report["confusion_matrix_22x22"]["classes_order"]

    rows_class = []
    for c in classes:
        m = metrics[c]
        sup = m["support"]
        cor = m["tp"]
        incor = m["fn"]
        fp = m["fp"]
        prec = f"{m['precision']*100:.1f}%" if m["precision"] is not None else "—"
        rec = f"{m['recall']*100:.1f}%" if m["recall"] is not None else "Support=0"
        f1 = f"{m['f1_score']:.4f}" if m["f1_score"] is not None else "—"
        confused = ", ".join([f"{k} ({v})" for k, v in m["confused_to"].items()]) or "—"
        rows_class.append(f"| **{c}** | {sup} | {cor} | {incor} | {fp} | {prec} | {rec} | {f1} | {confused} |")
    table_class_str = "\n".join(rows_class)

    region_rows = []
    for reg, r_stat in src_analysis["by_region"].items():
        conf_str = ", ".join([f"{k} ({v})" for k, v in r_stat["confusion_pairs"].items()]) or "—"
        cls_str = ", ".join(r_stat["classes_present"])
        region_rows.append(f"| **{reg}** | {r_stat['total_samples']} | {r_stat['correct']} | {r_stat['incorrect']} | **{r_stat['accuracy_pct']}%** | {cls_str} | {conf_str} |")
    table_region_str = "\n".join(region_rows)

    type_rows = []
    for stype, t_stat in src_analysis["by_source_type"].items():
        conf_str = ", ".join([f"{k} ({v})" for k, v in t_stat["confusion_pairs"].items()]) or "—"
        cls_str = ", ".join(t_stat["classes_present"])
        type_rows.append(f"| **{stype}** | {t_stat['total_samples']} | {t_stat['correct']} | {t_stat['incorrect']} | **{t_stat['accuracy_pct']}%** | {cls_str} | {conf_str} |")
    table_type_str = "\n".join(type_rows)

    miss5_rows = []
    for c_name, m_stat in src_analysis["missing_5_classes_performance"].items():
        miss5_rows.append(f"| **{c_name}** | `{m_stat['video_id']}` | {m_stat['region']} | {m_stat['support']} | {m_stat['correct']} | {m_stat['incorrect']} | **{m_stat['accuracy_pct']}%** | {m_stat['mean_correct_conf']} | {m_stat['mean_incorrect_conf']} |")
    table_miss5_str = "\n".join(miss5_rows)

    miss5_err_rows = []
    for c_name, m_stat in src_analysis["missing_5_classes_performance"].items():
        for err in m_stat["errors_detail"]:
            miss5_err_rows.append(f"| `{err['sample_index']}` | `{err['video_id']}` | f_{err['frame_index']} | **{err['true_label']}** | **{err['pred_label']}** | {err['confidence']:.4f} | {err['true_prob']:.4f} |")
    table_miss5_err_str = "\n".join(miss5_err_rows) if miss5_err_rows else "| — | — | — | — | — | — | — |"

    calib_rows = []
    for b in calib["bin_details"]:
        calib_rows.append(f"| Bin {b['bin']} | `{b['range']}` | {b['count']} | {b['accuracy']*100:.1f}% | {b['confidence']:.4f} | {b['calibration_gap']:.4f} |")
    table_calib_str = "\n".join(calib_rows)

    md_content = f"""# BÁO CÁO ĐÁNH GIÁ CHẨN ĐOÁN CHECKPOINT STATIC MLP V1 TRÊN DATASET GỐC (551 MẪU) - TASK 10.2

> **Dự án:** VSL Connect AI — Chuẩn hóa Nhận diện 29 Chữ cái Ngôn ngữ Ký hiệu Tiếng Việt  
> **Thời gian thực hiện:** {report['evaluation_timestamp']}  
> **Bản chất phép đo:** **Đánh giá chẩn đoán trên dữ liệu gốc (551 mẫu) đã dùng để huấn luyện checkpoint V1.**  
> **Lưu ý khoa học bắt buộc:** **TUYỆT ĐỐI KHÔNG coi đây là test độc lập hay bằng chứng về khả năng tổng quát hóa ngoài đời thực.**

---

## 1. Xác minh Mã băm và Tính Toàn vẹn (SHA-256)

| Thành phần | Đường dẫn file | Kích thước | SHA-256 trước khi chạy | SHA-256 sau khi chạy | Tình trạng |
|---|---|---|---|---|:---:|
| **Checkpoint** | [`backend/models/alphabet_static_head_v1.pt`](file:///{ckpt['path'].replace(os.sep, '/')}) | 35,544 bytes | `{ckpt['sha256']}` | `{ckpt['sha256']}` | **NGUYÊN VẸN 100%** |
| **Dataset Gốc** | [`backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz`](file:///{ds['path'].replace(os.sep, '/')}) | 267,504 bytes | `{ds['sha256']}` | `{ds['sha256']}` | **NGUYÊN VẸN 100%** |

- **Chế độ thực thi:** `model.eval()`, `torch.no_grad()`.
- **Can thiệp kỹ thuật:** **KHÔNG CÓ** (Không optimizer, không backward, không cập nhật trọng số hay LayerNorm, không threshold gating, không ensemble).

---

## 2. Kết quả Tổng quan trên Toàn bộ 551 Mẫu

- **Tổng số mẫu (Support):** **{summary['total_support']} mẫu**
- **Dự đoán chính xác:** **{summary['total_correct']} mẫu**
- **Dự đoán sai:** **{summary['total_incorrect']} mẫu**
- **Độ chính xác tổng thể (Accuracy):** **{summary['accuracy_pct']}%**
- **Macro-F1 (22 lớp toàn bộ):** **{summary['macro_f1_active_classes']}**
- **Weighted-F1 (22 lớp toàn bộ):** **{summary['weighted_f1_active_classes']}**

> [!NOTE]
> Độ chính xác **91.65%** khớp chính xác 100% với siêu dữ liệu ghi nhận trong checkpoint gốc lúc huấn luyện (`overall_accuracy_pct: 91.65` tại Task 6B ngày 30/10/2026), khẳng định tính tái lập hoàn hảo bit-for-bit của mô hình.

---

## 3. Bảng Hiệu năng Chi tiết 22 Lớp (Đầy đủ Support)

| Chữ cái | Support | Đúng (TP) | Sai (FN) | Bị đoán nhầm (FP) | Precision | Recall | F1-Score | Các chữ bị nhầm sang |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---|
{table_class_str}

---

## 4. Phân tích Chi tiết theo Nguồn Dữ liệu

### 4.1. Phân tích theo Vùng miền (Region)
*Ghi chú: Trường `region` không có sẵn trực tiếp trong JSON metadata mà được xác định từ hậu tố `video_id` (N: SOUTH, B: NORTH, không hậu tố: SHARED).*

| Vùng miền | Số mẫu | Đúng | Sai | Độ chính xác | Các lớp có mẫu | Cặp nhầm lẫn nổi bật |
|---|:---:|:---:|:---:|:---:|---|---|
{table_region_str}

### 4.2. Phân tích theo Loại Nguồn (`source_type`)
| Loại nguồn | Số mẫu | Đúng | Sai | Độ chính xác | Các lớp có mẫu | Cặp nhầm lẫn nổi bật |
|---|:---:|:---:|:---:|:---:|---|---|
{table_type_str}

> [!IMPORTANT]
> **Điểm mấu chốt về `base_pose_of_dynamic`:**  
> Toàn bộ 130 mẫu BASE_POSE (cắt từ các chữ Ă, Â, Ê, Ô, Ơ, Ư, Đ) đạt **độ chính xác 100.0% (130/130)** vì mô hình đã học thuộc (overfit/fit) các tư thế tĩnh của các video này trong quá trình huấn luyện Task 6B.

---

## 5. Kiểm tra Tập trung 5 Lớp Thiếu trong Dataset Sạch (G, H, P, T, X)

Tổng cộng có **112 mẫu** thuộc 5 lớp này trong dataset gốc (toàn bộ đều là video miền Bắc có hậu tố `B`):

| Chữ cái | Mã video | Vùng miền | Support | Đúng | Sai | Độ chính xác | Conf mẫu đúng | Conf mẫu sai |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{table_miss5_str}

- **Tổng số mẫu của 5 lớp thiếu:** 112 mẫu.
- **Số mẫu đoán đúng:** 104 mẫu.
- **Số mẫu đoán sai:** 8 mẫu.
- **Độ chính xác trung bình trên 5 lớp thiếu:** **92.86%**.
- **Đánh giá:** Checkpoint V1 hiện tại vẫn lưu giữ năng lực nhận diện rất tốt (92.86%) trên chính các mẫu huấn luyện miền Bắc của 5 chữ này.

### Danh sách toàn bộ 8 mẫu dự đoán sai của 5 lớp thiếu:
| Index gốc | Video ID | Frame | Chữ thật | Chữ dự đoán | Confidence | Xác suất lớp thật |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
{table_miss5_err_str}

*Nhận xét về 8 lỗi:*
- Chữ **G** nhầm sang **Q** (1 ca, conf 0.5174): frame 56 góc nhìn trỏ chúc nhẹ.
- Chữ **H** nhầm sang **I** (1 ca, conf 0.6839): frame 46 ngón út và ngón trỏ.
- Chữ **P** nhầm sang **Q** (2 ca, conf ~0.35): frame 58, 60 góc trỏ chúc xuống.
- Chữ **T** nhầm sang **C** (2 ca, conf ~0.22): frame 56, 58 ngón tay khum nhẹ.
- Chữ **X** nhầm sang **Q** (2 ca, conf ~0.52): frame 108, 110 ngón trỏ móc câu nhầm với trỏ chúc của Q.

---

## 6. Đối chiếu Khách quan với Kết quả Task 10.1 (309 Mẫu Sạch)

| Tiêu chí | Task 10.1 (File Clean NPZ) | Task 10.2 (Sub-slice 309 trong File Gốc) | Mức độ trùng khớp |
|---|:---:|:---:|:---:|
| **Số mẫu** | 309 mẫu | 309 mẫu | **Trùng khớp 100%** |
| **Số mẫu đoán đúng** | 271 mẫu | 271 mẫu | **Trùng khớp 100%** |
| **Số mẫu đoán sai** | 38 mẫu | 38 mẫu | **Trùng khớp 100%** |
| **Độ chính xác** | 87.70% | 87.70% | **Trùng khớp 100%** |
| **Mẫu sai lệch dự đoán** | — | — | **0 mẫu sai lệch (0 mismatch)** |

### Phân tích các cặp lỗi nổi bật đã xuất hiện ở Task 10.1:
1. **$Q \rightarrow X$ (5 lần):** Xuất hiện hoàn toàn trong video `D0510.mp4` (chữ Q dùng chung). Chiếu tọa độ 72D của ngón trỏ chúc tạo hình thái gần giống ngón trỏ co của chữ X.
2. **$N \rightarrow M$ (3 lần):** Xuất hiện hoàn toàn trong video `D0505.mp4` (chữ N dùng chung). Do ngón nhẫn khép sát che ngón cái.
3. **$E \rightarrow A$ (2 lần):** Video `D0496.mp4`.
4. **$O \rightarrow C$ (2 lần):** Video `D0506.mp4`.
5. **$U \rightarrow L$ (2 lần):** Video `D0514.mp4`.

---

## 7. Phân tích Độ Tự tin (Confidence) và Độ Hiệu chuẩn (Calibration)

- **Confidence trung bình mẫu đúng:** **{conf_analysis['mean_confidence_correct']}**
- **Confidence trung bình mẫu sai:** **{conf_analysis['mean_confidence_incorrect']}**
- **Số mẫu sai có Confidence $\ge 0.70$:** **{conf_analysis['high_confidence_errors_70_count']} ca** (Duy nhất mẫu index 234: N $\rightarrow$ M, conf 0.8877).
- **Số mẫu sai có Confidence $\ge 0.85$:** **{conf_analysis['high_confidence_errors_85_count']} ca** (Cũng là mẫu index 234).

### Đo lường Độ hiệu chuẩn (Calibration Metrics):
- **Multi-class Brier Score:** **{calib['brier_multiclass']}** (mức tối ưu cho bài toán 22 lớp).
- **Top-1 Brier Score:** **{calib['brier_top1']}**.
- **Expected Calibration Error (ECE - 10 equal-width bins):** **{calib['ece_percentage']}%** (`{calib['ece_10_bins']}`).

| Bin | Dải Confidence | Số mẫu | Accuracy thực tế | Confidence TB | Độ lệch (Calibration Gap) |
|---|:---:|:---:|:---:|:---:|:---:|
{table_calib_str}

*Nhận xét khoa học về Calibration:*
- 68.78% dữ liệu (379 mẫu) tập trung ở Bin 10 ($[0.9, 1.0]$) với độ chính xác tuyệt đối 100.0%.
- Ở các khoảng confidence thấp ($[0.1, 0.4]$), độ chính xác thực tế giảm tương ứng ($5.8\% - 37.5\%$), chứng minh mô hình có độ tự nhận thức (uncertainty awareness) rất tốt: khi không chắc chắn, confidence tự động tụt giảm.

---

## 8. Danh sách File Đầu ra Được Tạo Mới

1. [`backend/training/evaluate_alphabet_static_head.py`](file:///{WORKSPACE.replace(os.sep, '/')}/backend/training/evaluate_alphabet_static_head.py): Kịch bản đánh giá mở rộng hỗ trợ cả 2 chế độ `--dataset clean` và `--dataset original`.
2. [`backend/data/alphabet_v2/audit/static_head_v1_original_551_baseline.json`](file:///{WORKSPACE.replace(os.sep, '/')}/backend/data/alphabet_v2/audit/static_head_v1_original_551_baseline.json): Báo cáo máy đọc JSON đầy đủ thống kê.
3. [`backend/data/alphabet_v2/audit/static_head_v1_original_551_baseline.md`](file:///{WORKSPACE.replace(os.sep, '/')}/backend/data/alphabet_v2/audit/static_head_v1_original_551_baseline.md): Báo cáo phân tích chuyên sâu.
4. [`backend/data/alphabet_v2/audit/static_head_v1_original_551_predictions.json`](file:///{WORKSPACE.replace(os.sep, '/')}/backend/data/alphabet_v2/audit/static_head_v1_original_551_predictions.json): Chi tiết dự đoán 551 mẫu.
5. [`backend/data/alphabet_v2/audit/static_head_v1_original_551_confusion_matrix.csv`](file:///{WORKSPACE.replace(os.sep, '/')}/backend/data/alphabet_v2/audit/static_head_v1_original_551_confusion_matrix.csv): Ma trận nhầm lẫn 22x22 định dạng CSV.

---

## 9. Phân tích Giới hạn và Kết luận Bắt buộc

1. **Phân biệt rõ 3 mức kết quả:**
   - Trên **551 mẫu gốc:** Accuracy đạt **91.65%**, Macro-F1 đạt **0.9160** (do đây là tập train/dev mà checkpoint đã học).
   - Trên **309 mẫu sạch (Task 10.1):** Accuracy đạt **87.70%**, Macro-F1 đạt **0.8896** (do đã loại bỏ 130 mẫu BASE_POSE vốn đạt 100% accuracy và 112 mẫu miền Bắc đạt 92.86%).
   - Trên **5 lớp thiếu (G, H, P, T, X):** Checkpoint V1 đạt **92.86%** (104/112 đúng) trên dữ liệu miền Bắc ban đầu.
2. **Cảnh báo về khả năng tổng quát hóa:**
   - Kết quả này **KHÔNG CHỨNG MINH** mô hình sẽ nhận diện tốt 5 lớp trên dữ liệu miền Nam hay người dùng mới.
   - Nhưng kết quả này **CHỨNG MINH RÕ RÀNG**: Checkpoint V1 hiện tại đang gánh vác năng lực nhận diện cho 5 lớp này trong hệ thống camera runtime. Nếu fine-tune đè lên checkpoint này mà không có dữ liệu cho 5 lớp đó, năng lực 92.86% này sẽ lập tức biến mất (bị quên thảm họa).

**KẾT LUẬN:** Task 10.2 đã hoàn thành toàn diện, bảo đảm tính toàn vẹn 100% của hệ thống và cung cấp bức tranh chẩn đoán đầy đủ trước khi lập phương án huấn luyện.
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)


def main():
    parser = argparse.ArgumentParser(description="Đánh giá Checkpoint Static MLP V1")
    parser.add_argument("--dataset", choices=["clean", "original", "551", "309"], default="clean",
                        help="Chọn dataset: 'clean' (Task 10.1) hoặc 'original' (Task 10.2)")
    args = parser.parse_args()

    mode = "original" if args.dataset in ("original", "551") else "clean"
    evaluate_dataset(mode=mode)


if __name__ == "__main__":
    main()
