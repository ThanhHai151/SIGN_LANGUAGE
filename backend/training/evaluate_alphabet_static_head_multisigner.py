"""
Công cụ Đánh giá Độc lập Đa Người Ký cho VSLStaticAlphabetHead (Task 6B.2)
========================================================================
Dự án: VSL Connect AI
Mô-đun kiểm thử: backend/models/alphabet_static_head_v1.pt (22 lớp tư thế nền)

Nguyên tắc kỹ thuật:
- CHỈ ĐỌC (Read-Only): Tuyệt đối không gọi optimizer, không sửa trọng số checkpoint.
- Tách biệt hoàn toàn theo Signer (Signer-Independent Evaluation).
- Đánh giá đa chiều: Accuracy, Precision, Recall, F1, Confusion Matrix 22x22,
  Độ tin cậy (Confidence) của dự đoán đúng vs sai, Tỷ lệ từ chối (Rejection Rate).
- Phân tích sâu các cặp dễ xung đột: Q vs X, M vs N, R vs U, G vs Q.
"""

import os
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import json
import time
import argparse
import hashlib
import numpy as np
import torch

WORKSPACE = os.path.abspath(".")
if WORKSPACE not in sys.path:
    sys.path.insert(0, WORKSPACE)

from backend.models.alphabet_static_head import (
    VSLStaticAlphabetHead,
    STATIC_CLASSES_22,
    CLASS_TO_IDX_22,
    IDX_TO_CLASS_22,
    DEFAULT_HEAD_WEIGHTS_PATH
)
from backend.vision.feature_extractor import normalize_hand_landmarks

MISSING_10_LETTERS = ["G", "H", "K", "M", "N", "P", "Q", "R", "U", "X"]
RULE_12_LETTERS = ["A", "B", "C", "D", "E", "I", "L", "O", "S", "T", "V", "Y"]
COLLISION_PAIRS = [("Q", "X"), ("M", "N"), ("R", "U"), ("G", "Q")]

def compute_sha256(filepath):
    if not os.path.exists(filepath):
        return "NOT_FOUND"
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(8192):
            h.update(chunk)
    return h.hexdigest()

def verify_pipeline_integrity(weights_path: str) -> dict:
    """Kiểm tra tính toàn vẹn của mô hình và đường ống trước khi đánh giá"""
    assert os.path.exists(weights_path), f"Không tìm thấy checkpoint tại: {weights_path}"
    ckpt = torch.load(weights_path, map_location="cpu")
    
    sha_val = compute_sha256(weights_path)
    classes = ckpt.get("classes", [])
    input_dim = ckpt.get("input_dim", 0)
    num_classes = ckpt.get("num_classes", 0)
    
    is_valid = (
        classes == STATIC_CLASSES_22 and
        input_dim == 72 and
        num_classes == 22 and
        "model_state_dict" in ckpt
    )
    
    return {
        "is_valid": is_valid,
        "weights_path": weights_path,
        "sha256": sha_val,
        "input_dim": input_dim,
        "num_classes": num_classes,
        "classes_match": (classes == STATIC_CLASSES_22)
    }

def load_multisigner_dataset(dataset_path: str):
    """
    Tải tập dữ liệu đa người ký từ file .npz hoặc .json.
    Trả về: X (N, 72), y (N,), metadata_samples (list of dicts)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Tệp dữ liệu không tồn tại: {dataset_path}")
        
    if dataset_path.endswith(".npz"):
        data = np.load(dataset_path, allow_pickle=True)
        X = data["X"].astype(np.float32)
        y = data["y"].astype(np.int64)
        metadata = []
        if "metadata_json" in data:
            raw_meta = str(data["metadata_json"])
            metadata = json.loads(raw_meta)
        elif "metadata" in data:
            metadata = list(data["metadata"])
        return X, y, metadata

    elif dataset_path.endswith(".json"):
        with open(dataset_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        samples = obj.get("samples", obj) if isinstance(obj, dict) else obj
        
        X_list = []
        y_list = []
        meta_list = []
        for s in samples:
            label = s.get("letter_label") or s.get("label")
            if label not in CLASS_TO_IDX_22:
                continue
            feat = s.get("feature_vector")
            if feat is None and "raw_landmarks" in s:
                feat, _ = normalize_hand_landmarks(s["raw_landmarks"])
            if feat is None or len(feat) < 72:
                continue
            X_list.append(feat[:72])
            y_list.append(CLASS_TO_IDX_22[label])
            meta_list.append(s)
            
        return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.int64), meta_list
    else:
        raise ValueError(f"Định dạng tệp không được hỗ trợ: {dataset_path} (Yêu cầu .npz hoặc .json)")

def evaluate_multisigner_data(model: VSLStaticAlphabetHead, X: np.ndarray, y: np.ndarray, metadata: list, thresholds=[0.50, 0.55, 0.60, 0.70, 0.80]) -> dict:
    """
    Thực hiện đánh giá chỉ đọc trên tập dữ liệu đa người ký.
    """
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32)
        logits = model(X_tensor)
        probs = torch.softmax(logits, dim=-1).numpy()
        preds = np.argmax(probs, axis=-1)
        confs = np.max(probs, axis=-1)

    total_samples = len(y)
    overall_correct = int(np.sum(preds == y))
    overall_accuracy = round(100.0 * overall_correct / max(1, total_samples), 2)

    # 1. Ma trận nhầm lẫn 22 x 22 tổng thể
    cm_overall = np.zeros((22, 22), dtype=int)
    for t, p in zip(y, preds):
        cm_overall[t, p] += 1

    # 2. Đánh giá theo từng lớp
    per_class_metrics = {}
    for idx, ch in enumerate(STATIC_CLASSES_22):
        n_true = int(np.sum(y == idx))
        tp = int(cm_overall[idx, idx])
        fn = int(np.sum(cm_overall[idx, :]) - tp)
        fp = int(np.sum(cm_overall[:, idx]) - tp)

        recall = round(100.0 * tp / n_true, 2) if n_true > 0 else 0.0
        prec = round(100.0 * tp / (tp + fp), 2) if (tp + fp) > 0 else 0.0
        f1 = round(2 * prec * recall / (prec + recall), 2) if (prec + recall) > 0 else 0.0

        mask_c = (y == idx)
        confs_c = confs[mask_c] if np.any(mask_c) else np.array([])
        mean_conf = round(float(np.mean(confs_c)), 4) if len(confs_c) > 0 else 0.0

        correct_confs = [float(confs[i]) for i in range(total_samples) if y[i] == idx and preds[i] == idx]
        wrong_confs = [float(confs[i]) for i in range(total_samples) if y[i] == idx and preds[i] != idx]

        confused_to = {}
        for p_idx in range(22):
            if p_idx != idx and cm_overall[idx, p_idx] > 0:
                confused_to[STATIC_CLASSES_22[p_idx]] = int(cm_overall[idx, p_idx])

        per_class_metrics[ch] = {
            "total_samples": n_true,
            "tp": tp,
            "fn": fn,
            "fp": fp,
            "recall_pct": recall,
            "precision_pct": prec,
            "f1_pct": f1,
            "mean_confidence": mean_conf,
            "mean_correct_confidence": round(float(np.mean(correct_confs)), 4) if correct_confs else None,
            "mean_wrong_confidence": round(float(np.mean(wrong_confs)), 4) if wrong_confs else None,
            "confused_to": confused_to,
            "is_missing_10": ch in MISSING_10_LETTERS,
            "is_rule_12": ch in RULE_12_LETTERS
        }

    # 3. Phân tách theo từng Signer (Signer-Independent Breakdown)
    signer_breakdown = {}
    if metadata and len(metadata) == total_samples:
        signers = sorted(list(set(m.get("signer_id", "unknown_signer") for m in metadata)))
        for s_id in signers:
            s_indices = [i for i, m in enumerate(metadata) if m.get("signer_id", "unknown_signer") == s_id]
            y_s = y[s_indices]
            preds_s = preds[s_indices]
            confs_s = confs[s_indices]

            s_correct = int(np.sum(preds_s == y_s))
            s_total = len(s_indices)
            s_acc = round(100.0 * s_correct / max(1, s_total), 2)
            s_mean_conf = round(float(np.mean(confs_s)), 4) if s_total > 0 else 0.0

            # Confusion matrix của từng signer
            cm_s = np.zeros((22, 22), dtype=int)
            for t, p in zip(y_s, preds_s):
                cm_s[t, p] += 1

            signer_breakdown[s_id] = {
                "signer_id": s_id,
                "total_samples": s_total,
                "correct_samples": s_correct,
                "accuracy_pct": s_acc,
                "mean_confidence": s_mean_conf,
                "confusion_matrix": cm_s.tolist()
            }

    # 4. Đánh giá tỷ lệ từ chối (Rejection Rate) theo các ngưỡng
    rejection_analysis = {}
    for th in thresholds:
        accepted_mask = (confs >= th)
        n_accepted = int(np.sum(accepted_mask))
        n_rejected = total_samples - n_accepted
        rejection_rate = round(100.0 * n_rejected / max(1, total_samples), 2)

        if n_accepted > 0:
            acc_on_accepted = round(100.0 * np.sum(preds[accepted_mask] == y[accepted_mask]) / n_accepted, 2)
        else:
            acc_on_accepted = 0.0

        rejection_analysis[f"threshold_{th:.2f}"] = {
            "threshold": th,
            "accepted_samples": n_accepted,
            "rejected_samples": n_rejected,
            "rejection_rate_pct": rejection_rate,
            "accuracy_on_accepted_pct": acc_on_accepted
        }

    # 5. Phân tích các cặp xung đột đặc thù
    collision_results = {}
    for pair in COLLISION_PAIRS:
        c1, c2 = pair
        idx1, idx2 = CLASS_TO_IDX_22[c1], CLASS_TO_IDX_22[c2]
        c1_to_c2 = int(cm_overall[idx1, idx2])
        c2_to_c1 = int(cm_overall[idx2, idx1])
        collision_results[f"{c1}_vs_{c2}"] = {
            f"{c1}_misclassified_as_{c2}": c1_to_c2,
            f"{c2}_misclassified_as_{c1}": c2_to_c1,
            f"{c1}_total": int(np.sum(y == idx1)),
            f"{c2}_total": int(np.sum(y == idx2))
        }

    return {
        "total_samples": total_samples,
        "overall_correct": overall_correct,
        "overall_accuracy_pct": overall_accuracy,
        "confusion_matrix_22x22": cm_overall.tolist(),
        "per_class_metrics": per_class_metrics,
        "missing_10_summary": {ch: per_class_metrics[ch] for ch in MISSING_10_LETTERS},
        "signer_breakdown": signer_breakdown,
        "rejection_analysis": rejection_analysis,
        "collision_analysis": collision_results
    }

def print_dry_run_guide(integrity_info: dict):
    """In hướng dẫn và trạng thái sẵn sàng khi chưa có dữ liệu thực tế"""
    print("================================================================================")
    print("CHẾ ĐỘ KIỂM TRA SẴN SÀNG (DRY-RUN / INTEGRITY CHECK)")
    print("================================================================================")
    print(f"Trạng thái Checkpoint: {'[OK] HỢP LỆ' if integrity_info['is_valid'] else '[LỖI] KHÔNG HỢP LỆ'}")
    print(f"Đường dẫn Checkpoint: {integrity_info['weights_path']}")
    print(f"Mã băm SHA-256:       {integrity_info['sha256']}")
    print(f"Số chiều đầu vào:     {integrity_info['input_dim']} (Khớp normalize_hand_landmarks)")
    print(f"Số lớp đầu ra:        {integrity_info['num_classes']} (Khớp đúng 22 nhãn tư thế nền VSL)")
    print("\n--- HƯỚNG DẪN THỰC THI THU THẬP VÀ ĐÁNH GIÁ ---")
    print("1. Thu thập video từ 3 - 5 người ký độc lập theo hướng dẫn tại:")
    print("   backend/data/alphabet_v2/multi_signer/README.md")
    print("2. Định dạng dữ liệu theo chuẩn:")
    print("   backend/data/alphabet_v2/multi_signer/collection_schema.json")
    print("3. Chạy lệnh đánh giá chính thức:")
    print("   python backend/training/evaluate_alphabet_static_head_multisigner.py --dataset_path <path_to_npz_or_json>")
    print("================================================================================")

def main():
    parser = argparse.ArgumentParser(description="Đánh giá Độc lập Đa Người ký cho VSLStaticAlphabetHead")
    parser.add_argument("--dataset_path", type=str, default=None, help="Đường dẫn tới tệp .npz hoặc .json chứa dữ liệu đa người ký")
    parser.add_argument("--weights_path", type=str, default=DEFAULT_HEAD_WEIGHTS_PATH, help="Đường dẫn tới checkpoint alphabet_static_head_v1.pt")
    parser.add_argument("--dry_run", action="store_true", help="Chạy kiểm tra trạng thái sẵn sàng của công cụ mà không cần dữ liệu")
    args = parser.parse_args()

    print("================================================================================")
    print("CÔNG CỤ ĐÁNH GIÁ ĐỘC LẬP ĐA NGƯỜI KÝ (VSLStaticAlphabetHead Multi-Signer Benchmark)")
    print("================================================================================")

    # 1. Kiểm tra tính toàn vẹn của Checkpoint
    integrity = verify_pipeline_integrity(args.weights_path)
    if not integrity["is_valid"]:
        print("[LỖI] Checkpoint không đạt chuẩn toàn vẹn kỹ thuật!")
        sys.exit(1)

    if args.dry_run or args.dataset_path is None or not os.path.exists(args.dataset_path):
        print("[THÔNG BÁO] Chưa truyền tập dữ liệu đa người ký thực tế hoặc đang chạy ở chế độ Dry-run.")
        print_dry_run_guide(integrity)
        return

    # 2. Tải mô hình
    print(f"[1/3] Nạp mô hình từ: {args.weights_path}")
    model = VSLStaticAlphabetHead(input_dim=72, hidden_dim1=64, hidden_dim2=32, num_classes=22)
    ckpt = torch.load(args.weights_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Tải dữ liệu đa người ký
    print(f"[2/3] Nạp dữ liệu đa người ký từ: {args.dataset_path}")
    X, y, meta = load_multisigner_dataset(args.dataset_path)
    print(f"      Đã nạp {len(X)} mẫu đặc trưng 72D.")

    # 4. Thực hiện đánh giá
    print("[3/3] Đang tiến hành đánh giá chỉ đọc (Read-only Evaluation)...")
    results = evaluate_multisigner_data(model, X, y, meta)

    # 5. Xuất báo cáo
    report_json_path = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2", "audit", "multisigner_evaluation_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("\n================================================================================")
    print("KẾT QUẢ ĐÁNH GIÁ TỔNG THỂ")
    print("================================================================================")
    print(f"Tổng số mẫu đánh giá: {results['total_samples']}")
    print(f"Độ chính xác tổng thể: {results['overall_accuracy_pct']}% ({results['overall_correct']}/{results['total_samples']})")
    
    if results["signer_breakdown"]:
        print("\n--- ĐỘ CHÍNH XÁC THEO TỪNG SIGNER (SIGNER-INDEPENDENT) ---")
        for s_id, s_info in results["signer_breakdown"].items():
            print(f"  Signer {s_id:15s}: {s_info['accuracy_pct']:6.2f}% ({s_info['correct_samples']}/{s_info['total_samples']} mẫu, Conf TB: {s_info['mean_confidence']:.4f})")

    print("\n--- PHÂN TÍCH CÁC CẶP XUNG ĐỘT TRỌNG ĐIỂM ---")
    for pair_name, c_info in results["collision_analysis"].items():
        print(f"  {pair_name}: {c_info}")

    print(f"\n-> Đã lưu báo cáo JSON: {report_json_path}")
    print("================================================================================")

if __name__ == "__main__":
    main()

