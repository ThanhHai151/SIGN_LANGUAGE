"""
Script tạo Dataset Tĩnh Chuẩn Miền Nam (Clean South V1) - Task 8
================================================================
Trích xuất chính xác 309 mẫu tĩnh đủ điều kiện (293 SHARED + 16 SOUTH)
từ backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz
sang backend/data/alphabet_v2/datasets/vsl_alphabet_static_south_clean_v1.npz.

Bảo đảm:
1. Giữ nguyên 100% file gốc vsl_alphabet_static_v2.npz (kiểm tra hash SHA-256).
2. Không thay đổi dữ liệu feature, label, landmarks, metadata gốc.
3. Không tự sinh mẫu, không sửa threshold, không huấn luyện mô hình.
4. Loại trừ hoàn toàn 193 mẫu miền Bắc và 49 mẫu BASE_POSE chờ thẩm định.
"""

import os
import sys
import json
import hashlib
import numpy as np
from collections import Counter

if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

WORKSPACE = os.path.abspath(".")
DATASETS_DIR = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2", "datasets")
SRC_NPZ_PATH = os.path.join(DATASETS_DIR, "vsl_alphabet_static_v2.npz")
OUT_NPZ_PATH = os.path.join(DATASETS_DIR, "vsl_alphabet_static_south_clean_v1.npz")
VERIFY_REPORT_JSON = os.path.join(DATASETS_DIR, "vsl_alphabet_static_south_clean_v1_report.json")

# 1. Các dải index bắt buộc theo quy định Task 8
ELIGIBLE_INDEX_RANGES = [
    (0, 98),     # 99 mẫu: A (22), B (22), C (20), D (20), E (15)
    (146, 256),  # 111 mẫu: I (18), K (17), L (16), M (19), N (21), O (20)
    (277, 324),  # 48 mẫu: Q (19), R (14), S (15)
    (345, 379),  # 35 mẫu: U (18), V (17)
    (405, 420),  # 16 mẫu: Y (16)
]

EXCLUDED_NORTH_RANGES = [
    (99, 145),   # 47 mẫu: G (29), H (18)
    (257, 276),  # 20 mẫu: P (20)
    (325, 344),  # 20 mẫu: T (20)
    (380, 404),  # 25 mẫu: X (25)
    (421, 501),  # 81 mẫu: BASE_POSE Ă/Â (42), Ê (20), Ô (19)
]

PENDING_RANGES = [
    (502, 550),  # 49 mẫu: BASE_POSE Ơ (9), Ư (14), Đ (26)
]

EXPECTED_17_CLASSES = {
    "A": 22, "B": 22, "C": 20, "D": 20, "E": 15,
    "I": 18, "K": 17, "L": 16, "M": 19, "N": 21, "O": 20,
    "Q": 19, "R": 14, "S": 15, "U": 18, "V": 17, "Y": 16
}
MISSING_5_CLASSES = ["G", "H", "P", "T", "X"]


def compute_sha256(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_and_verify():
    print("=" * 80)
    print("BẮT ĐẦU TASK 8: XÂY DỰNG DATASET TĨNH CHUẨN MIỀN NAM (309 MẪU)")
    print("=" * 80)

    # Bước 1: Kiểm tra tính sẵn sàng và tính hash file gốc
    assert os.path.exists(SRC_NPZ_PATH), f"Không tìm thấy file nguồn: {SRC_NPZ_PATH}"
    initial_src_sha256 = compute_sha256(SRC_NPZ_PATH)
    print(f"1. File nguồn: {SRC_NPZ_PATH}")
    print(f"   Kích thước: {os.path.getsize(SRC_NPZ_PATH):,} bytes")
    print(f"   SHA-256 ban đầu: {initial_src_sha256}")

    # Bước 2: Tải dữ liệu gốc
    src_data = np.load(SRC_NPZ_PATH, allow_pickle=True)
    src_X = src_data["X"]
    src_y = src_data["y"]
    src_classes = src_data["classes"]
    src_landmarks = src_data["landmarks"]
    src_meta_str = str(src_data["metadata_json"])
    src_meta = json.loads(src_meta_str)

    print("\n2. Cấu trúc mảng file gốc:")
    print(f"   - X: shape={src_X.shape}, dtype={src_X.dtype}")
    print(f"   - y: shape={src_y.shape}, dtype={src_y.dtype}")
    print(f"   - classes: shape={src_classes.shape}, dtype={src_classes.dtype}")
    print(f"   - landmarks: shape={src_landmarks.shape}, dtype={src_landmarks.dtype}")
    print(f"   - metadata_json: {len(src_meta)} records")

    assert src_X.shape == (551, 72)
    assert src_y.shape == (551,)
    assert src_landmarks.shape == (551, 21, 3)
    assert len(src_meta) == 551

    # Bước 3: Xác thực dải index
    keep_indices = []
    for start, end in ELIGIBLE_INDEX_RANGES:
        keep_indices.extend(range(start, end + 1))
    
    assert len(keep_indices) == 309, f"Tổng số keep_indices không bằng 309: {len(keep_indices)}"
    assert sorted(keep_indices) == keep_indices, "Indices không giữ nguyên thứ tự tăng dần!"
    assert len(set(keep_indices)) == 309, "Phát hiện index bị trùng lặp!"

    excluded_north_indices = []
    for start, end in EXCLUDED_NORTH_RANGES:
        excluded_north_indices.extend(range(start, end + 1))
    assert len(excluded_north_indices) == 193, f"Tổng số excluded_north không bằng 193: {len(excluded_north_indices)}"

    pending_indices = []
    for start, end in PENDING_RANGES:
        pending_indices.extend(range(start, end + 1))
    assert len(pending_indices) == 49, f"Tổng số pending không bằng 49: {len(pending_indices)}"

    # Kiểm tra tính phân hoạch hoàn hảo [0, 550]
    all_indices = set(keep_indices) | set(excluded_north_indices) | set(pending_indices)
    assert len(all_indices) == 551, f"Tổng số index phân hoạch không đủ 551: {len(all_indices)}"
    assert all_indices == set(range(551)), "Tập hợp index không khớp chính xác dải [0, 550]!"
    assert len(set(keep_indices) & set(excluded_north_indices)) == 0, "Trùng giữa keep và north!"
    assert len(set(keep_indices) & set(pending_indices)) == 0, "Trùng giữa keep và pending!"
    assert len(set(excluded_north_indices) & set(pending_indices)) == 0, "Trùng giữa north và pending!"
    print("\n3. Kiểm tra dải index: ĐẠT")
    print(f"   - Keep indices: {len(keep_indices)} mẫu (309)")
    print(f"   - Excluded North indices: {len(excluded_north_indices)} mẫu (193)")
    print(f"   - Excluded Pending indices: {len(pending_indices)} mẫu (49)")
    print("   - Tổng phân hoạch: 309 + 193 + 49 = 551 mẫu (100% bao phủ không giao nhau)")

    # Bước 4: Trích xuất và xây dựng dataset sạch
    clean_X = src_X[keep_indices].copy()
    clean_y = src_y[keep_indices].copy()
    clean_landmarks = src_landmarks[keep_indices].copy()
    clean_classes = src_classes.copy()  # Giữ danh sách 22 lớp gốc để đảm bảo tương thích 100% với model architecture
    clean_meta_list = [src_meta[i] for i in keep_indices]
    clean_meta_str = json.dumps(clean_meta_list, ensure_ascii=False)

    print("\n4. Lưu file dataset sạch mới...")
    np.savez_compressed(
        OUT_NPZ_PATH,
        X=clean_X,
        y=clean_y,
        classes=clean_classes,
        landmarks=clean_landmarks,
        metadata_json=clean_meta_str
    )
    print(f"   => Đã lưu thành công tại: {OUT_NPZ_PATH}")
    print(f"   Kích thước file sạch: {os.path.getsize(OUT_NPZ_PATH):,} bytes")
    out_sha256 = compute_sha256(OUT_NPZ_PATH)
    print(f"   SHA-256 file sạch: {out_sha256}")

    # Bước 5: Kiểm tra tính toàn vẹn của file gốc
    post_src_sha256 = compute_sha256(SRC_NPZ_PATH)
    assert initial_src_sha256 == post_src_sha256, "CẢNH BÁO: File nguồn vsl_alphabet_static_v2.npz đã bị thay đổi!"
    print(f"\n5. Kiểm tra an toàn file gốc: ĐẠT")
    print(f"   SHA-256 file nguồn trước và sau hoàn toàn khớp: {post_src_sha256}")

    # Bước 6: Kiểm tra chi tiết dữ liệu file sạch vừa tạo
    print("\n6. Đọc lại và kiểm tra toàn diện file clean:")
    clean_data = np.load(OUT_NPZ_PATH, allow_pickle=True)
    c_X = clean_data["X"]
    c_y = clean_data["y"]
    c_classes = clean_data["classes"]
    c_lms = clean_data["landmarks"]
    c_meta = json.loads(str(clean_data["metadata_json"]))

    # A. Kiểm tra số lượng
    assert c_X.shape == (309, 72), f"Sai shape X: {c_X.shape}"
    assert c_y.shape == (309,), f"Sai shape y: {c_y.shape}"
    assert c_lms.shape == (309, 21, 3), f"Sai shape landmarks: {c_lms.shape}"
    assert c_classes.shape == (22,), f"Sai shape classes: {c_classes.shape}"
    assert len(c_meta) == 309, f"Sai số lượng metadata: {len(c_meta)}"

    # B. Kiểm tra phân bố lớp
    class_counter = Counter()
    video_counter = Counter()
    region_counter = Counter()
    source_type_counter = Counter()

    for idx, (y_val, m) in enumerate(zip(c_y, c_meta)):
        lbl_from_y = c_classes[y_val]
        lbl_from_meta = m.get("label") or m.get("source_letter")
        assert lbl_from_y == lbl_from_meta, f"Mẫu {idx}: label từ y ({lbl_from_y}) != label từ meta ({lbl_from_meta})"
        
        class_counter[lbl_from_y] += 1
        video_counter[m["video_id"]] += 1
        source_type_counter[m["source_type"]] += 1

        vid = m["video_id"]
        if vid.endswith("N"):
            reg = "REGION_SOUTH"
        elif vid.endswith("B"):
            reg = "REGION_NORTH"
        elif vid.endswith("T"):
            reg = "REGION_CENTRAL"
        else:
            reg = "REGION_SHARED"
        region_counter[reg] += 1

    assert dict(class_counter) == EXPECTED_17_CLASSES, f"Sai phân bố 17 lớp: {dict(class_counter)}"
    assert all(class_counter[c] == 0 for c in MISSING_5_CLASSES), "Phát hiện lớp thiếu có mẫu!"
    
    # C. Kiểm tra vùng miền
    assert region_counter["REGION_SHARED"] == 293, f"Sai số mẫu SHARED: {region_counter['REGION_SHARED']}"
    assert region_counter["REGION_SOUTH"] == 16, f"Sai số mẫu SOUTH: {region_counter['REGION_SOUTH']}"
    assert region_counter["REGION_NORTH"] == 0, f"Có mẫu NORTH trong clean: {region_counter['REGION_NORTH']}"
    assert region_counter["REGION_CENTRAL"] == 0, f"Có mẫu CENTRAL trong clean: {region_counter['REGION_CENTRAL']}"

    # D. Kiểm tra nguồn dữ liệu (100% static_video, 0% base_pose_of_dynamic)
    assert source_type_counter["static_video"] == 309, f"Sai số static_video: {source_type_counter['static_video']}"
    assert source_type_counter.get("base_pose_of_dynamic", 0) == 0, "Có mẫu base_pose_of_dynamic lọt vào clean!"

    # E. Kiểm tra tính toàn vẹn từng mẫu (Feature, Label, Landmark, Metadata)
    for clean_i, orig_i in enumerate(keep_indices):
        # Feature khớp tuyệt đối
        assert np.array_equal(c_X[clean_i], src_X[orig_i]), f"Mẫu {clean_i} (gốc {orig_i}): X không khớp!"
        # Label index khớp tuyệt đối
        assert c_y[clean_i] == src_y[orig_i], f"Mẫu {clean_i} (gốc {orig_i}): y không khớp!"
        # Landmark khớp tuyệt đối
        assert np.array_equal(c_lms[clean_i], src_landmarks[orig_i]), f"Mẫu {clean_i} (gốc {orig_i}): landmarks không khớp!"
        # Metadata khớp từng trường
        assert c_meta[clean_i] == src_meta[orig_i], f"Mẫu {clean_i} (gốc {orig_i}): metadata không khớp!"

    print("   - [A] Số lượng mẫu: 309/309 ĐẠT")
    print("   - [B] Phân bố 17 lớp: Khớp 100% bảng yêu cầu (5 lớp thiếu G, H, P, T, X có đúng 0 mẫu)")
    print("   - [C] Phân bố vùng miền: 293 SHARED + 16 SOUTH + 0 NORTH + 0 CENTRAL ĐẠT")
    print("   - [D] Phân bố source_type: 309 static_video + 0 base_pose_of_dynamic ĐẠT")
    print("   - [E] Tính toàn vẹn mảng: 100% X, y, landmarks, metadata khớp hoàn hảo với mẫu nguồn ĐẠT")

    # Lưu báo cáo JSON xác minh
    report_dict = {
        "status": "SUCCESS",
        "task": "TASK_8_BUILD_CLEAN_SOUTH_DATASET",
        "source_file": {
            "path": SRC_NPZ_PATH,
            "sha256": initial_src_sha256,
            "total_samples": 551,
            "status": "UNTOUCHED"
        },
        "clean_file": {
            "path": OUT_NPZ_PATH,
            "sha256": out_sha256,
            "total_samples": 309,
            "num_classes_in_classes_array": len(c_classes),
            "num_classes_active": 17,
            "classes_active": list(EXPECTED_17_CLASSES.keys()),
            "classes_missing_0_samples": MISSING_5_CLASSES,
            "shapes": {
                "X": list(c_X.shape),
                "y": list(c_y.shape),
                "landmarks": list(c_lms.shape),
                "classes": list(c_classes.shape)
            },
            "dtypes": {
                "X": str(c_X.dtype),
                "y": str(c_y.dtype),
                "landmarks": str(c_lms.dtype),
                "classes": str(c_classes.dtype)
            }
        },
        "sample_counts": {
            "original_total": 551,
            "clean_total": 309,
            "eligible_shared": region_counter["REGION_SHARED"],
            "eligible_south": region_counter["REGION_SOUTH"],
            "excluded_north": len(excluded_north_indices),
            "excluded_pending_base_pose": len(pending_indices)
        },
        "class_distribution_17": dict(class_counter),
        "video_distribution": dict(video_counter),
        "region_distribution": dict(region_counter),
        "source_type_distribution": dict(source_type_counter)
    }

    with open(VERIFY_REPORT_JSON, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, ensure_ascii=False, indent=2)
    print(f"\n=> Đã ghi báo cáo xác minh máy đọc tại: {VERIFY_REPORT_JSON}")
    print("TASK 8 HOÀN THÀNH XUẤT SẮC VÀ ĐẠT 100% TIÊU CHUẨN KIỂM TOÀN VẸN!")


if __name__ == "__main__":
    build_and_verify()

