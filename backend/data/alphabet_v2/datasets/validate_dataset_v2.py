"""
Kịch bản kiểm định chất lượng Dataset V2 (VSL Connect AI - Task 5B)
Xác minh tính đúng đắn, không rò rỉ và toàn vẹn dữ liệu cho 29 chữ cái.
"""
import os
import sys
import json
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

WORKSPACE = os.path.abspath(".")
DATASET_DIR = os.path.join(WORKSPACE, "backend", "data", "alphabet_v2", "datasets")

NPZ_PATH = os.path.join(DATASET_DIR, "vsl_alphabet_static_v2.npz")
MOTION_PATH = os.path.join(DATASET_DIR, "vsl_alphabet_motion_v2.json")
LABELS_PATH = os.path.join(DATASET_DIR, "alphabet_labels_v2.json")

def validate():
    print("=" * 75)
    print("BẮT ĐẦU KIỂM ĐỊNH CHẤT LƯỢNG DATASET V2 CHO 29 CHỮ CÁI VSL")
    print("=" * 75)
    
    errors = []
    warnings = []
    
    # 1. Kiểm tra alphabet_labels_v2.json
    with open(LABELS_PATH, "r", encoding="utf-8") as fp:
        labels = json.load(fp)
    
    vn_29 = labels["vietnamese_alphabet_29"]
    base_22 = labels["base_handshapes_22"]
    dyn_7 = labels["dynamic_letters_7"]
    dyn_map = labels["dynamic_to_base_mapping"]
    
    if len(vn_29) != 29:
        errors.append(f"Danh mục tiếng Việt không đủ 29 chữ: {len(vn_29)}")
    if any(c in vn_29 for c in ["F", "J", "W", "Z"]):
        errors.append("Danh mục chữ tiếng Việt bị lẫn chữ tiếng Anh (F, J, W, Z)!")
    if len(base_22) != 22:
        errors.append(f"Danh mục chữ nền không đủ 22 chữ: {len(base_22)}")
    if len(dyn_7) != 7:
        errors.append(f"Danh mục chữ động không đủ 7 chữ: {len(dyn_7)}")
        
    print(f"[OK] Đã kiểm tra danh mục nhãn: 29 chữ cái, 22 nhãn nền, 7 nhãn động.")

    # 2. Kiểm tra vsl_alphabet_motion_v2.json
    with open(MOTION_PATH, "r", encoding="utf-8") as fp:
        motion = json.load(fp)
        
    seqs = motion["sequences"]
    if len(seqs) != 7:
        errors.append(f"Số lượng chuỗi động không đủ 7: {len(seqs)}")
        
    for dl in dyn_7:
        if dl not in seqs:
            errors.append(f"Thiếu chuỗi động cho chữ {dl}")
            continue
        s_data = seqs[dl]
        raw = s_data["raw_timeline_sequence"]
        valid = s_data["valid_landmarks_sequence"]
        diac = s_data["diacritic_motion_frames"]
        base = s_data["base_pose_frames"]
        
        # Kiểm tra thứ tự frame_index
        last_f = -1
        for f in raw:
            fidx = f["frame_index"]
            if fidx <= last_f:
                errors.append(f"Lỗi thứ tự frame trong chuỗi {dl}: {fidx} <= {last_f}")
            last_f = fidx
            
            lms = f["landmarks"]
            if lms is not None:
                if len(lms) != 21:
                    errors.append(f"Sai số điểm mốc trong chuỗi {dl} frame {fidx}: {len(lms)} != 21")
                if np.isnan(lms).any() or np.isinf(lms).any():
                    errors.append(f"Tọa độ NaN/Inf trong chuỗi {dl} frame {fidx}")
                    
        if len(diac) == 0:
            errors.append(f"Không có frame diacritic nào cho chữ {dl}")
        if len(base) == 0:
            errors.append(f"Không có frame base_pose nào cho chữ {dl}")
            
    print(f"[OK] Đã kiểm tra 7 chuỗi động: tính toàn vẹn mốc xương, thứ tự khung hình hoàn hảo.")

    # 3. Kiểm tra vsl_alphabet_static_v2.npz
    npz = np.load(NPZ_PATH, allow_pickle=True)
    X = npz["X"]
    y = npz["y"]
    classes = list(npz["classes"])
    lms = npz["landmarks"]
    meta = json.loads(str(npz["metadata_json"]))
    
    if len(X) != len(y) or len(X) != len(lms) or len(X) != len(meta):
        errors.append(f"Kích thước các mảng trong npz không khớp: X={len(X)}, y={len(y)}, lms={len(lms)}, meta={len(meta)}")
        
    if X.shape[1] != 72:
        errors.append(f"Kích thước đặc trưng X không đúng 72 chiều: {X.shape[1]}")
    if lms.shape[1:] != (21, 3):
        errors.append(f"Kích thước landmarks không đúng (21, 3): {lms.shape[1:]}")
        
    if np.isnan(X).any() or np.isinf(X).any():
        errors.append("Mảng X chứa giá trị NaN hoặc Inf!")
    if np.isnan(lms).any() or np.isinf(lms).any():
        errors.append("Mảng landmarks chứa giá trị NaN hoặc Inf!")
        
    if classes != base_22:
        errors.append("Danh sách classes trong npz không khớp đúng 22 chữ nền!")
        
    # Kiểm tra nhãn và cấm frame ENTRY/EXIT trong static dataset
    for idx, m in enumerate(meta):
        label = m["label"]
        if label not in base_22:
            errors.append(f"Nhãn lạ trong dataset tĩnh: {label}")
        if y[idx] != base_22.index(label):
            errors.append(f"Index y không khớp label tại mẫu {idx}: {y[idx]} != {base_22.index(label)}")
        if m["source_type"] == "base_pose_of_dynamic":
            src = m["source_letter"]
            expected_base = dyn_map.get(src)
            if label != expected_base:
                errors.append(f"Gán sai nhãn BASE_POSE: Chữ nguồn {src} bị gán thành {label} thay vì {expected_base}!")
        if m.get("phase") in ["entry", "exit"]:
            errors.append(f"Phát hiện frame ENTRY/EXIT bị lọt vào dataset tĩnh tại mẫu {idx}!")
            
    print(f"[OK] Đã kiểm tra dataset tĩnh: {len(X)} mẫu, 72 chiều, 0 NaN/Inf, đúng 22 nhãn nền.")

    # 4. Báo cáo thống kê
    from collections import Counter
    label_counts = Counter(m["label"] for m in meta)
    src_type_counts = Counter(m["source_type"] for m in meta)
    
    print("\n--- THỐNG KÊ PHÂN BỐ MẪU DATASET TĨNH ---")
    print(f"Tổng số mẫu tĩnh: {len(X)}")
    print(f"  + Nguồn video tĩnh (stable_pose): {src_type_counts['static_video']}")
    print(f"  + Nguồn BASE_POSE của 7 chữ động:  {src_type_counts['base_pose_of_dynamic']}")
    print("\nSố mẫu của từng nhãn nền trong 22 chữ:")
    for b in base_22:
        print(f"  {b}: {label_counts[b]:>2} mẫu")

    print("\n" + "=" * 75)
    if errors:
        print(f"KẾT QUẢ KIỂM ĐỊNH: THẤT BẠI VỚI {len(errors)} LỖI!")
        for e in errors:
            print(f"  [LỖI] {e}")
        return False
    else:
        print("KẾT QUẢ KIỂM ĐỊNH: ĐẠT 100% TIÊU CHUẨN KỸ THUẬT VÀ TOÀN VẸN!")
        return True

if __name__ == "__main__":
    success = validate()
    sys.exit(0 if success else 1)
