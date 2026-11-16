import json
import os
import numpy as np

WORKSPACE = "d:/SIGN_LANGUAGE"

# Danh sách 29 chữ cái chuẩn tiếng Việt
VN_29 = [
    "A", "Ă", "Â", "B", "C", "D", "Đ", "E", "Ê", 
    "G", "H", "I", "K", "L", "M", "N", "O", "Ô", 
    "Ơ", "P", "Q", "R", "S", "T", "U", "Ư", "V", 
    "X", "Y"
]

# 1. Đọc metadata video
with open(os.path.join(WORKSPACE, "backend/data/alphabet_v2/metadata/videos_metadata.json"), "r", encoding="utf-8") as f:
    video_meta = json.load(f)

# 2. Đọc static landmarks
static_dir = os.path.join(WORKSPACE, "backend/data/alphabet_v2/static_landmarks")
# 3. Đọc motion json
with open(os.path.join(WORKSPACE, "backend/data/alphabet_v2/datasets/vsl_alphabet_motion_v2.json"), "r", encoding="utf-8") as f:
    motion_data = json.load(f)

# 4. Đọc static npz
data_static = np.load(os.path.join(WORKSPACE, "backend/data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"), allow_pickle=True)
static_classes = list(data_static["classes"])
static_y = data_static["y"]

# 5. Đọc classifier
from backend.vision.alphabet_classifier import VSLAlphabetClassifier
c = VSLAlphabetClassifier()

# Evaluate classifier on static npz
stats_npz = {}
for idx, target in enumerate(static_classes):
    t_idx = idx
    tp = fn = fp = 0
    confused_to = {}
    for i, (lm, label) in enumerate(zip(data_static["landmarks"], static_y)):
        pred, _ = c.classify(lm.tolist())
        if label == t_idx:
            if pred == target:
                tp += 1
            else:
                fn += 1
                confused_to[pred] = confused_to.get(pred, 0) + 1
        elif pred == target:
            fp += 1
    stats_npz[target] = {
        "tp": tp, "fn": fn, "fp": fp, "total": tp + fn,
        "confused_to": confused_to
    }

# 6. Kiểm tra các rules trong VSLAlphabetClassifier
classifier_rules = {
    "A": "Có rule riêng (_is_a / matches.append('A'))",
    "B": "Có rule riêng (matches.append('B'))",
    "C": "Có rule riêng (matches.append('C'))",
    "D": "Có rule riêng (matches.append('D'))",
    "E": "Có rule riêng (matches.append('E'))",
    "G": "Có rule riêng (matches.append('G'))",
    "H": "Có rule riêng (matches.append('H'))",
    "I": "Có rule riêng (matches.append('I'))",
    "K": "Có rule riêng (matches.append('K'))",
    "L": "Có rule riêng (matches.append('L'))",
    "M": "Chưa có rule hình học riêng (bị chặn an toàn)",
    "N": "Chưa có rule hình học riêng (bị chặn an toàn)",
    "O": "Có rule riêng (matches.append('O'))",
    "P": "Có rule riêng (matches.append('P'))",
    "Q": "Có rule riêng (matches.append('Q'))",
    "R": "Có rule riêng (matches.append('R'))",
    "S": "Có rule riêng (matches.append('S'))",
    "T": "Có rule riêng (matches.append('T'))",
    "U": "Có rule riêng (matches.append('U'))",
    "V": "Có rule riêng (matches.append('V'))",
    "X": "Có rule riêng (matches.append('X'))",
    "Y": "Có rule riêng (matches.append('Y'))",
    "Ă": "Chữ động - Chưa có rule tĩnh trong VSLAlphabetClassifier",
    "Â": "Chữ động - Chưa có rule tĩnh trong VSLAlphabetClassifier",
    "Ê": "Chữ động - Chưa có rule tĩnh trong VSLAlphabetClassifier",
    "Ô": "Chữ động - Chưa có rule tĩnh trong VSLAlphabetClassifier",
    "Ơ": "Chữ động - Chưa có rule tĩnh trong VSLAlphabetClassifier",
    "Ư": "Chữ động - Chưa có rule tĩnh trong VSLAlphabetClassifier",
    "Đ": "Chữ động - Chưa có rule tĩnh trong VSLAlphabetClassifier",
}

# 7. Tổng hợp 29 chữ
report_29 = []
for ch in VN_29:
    # Video info
    vid_info = video_meta.get(ch, {})
    vid_id = vid_info.get("video_id", "N/A")
    total_frames = vid_info.get("total_frames", 0)
    
    # Dataset info
    is_static = ch in static_classes
    is_motion = ch in motion_data.get("sequences", {})
    
    npz_count = stats_npz.get(ch, {}).get("total", 0) if is_static else 0
    tp = stats_npz.get(ch, {}).get("tp", 0) if is_static else 0
    fp = stats_npz.get(ch, {}).get("fp", 0) if is_static else 0
    confused = stats_npz.get(ch, {}).get("confused_to", {}) if is_static else {}
    
    # Motion info
    motion_info = motion_data.get("sequences", {}).get(ch, {})
    diacritic_frames = len(motion_info.get("diacritic_motion_frames", []))
    base_pose_frames = len(motion_info.get("base_pose_frames", []))
    base_letter = motion_info.get("base_handshape", "N/A")
    
    has_data = is_static or is_motion
    enough_data = (npz_count >= 14) or (diacritic_frames >= 9)
    
    # Confusable letters
    confusable_notes = []
    if ch in ["S", "T"]: confusable_notes.append("S và T (ngón cái kẹp giữa các ngón)")
    if ch in ["C", "O"]: confusable_notes.append("C và O (độ mở khép ngón)")
    if ch in ["D", "L"]: confusable_notes.append("D và L (ngón trỏ và ngón cái)")
    if ch in ["A", "E"]: confusable_notes.append("A và E (ngón cái ôm nắm đấm)")
    if ch in ["U", "V"]: confusable_notes.append("U và V (độ mở 2 ngón)")
    if ch in ["G", "Q"]: confusable_notes.append("G và Q (phương ngang vs phương chúc xuống)")
    if ch in ["H", "V", "R"]: confusable_notes.append("H, V, R (2 ngón trỏ và giữa)")
    if ch in ["P", "K"]: confusable_notes.append("P và K (tư thế ngón giống nhau, khác hướng cẳng tay)")
    if ch in ["M", "N"]: confusable_notes.append("M và N (3 ngón vs 2 ngón đè lên ngón cái)")
    if is_motion: confusable_notes.append(f"Dễ nhầm với chữ nền '{base_letter}' nếu không phát hiện được nét vẽ dấu")
    
    report_29.append({
        "letter": ch,
        "type": "Tĩnh (Base)" if is_static else "Động (Diacritic)",
        "video_id": vid_id,
        "total_video_frames": total_frames,
        "npz_samples": npz_count,
        "motion_diacritic_samples": diacritic_frames,
        "has_data": has_data,
        "enough_data": enough_data,
        "rule_type": "Geometric Rule + MLP" if (is_static and ch not in ["M", "N"]) else ("Chỉ MLP (Chưa có Rule)" if ch in ["M", "N"] else "Diacritic FSM (Chữ động)"),
        "recall_on_npz": f"{tp}/{npz_count} ({tp/npz_count*100:.1f}%)" if npz_count > 0 else "N/A (Chữ động)",
        "fp_on_npz": fp if is_static else "N/A",
        "confusable": ", ".join(confusable_notes) if confusable_notes else "Ít nhầm lẫn"
    })

print(f"{'Chữ':<4} {'Loại':<18} {'Video':<8} {'NPZ':<6} {'DiacFrames':<12} {'Có Dữ Liệu?':<12} {'Đủ Đánh Giá?':<14} {'Khả Năng Nhận Diện Hiện Tại':<26}")
print("-" * 110)
for r in report_29:
    print(f"{r['letter']:<4} {r['type']:<18} {r['video_id']:<8} {r['npz_samples']:<6} {r['motion_diacritic_samples']:<12} {str(r['has_data']):<12} {str(r['enough_data']):<14} {r['rule_type']:<26}")

