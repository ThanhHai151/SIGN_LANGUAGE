"""
Kiểm thử tự động cho A01.1: Words Holistic Runtime Integration & Fallback Blocking.
Kiểm tra:
1. Tính toàn vẹn của mô hình SSTCN_Attention và từ điển 102 lớp (Word class 39..101).
2. Kiểm tra hàm analyze_sequence_landmarks:
   - Phát hiện đúng 'holistic_raw' khi có chuyển động pose thật.
   - Phát hiện đúng 'hands_fallback_pose' khi vai cố định [0.65, 0.45] & [0.35, 0.45] hoặc >= 25 khớp tĩnh.
   - Phát hiện đúng 'holistic_pose_missing' khi pose toàn 0.
   - Phát hiện đúng 'both_hands_missing' khi 2 bàn tay toàn 0.
3. Kiểm tra recognizer.predict_sequence ở mode 'words':
   - Chặn pose giả, trả về fallback_blocked=True, sign='NEUTRAL', confidence=0.0, không crash.
   - Chặn chuỗi mất pose, trả về fallback_blocked=True, sign='NEUTRAL', confidence=0.0, không crash.
   - Chấp nhận chuỗi holistic_raw hợp lệ, suy luận ra Top-1 và Top-5 thuộc lớp 39..101.
4. Kiểm tra tính độc lập của các mode khác:
   - mode='letters' / instant landmarks không bị ảnh hưởng.
   - mode='digits' không bị ảnh hưởng.
   - mode='sentence' không bị ảnh hưởng.
5. Kiểm tra ghi log word_debugger khi có phiên test:
   - Ghi nhận event FALLBACK_BLOCKED.
   - Ghi nhận event PREDICTION khi frame hợp lệ.
"""
import os
import sys
import numpy as np
import torch

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from backend.models.sstcn_model import load_sstcn_model
from backend.models.vsl_102_dictionary import SIGN_DICT_102, SIGN_METADATA_102
from backend.vision.recognizer import VSLRecognizer
from backend.vision.word_debugger import word_debugger, analyze_sequence_landmarks

passed_tests = []
failed_tests = []

def check(desc: str, condition: bool, extra: str = ""):
    if condition:
        print(f"  [PASS] {desc}")
        passed_tests.append(desc)
    else:
        print(f"  [FAIL] {desc} -> {extra}")
        failed_tests.append(f"{desc} ({extra})")


print("=" * 70)
print("TEST 1: Kiểm tra cấu trúc 102 lớp & Word Class Range (39..101)")
print("=" * 70)
check("Từ điển SIGN_DICT_102 đủ 102 lớp model (0..101)", len([k for k in SIGN_DICT_102 if 0 <= k <= 101]) == 102)
check("Lớp 39 là từ vựng đầu tiên ('Thời gian')", SIGN_DICT_102[39] == "Thời gian")
check("Lớp 101 là từ vựng cuối cùng ('Bộ đội')", SIGN_DICT_102[101] == "Bộ đội")
check("Đúng 63 từ vựng trong khoảng [39, 101]", len([k for k in SIGN_DICT_102 if 39 <= k <= 101]) == 63)

print("\n" + "=" * 70)
print("TEST 2: Phân tích Landmark & Phát hiện Pose Giả / Pose Thật")
print("=" * 70)
# A. Tạo mẫu Holistic Thật (có chuyển động vai, khuỷu tay và bàn tay)
t = np.linspace(0, 1, 30)[:, None, None]
holistic_sample = np.zeros((30, 75, 3), dtype=np.float32)
# Pose: Vai trái (11) và vai phải (12) chuyển động tự nhiên
holistic_sample[:, 11, :2] = np.array([0.62, 0.40]) + 0.03 * np.sin(t * 5)[:, 0, :]
holistic_sample[:, 12, :2] = np.array([0.38, 0.40]) - 0.02 * np.cos(t * 5)[:, 0, :]
# Khuỷu tay và cổ tay
holistic_sample[:, 13, :2] = np.array([0.70, 0.60]) + 0.05 * np.sin(t * 3)[:, 0, :]
holistic_sample[:, 14, :2] = np.array([0.30, 0.60]) - 0.05 * np.sin(t * 3)[:, 0, :]
holistic_sample[:, 15, :2] = np.array([0.65, 0.75]) + 0.04 * np.sin(t * 4)[:, 0, :]
holistic_sample[:, 16, :2] = np.array([0.35, 0.75]) - 0.04 * np.sin(t * 4)[:, 0, :]
# Tay phải (54..74) có chuyển động
holistic_sample[:, 54:75, 0] = 0.35 + 0.05 * np.random.randn(30, 21) * 0.02
holistic_sample[:, 54:75, 1] = 0.75 + 0.05 * np.random.randn(30, 21) * 0.02

stats_real = analyze_sequence_landmarks(holistic_sample, debug_meta={"tracker_engine": "holistic"})
check("Nhận diện đúng 'holistic_raw'", stats_real["landmarks_source"] == "holistic_raw")
check("Không phải pose giả", stats_real["is_synthetic_pose"] is False)
check("Vai động (is_static_shoulders = False)", stats_real["is_static_shoulders"] is False)
check("Hợp lệ cho SSTCN words (is_valid_for_words_sstcn = True)", stats_real["is_valid_for_words_sstcn"] is True)

# B. Tạo mẫu Hands Fallback Pose (vai tĩnh [0.65, 0.45] & [0.35, 0.45], 29 khớp tĩnh)
fallback_sample = np.zeros((30, 75, 3), dtype=np.float32)
fallback_sample[:, 11, :2] = [0.65, 0.45]  # DEFAULT_POSE vai trái
fallback_sample[:, 12, :2] = [0.35, 0.45]  # DEFAULT_POSE vai phải
fallback_sample[:, 0, :2] = [0.5, 0.2]     # Mũi
for i in range(1, 11):
    fallback_sample[:, i, :2] = [0.5, 0.2]
fallback_sample[:, 13, :2] = [0.72, 0.65]  # Khuỷu trái tĩnh
fallback_sample[:, 14, :2] = [0.28, 0.65]  # Khuỷu phải tĩnh
fallback_sample[:, 15, :2] = [0.68, 0.80]  # Cổ tay trái tĩnh
fallback_sample[:, 16, :2] = [0.32, 0.80]  # Cổ tay phải tĩnh
for i in range(17, 33):
    fallback_sample[:, i, :2] = [0.5, 0.8] # Khớp dưới tĩnh
fallback_sample[:, 54:75, 0] = 0.32
fallback_sample[:, 54:75, 1] = 0.80

stats_fake = analyze_sequence_landmarks(fallback_sample, debug_meta={"tracker_engine": "hands"})
check("Nhận diện đúng 'hands_fallback_pose'", stats_fake["landmarks_source"] == "hands_fallback_pose")
check("Xác nhận is_synthetic_pose = True", stats_fake["is_synthetic_pose"] is True)
check("Vai tĩnh (is_static_shoulders = True)", stats_fake["is_static_shoulders"] is True)
check("Bị chặn cho SSTCN words (is_valid_for_words_sstcn = False)", stats_fake["is_valid_for_words_sstcn"] is False)
check("Lý do chặn đúng: 'hands_fallback_pose_blocked'", stats_fake["blocked_reason"] == "hands_fallback_pose_blocked")

# C. Tạo mẫu mất Pose toàn bộ (pose bằng 0)
missing_pose_sample = np.zeros((30, 75, 3), dtype=np.float32)
missing_pose_sample[:, 54:75, :] = 0.5  # Chỉ có bàn tay
stats_miss_pose = analyze_sequence_landmarks(missing_pose_sample, debug_meta={"tracker_engine": "holistic"})
check("Nhận diện đúng 'holistic_pose_missing'", stats_miss_pose["landmarks_source"] == "holistic_pose_missing")
check("Bị chặn do mất pose thân trên", stats_miss_pose["blocked_reason"] == "holistic_pose_missing")

# D. Tạo mẫu mất cả 2 bàn tay (hai bàn tay bằng 0)
missing_hands_sample = np.zeros((30, 75, 3), dtype=np.float32)
missing_hands_sample[:, :33, :] = holistic_sample[:, :33, :]  # Có pose nhưng không có tay
stats_miss_hands = analyze_sequence_landmarks(missing_hands_sample, debug_meta={"tracker_engine": "holistic"})
check("Bị chặn do mất cả 2 bàn tay", stats_miss_hands["blocked_reason"] == "both_hands_missing")


print("\n" + "=" * 70)
print("TEST 3: Kiểm thử Chốt Chặn Runtime tại recognizer.predict_sequence")
print("=" * 70)
recognizer = VSLRecognizer()

# A. Thử gửi fallback_sample vào mode 'words'
res_blocked = recognizer.predict_sequence(
    sequence_data=fallback_sample,
    threshold=0.65,
    mode="words",
    debug_meta={"tracker_engine": "hands"}
)
check("Chặn thành công không gây crash", res_blocked.get("success") is True)
check("Có cờ fallback_blocked = True", res_blocked.get("fallback_blocked") is True)
check("Sign trả về là 'NEUTRAL'", res_blocked.get("sign") == "NEUTRAL")
check("Confidence = 0.0", res_blocked.get("confidence") == 0.0)
check("Không chốt token (confirmed = False)", res_blocked.get("confirmed") is False)
check("Nhãn hiển thị rõ ràng 'Đã chặn Pose giả'", "Đã chặn Pose giả" in res_blocked.get("label", ""))

# B. Thử gửi missing_pose_sample vào mode 'words'
res_miss_pose = recognizer.predict_sequence(
    sequence_data=missing_pose_sample,
    threshold=0.65,
    mode="words",
    debug_meta={"tracker_engine": "holistic"}
)
check("Chặn missing pose thành công không gây crash", res_miss_pose.get("fallback_blocked") is True)
check("Lý do chặn đúng 'holistic_pose_missing'", res_miss_pose.get("blocked_reason") == "holistic_pose_missing")

# C. Thử gửi holistic_sample thật vào mode 'words'
res_real = recognizer.predict_sequence(
    sequence_data=holistic_sample,
    threshold=0.65,
    mode="words",
    debug_meta={"tracker_engine": "holistic"}
)
check("Suy luận holistic thật thành công", res_real.get("success") is True)
check("Không bị chặn fallback (fallback_blocked absent/False)", not res_real.get("fallback_blocked"))
check("Có Top-5 dự đoán", len(res_real.get("top_k", [])) == 5)
top1_idx = res_real["top_k"][0]["index"]
check("Top-1 dự đoán nằm trong khoảng 39..101 (Words)", 39 <= top1_idx <= 101, f"top1_idx = {top1_idx}")


print("\n" + "=" * 70)
print("TEST 4: Kiểm tra Tính Độc Lập của các Chế độ Khác (letters, digits, sentence)")
print("=" * 70)
# Mock 21 landmarks bàn tay chữ V
v_landmarks = [[0.5, 0.7, 0.0] for _ in range(21)]
v_landmarks[8] = [0.45, 0.3, 0.0]   # Index tip duỗi
v_landmarks[12] = [0.55, 0.3, 0.0]  # Middle tip duỗi
v_landmarks[16] = [0.5, 0.6, 0.0]   # Ring co
v_landmarks[20] = [0.5, 0.65, 0.0]  # Pinky co

res_letters = recognizer.predict_landmarks(
    landmarks_list=v_landmarks,
    multi_hands=None,
    mode="letters"
)
check("Mode 'letters' hoạt động bình thường", isinstance(res_letters, dict) and "sign" in res_letters)
check("Mode 'letters' trả về kết quả định dạng chuẩn", "label" in res_letters and "confidence" in res_letters)

res_digits = recognizer.predict_landmarks(
    landmarks_list=v_landmarks,
    multi_hands=None,
    mode="digits"
)
check("Mode 'digits' hoạt động bình thường", isinstance(res_digits, dict) and "sign" in res_digits)

# Kiểm tra predict_sequence ở mode 'sentence' hoặc mode khác
res_sentence = recognizer.predict_sequence(
    sequence_data=holistic_sample,
    threshold=0.65,
    mode="sentence"
)
check("Mode 'sentence' hoạt động bình thường", res_sentence.get("success") is True)


print("\n" + "=" * 70)
print("TEST 5: Kiểm tra Logging Word Debug với Event Mới (FALLBACK_BLOCKED)")
print("=" * 70)
# Bắt đầu phiên debug giả lập cho từ 'Bạn' (class 60)
session_res = word_debugger.start_session(target_class=60, target_word="Bạn", trial=99)
check("Bắt đầu session debug thành công", session_res.get("success") is True)

# Gửi chuỗi fallback_sample để kích hoạt log chặn
recognizer.predict_sequence(sequence_data=fallback_sample, mode="words")

# Gửi chuỗi holistic thật để kích hoạt log prediction
recognizer.predict_sequence(sequence_data=holistic_sample, mode="words")

status = word_debugger.get_status()
check("Đếm đúng số lần bị chặn (blocked_fallback_count >= 1)", status.get("blocked_fallback_count", 0) >= 1)
check("Đếm đúng số lần dự đoán hợp lệ (seq_count >= 1)", status.get("seq_count", 0) >= 1)

# Kết thúc session
stop_res = word_debugger.stop_session()
check("Kết thúc session debug thành công", stop_res.get("success") is True)

# Dọn dẹp thư mục trial 99 nếu có
trial_dir = stop_res.get("session_dir")
if trial_dir and os.path.exists(trial_dir):
    import shutil
    shutil.rmtree(trial_dir, ignore_errors=True)
    print(f"  [INFO] Đã dọn dẹp thư mục test: {trial_dir}")


print("\n" + "=" * 70)
print(f"TỔNG KẾT: {len(passed_tests)} PASS, {len(failed_tests)} FAIL")
print("=" * 70)
if failed_tests:
    print("Các bài test thất bại:")
    for f in failed_tests:
        print(f"  - {f}")
    sys.exit(1)
else:
    print("Tất cả các bài test tự động cho A01.1 đã PASS 100%!")
    sys.exit(0)
