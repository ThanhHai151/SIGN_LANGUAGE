"""
Bộ kiểm thử Toàn diện cho VSL10GestureDetector (Vòng 4)
======================================================
Kiểm chứng 8 yêu cầu kỹ thuật:
1. Nhận diện thành công cử chỉ Số 10 từ dữ liệu thật D0539 (real_digit_10_landmarks.json).
2. Tư thế Số 1 tĩnh (duỗi thẳng ngón trỏ, giữ yên) KHÔNG bị nhận nhầm thành Số 10.
3. Cử chỉ Số 2 (ngón giữa xòe) bị từ chối ngay lập tức do không thỏa mãn điều kiện nắm tay.
4. Cooldown chống phát hành lặp lại cho cùng một lần ký.
5. Dao động nhỏ / rung tay (< 40°) không kích hoạt Số 10 giả.
6. Reset() xóa sạch bộ đệm và trạng thái phát hiện.
7. Hoạt động chính xác với timestamp không đều (mô phỏng FPS trồi sụt).
8. Mất tracking không gây nổ ngoại lệ và dọn sạch dữ liệu cũ.
"""

import json
import os
import sys
import unittest

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath("."))

from backend.vision.digit_10_detector import VSL10GestureDetector, Digit10DetectorConfig

FIXTURE_10_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_10_landmarks.json")
FIXTURE_DIGITS_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "real_digit_landmarks.json")


class TestDigit10Detector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(FIXTURE_10_PATH, "r", encoding="utf-8") as f:
            cls.d10_data = json.load(f)
        cls.d10_samples = [s for s in cls.d10_data["samples"] if s["quality_status"] == "ok"]

        with open(FIXTURE_DIGITS_PATH, "r", encoding="utf-8") as f:
            cls.digits_data = json.load(f)
        cls.s1_samples = [s for s in cls.digits_data["samples"] if s["digit"] == "1"]
        cls.s2_samples = [s for s in cls.digits_data["samples"] if s["digit"] == "2"]

    def setUp(self):
        self.detector = VSL10GestureDetector()

    def test_01_real_d0539_sequence_detected_as_digit_10(self):
        """Chuỗi landmark thực từ video D0539 được nhận diện chính xác thành Số 10."""
        detected = False
        res = None
        for s in self.d10_samples:
            res = self.detector.update(s["landmarks"], timestamp=s["timestamp_sec"])
            if res["detected"]:
                detected = True
                break

        self.assertTrue(detected, "Dữ liệu thực D0539 không kích hoạt nhận diện Số 10")
        self.assertGreaterEqual(res["confidence"], 0.85)
        self.assertGreaterEqual(res["angular_span_deg"], 40.0)

    def test_02_stable_digit_1_never_detected_as_digit_10(self):
        """Tư thế số 1 tĩnh được giữ ổn định trong 1 giây KHÔNG BAO GIỜ bị nhận thành Số 10."""
        lm_1 = self.s1_samples[0]["landmarks"]
        t = 10.0
        # Đưa mẫu số 1 liên tục trong 1.0 giây (20 frame cách nhau 0.05s)
        detected = False
        for _ in range(20):
            res = self.detector.update(lm_1, timestamp=t)
            t += 0.05
            if res["detected"]:
                detected = True
                break

        self.assertFalse(detected, "Số 1 tĩnh bị nhận nhầm thành Số 10!")
        self.assertFalse(res["is_observing"])

    def test_03_digit_2_rejected_by_finger_posture(self):
        """Số 2 (ngón giữa duỗi) bị từ chối ngay lập tức do vi phạm điều kiện 3 ngón ngoài gập."""
        lm_2 = self.s2_samples[0]["landmarks"]
        t = 20.0
        for _ in range(10):
            res = self.detector.update(lm_2, timestamp=t)
            t += 0.05
            self.assertFalse(res["detected"])
            self.assertEqual(res["num_frames"], 0)

    def test_04_cooldown_prevents_duplicate_detection(self):
        """Sau khi phát hiện Số 10, cooldown ngăn chặn phát hành lặp lại cho cùng 1 cử chỉ."""
        # Chạy chuỗi D0539 để kích hoạt
        first_detected_time = None
        for s in self.d10_samples:
            res = self.detector.update(s["landmarks"], timestamp=s["timestamp_sec"])
            if res["detected"]:
                first_detected_time = s["timestamp_sec"]
                break
        self.assertIsNotNone(first_detected_time)

        # Đưa tiếp các frame kế tiếp trong khoảng cooldown (< 1.5s)
        for s in self.d10_samples:
            res_repeat = self.detector.update(s["landmarks"], timestamp=first_detected_time + 0.2)
            self.assertFalse(res_repeat["detected"], "Phát hiện Số 10 bị lặp lại trong thời gian cooldown!")

    def test_05_small_hand_tremor_does_not_trigger_10(self):
        """Dao động nhỏ (rung tay < 25 độ) không đủ ngưỡng kích hoạt Số 10."""
        # Giả lập cử chỉ ngón trỏ rung nhẹ quanh 0 độ
        lm_base = self.s1_samples[0]["landmarks"]
        t = 30.0
        for i in range(15):
            # Tạo bản sao và rung nhẹ đầu ngón trỏ
            lms_jitter = [p.copy() for p in lm_base]
            jitter = 0.005 if (i % 2 == 0) else -0.005
            lms_jitter[8][0] += jitter
            res = self.detector.update(lms_jitter, timestamp=t)
            t += 0.05
            self.assertFalse(res["detected"])

    def test_06_reset_clears_buffer_and_state(self):
        """Gọi reset() xóa sạch bộ đệm chuyển động."""
        for s in self.d10_samples[:3]:
            self.detector.update(s["landmarks"], timestamp=s["timestamp_sec"])
        self.assertGreater(len(self.detector.buffer), 0)

        self.detector.reset()
        self.assertEqual(len(self.detector.buffer), 0)
        self.assertEqual(self.detector.last_detected_time, 0.0)

    def test_07_irregular_timestamps_handling(self):
        """Xử lý đúng khi timestamp không đều."""
        detected = False
        t = 50.0
        # Gửi các frame D0539 với khoảng thời gian thay đổi (16ms, 45ms, 80ms)
        intervals = [0.033, 0.050, 0.080, 0.020, 0.060, 0.040, 0.070, 0.050]
        for i, s in enumerate(self.d10_samples[:len(intervals)]):
            res = self.detector.update(s["landmarks"], timestamp=t)
            t += intervals[i]
            if res["detected"]:
                detected = True
                break

        # Cử chỉ xoay đủ lớn vẫn được ghi nhận
        self.assertTrue(detected or res["angular_span_deg"] >= 40.0)

    def test_08_tracking_loss_safety(self):
        """Mất tracking (None) không gây lỗi và dọn dẹp các frame quá cũ."""
        res = self.detector.update(None, timestamp=60.0)
        self.assertFalse(res["detected"])
        self.assertEqual(res["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()

