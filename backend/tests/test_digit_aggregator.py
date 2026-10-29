"""
Bộ kiểm thử Toàn diện cho VSLSequentialDigitAggregator (Vòng 2)
==============================================================
Kiểm chứng 12 yêu cầu kỹ thuật:
1. Ghép 1 rồi 0 thành 10.
2. Ghép 2 rồi 0 thành 20.
3. Ghép 9 rồi 0 thành 90.
4. Một chữ số đứng riêng phát hành đúng sau timeout.
5. Giữ một chữ số liên tục nhiều giây không sinh bản sao lặp lại.
6. Chữ số lặp lại hợp lệ (như 11, 22) khi có động tác thả tay (NEUTRAL) ở giữa.
7. Nhiễu ngắn hạn và NEUTRAL không sinh số giả.
8. Gọi reset() xóa sạch bộ đệm và trạng thái về IDLE.
9. Hoạt động chính xác với timestamp không đều (mô phỏng FPS trồi sụt).
10. Từ chối các nhãn có độ tin cậy thấp hơn min_confidence.
11. Phân biệt rõ kết quả tạm thời (provisional) và đã chốt (committed).
12. Giới hạn số chữ số tối đa (max_digits cap) tự động chốt.
"""

import unittest
from backend.vision.digit_aggregator import (
    VSLSequentialDigitAggregator,
    DigitAggregatorConfig,
    AggregatorState,
)


class TestVSLSequentialDigitAggregator(unittest.TestCase):
    def setUp(self):
        # Cấu hình chuẩn theo yêu cầu: T_hold = 0.4s, T_timeout = 1.5s
        self.config = DigitAggregatorConfig(
            t_hold=0.4,
            t_timeout=1.5,
            min_confidence=0.70,
            max_digits=5,
        )
        self.agg = VSLSequentialDigitAggregator(config=self.config)

    def test_01_composition_1_then_0_creates_10(self):
        """1 rồi 0 tạo 10 sau khi hết thời gian timeout."""
        t = 0.0
        # Bước 1: Ký số 1 ổn định trong 0.4s (8 frames cách nhau 0.05s)
        for _ in range(9):
            out = self.agg.update(sign="1", confidence=0.9, timestamp=t)
            t += 0.05

        self.assertEqual(out.accumulated_digits, ["1"])
        self.assertEqual(out.provisional_number, "1")
        self.assertFalse(out.is_committed)

        # Bước 2: Thả tay sang NEUTRAL trong 0.2s
        for _ in range(4):
            out = self.agg.update(sign="NEUTRAL", confidence=0.0, timestamp=t)
            t += 0.05

        self.assertEqual(out.accumulated_digits, ["1"])
        self.assertEqual(out.provisional_number, "1")

        # Bước 3: Ký số 0 ổn định trong 0.4s
        for _ in range(9):
            out = self.agg.update(sign="0", confidence=0.9, timestamp=t)
            t += 0.05

        self.assertEqual(out.accumulated_digits, ["1", "0"])
        self.assertEqual(out.provisional_number, "10")
        self.assertFalse(out.is_committed)

        # Bước 4: Hạ tay chờ timeout (1.5s)
        # Tại t_timeout, hệ thống phải phát hành số 10
        timeout_fired = False
        committed_val = None
        for _ in range(35):  # 35 * 0.05 = 1.75s > 1.5s
            out = self.agg.update(sign="NEUTRAL", confidence=0.0, timestamp=t)
            t += 0.05
            if out.is_committed:
                timeout_fired = True
                committed_val = out.committed_number
                break

        self.assertTrue(timeout_fired, "Aggregator không kích hoạt timeout để chốt số")
        self.assertEqual(committed_val, "10")

    def test_02_composition_2_then_0_creates_20(self):
        """2 rồi 0 tạo 20."""
        t = 10.0
        # Giữ số 2 trong 0.45s
        for _ in range(10):
            self.agg.update("2", 0.85, timestamp=t)
            t += 0.05
        # Thả tay 0.2s
        for _ in range(4):
            self.agg.update("NEUTRAL", 0.0, timestamp=t)
            t += 0.05
        # Giữ số 0 trong 0.45s
        for _ in range(10):
            out = self.agg.update("0", 0.85, timestamp=t)
            t += 0.05
        self.assertEqual(out.provisional_number, "20")

        # Chờ timeout 1.6s
        committed = None
        for _ in range(35):
            out = self.agg.update("NEUTRAL", 0.0, timestamp=t)
            t += 0.05
            if out.is_committed:
                committed = out.committed_number
                break
        self.assertEqual(committed, "20")

    def test_03_composition_9_then_0_creates_90(self):
        """9 rồi 0 tạo 90."""
        t = 20.0
        for _ in range(10):
            self.agg.update("9", 0.95, timestamp=t)
            t += 0.05
        for _ in range(4):
            self.agg.update("NEUTRAL", 0.0, timestamp=t)
            t += 0.05
        for _ in range(10):
            out = self.agg.update("0", 0.95, timestamp=t)
            t += 0.05
        self.assertEqual(out.provisional_number, "90")

        # Chờ timeout
        committed = None
        for _ in range(35):
            out = self.agg.update("NEUTRAL", 0.0, timestamp=t)
            t += 0.05
            if out.is_committed:
                committed = out.committed_number
                break
        self.assertEqual(committed, "90")

    def test_04_single_digit_committed_after_timeout(self):
        """Một chữ số đứng riêng (ví dụ 7) được phát hành đúng sau timeout."""
        t = 30.0
        # Giữ số 7 trong 0.45s
        for _ in range(10):
            out = self.agg.update("7", 0.88, timestamp=t)
            t += 0.05
        self.assertEqual(out.accumulated_digits, ["7"])
        self.assertFalse(out.is_committed)

        # Hạ tay, chờ timeout
        committed = None
        for _ in range(35):
            out = self.agg.update("NEUTRAL", 0.0, timestamp=t)
            t += 0.05
            if out.is_committed:
                committed = out.committed_number
                break
        self.assertEqual(committed, "7")

    def test_05_holding_single_digit_long_does_not_duplicate(self):
        """Giữ một chữ số liên tục 5 giây KHÔNG sinh thêm chữ số lặp lại."""
        t = 40.0
        # Người dùng giữ tay liên tục ở số 1 suốt 5.0 giây
        commit_count = 0
        committed_values = []
        for _ in range(100):  # 100 * 0.05 = 5.0s
            out = self.agg.update("1", 0.92, timestamp=t)
            t += 0.05
            if out.is_committed:
                commit_count += 1
                committed_values.append(out.committed_number)

        # Chỉ được chốt số 1 đúng 1 lần (sau khi timeout kích hoạt), không lặp thành 11111...
        self.assertEqual(commit_count, 1)
        self.assertEqual(committed_values, ["1"])

    def test_06_repeated_digits_with_release_creates_11(self):
        """Ký số 1, thả tay (NEUTRAL), rồi ký tiếp số 1 -> ghép thành 11 hợp lệ."""
        t = 50.0
        # Chữ số 1 thứ nhất (0.45s)
        for _ in range(10):
            self.agg.update("1", 0.9, timestamp=t)
            t += 0.05

        # Thả tay rõ ràng (0.2s)
        for _ in range(4):
            self.agg.update("NEUTRAL", 0.0, timestamp=t)
            t += 0.05

        # Chữ số 1 thứ hai (0.45s)
        for _ in range(10):
            out = self.agg.update("1", 0.9, timestamp=t)
            t += 0.05

        self.assertEqual(out.accumulated_digits, ["1", "1"])
        self.assertEqual(out.provisional_number, "11")

        # Chờ timeout
        committed = None
        for _ in range(35):
            out = self.agg.update("NEUTRAL", 0.0, timestamp=t)
            t += 0.05
            if out.is_committed:
                committed = out.committed_number
                break
        self.assertEqual(committed, "11")

    def test_07_noise_and_neutral_rejection(self):
        """Ký hiệu thoáng qua không đủ T_hold (0.4s) bị bỏ qua, không tạo số giả."""
        t = 60.0
        # Nhiễu 0.1s của số 4
        for _ in range(2):
            self.agg.update("4", 0.85, timestamp=t)
            t += 0.05
        # Trở lại NEUTRAL
        for _ in range(5):
            out = self.agg.update("NEUTRAL", 0.0, timestamp=t)
            t += 0.05

        self.assertEqual(out.accumulated_digits, [])
        self.assertIsNone(out.provisional_number)
        self.assertFalse(out.is_committed)

    def test_08_reset_clears_all_buffers(self):
        """Hàm reset() đưa trạng thái về IDLE và xóa sạch các chữ số đang gom."""
        t = 70.0
        for _ in range(10):
            self.agg.update("8", 0.9, timestamp=t)
            t += 0.05
        self.assertEqual(len(self.agg.accumulated_digits), 1)

        self.agg.reset()
        self.assertEqual(self.agg.state, AggregatorState.IDLE)
        self.assertEqual(self.agg.accumulated_digits, [])
        self.assertIsNone(self.agg.current_candidate)

    def test_09_irregular_timestamps(self):
        """Hoạt động chuẩn xác khi khoảng cách thời gian giữa các frame không đều."""
        # Giả lập FPS biến động: 16ms, 45ms, 80ms, 20ms... Tổng > 0.4s
        intervals = [0.016, 0.045, 0.080, 0.020, 0.060, 0.050, 0.070, 0.060]
        t = 80.0
        for dt in intervals:
            self.agg.update("3", 0.9, timestamp=t)
            t += dt
        # Frame tại thời điểm t >= 80.401s
        out = self.agg.update("3", 0.9, timestamp=t)

        self.assertEqual(out.accumulated_digits, ["3"])
        self.assertEqual(out.provisional_number, "3")

    def test_10_confidence_threshold_rejection(self):
        """Ký hiệu có confidence thấp hơn min_confidence (0.70) bị xem là NEUTRAL."""
        t = 90.0
        # Ký số 5 nhưng confidence chỉ 0.65 trong 1 giây
        for _ in range(20):
            out = self.agg.update("5", 0.65, timestamp=t)
            t += 0.05

        self.assertEqual(out.accumulated_digits, [])
        self.assertIsNone(out.provisional_number)

    def test_11_provisional_vs_committed_distinction(self):
        """Phân biệt rõ ràng giữa kết quả tạm thời (provisional) và đã chốt (committed)."""
        t = 100.0
        # Trong lúc đang giữ số 4
        out1 = self.agg.update("4", 0.9, timestamp=t)
        self.assertEqual(out1.state, AggregatorState.HOLDING)
        self.assertFalse(out1.is_committed)
        self.assertIsNone(out1.committed_number)

        # Sau khi đủ T_hold
        t += 0.45
        out2 = self.agg.update("4", 0.9, timestamp=t)
        self.assertEqual(out2.provisional_number, "4")
        self.assertFalse(out2.is_committed)
        self.assertIsNone(out2.committed_number)

        # Sau khi hết T_timeout (1.5s)
        t += 1.6
        out3 = self.agg.update("NEUTRAL", 0.0, timestamp=t)
        self.assertTrue(out3.is_committed)
        self.assertEqual(out3.committed_number, "4")

    def test_12_max_digits_cap_auto_commits(self):
        """Đạt giới hạn max_digits (5) tự động chốt mà không cần chờ timeout."""
        t = 110.0
        # Ký liên tiếp 1, 2, 3, 4, 5
        digits = ["1", "2", "3", "4", "5"]
        last_out = None
        for i, d in enumerate(digits):
            for _ in range(9):  # 0.45s giữ
                last_out = self.agg.update(d, 0.9, timestamp=t)
                t += 0.05
            if i < len(digits) - 1:
                for _ in range(4):  # 0.2s thả
                    self.agg.update("NEUTRAL", 0.0, timestamp=t)
                    t += 0.05

        # Ngay khi ký số thứ 5 đạt T_hold, đạt max_digits=5 -> Tự động chốt
        self.assertTrue(last_out.is_committed)
        self.assertEqual(last_out.committed_number, "12345")

    def test_13_non_monotonic_timestamp_safety(self):
        """Bảo vệ an toàn khi timestamp bị nhảy lùi hoặc âm."""
        out1 = self.agg.update("1", 0.9, timestamp=10.0)
        out2 = self.agg.update("1", 0.9, timestamp=8.0)  # Nhảy lùi thời gian
        self.assertIsNotNone(out2)


if __name__ == "__main__":
    unittest.main()
