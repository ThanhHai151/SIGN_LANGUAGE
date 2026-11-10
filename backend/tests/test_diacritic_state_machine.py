"""
Kiểm thử Unit Tests cho DiacriticStateMachine (Task 7C).
=========================================================
Bao phủ toàn diện 15 trường hợp kiểm thử logic:
1. A giữ yên -> A.
2. A mở hai ngón V -> Ă.
3. A nâng tay vẽ nón -> Â.
4. E nâng tay vẽ nón -> Ê.
5. O nâng tay vẽ nón -> Ô.
6. O thay đổi hình thái (vẩy râu ngón út) -> Ơ.
7. U duỗi ngón út -> Ư.
8. D gập góc ngón trỏ -> Đ (xung gập ngón trỏ).
9. Chặn pha đưa tay (ENTRY Guard): di chuyển nhanh lúc đầu không kích hoạt dấu.
10. Chặn pha hạ tay (EXIT Guard): cổ tay rơi nhanh không kích hoạt dấu.
11. Mất landmarks / lỗi đầu vào: trả về NEUTRAL an toàn, reset trạng thái.
12. Đổi nhóm chữ nền (Base switch): A đổi sang D tự động reset và không xung đột chéo.
13. Chuyển động dưới ngưỡng: giữ nguyên chữ nền.
14. Timestamp nhảy lùi: reset an toàn, không xác nhận dấu.
15. Chữ ngoài 5 nhóm nền (B, C, G...): trả kết quả gốc tức thì, không xử lý FSM.
"""

import sys
import os
import unittest

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.vision.diacritic_state_machine import (
    DiacriticStateMachine,
    STATE_IDLE,
    STATE_BASE_CANDIDATE,
    STATE_BASE_STABLE,
    STATE_DIACRITIC_CANDIDATE,
    STATE_DIACRITIC_CONFIRMED,
    STATE_COOLDOWN,
)

def make_dummy_landmarks(
    wrist_y=0.55,
    index_tip_y=0.35,
    middle_tip_y=0.35,
    pinky_tip_y=0.45,
    index_tip_x=0.48,
    middle_tip_x=0.52,
    pinky_tip_x=0.51
):
    """Tạo bộ 21 landmarks giả lập có thể điều chỉnh tọa độ các ngón."""
    lm = []
    # 0: Wrist
    lm.append([0.50, wrist_y, 0.0])
    # 1-4: Thumb
    lm.append([0.46, wrist_y - 0.05, 0.0])
    lm.append([0.43, wrist_y - 0.10, 0.0])
    lm.append([0.42, wrist_y - 0.14, 0.0])
    lm.append([0.45, wrist_y - 0.16, 0.0]) # Thumb tip (4)
    # 5-8: Index
    lm.append([0.47, wrist_y - 0.15, 0.0]) # Index MCP (5)
    lm.append([0.47, wrist_y - 0.22, 0.0]) # Index PIP (6)
    lm.append([0.47, wrist_y - 0.28, 0.0]) # Index DIP (7)
    lm.append([index_tip_x, index_tip_y, 0.0]) # Index TIP (8)
    # 9-12: Middle
    lm.append([0.50, wrist_y - 0.16, 0.0]) # Middle MCP (9)
    lm.append([0.50, wrist_y - 0.23, 0.0])
    lm.append([0.50, wrist_y - 0.29, 0.0])
    lm.append([middle_tip_x, middle_tip_y, 0.0]) # Middle TIP (12)
    # 13-16: Ring
    lm.append([0.53, wrist_y - 0.15, 0.0])
    lm.append([0.53, wrist_y - 0.20, 0.0])
    lm.append([0.53, wrist_y - 0.24, 0.0])
    lm.append([0.52, wrist_y - 0.20, 0.0]) # Ring TIP (16)
    # 17-20: Pinky
    lm.append([0.55, wrist_y - 0.13, 0.0])
    lm.append([0.55, wrist_y - 0.17, 0.0])
    lm.append([0.55, wrist_y - 0.20, 0.0])
    lm.append([pinky_tip_x, pinky_tip_y, 0.0]) # Pinky TIP (20)
    return lm

class TestDiacriticStateMachine(unittest.TestCase):
    def setUp(self):
        self.fsm = DiacriticStateMachine(
            min_base_stable_frames=4,
            min_base_stable_sec=0.10,
            cooldown_sec=0.30
        )

    # 1. Test A giữ yên -> A
    def test_case_1_base_a_static_hold(self):
        """Giữ thế tay A ổn định mà không có chuyển động -> Trả về A."""
        t = 1.0
        for _ in range(6):
            lm = make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, middle_tip_y=0.45)
            res = self.fsm.update("A", 0.95, lm, timestamp=t)
            t += 0.04
            self.assertEqual(res["final_label"], "A")
            self.assertEqual(res["base_label"], "A")

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)
        self.assertIsNotNone(self.fsm.baseline)

    # 2. Test A mở hai ngón V -> Ă
    def test_case_2_a_open_v_to_a_breve(self):
        """Từ thế tay A, mở xòe 2 ngón trỏ và giữa -> Xác nhận Ă."""
        t = 1.0
        for _ in range(5):
            lm_a = make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.49, middle_tip_x=0.51, index_tip_y=0.45, middle_tip_y=0.45)
            self.fsm.update("A", 0.95, lm_a, timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # Mở xòe 2 ngón (khoảng cách tip x tăng mạnh: 0.38 và 0.62)
        res1 = self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.38, middle_tip_x=0.62, index_tip_y=0.30, middle_tip_y=0.30), timestamp=t)
        t += 0.04
        self.assertEqual(res1["state"], STATE_DIACRITIC_CANDIDATE)

        res2 = self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.38, middle_tip_x=0.62, index_tip_y=0.30, middle_tip_y=0.30), timestamp=t)
        self.assertEqual(res2["final_label"], "Ă")
        self.assertEqual(res2["state"], STATE_DIACRITIC_CONFIRMED)
        self.assertEqual(res2["source"], "diacritic_fsm")

    # 3. Test A nâng tay vẽ nón -> Â
    def test_case_3_a_lift_to_a_circumflex(self):
        """Từ thế tay A, nâng tay lên cao vẽ nón -> Xác nhận Â."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.56, index_tip_y=0.45, middle_tip_y=0.45), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # Nâng cổ tay lên (y giảm từ 0.56 xuống 0.50, delta=0.06 > 0.035, vy < -0.10)
        res1 = self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.51, index_tip_y=0.40, middle_tip_y=0.40), timestamp=t)
        t += 0.04
        res2 = self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.50, index_tip_y=0.39, middle_tip_y=0.39), timestamp=t)

        self.assertEqual(res2["final_label"], "Â")
        self.assertEqual(res2["state"], STATE_DIACRITIC_CONFIRMED)

    # 4. Test E nâng tay vẽ nón -> Ê
    def test_case_4_e_lift_to_e_circumflex(self):
        """Từ thế tay E, nâng tay lên cao vẽ nón -> Xác nhận Ê."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("E", 0.95, make_dummy_landmarks(wrist_y=0.57), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        res1 = self.fsm.update("E", 0.95, make_dummy_landmarks(wrist_y=0.52), timestamp=t)
        t += 0.04
        res2 = self.fsm.update("E", 0.95, make_dummy_landmarks(wrist_y=0.51), timestamp=t)

        self.assertEqual(res2["final_label"], "Ê")
        self.assertEqual(res2["state"], STATE_DIACRITIC_CONFIRMED)

    # 5. Test O nâng tay vẽ nón -> Ô
    def test_case_5_o_lift_to_o_circumflex(self):
        """Từ thế tay O tròn khép, nâng tay lên cao vẽ nón -> Xác nhận Ô."""
        t = 1.0
        # O khép tròn: các ngón tụ lại gần nhau (index 0.48, pinky 0.51)
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.58, index_tip_x=0.48, pinky_tip_x=0.51, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # Nâng lên tạo nón mà vẫn giữ O khép tròn
        res1 = self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.53, index_tip_x=0.48, pinky_tip_x=0.51, index_tip_y=0.40, pinky_tip_y=0.40), timestamp=t)
        t += 0.04
        res2 = self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.52, index_tip_x=0.48, pinky_tip_x=0.51, index_tip_y=0.39, pinky_tip_y=0.39), timestamp=t)

        self.assertEqual(res2["final_label"], "Ô")
        self.assertEqual(res2["state"], STATE_DIACRITIC_CONFIRMED)

    # 6. Test O vẩy râu ngón út -> Ơ
    def test_case_6_o_horn_to_o_horn(self):
        """Từ thế tay O tròn khép, ngón út vươn tách xa -> Xác nhận Ơ."""
        t = 1.0
        for _ in range(5):
            # O khép tròn: ngón út ở gần ngón trỏ (x=0.50, y=0.45)
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.48, pinky_tip_x=0.50, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # Ngón út vươn tách xa ra ngoài (pinky_tip_x=0.68, y=0.30)
        lm_horn = make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.48, pinky_tip_x=0.68, index_tip_y=0.45, pinky_tip_y=0.30)

        res1 = self.fsm.update("O", 0.95, lm_horn, timestamp=t)
        t += 0.04
        res2 = self.fsm.update("O", 0.95, lm_horn, timestamp=t)

        self.assertEqual(res2["final_label"], "Ơ")
        self.assertEqual(res2["state"], STATE_DIACRITIC_CONFIRMED)

    # 7. Test U duỗi ngón út -> Ư
    def test_case_7_u_pinky_extend_to_u_horn(self):
        """Từ thế tay U (ngón út gập), ngón út duỗi thẳng -> Xác nhận Ư."""
        t = 1.0
        for _ in range(5):
            # U: ngón út gập (pinky_tip_y=0.48, gần wrist)
            self.fsm.update("U", 0.95, make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.48), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # Ngón út duỗi thẳng lên (pinky_tip_y=0.30 < middle_joint_y=0.38)
        lm_u_horn = make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.30)
        res1 = self.fsm.update("U", 0.95, lm_u_horn, timestamp=t)
        t += 0.04
        res2 = self.fsm.update("U", 0.95, lm_u_horn, timestamp=t)

        self.assertEqual(res2["final_label"], "Ư")
        self.assertEqual(res2["state"], STATE_DIACRITIC_CONFIRMED)

    # 8. Test D gập góc ngón trỏ -> Đ
    def test_case_8_d_stroke_to_d_stroke(self):
        """Từ thế tay D (ngón trỏ thẳng), ngón trỏ gập đốt giữa -> Xác nhận Đ."""
        t = 1.0
        for _ in range(5):
            # D: ngón trỏ duỗi thẳng
            self.fsm.update("D", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.25), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # Ngón trỏ gập gãy đốt giữa (tọa độ ngón trỏ biến dạng gập góc)
        lm_d_hook = make_dummy_landmarks(wrist_y=0.55)
        lm_d_hook[6] = [0.47, 0.45, 0.0]
        lm_d_hook[7] = [0.55, 0.45, 0.0]
        lm_d_hook[8] = [0.55, 0.40, 0.0]

        # Đ chỉ cần xung 1 frame là được xác nhận
        res = self.fsm.update("D", 0.95, lm_d_hook, timestamp=t)
        self.assertEqual(res["final_label"], "Đ")
        self.assertEqual(res["state"], STATE_DIACRITIC_CONFIRMED)

    # 9. Chặn pha đưa tay (ENTRY Guard)
    def test_case_9_entry_guard_rejects_premature_diacritic(self):
        """Bàn tay mới vọt lên từ ngoài khung hình (vy rất lớn) không được kích hoạt dấu."""
        t = 1.0
        # Frame 1: Cổ tay ở dưới mép (y=0.85)
        self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.85), timestamp=t)
        t += 0.04
        # Frame 2: Cổ tay vọt nhanh lên y=0.55 (vy = (0.55 - 0.85)/0.04 = -7.5/s)
        res = self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=t)

        self.assertNotEqual(res["state"], STATE_DIACRITIC_CONFIRMED)
        self.assertIn("ENTRY Guard", res["reason"])

    # 10. Chặn pha hạ tay (EXIT Guard)
    def test_case_10_exit_guard_rejects_dropping_hand(self):
        """Khi cổ tay rơi nhanh xuống (vy > 0.65/s) hoặc y > 0.75, khóa nhận diện dấu."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # Cổ tay rơi mạnh xuống y=0.78
        res = self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.78), timestamp=t)
        self.assertEqual(res["state"], STATE_IDLE)
        self.assertIn("EXIT Guard", res["reason"])

    # 11. Mất landmarks -> Trả về NEUTRAL, reset FSM
    def test_case_11_lost_landmarks_safe_neutral(self):
        """Khi mất bàn tay (None, rỗng, NaN), FSM trả về NEUTRAL và reset về IDLE."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # 1. Landmarks = None
        res_none = self.fsm.update("A", 0.95, None, timestamp=t)
        self.assertEqual(res_none["final_label"], "NEUTRAL")
        self.assertEqual(res_none["state"], STATE_IDLE)

        # 2. Landmarks chứa NaN
        nan_lm = make_dummy_landmarks(wrist_y=0.55)
        nan_lm[0][0] = float("nan")
        res_nan = self.fsm.update("A", 0.95, nan_lm, timestamp=t + 0.04)
        self.assertEqual(res_nan["final_label"], "NEUTRAL")

    # 12. Đổi nhóm chữ nền (Base switch) -> Reset đúng
    def test_case_12_base_switch_resets_cleanly(self):
        """Đang theo dõi A mà người dùng đổi sang D, FSM lập tức reset và không tạo xung đột chéo."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)
        self.assertEqual(self.fsm.current_base, "A")

        # Đổi sang D (sau 2 frames xác nhận chuyển chữ - Task 7E.4B)
        self.fsm.update("D", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
        t += 0.04
        res_d = self.fsm.update("D", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
        # FSM phải reset base cũ
        self.assertNotEqual(self.fsm.current_base, "A")
        self.assertEqual(res_d["final_label"], "D")

    # 13. Chuyển động dưới ngưỡng -> Giữ nguyên chữ nền
    def test_case_13_sub_threshold_motion_keeps_base(self):
        """Chuyển động rung nhẹ dưới ngưỡng (vd lift=0.015 < 0.035) vẫn giữ nguyên chữ A."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.56, index_tip_y=0.45, middle_tip_y=0.45), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # Rung nhẹ lên y=0.545 (lift=0.015, chưa đạt ngưỡng 0.035)
        res = self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.545, index_tip_y=0.435, middle_tip_y=0.435), timestamp=t)
        self.assertEqual(res["final_label"], "A")
        self.assertEqual(res["state"], STATE_BASE_STABLE)

    # 14. Timestamp nhảy lùi -> Reset an toàn
    def test_case_14_backwards_timestamp_safe_reset(self):
        """Timestamp đột ngột nhảy lùi (lỗi hệ thống client), FSM reset an toàn."""
        self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=10.0)
        self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=10.04)

        # Nhảy lùi về 5.0
        res = self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=5.0)
        self.assertEqual(res["state"], STATE_BASE_CANDIDATE)
        self.assertEqual(self.fsm.candidate_base_count, 1)

    # 15. Chữ ngoài 5 nhóm nền (B, C, G...) không bị tác động
    def test_case_15_pure_static_letters_passthrough(self):
        """Chữ cái tĩnh thuần túy (B, C, G, H...) không thuộc 5 nhóm nền thì đi thẳng qua, FSM ở IDLE."""
        non_base_letters = ["B", "C", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "X", "Y"]
        for letter in non_base_letters:
            res = self.fsm.update(letter, 0.98, make_dummy_landmarks(), timestamp=1.0, source="geometric_rule")
            self.assertEqual(res["final_label"], letter)
            self.assertEqual(res["state"], STATE_IDLE)
            self.assertIsNone(res["diacritic_detected"])

    # 16. (Task 7E.3) Transient NEUTRAL bảo toàn baseline
    def test_case_16_transient_neutral_preserves_baseline(self):
        """Frame NEUTRAL tạm thời không làm mất baseline của FSM đang ở BASE_STABLE."""
        t = 1.0
        # Dáng O khép tròn: index_tip_y=0.45, pinky_tip_y=0.45
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04
        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)
        self.assertTrue(self.fsm.baseline is not None)

        # Frame 6: Transient NEUTRAL với landmarks hợp lệ
        res_transient = self.fsm.update_transient(make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
        self.assertEqual(res_transient["final_label"], "NEUTRAL")
        self.assertEqual(res_transient["state"], STATE_BASE_STABLE)
        self.assertTrue(self.fsm.baseline is not None)
        self.assertEqual(self.fsm.current_base, "O")

        # Frame 7: O quay lại -> FSM vẫn ở BASE_STABLE, không cần học lại từ đầu
        t += 0.04
        res_back = self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
        self.assertEqual(res_back["state"], STATE_BASE_STABLE)
        self.assertEqual(res_back["final_label"], "O")

    # 17. (Task 7E.3) Transient NEUTRAL không tạo nhãn nền giả hay confidence giả
    def test_case_17_transient_neutral_does_not_fabricate_label(self):
        """Frame transient NEUTRAL không được tự gán nhãn nền giả hay confidence giả."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("A", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
            t += 0.04

        # Frame transient
        res = self.fsm.update_transient(make_dummy_landmarks(wrist_y=0.55), timestamp=t)
        self.assertEqual(res["final_label"], "NEUTRAL")
        self.assertEqual(res["confidence"], 0.0)
        self.assertEqual(res["source"], "none")

    # 18. (Task 7E.3) Phát hiện dấu thành công ngay trong pha transient
    def test_case_18_transient_neutral_detects_diacritic(self):
        """Trong pha transient NEUTRAL, chuyển động vẽ nón vẫn được phát hiện và xác nhận."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.58, index_tip_x=0.48, pinky_tip_x=0.51, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        # Nâng tay lên vẽ nón trong khi nhãn phân loại tĩnh là NEUTRAL (vy <= -0.10)
        lm_lift1 = make_dummy_landmarks(wrist_y=0.53, index_tip_x=0.48, pinky_tip_x=0.51, index_tip_y=0.40, pinky_tip_y=0.40)
        res1 = self.fsm.update_transient(lm_lift1, timestamp=t)
        t += 0.04
        lm_lift2 = make_dummy_landmarks(wrist_y=0.52, index_tip_x=0.48, pinky_tip_x=0.51, index_tip_y=0.39, pinky_tip_y=0.39)
        res2 = self.fsm.update_transient(lm_lift2, timestamp=t)

        self.assertEqual(res2["state"], STATE_DIACRITIC_CONFIRMED)
        self.assertEqual(res2["final_label"], "Ô")
        self.assertEqual(res2["diacritic_detected"], "Ô")

    # 19. (Task 7E.3) Transient kéo dài quá max_diacritic_search_sec -> Reset FSM
    def test_case_19_transient_timeout_resets_fsm(self):
        """Nếu nhãn NEUTRAL kéo dài quá 0.65s mà không có dấu, FSM phải reset về IDLE."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        # Nhảy thời gian quá max_diacritic_search_sec (0.65s)
        t += 0.70
        res = self.fsm.update_transient(make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
        self.assertEqual(res["state"], STATE_IDLE)
        self.assertIsNone(self.fsm.baseline)

    # 20. (Task 7E.3) Mất landmarks thực sự trong transient -> Reset lập tức
    def test_case_20_transient_lost_landmarks_resets_immediately(self):
        """Mất landmarks thực sự (None hoặc rỗng) trong pha transient phải reset FSM ngay."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        res = self.fsm.update_transient(None, timestamp=t)
        self.assertEqual(res["state"], STATE_IDLE)
        self.assertIsNone(self.fsm.baseline)

    # 21. (Task 7E.3) Chuyển rõ ràng sang chữ tĩnh khác -> Reset lập tức
    def test_case_21_switch_to_other_letter_resets_immediately(self):
        """Khi người ký đổi từ O sang B (chữ tĩnh ngoài nhóm), FSM phải reset lập tức."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        # Ký chữ B (sau 2 frames xác nhận chuyển chữ - Task 7E.4B)
        self.fsm.update("B", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
        t += 0.04
        res = self.fsm.update("B", 0.95, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
        self.assertEqual(res["state"], STATE_IDLE)
        self.assertIsNone(self.fsm.baseline)
        self.assertEqual(res["final_label"], "B")

    # 22. (Task 7E.4A) O đang BASE_STABLE -> 1 frame Rule I không làm mất baseline
    def test_case_22_o_stable_competing_rule_i_preserves_baseline(self):
        """Khi FSM đang ở BASE_STABLE với O, 1 frame Rule I xuất hiện không được reset baseline."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)
        self.assertTrue(self.fsm.baseline is not None)

        # 1 frame Rule I xuất hiện (ngón út chưa đủ tách xa để thành Ơ)
        lm_competing = make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45)
        res = self.fsm.update_competing("I", 0.95, lm_competing, timestamp=t)

        self.assertEqual(res["state"], STATE_BASE_STABLE)
        self.assertEqual(res["final_label"], "NEUTRAL")
        self.assertTrue(res.get("suppressed"))
        self.assertTrue(self.fsm.baseline is not None)
        self.assertEqual(self.fsm.current_base, "O")

    # 23. (Task 7E.4A) U đang BASE_STABLE -> Rule I xuất hiện khi ngón út mở -> FSM xác nhận Ư
    def test_case_23_u_stable_competing_rule_i_detects_u_horn(self):
        """Khi U đang ổn định, ngón út mở tạo râu Ư khiến Rule I MATCH -> FSM vẫn xác nhận Ư thành công."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("U", 0.95, make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # Ngón út duỗi thẳng (pinky_curl <= 0.35) làm Rule I MATCH
        lm_u_horn = make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.30)
        res1 = self.fsm.update_competing("I", 0.95, lm_u_horn, timestamp=t)
        t += 0.04
        res2 = self.fsm.update_competing("I", 0.95, lm_u_horn, timestamp=t)

        self.assertEqual(res2["state"], STATE_DIACRITIC_CONFIRMED)
        self.assertEqual(res2["final_label"], "Ư")
        self.assertEqual(res2["diacritic_detected"], "Ư")

    # 24. (Task 7E.4A) Ư vừa xác nhận (COOLDOWN) -> Rule I xuất hiện -> Không bị I cướp nhãn
    def test_case_24_confirmed_u_horn_cooldown_protected_from_rule_i(self):
        """Sau khi Ư được xác nhận, trong suốt cooldown, các frame Rule I kế tiếp không được ghi đè Ư."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("U", 0.95, make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        # Xác nhận Ư
        lm_u_horn = make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.30)
        self.fsm.update_competing("I", 0.95, lm_u_horn, timestamp=t)
        t += 0.04
        res_conf = self.fsm.update_competing("I", 0.95, lm_u_horn, timestamp=t)
        self.assertEqual(res_conf["state"], STATE_DIACRITIC_CONFIRMED)

        # Frame tiếp theo vẫn giữ ngón út duỗi -> Rule I MATCH
        t += 0.04
        res_cool = self.fsm.update_competing("I", 0.95, lm_u_horn, timestamp=t)
        self.assertEqual(res_cool["state"], STATE_COOLDOWN)
        self.assertEqual(res_cool["final_label"], "Ư")
        self.assertEqual(res_cool["diacritic_detected"], "Ư")

    # 25. (Task 7E.4A) Người ký chủ động chuyển sang chữ I -> Nhận diện I sau 3 frames
    def test_case_25_intentional_switch_to_i_after_consecutive_frames(self):
        """Nếu người ký giữ nguyên dáng chữ I liên tiếp 3 frames mà không tạo dấu, FSM reset và nhận I."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        # Giữ dáng chữ I (không tạo Ơ)
        lm_i = make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45)

        # Frame 1: chặn tạm thời
        r1 = self.fsm.update_competing("I", 0.95, lm_i, timestamp=t)
        self.assertEqual(r1["final_label"], "NEUTRAL")
        self.assertEqual(r1["state"], STATE_BASE_STABLE)

        # Frame 2: chặn tạm thời
        t += 0.04
        r2 = self.fsm.update_competing("I", 0.95, lm_i, timestamp=t)
        self.assertEqual(r2["final_label"], "NEUTRAL")

        # Frame 3: đạt max_competing_frames (3) -> Chủ động chuyển sang chữ I!
        t += 0.04
        r3 = self.fsm.update_competing("I", 0.95, lm_i, timestamp=t)
        self.assertEqual(r3["state"], STATE_IDLE)
        self.assertEqual(r3["final_label"], "I")
        self.assertEqual(r3["confidence"], 0.95)
        self.assertIsNone(self.fsm.baseline)

    # 26. (Task 7E.4A) A đang BASE_STABLE -> Rule V cạnh tranh -> FSM xác nhận Ă
    def test_case_26_a_stable_competing_rule_v_detects_a_breve(self):
        """Khi A đang BASE_STABLE, mở 2 ngón tay làm dấu Ă (Rule V MATCH) -> FSM xác nhận Ă thành công."""
        t = 1.0
        for _ in range(5):
            lm_a = make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.49, middle_tip_x=0.51, index_tip_y=0.45, middle_tip_y=0.45)
            self.fsm.update("A", 0.95, lm_a, timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)

        # Mở ngón trỏ và ngón giữa (Ă) khiến Rule V MATCH
        lm_breve = make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.38, middle_tip_x=0.62, index_tip_y=0.25, middle_tip_y=0.25)
        res1 = self.fsm.update_competing("V", 0.95, lm_breve, timestamp=t)
        t += 0.04
        res2 = self.fsm.update_competing("V", 0.95, lm_breve, timestamp=t)

        self.assertEqual(res2["state"], STATE_DIACRITIC_CONFIRMED)
        self.assertEqual(res2["final_label"], "Ă")
        self.assertEqual(res2["diacritic_detected"], "Ă")

    # 27. (Task 7E.4A) Hết timeout tìm kiếm dấu -> Kết thúc ưu tiên FSM và nhận diện I
    def test_case_27_timeout_terminates_competing_priority(self):
        """Khi hết thời gian tìm kiếm dấu (0.65s), frame Rule I xuất hiện sẽ lập tức được nhận diện là I."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        # Nhảy thời gian quá 0.65s
        t += 0.70
        lm_i = make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45)
        res = self.fsm.update_competing("I", 0.95, lm_i, timestamp=t)

        self.assertEqual(res["state"], STATE_IDLE)
        self.assertEqual(res["final_label"], "I")
        self.assertIsNone(self.fsm.baseline)

    # 28. (Task 7E.4B) O đang BASE_STABLE -> 1 frame E xuất hiện thoáng qua -> O quay lại: không mất baseline
    def test_case_28_o_accumulating_e_glitch_1_frame_preserves_o(self):
        """O đang BASE_STABLE, frame E thoáng qua (1 frame) không reset FSM, khi O quay lại baseline O được bảo toàn."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)
        self.assertEqual(self.fsm.current_base, "O")
        self.assertTrue(self.fsm.baseline is not None)

        # Frame thoáng qua nhận diện thành E (1 frame duy nhất)
        res_e = self.fsm.update("E", 0.90, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
        self.assertEqual(res_e["final_label"], "E")  # Vẫn xuất nhãn E tức thì cho người dùng
        self.assertEqual(self.fsm.current_base, "O")  # Baseline O vẫn được giữ
        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)
        self.assertTrue(self.fsm.baseline is not None)

        # Frame tiếp theo O quay trở lại
        t += 0.04
        res_o = self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
        self.assertEqual(res_o["final_label"], "O")
        self.assertEqual(self.fsm.current_base, "O")
        self.assertIsNone(self.fsm.switch_candidate_base)
        self.assertEqual(self.fsm.switch_candidate_count, 0)

    # 29. (Task 7E.4B) O chuyển sang E liên tiếp đủ điều kiện -> Kết thúc nhóm O và chuyển chữ đúng
    def test_case_29_o_to_e_consecutive_frames_switches_base(self):
        """Khi người ký thực sự chuyển từ O sang E liên tiếp (>= min_switch_confirm_frames), FSM kết thúc O và chuyển sang E."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        self.assertEqual(self.fsm.current_base, "O")

        # Frame 1 của E: bắt đầu ghi nhận candidate
        res_e1 = self.fsm.update("E", 0.92, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
        self.assertEqual(res_e1["final_label"], "E")
        self.assertEqual(self.fsm.switch_candidate_base, "E")
        self.assertEqual(self.fsm.switch_candidate_count, 1)

        # Frame 2 của E: đủ 2 frames xác nhận chuyển chữ -> reset baseline O và chuyển sang E
        t += 0.04
        res_e2 = self.fsm.update("E", 0.92, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
        self.assertEqual(res_e2["final_label"], "E")
        self.assertNotEqual(self.fsm.current_base, "O")

    # 30. (Task 7E.4B) Quá thời gian tìm kiếm dấu (timeout > 0.65s) -> Reset FSM ngay khi có chữ khác
    def test_case_30_o_glitch_timeout_resets_fsm(self):
        """Nếu đã vượt quá max_diacritic_search_sec kể từ lúc đạt BASE_STABLE, frame chữ khác sẽ reset FSM ngay."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        # Thời gian trôi qua 0.70s (> 0.65s)
        t += 0.70
        res_e = self.fsm.update("E", 0.90, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
        self.assertNotEqual(self.fsm.current_base, "O")
        self.assertIsNone(self.fsm.baseline)

    # 31. (Task 7E.4B) Nhóm U và A hoạt động hoàn toàn bình thường
    def test_case_31_u_and_a_unaffected_by_switch_tolerance(self):
        """Nhóm U và A không bị ảnh hưởng tiêu cực bởi cơ chế dung sai chuyển nhãn."""
        t = 1.0
        # Nhóm U: tạo Ư bằng duỗi ngón út (U base có pinky_tip_y=0.48)
        for _ in range(5):
            self.fsm.update("U", 0.95, make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.48), timestamp=t)
            t += 0.04
        self.assertEqual(self.fsm.state, STATE_BASE_STABLE)
        self.assertEqual(self.fsm.current_base, "U")

        # Duỗi ngón út lên pinky_tip_y=0.30
        lm_u_horn = make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.30)
        res_u1 = self.fsm.update("U", 0.95, lm_u_horn, timestamp=t)
        t += 0.04
        res_u2 = self.fsm.update("U", 0.95, lm_u_horn, timestamp=t)
        self.assertEqual(res_u2["final_label"], "Ư")
        self.assertEqual(res_u2["state"], STATE_DIACRITIC_CONFIRMED)

    # 32. (Task 7E.4B) Mất landmarks thật sự trong lúc có candidate switch -> Reset lập tức
    def test_case_32_lost_landmarks_during_switch_candidate_resets_immediately(self):
        """Nếu landmarks bị mất thật sự (None) thì FSM phải reset lập tức, không chờ đợi frames."""
        t = 1.0
        for _ in range(5):
            self.fsm.update("O", 0.95, make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), timestamp=t)
            t += 0.04

        # Frame 1 của E
        self.fsm.update("E", 0.90, make_dummy_landmarks(wrist_y=0.55), timestamp=t)
        self.assertEqual(self.fsm.switch_candidate_count, 1)

        # Mất landmarks
        t += 0.04
        res_none = self.fsm.update("E", 0.90, None, timestamp=t)
        self.assertEqual(res_none["state"], STATE_IDLE)
        self.assertIsNone(self.fsm.baseline)
        self.assertIsNone(self.fsm.switch_candidate_base)
        self.assertEqual(self.fsm.switch_candidate_count, 0)


if __name__ == "__main__":
    unittest.main()
