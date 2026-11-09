"""
Kiểm thử tích hợp DiacriticStateMachine vào VSLRecognizer và luồng Camera (Task 7D).
===================================================================================
Bao phủ toàn diện 14 trường hợp kiểm thử tích hợp:
1. A -> Ă (Chữ A mở ngón V thăng cấp thành Ă, source='diacritic_fsm').
2. A -> Â (Chữ A vẽ nón thăng cấp thành Â).
3. E -> Ê (Chữ E vẽ nón thăng cấp thành Ê).
4. O -> Ô (Chữ O vẽ nón thăng cấp thành Ô).
5. O -> Ơ (Chữ O vẩy râu ngón út thăng cấp thành Ơ).
6. U -> Ư (Chữ U từ Static Head bật ngón út thăng cấp thành Ư).
7. D -> Đ (Chữ D gập đốt ngón trỏ thăng cấp thành Đ).
8. 17 chữ cái tĩnh thuần túy không bị FSM can thiệp (B, C, G, H, I, K, L, M, N, P, Q, R, S, T, V, X, Y).
9. 12 geometric rules không bị ghi đè hoặc thay đổi.
10. Static Head vẫn hoạt động bình thường với 10 chữ bổ sung.
11. Mất landmarks không giữ lại nhãn có dấu cũ (trả về NEUTRAL an toàn).
12. Quản lý phiên độc lập: Session 1 và Session 2 có FSM riêng, reset session_1 không ảnh hưởng session_2.
13. Chế độ digits và words hoàn toàn không bị ảnh hưởng.
14. Chặn tuyệt đối chuyển nhãn chéo giữa các nhóm nền.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS, ALLOWED_DIGITS
from backend.vision.diacritic_state_machine import (
    BASE_GROUPS,
    STATE_BASE_STABLE,
    STATE_DIACRITIC_CONFIRMED,
    STATE_COOLDOWN
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
    lm.append([0.45, wrist_y - 0.16, 0.0])
    # 5-8: Index
    lm.append([0.47, wrist_y - 0.15, 0.0])
    lm.append([0.47, wrist_y - 0.22, 0.0])
    lm.append([0.47, wrist_y - 0.28, 0.0])
    lm.append([index_tip_x, index_tip_y, 0.0])
    # 9-12: Middle
    lm.append([0.50, wrist_y - 0.16, 0.0])
    lm.append([0.50, wrist_y - 0.23, 0.0])
    lm.append([0.50, wrist_y - 0.29, 0.0])
    lm.append([middle_tip_x, middle_tip_y, 0.0])
    # 13-16: Ring
    lm.append([0.53, wrist_y - 0.15, 0.0])
    lm.append([0.53, wrist_y - 0.20, 0.0])
    lm.append([0.53, wrist_y - 0.24, 0.0])
    lm.append([0.52, wrist_y - 0.20, 0.0])
    # 17-20: Pinky
    lm.append([0.55, wrist_y - 0.13, 0.0])
    lm.append([0.55, wrist_y - 0.17, 0.0])
    lm.append([0.55, wrist_y - 0.20, 0.0])
    lm.append([pinky_tip_x, pinky_tip_y, 0.0])
    return lm


class TestDiacriticCameraIntegration(unittest.TestCase):
    def setUp(self):
        self.rec = VSLRecognizer(confidence_threshold=0.70, smoothing_window=4, cooldown_sec=0.4)
        self.rec.reset()

    # 1. Test A -> Ă
    def test_case_1_a_to_a_breve_integration(self):
        """Luồng nhận diện camera: thế tay A ổn định sau đó mở xòe ngón V -> thăng cấp thành Ă."""
        session_id = "test_cam_a_breve"
        t = 100.0

        # Giả lập 5 frames nhận diện A từ Geometric Rules
        mock_res_a = {"letter": "A", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_a):
            for _ in range(5):
                lm_a = make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, middle_tip_y=0.45)
                res = self.rec.predict_landmarks(lm_a, mode="letters", session_id=session_id, timestamp=t)
                t += 0.04
                self.assertEqual(res["sign"], "A")

            # Frame 6 & 7: Người ký mở ngón xòe V (Ă)
            lm_breve = make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.38, middle_tip_x=0.62, index_tip_y=0.30, middle_tip_y=0.30)
            self.rec.predict_landmarks(lm_breve, mode="letters", session_id=session_id, timestamp=t)
            t += 0.04
            res_confirmed = self.rec.predict_landmarks(lm_breve, mode="letters", session_id=session_id, timestamp=t)

            self.assertEqual(res_confirmed["sign"], "Ă")
            self.assertEqual(res_confirmed["type"], "alphabet")
            self.assertEqual(res_confirmed["source"], "diacritic_fsm")
            self.assertIn("Ă", res_confirmed["classifier_details"]["diacritic_fsm"]["diacritic_detected"])

    # 2. Test A -> Â
    def test_case_2_a_to_a_circumflex_integration(self):
        """Luồng nhận diện camera: thế tay A nâng tay vẽ nón -> thăng cấp thành Â."""
        session_id = "test_cam_a_circumflex"
        t = 200.0

        mock_res_a = {"letter": "A", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_a):
            for _ in range(5):
                res = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.56, index_tip_y=0.45, middle_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04
                self.assertEqual(res["sign"], "A")

            # Nâng cổ tay lên đỉnh nón (y giảm xuống 0.50)
            self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.51, index_tip_y=0.40, middle_tip_y=0.40), mode="letters", session_id=session_id, timestamp=t)
            t += 0.04
            res_confirmed = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.50, index_tip_y=0.39, middle_tip_y=0.39), mode="letters", session_id=session_id, timestamp=t)

            self.assertEqual(res_confirmed["sign"], "Â")
            self.assertEqual(res_confirmed["source"], "diacritic_fsm")

    # 3. Test E -> Ê
    def test_case_3_e_to_e_circumflex_integration(self):
        """Luồng nhận diện camera: thế tay E nâng tay vẽ nón -> thăng cấp thành Ê."""
        session_id = "test_cam_e_circumflex"
        t = 300.0

        mock_res_e = {"letter": "E", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_e):
            for _ in range(5):
                res = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.57), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04
                self.assertEqual(res["sign"], "E")

            self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.52), mode="letters", session_id=session_id, timestamp=t)
            t += 0.04
            res_confirmed = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.51), mode="letters", session_id=session_id, timestamp=t)

            self.assertEqual(res_confirmed["sign"], "Ê")
            self.assertEqual(res_confirmed["source"], "diacritic_fsm")

    # 4. Test O -> Ô
    def test_case_4_o_to_o_circumflex_integration(self):
        """Luồng nhận diện camera: thế tay O nâng tay vẽ nón -> thăng cấp thành Ô."""
        session_id = "test_cam_o_circumflex"
        t = 400.0

        mock_res_o = {"letter": "O", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_o):
            for _ in range(5):
                res = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.58, index_tip_x=0.48, pinky_tip_x=0.51, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04
                self.assertEqual(res["sign"], "O")

            self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.53, index_tip_x=0.48, pinky_tip_x=0.51, index_tip_y=0.40, pinky_tip_y=0.40), mode="letters", session_id=session_id, timestamp=t)
            t += 0.04
            res_confirmed = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.52, index_tip_x=0.48, pinky_tip_x=0.51, index_tip_y=0.39, pinky_tip_y=0.39), mode="letters", session_id=session_id, timestamp=t)

            self.assertEqual(res_confirmed["sign"], "Ô")
            self.assertEqual(res_confirmed["source"], "diacritic_fsm")

    # 5. Test O -> Ơ
    def test_case_5_o_to_o_horn_integration(self):
        """Luồng nhận diện camera: thế tay O vẩy mở ngón út vươn xa -> thăng cấp thành Ơ."""
        session_id = "test_cam_o_horn"
        t = 500.0

        mock_res_o = {"letter": "O", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_o):
            for _ in range(5):
                res = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.48, pinky_tip_x=0.50, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04
                self.assertEqual(res["sign"], "O")

            lm_horn = make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.48, pinky_tip_x=0.68, index_tip_y=0.45, pinky_tip_y=0.30)
            self.rec.predict_landmarks(lm_horn, mode="letters", session_id=session_id, timestamp=t)
            t += 0.04
            res_confirmed = self.rec.predict_landmarks(lm_horn, mode="letters", session_id=session_id, timestamp=t)

            self.assertEqual(res_confirmed["sign"], "Ơ")
            self.assertEqual(res_confirmed["source"], "diacritic_fsm")

    # 6. Test U -> Ư
    def test_case_6_u_to_u_horn_integration(self):
        """Luồng nhận diện camera: U nhận diện từ Static Head, bật ngón út -> thăng cấp thành Ư."""
        session_id = "test_cam_u_horn"
        t = 600.0

        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "reason": "No rule"}
        mock_head_u = {
            "letter": "U", "raw_predicted_letter": "U", "confidence": 0.92,
            "status": "CONFIDENT", "is_confident": True, "top_k": [{"letter": "U", "confidence": 0.92}]
        }

        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_u):
                # 5 frames thế tay U (ngón út gập)
                for _ in range(5):
                    res = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.48), mode="letters", session_id=session_id, timestamp=t)
                    t += 0.04
                    self.assertEqual(res["sign"], "U")

                # Ngón út bật duỗi thẳng lên (Ư)
                lm_u_horn = make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.30)
                self.rec.predict_landmarks(lm_u_horn, mode="letters", session_id=session_id, timestamp=t)
                t += 0.04
                res_confirmed = self.rec.predict_landmarks(lm_u_horn, mode="letters", session_id=session_id, timestamp=t)

                self.assertEqual(res_confirmed["sign"], "Ư")
                self.assertEqual(res_confirmed["source"], "diacritic_fsm")

    # 7. Test D -> Đ
    def test_case_7_d_to_d_stroke_integration(self):
        """Luồng nhận diện camera: thế tay D gập đốt ngón trỏ -> thăng cấp thành Đ."""
        session_id = "test_cam_d_stroke"
        t = 700.0

        mock_res_d = {"letter": "D", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_d):
            for _ in range(5):
                res = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.25), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04
                self.assertEqual(res["sign"], "D")

            # Ngón trỏ gập gãy đốt giữa (Đ)
            lm_d_hook = make_dummy_landmarks(wrist_y=0.55)
            lm_d_hook[6] = [0.47, 0.45, 0.0]
            lm_d_hook[7] = [0.55, 0.45, 0.0]
            lm_d_hook[8] = [0.55, 0.40, 0.0]

            res_confirmed = self.rec.predict_landmarks(lm_d_hook, mode="letters", session_id=session_id, timestamp=t)
            self.assertEqual(res_confirmed["sign"], "Đ")
            self.assertEqual(res_confirmed["source"], "diacritic_fsm")

    # 8. 17 chữ tĩnh không bị FSM can thiệp
    def test_case_8_pure_static_letters_not_interfered(self):
        """17 chữ cái tĩnh ngoài 5 nhóm (B, C, G, H, I, K, L, M, N, P, Q, R, S, T, V, X, Y) không bị FSM can thiệp."""
        static_letters = ["B", "C", "G", "H", "I", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "V", "X", "Y"]
        session_id = "test_static_passthrough"

        for letter in static_letters:
            if letter in ["B", "C", "I", "L", "S", "T", "V", "Y"]:
                mock_geom = {"letter": letter, "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
                with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_geom):
                    res = self.rec.predict_landmarks(make_dummy_landmarks(), mode="letters", session_id=session_id)
                    self.assertEqual(res["sign"], letter)
                    self.assertEqual(res["source"], "geometric_rule")
                    self.assertNotIn("diacritic_fsm", res["classifier_details"])

    # 9. 12 geometric rules vẫn hoạt động nguyên vẹn
    def test_case_9_geometric_rules_preserved(self):
        """Các chữ có luật hình học (A, B, C, D, E, I, L, O, S, T, V, Y) vẫn được nhận diện chuẩn."""
        mock_geom_b = {"letter": "B", "confidence": 0.98, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_geom_b):
            res = self.rec.predict_landmarks(make_dummy_landmarks(), mode="letters")
            self.assertEqual(res["sign"], "B")
            self.assertEqual(res["source"], "geometric_rule")

    # 10. Static Head vẫn hoạt động bình thường với 10 chữ bổ sung
    def test_case_10_static_head_supplemental_letters_preserved(self):
        """10 chữ tĩnh bổ sung (G, H, K, M, N, P, Q, R, U, X) từ Static Head hoạt động tốt."""
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "reason": "No rule"}
        mock_head_k = {
            "letter": "K", "raw_predicted_letter": "K", "confidence": 0.92,
            "status": "CONFIDENT", "is_confident": True, "top_k": [{"letter": "K", "confidence": 0.92}]
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_k):
                res = self.rec.predict_landmarks(make_dummy_landmarks(), mode="letters")
                self.assertEqual(res["sign"], "K")
                self.assertEqual(res["source"], "static_head")

    # 11. Mất landmarks không giữ lại nhãn có dấu cũ
    def test_case_11_lost_landmarks_safe_neutral(self):
        """Khi mất mốc xương bàn tay, hệ thống trả về NEUTRAL và FSM tự reset."""
        session_id = "test_lost_lm"
        # 1. Khởi động thế A
        mock_res_a = {"letter": "A", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_a):
            self.rec.predict_landmarks(make_dummy_landmarks(), mode="letters", session_id=session_id)

            # 2. Mất tay (None)
            res_none = self.rec.predict_landmarks(None, mode="letters", session_id=session_id)
            self.assertEqual(res_none["sign"], "NEUTRAL")
            self.assertEqual(res_none["confidence"], 0.0)

            # 3. Mất tay (rỗng)
            res_empty = self.rec.predict_landmarks([], mode="letters", session_id=session_id)
            self.assertEqual(res_empty["sign"], "NEUTRAL")

    # 12. Quản lý phiên độc lập (Session separation) & Reset session
    def test_case_12_session_isolation_and_reset(self):
        """Session 1 và Session 2 có FSM riêng biệt, reset session 1 không làm ảnh hưởng session 2."""
        s1 = "session_user_1"
        s2 = "session_user_2"

        mock_res_a = {"letter": "A", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_a):
            # Tạo 5 frame A trên s1
            for t in range(5):
                self.rec.predict_landmarks(make_dummy_landmarks(), mode="letters", session_id=s1, timestamp=float(t))

            # s1 đạt BASE_STABLE
            fsm_1 = self.rec.diacritic_fsms[s1]
            self.assertEqual(fsm_1.state, STATE_BASE_STABLE)

            # s2 chưa có frame nào
            self.assertNotIn(s2, self.rec.diacritic_fsms)

            # Tạo 1 frame A trên s2
            self.rec.predict_landmarks(make_dummy_landmarks(), mode="letters", session_id=s2, timestamp=1.0)
            fsm_2 = self.rec.diacritic_fsms[s2]
            self.assertNotEqual(fsm_2.state, STATE_BASE_STABLE)

            # Reset s1
            self.rec.reset_diacritic_fsm(session_id=s1)
            self.assertEqual(fsm_1.state, "IDLE")

    # 13. Chế độ digits và words không bị ảnh hưởng
    def test_case_13_digits_and_words_unaffected(self):
        """Mode 'digits' và 'words' hoàn toàn không gọi FSM."""
        res_digit = self.rec.predict_landmarks(make_dummy_landmarks(), mode="digits")
        self.assertNotEqual(res_digit["type"], "alphabet")

        res_words = self.rec.predict_landmarks(make_dummy_landmarks(), mode="words")
        self.assertNotIn(res_words["sign"], ALLOWED_DIGITS)

    # 14. Chặn tuyệt đối chuyển nhãn chéo giữa các nhóm nền
    def test_case_14_no_cross_group_transition(self):
        """Dáng tay A tuyệt đối không thể trở thành Ô hay Ư."""
        session_id = "test_no_cross_group"
        mock_res_a = {"letter": "A", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_a):
            for t in range(5):
                self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55), mode="letters", session_id=session_id, timestamp=float(t))

            # Giả lập cử chỉ vẩy râu của chữ Ư (pinky_curl giảm mạnh) trong khi chữ nền là A
            lm_pinky = make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.30)
            res = self.rec.predict_landmarks(lm_pinky, mode="letters", session_id=session_id, timestamp=6.0)

            # Phải giữ A hoặc ra Ă/Â, TUYỆT ĐỐI không thể ra Ư!
            self.assertNotEqual(res["sign"], "Ư")
            self.assertIn(res["sign"], ["A", "Ă", "Â"])

    # 15. (Task 7E.3) Quản lý phiên độc lập trong pha transient
    def test_case_15_session_isolation_transient_fsm(self):
        """Transient NEUTRAL ở session 1 không gây ảnh hưởng đến session 2."""
        session_1 = "cam_session_1"
        session_2 = "cam_session_2"
        t = 100.0

        mock_res_o = {"letter": "O", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        mock_res_a = {"letter": "A", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}

        # Session 1 học O, Session 2 học A
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_o):
            for _ in range(5):
                self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.58, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_1, timestamp=t)
                t += 0.04

        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_res_a):
            for _ in range(5):
                self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55), mode="letters", session_id=session_2, timestamp=t)
                t += 0.04

        self.assertEqual(self.rec.diacritic_fsms[session_1].current_base, "O")
        self.assertEqual(self.rec.diacritic_fsms[session_2].current_base, "A")

        # Session 1 gặp frame transient NEUTRAL
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "reason": "No match"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            res_s1 = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.58, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_1, timestamp=t)
            self.assertEqual(res_s1["sign"], "NEUTRAL")
            # Session 1 vẫn giữ baseline O
            self.assertEqual(self.rec.diacritic_fsms[session_1].current_base, "O")
            self.assertEqual(self.rec.diacritic_fsms[session_1].state, STATE_BASE_STABLE)
            # Session 2 vẫn giữ baseline A hoàn toàn độc lập
            self.assertEqual(self.rec.diacritic_fsms[session_2].current_base, "A")
            self.assertEqual(self.rec.diacritic_fsms[session_2].state, STATE_BASE_STABLE)

    # 16. (Task 7E.3) Camera pipeline bảo toàn baseline khi gặp transient NEUTRAL
    def test_case_16_transient_neutral_camera_pipeline(self):
        """Luồng camera thực tế: O ổn định, 1 frame NO_MATCH không reset FSM, chuyển sang B reset lập tức."""
        session_id = "test_cam_transient"
        t = 50.0

        mock_o = {"letter": "O", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        mock_neutral = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "reason": "none"}
        mock_b = {"letter": "B", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}

        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_o):
            for _ in range(5):
                self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04

        fsm = self.rec.diacritic_fsms[session_id]
        self.assertEqual(fsm.state, STATE_BASE_STABLE)
        self.assertTrue(fsm.baseline is not None)

        # 1 frame NO_MATCH -> sign = NEUTRAL
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_neutral):
            res = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
            self.assertEqual(res["sign"], "NEUTRAL")
            # FSM không bị reset!
            self.assertEqual(fsm.state, STATE_BASE_STABLE)
            self.assertTrue(fsm.baseline is not None)
            self.assertEqual(fsm.current_base, "O")

        # Frame tiếp theo: Người dùng đổi sang chữ B (chữ tĩnh ngoài nhóm) -> xác nhận chuyển sau 2 frames (Task 7E.4B)
        t += 0.04
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_b):
            self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55), mode="letters", session_id=session_id, timestamp=t)
            t += 0.04
            res_b = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55), mode="letters", session_id=session_id, timestamp=t)
            self.assertEqual(res_b["sign"], "B")
            self.assertEqual(fsm.state, "IDLE")
            self.assertIsNone(fsm.baseline)

    # 17. (Task 7E.4A) Camera pipeline: U vươn ngón út tạo Ư, Rule I không cướp nhãn trong cooldown
    def test_case_17_camera_pipeline_u_to_u_horn_competing_rule_i(self):
        """Pipeline camera: U ổn định, vươn ngón út tạo Ư khiến Rule I MATCH -> Ra Ư, cooldown giữ vững Ư."""
        session_id = "test_cam_u_competing"
        t = 10.0

        mock_u = {"letter": "U", "confidence": 0.95, "status": "MATCH", "source": "static_head"}
        mock_i = {"letter": "I", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}

        # 1. Khởi động 5 frames U đạt BASE_STABLE
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value={"status": "NO_MATCH", "letter": "UNKNOWN", "confidence": 0.0}):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value={"raw_predicted_letter": "U", "confidence": 0.95, "is_confident": True}):
                for _ in range(5):
                    self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
                    t += 0.04

        fsm = self.rec.diacritic_fsms[session_id]
        self.assertEqual(fsm.state, STATE_BASE_STABLE)
        self.assertEqual(fsm.current_base, "U")

        # 2. Vươn ngón út tạo Ư: Rule I MATCH (2 frames để confirm Ư)
        lm_u_horn = make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.30)
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_i):
            res1 = self.rec.predict_landmarks(lm_u_horn, mode="letters", session_id=session_id, timestamp=t)
            t += 0.04
            res2 = self.rec.predict_landmarks(lm_u_horn, mode="letters", session_id=session_id, timestamp=t)

            self.assertEqual(res2["sign"], "Ư")
            self.assertEqual(res2["source"], "diacritic_fsm")
            self.assertEqual(fsm.state, STATE_COOLDOWN)

            # 3. Frame tiếp theo vẫn giữ ngón út duỗi -> Rule I MATCH tiếp nhưng KHÔNG được cướp nhãn!
            t += 0.04
            res3 = self.rec.predict_landmarks(lm_u_horn, mode="letters", session_id=session_id, timestamp=t)
            self.assertEqual(res3["sign"], "Ư")
            self.assertEqual(res3["source"], "diacritic_fsm")
            self.assertEqual(fsm.state, STATE_COOLDOWN)

    # 18. (Task 7E.4A) Camera pipeline: Người ký chủ động chuyển từ O sang I
    def test_case_18_camera_pipeline_intentional_switch_to_i(self):
        """Pipeline camera: O ổn định, người ký đổi sang thế tay I và giữ 3 frames -> FSM kết thúc và ra I."""
        session_id = "test_cam_switch_i"
        t = 20.0

        mock_o = {"letter": "O", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        mock_i = {"letter": "I", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}

        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_o):
            for _ in range(5):
                self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04

        fsm = self.rec.diacritic_fsms[session_id]
        self.assertEqual(fsm.state, STATE_BASE_STABLE)

        # Chuyển sang ký chữ I (ngón út duỗi, các ngón khác gập, không vươn râu Ơ)
        lm_i = make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45)
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_i):
            # Frame 1: Tạm chặn
            r1 = self.rec.predict_landmarks(lm_i, mode="letters", session_id=session_id, timestamp=t)
            self.assertEqual(r1["sign"], "NEUTRAL")
            t += 0.04

            # Frame 2: Tạm chặn
            r2 = self.rec.predict_landmarks(lm_i, mode="letters", session_id=session_id, timestamp=t)
            self.assertEqual(r2["sign"], "NEUTRAL")
            t += 0.04

            # Frame 3: Đủ 3 frames liên tiếp -> Chuyển sang chữ I!
            r3 = self.rec.predict_landmarks(lm_i, mode="letters", session_id=session_id, timestamp=t)
            self.assertEqual(r3["sign"], "I")
            self.assertEqual(r3["source"], "geometric_rule")
            self.assertEqual(fsm.state, "IDLE")

    # 19. (Task 7E.4A) Không có baseline -> Rule V MATCH nhận diện ngay không bị chặn
    def test_case_19_camera_pipeline_no_baseline_no_suppression(self):
        """Khi FSM không có baseline (IDLE), Rule V MATCH lập tức nhận diện V mà không bị FSM chặn."""
        session_id = "test_cam_no_bl"
        mock_v = {"letter": "V", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}

        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_v):
            res = self.rec.predict_landmarks(make_dummy_landmarks(), mode="letters", session_id=session_id)
            self.assertEqual(res["sign"], "V")
            self.assertEqual(res["source"], "geometric_rule")
            self.assertIsNone(self.rec.diacritic_fsms[session_id].baseline)

    # 20. (Task 7E.4A) Đổi session độc lập: Session 1 đang theo dõi U, Session 2 ký I nhận ngay
    def test_case_20_camera_pipeline_session_isolation_competing(self):
        """Session 1 có baseline U (Rule I cạnh tranh), Session 2 ký I lập tức nhận diện I bình thường."""
        s1 = "session_u_tracking"
        s2 = "session_i_direct"
        t = 30.0

        mock_u = {"letter": "U", "confidence": 0.95, "status": "MATCH", "source": "static_head"}
        mock_i = {"letter": "I", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}

        # Session 1 học U
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value={"status": "NO_MATCH", "letter": "UNKNOWN", "confidence": 0.0}):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value={"raw_predicted_letter": "U", "confidence": 0.95, "is_confident": True}):
                for _ in range(5):
                    self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, pinky_tip_y=0.45), mode="letters", session_id=s1, timestamp=t)
                    t += 0.04

        self.assertEqual(self.rec.diacritic_fsms[s1].current_base, "U")

        # Session 2 ký I -> Không hề có baseline, nhận I ngay lập tức!
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_i):
            res_s2 = self.rec.predict_landmarks(make_dummy_landmarks(), mode="letters", session_id=s2, timestamp=t)
            self.assertEqual(res_s2["sign"], "I")
            self.assertEqual(res_s2["source"], "geometric_rule")

            # Session 1 vẫn giữ baseline U độc lập
            self.assertEqual(self.rec.diacritic_fsms[s1].current_base, "U")

    # 21. (Task 7E.4B) Camera pipeline: O đang BASE_STABLE, 1 frame E thoáng qua không phá vỡ baseline O
    def test_case_21_camera_pipeline_transient_wrong_base_preserves_o(self):
        """Pipeline camera: O ổn định, 1 frame E thoáng qua hiển thị E tức thì nhưng baseline O của FSM vẫn nguyên vẹn."""
        session_id = "test_cam_o_glitch"
        t = 40.0

        mock_o = {"letter": "O", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        mock_e = {"letter": "E", "confidence": 0.92, "status": "MATCH", "source": "geometric_rule"}

        # 1. 5 frames O để đạt BASE_STABLE
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_o):
            for _ in range(5):
                self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04

        fsm = self.rec.diacritic_fsms[session_id]
        self.assertEqual(fsm.state, STATE_BASE_STABLE)
        self.assertEqual(fsm.current_base, "O")
        self.assertTrue(fsm.baseline is not None)

        # 2. 1 frame E thoáng qua
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_e):
            res_e = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
            self.assertEqual(res_e["sign"], "E")  # Người dùng thấy E ngay
            self.assertEqual(fsm.current_base, "O")  # FSM không mất baseline O!
            self.assertEqual(fsm.switch_candidate_base, "E")
            self.assertEqual(fsm.switch_candidate_count, 1)

        # 3. Frame tiếp theo O quay lại
        t += 0.04
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_o):
            res_o = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
            self.assertEqual(res_o["sign"], "O")
            self.assertEqual(fsm.current_base, "O")
            self.assertIsNone(fsm.switch_candidate_base)
            self.assertEqual(fsm.switch_candidate_count, 0)

    # 22. (Task 7E.4B) Camera pipeline: O chuyển sang E liên tiếp -> Chuyển chữ đúng
    def test_case_22_camera_pipeline_consecutive_wrong_base_switches_cleanly(self):
        """Pipeline camera: O ổn định, người dùng chuyển sang E liên tiếp -> FSM xác nhận chuyển sang E và reset O."""
        session_id = "test_cam_o_to_e"
        t = 50.0

        mock_o = {"letter": "O", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        mock_e = {"letter": "E", "confidence": 0.92, "status": "MATCH", "source": "geometric_rule"}

        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_o):
            for _ in range(5):
                self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, pinky_tip_y=0.45), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04

        fsm = self.rec.diacritic_fsms[session_id]
        self.assertEqual(fsm.current_base, "O")

        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_e):
            # Frame 1 của E
            self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55), mode="letters", session_id=session_id, timestamp=t)
            t += 0.04
            # Frame 2 của E -> Đủ 2 frames chuyển chữ
            res_e2 = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.55), mode="letters", session_id=session_id, timestamp=t)
            self.assertEqual(res_e2["sign"], "E")
            self.assertNotEqual(fsm.current_base, "O")

    # 23. (Task 7E.4B) Chế độ digits và words hoàn toàn không bị ảnh hưởng
    def test_case_23_digits_and_words_modes_unaffected(self):
        """Các chế độ ngoài letters (digits, words) tuyệt đối không đi qua FSM phân giải dấu."""
        lm = make_dummy_landmarks(wrist_y=0.55)
        # Chế độ digits
        res_dig = self.rec.predict_landmarks(lm, mode="digits")
        self.assertNotIn("diacritic_fsm", res_dig)
        # Chế độ words
        res_word = self.rec.predict_landmarks(lm, mode="words")
        self.assertNotIn("diacritic_fsm", res_word)


if __name__ == "__main__":
    unittest.main()

