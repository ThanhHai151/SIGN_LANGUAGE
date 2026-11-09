"""
Kiểm thử Gated Static Head Fallback cho 3 chữ nền A, D, E (Task 7E.5B).
========================================================================
Bao phủ 4 nhóm kiểm thử:
- Category A (Fallback Active):
    1. Geom NO_MATCH, Head "A" conf 0.85 -> sign="A", source="static_head_fallback"
    2. Geom UNKNOWN, Head "D" conf 0.80 -> sign="D", source="static_head_fallback"
    3. Geom NO_MATCH, Head "E" conf 0.70 -> sign="E", source="static_head_fallback"
    4. Fallback letter được truyền vào FSM session
- Category B (Fallback Rejected):
    5. Head "A" conf 0.69 (< 0.70) -> rejected, NEUTRAL
    6. Head predicts "B" conf 0.90 -> rejected (không thuộc A/D/E), NEUTRAL
    7. Head predicts "G" conf 0.80 -> sử dụng nhánh static_head chuẩn (supplemental set)
    8. Landmarks is None -> không gọi fallback, NEUTRAL
    9. Head predict_feature raises Exception -> xử lý an toàn, NEUTRAL
- Category C (Geometric Absolute Priority):
    10. Geom MATCH "A" (0.95), Head "D" (0.99) -> "A", source="geometric_rule"
    11. Geom MATCH "I" (0.95), Head "A" (0.99) -> "I", source="geometric_rule"
    12. Geom MATCH "V" (0.95), Head "A" (0.99) -> "V", source="geometric_rule"
    13. Geom MATCH "D" (0.90), Head "E" (0.99) -> "D", source="geometric_rule"
- Category D (FSM Progression from Fallback Base):
    14. 5 frames Fallback A -> FSM đạt BASE_STABLE (current_base="A")
    15. 5 frames Fallback E -> FSM đạt BASE_STABLE (current_base="E")
    16. 5 frames Fallback D -> FSM đạt BASE_STABLE (current_base="D")
    17. Fallback A đạt stable -> Rule V xuất hiện -> thăng cấp thành Ă (source="diacritic_fsm")
    18. Fallback D đạt stable -> ngón trỏ gập đốt -> thăng cấp thành Đ (source="diacritic_fsm")
    19. Fallback A đạt stable -> vẽ nón -> thăng cấp thành Â (source="diacritic_fsm")
    20. Fallback E đạt stable -> vẽ nón -> thăng cấp thành Ê (source="diacritic_fsm")
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.vision.recognizer import (
    VSLRecognizer,
    ALLOWED_FALLBACK_BASE_LETTERS,
    DEFAULT_FALLBACK_BASE_THRESHOLD,
)
from backend.vision.diacritic_state_machine import STATE_BASE_STABLE


def make_dummy_landmarks(
    wrist_y=0.55,
    index_tip_y=0.35,
    middle_tip_y=0.35,
    pinky_tip_y=0.45,
    index_tip_x=0.48,
    middle_tip_x=0.52,
    pinky_tip_x=0.51,
):
    """Tạo bộ 21 landmarks giả lập."""
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


class TestStaticHeadFallback(unittest.TestCase):
    def setUp(self):
        self.rec = VSLRecognizer(confidence_threshold=0.70, smoothing_window=4, cooldown_sec=0.4)
        self.rec.reset()

    # =========================================================================
    # Category A: Fallback Active
    # =========================================================================
    def test_a1_fallback_active_letter_a(self):
        """Geom NO_MATCH, Head 'A' conf 0.85 -> sign='A', source='static_head_fallback'."""
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_a = {
            "letter": "A",
            "raw_predicted_letter": "A",
            "confidence": 0.85,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "A", "confidence": 0.85}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_a):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_fallback_a", timestamp=10.0)

                self.assertEqual(res["sign"], "A")
                self.assertEqual(res["source"], "static_head_fallback")
                self.assertEqual(res["prediction_source"], "static_head_fallback")
                self.assertAlmostEqual(res["confidence"], 0.85, places=2)

    def test_a2_fallback_active_letter_d(self):
        """Geom UNKNOWN, Head 'D' conf 0.80 -> sign='D', source='static_head_fallback'."""
        mock_unknown = {"letter": "UNKNOWN", "confidence": 0.0, "status": "UNKNOWN", "source": "none"}
        mock_head_d = {
            "letter": "D",
            "raw_predicted_letter": "D",
            "confidence": 0.80,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "D", "confidence": 0.80}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_unknown):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_d):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_fallback_d", timestamp=20.0)

                self.assertEqual(res["sign"], "D")
                self.assertEqual(res["source"], "static_head_fallback")
                self.assertEqual(res["prediction_source"], "static_head_fallback")
                self.assertAlmostEqual(res["confidence"], 0.80, places=2)

    def test_a3_fallback_active_letter_e(self):
        """Geom NO_MATCH, Head 'E' conf 0.70 (ngưỡng chính xác) -> sign='E', source='static_head_fallback'."""
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_e = {
            "letter": "E",
            "raw_predicted_letter": "E",
            "confidence": 0.70,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "E", "confidence": 0.70}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_e):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_fallback_e", timestamp=30.0)

                self.assertEqual(res["sign"], "E")
                self.assertEqual(res["source"], "static_head_fallback")
                self.assertAlmostEqual(res["confidence"], 0.70, places=2)

    def test_a4_fallback_feeds_fsm_session(self):
        """Fallback base letter được truyền trực tiếp vào FSM session và ghi nhận candidate_base / current_base."""
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_a = {
            "letter": "A",
            "raw_predicted_letter": "A",
            "confidence": 0.88,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "A", "confidence": 0.88}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_a):
                lm = make_dummy_landmarks()
                self.rec.predict_landmarks(lm, mode="letters", session_id="test_fsm_feed", timestamp=40.0)

                fsm = self.rec.diacritic_fsms["test_fsm_feed"]
                self.assertEqual(fsm.candidate_base, "A")
                self.assertEqual(fsm.candidate_base_count, 1)

    # =========================================================================
    # Category B: Fallback Rejected
    # =========================================================================
    def test_b1_fallback_rejected_low_confidence(self):
        """Head 'A' conf 0.69 (< 0.70) -> rejected, trả về NEUTRAL."""
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_a = {
            "letter": "A",
            "raw_predicted_letter": "A",
            "confidence": 0.69,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "A", "confidence": 0.69}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_a):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_rej_low_conf", timestamp=50.0)

                self.assertEqual(res["sign"], "NEUTRAL")
                self.assertEqual(res["source"], "none")

    def test_b2_fallback_rejected_non_target_letter(self):
        """Head dự đoán 'B' conf 0.90 -> không thuộc {A, D, E} và không thuộc supplemental -> rejected."""
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_b = {
            "letter": "B",
            "raw_predicted_letter": "B",
            "confidence": 0.90,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "B", "confidence": 0.90}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_b):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_rej_b", timestamp=60.0)

                self.assertEqual(res["sign"], "NEUTRAL")
                self.assertEqual(res["source"], "none")

    def test_b2b_fallback_rejects_o_when_geometry_does_not_match(self):
        """O remains outside both static-head fallback allowlists."""
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_o = {
            "letter": "O",
            "raw_predicted_letter": "O",
            "confidence": 0.99,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "O", "confidence": 0.99}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_o):
                res = self.rec.predict_landmarks(
                    make_dummy_landmarks(), mode="letters", session_id="test_rej_o", timestamp=65.0
                )

        self.assertEqual(res["sign"], "NEUTRAL")
        self.assertEqual(res["source"], "none")

    def test_b3_supplemental_letters_still_use_static_head(self):
        """Head dự đoán 'G' conf 0.80 -> sử dụng nhánh static_head chuẩn, source='static_head'."""
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_g = {
            "letter": "G",
            "raw_predicted_letter": "G",
            "confidence": 0.80,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "G", "confidence": 0.80}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_g):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_supp_g", timestamp=70.0)

                self.assertEqual(res["sign"], "G")
                self.assertEqual(res["source"], "static_head")
                self.assertEqual(res["prediction_source"], "static_head")

    def test_b4_no_landmarks_returns_neutral(self):
        """Landmarks is None / empty -> không gọi fallback, trả về NEUTRAL an toàn."""
        res = self.rec.predict_landmarks(None, mode="letters", session_id="test_no_lm", timestamp=80.0)
        self.assertEqual(res["sign"], "NEUTRAL")

    def test_b5_head_exception_handled_gracefully(self):
        """Nếu predict_feature raise Exception -> không crash, trả về NEUTRAL an toàn."""
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", side_effect=RuntimeError("Model error")):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_head_err", timestamp=90.0)

                self.assertEqual(res["sign"], "NEUTRAL")
                self.assertEqual(res["source"], "none")

    # =========================================================================
    # Category C: Geometric Absolute Priority
    # =========================================================================
    def test_c1_geometric_priority_over_head_a_vs_d(self):
        """Geom MATCH 'A' (0.95), Head 'D' (0.99) -> chọn 'A', source='geometric_rule'."""
        mock_match_a = {"letter": "A", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        mock_head_d = {
            "letter": "D",
            "raw_predicted_letter": "D",
            "confidence": 0.99,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "D", "confidence": 0.99}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_match_a):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_d):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_prio_a_d", timestamp=100.0)

                self.assertEqual(res["sign"], "A")
                self.assertEqual(res["source"], "geometric_rule")
                self.assertEqual(res["prediction_source"], "geometric_rule")

    def test_c2_geometric_priority_i_vs_head_a(self):
        """Geom MATCH 'I' (0.95), Head 'A' (0.99) -> chọn 'I', source='geometric_rule'."""
        mock_match_i = {"letter": "I", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        mock_head_a = {
            "letter": "A",
            "raw_predicted_letter": "A",
            "confidence": 0.99,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "A", "confidence": 0.99}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_match_i):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_a):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_prio_i_a", timestamp=110.0)

                self.assertEqual(res["sign"], "I")
                self.assertEqual(res["source"], "geometric_rule")

    def test_c3_geometric_priority_v_vs_head_a(self):
        """Geom MATCH 'V' (0.95), Head 'A' (0.99) -> chọn 'V', source='geometric_rule'."""
        mock_match_v = {"letter": "V", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        mock_head_a = {
            "letter": "A",
            "raw_predicted_letter": "A",
            "confidence": 0.99,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "A", "confidence": 0.99}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_match_v):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_a):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_prio_v_a", timestamp=120.0)

                self.assertEqual(res["sign"], "V")
                self.assertEqual(res["source"], "geometric_rule")

    def test_c4_geometric_priority_d_vs_head_e(self):
        """Geom MATCH 'D' (0.90), Head 'E' (0.99) -> chọn 'D', source='geometric_rule'."""
        mock_match_d = {"letter": "D", "confidence": 0.90, "status": "MATCH", "source": "geometric_rule"}
        mock_head_e = {
            "letter": "E",
            "raw_predicted_letter": "E",
            "confidence": 0.99,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "E", "confidence": 0.99}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_match_d):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_e):
                lm = make_dummy_landmarks()
                res = self.rec.predict_landmarks(lm, mode="letters", session_id="test_prio_d_e", timestamp=130.0)

                self.assertEqual(res["sign"], "D")
                self.assertEqual(res["source"], "geometric_rule")

    # =========================================================================
    # Category D: FSM Progression from Fallback Base
    # =========================================================================
    def test_d1_fallback_a_achieves_base_stable(self):
        """5 frames Fallback A -> FSM đạt BASE_STABLE (current_base='A')."""
        session_id = "test_fsm_a_stable"
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_a = {
            "letter": "A",
            "raw_predicted_letter": "A",
            "confidence": 0.88,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "A", "confidence": 0.88}],
        }
        t = 200.0
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_a):
                for _ in range(5):
                    lm = make_dummy_landmarks(wrist_y=0.55)
                    self.rec.predict_landmarks(lm, mode="letters", session_id=session_id, timestamp=t)
                    t += 0.04

        fsm = self.rec.diacritic_fsms[session_id]
        self.assertEqual(fsm.state, STATE_BASE_STABLE)
        self.assertEqual(fsm.current_base, "A")

    def test_d2_fallback_e_achieves_base_stable(self):
        """5 frames Fallback E -> FSM đạt BASE_STABLE (current_base='E')."""
        session_id = "test_fsm_e_stable"
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_e = {
            "letter": "E",
            "raw_predicted_letter": "E",
            "confidence": 0.85,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "E", "confidence": 0.85}],
        }
        t = 300.0
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_e):
                for _ in range(5):
                    lm = make_dummy_landmarks(wrist_y=0.55)
                    self.rec.predict_landmarks(lm, mode="letters", session_id=session_id, timestamp=t)
                    t += 0.04

        fsm = self.rec.diacritic_fsms[session_id]
        self.assertEqual(fsm.state, STATE_BASE_STABLE)
        self.assertEqual(fsm.current_base, "E")

    def test_d3_fallback_d_achieves_base_stable(self):
        """5 frames Fallback D -> FSM đạt BASE_STABLE (current_base='D')."""
        session_id = "test_fsm_d_stable"
        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_d = {
            "letter": "D",
            "raw_predicted_letter": "D",
            "confidence": 0.82,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "D", "confidence": 0.82}],
        }
        t = 400.0
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_d):
                for _ in range(5):
                    lm = make_dummy_landmarks(wrist_y=0.55)
                    self.rec.predict_landmarks(lm, mode="letters", session_id=session_id, timestamp=t)
                    t += 0.04

        fsm = self.rec.diacritic_fsms[session_id]
        self.assertEqual(fsm.state, STATE_BASE_STABLE)
        self.assertEqual(fsm.current_base, "D")

    def test_d4_fallback_a_to_a_breve(self):
        """Fallback A đạt BASE_STABLE -> Rule V xuất hiện -> thăng cấp thành Ă."""
        session_id = "test_fallback_a_to_breve"
        t = 500.0

        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_a = {
            "letter": "A",
            "raw_predicted_letter": "A",
            "confidence": 0.88,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "A", "confidence": 0.88}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_a):
                for _ in range(5):
                    lm_a = make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.45, middle_tip_y=0.45)
                    res = self.rec.predict_landmarks(lm_a, mode="letters", session_id=session_id, timestamp=t)
                    t += 0.04
                    self.assertEqual(res["sign"], "A")
                    self.assertEqual(res["source"], "static_head_fallback")

        # Frame 6 & 7: Rule V xuất hiện khi người ký xòe ngón
        mock_match_v = {"letter": "V", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_match_v):
            lm_breve = make_dummy_landmarks(wrist_y=0.55, index_tip_x=0.38, middle_tip_x=0.62, index_tip_y=0.30, middle_tip_y=0.30)
            self.rec.predict_landmarks(lm_breve, mode="letters", session_id=session_id, timestamp=t)
            t += 0.04
            res_confirmed = self.rec.predict_landmarks(lm_breve, mode="letters", session_id=session_id, timestamp=t)

            self.assertEqual(res_confirmed["sign"], "Ă")
            self.assertEqual(res_confirmed["source"], "diacritic_fsm")

    def test_d5_fallback_d_to_d_stroke(self):
        """Fallback D đạt BASE_STABLE -> ngón trỏ gập đốt -> thăng cấp thành Đ."""
        session_id = "test_fallback_d_to_stroke"
        t = 600.0

        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_d = {
            "letter": "D",
            "raw_predicted_letter": "D",
            "confidence": 0.85,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "D", "confidence": 0.85}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_d):
                for _ in range(5):
                    lm_d = make_dummy_landmarks(wrist_y=0.55, index_tip_y=0.25)
                    res = self.rec.predict_landmarks(lm_d, mode="letters", session_id=session_id, timestamp=t)
                    t += 0.04
                    self.assertEqual(res["sign"], "D")
                    self.assertEqual(res["source"], "static_head_fallback")

                # Frame gập đốt ngón trỏ (vẫn duy trì Static Head predict D)
                lm_d_hook = make_dummy_landmarks(wrist_y=0.55)
                lm_d_hook[6] = [0.47, 0.45, 0.0]
                lm_d_hook[7] = [0.55, 0.45, 0.0]
                lm_d_hook[8] = [0.55, 0.40, 0.0]

                res_confirmed = self.rec.predict_landmarks(lm_d_hook, mode="letters", session_id=session_id, timestamp=t)
                self.assertEqual(res_confirmed["sign"], "Đ")
                self.assertEqual(res_confirmed["source"], "diacritic_fsm")

    def test_d6_fallback_a_to_a_circumflex(self):
        """Fallback A đạt BASE_STABLE -> nâng tay vẽ nón -> thăng cấp thành Â."""
        session_id = "test_fallback_a_to_circumflex"
        t = 700.0

        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_a = {
            "letter": "A",
            "raw_predicted_letter": "A",
            "confidence": 0.86,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "A", "confidence": 0.86}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_a):
                for _ in range(5):
                    lm_a = make_dummy_landmarks(wrist_y=0.56, index_tip_y=0.45, middle_tip_y=0.45)
                    res = self.rec.predict_landmarks(lm_a, mode="letters", session_id=session_id, timestamp=t)
                    t += 0.04
                    self.assertEqual(res["sign"], "A")
                    self.assertEqual(res["source"], "static_head_fallback")

                # Nâng cổ tay lên đỉnh nón
                self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.51, index_tip_y=0.40, middle_tip_y=0.40), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04
                res_confirmed = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.50, index_tip_y=0.39, middle_tip_y=0.39), mode="letters", session_id=session_id, timestamp=t)

                self.assertEqual(res_confirmed["sign"], "Â")
                self.assertEqual(res_confirmed["source"], "diacritic_fsm")

    def test_d7_fallback_e_to_e_circumflex(self):
        """Fallback E đạt BASE_STABLE -> nâng tay vẽ nón -> thăng cấp thành Ê."""
        session_id = "test_fallback_e_to_circumflex"
        t = 800.0

        mock_no_match = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        mock_head_e = {
            "letter": "E",
            "raw_predicted_letter": "E",
            "confidence": 0.85,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "E", "confidence": 0.85}],
        }
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_e):
                for _ in range(5):
                    lm_e = make_dummy_landmarks(wrist_y=0.57)
                    res = self.rec.predict_landmarks(lm_e, mode="letters", session_id=session_id, timestamp=t)
                    t += 0.04
                    self.assertEqual(res["sign"], "E")
                    self.assertEqual(res["source"], "static_head_fallback")

                # Nâng cổ tay lên đỉnh nón
                self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.52), mode="letters", session_id=session_id, timestamp=t)
                t += 0.04
                res_confirmed = self.rec.predict_landmarks(make_dummy_landmarks(wrist_y=0.51), mode="letters", session_id=session_id, timestamp=t)

                self.assertEqual(res_confirmed["sign"], "Ê")
                self.assertEqual(res_confirmed["source"], "diacritic_fsm")


if __name__ == "__main__":
    unittest.main()
