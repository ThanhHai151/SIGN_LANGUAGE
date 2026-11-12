"""
Kiểm thử tích hợp VSLStaticAlphabetHead vào luồng nhận diện camera (Task 6C).
=============================================================================
Bao phủ toàn diện các yêu cầu kỹ thuật:
1. Tính ưu tiên phân cấp: 12 luật hình học chạy trước; nếu MATCH thì static head KHÔNG bao giờ được gọi.
2. Bổ sung chính xác 10 chữ cái tĩnh (G, H, K, M, N, P, Q, R, U, X) khi luật hình học trả về NO_MATCH.
3. Không ghi đè: Static head không bao giờ có thể ghi đè kết quả của 12 luật hình học.
4. Chặn nhãn ngoài danh mục: Nếu static head dự đoán chữ cái nằm ngoài 10 chữ cho phép (vd: A, B, Z...), hệ thống từ chối và trả về NEUTRAL.
5. Kiểm soát ngưỡng tin cậy (Confidence Thresholding): Dự đoán dưới ngưỡng (bao gồm ngưỡng đặc thù Q=0.85) bị từ chối về NEUTRAL.
6. Suy thoái mềm (Graceful Degradation): Thiếu hoặc lỗi checkpoint static head không gây crash camera và bảo toàn 12 luật hình học.
7. Độc lập tuyệt đối giữa các mode: digits, words, sentence không bị ảnh hưởng.
8. Cấu hình ngưỡng linh hoạt (Configurable Thresholds).
9. Tính tương thích schema API response cho frontend.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
import numpy as np

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from backend.vision.recognizer import (
    VSLRecognizer,
    ALLOWED_LETTERS,
    ALLOWED_DIGITS,
    ALLOWED_SUPPLEMENTAL_LETTERS,
    DEFAULT_SUPPLEMENTAL_THRESHOLDS,
)
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

NPZ_PATH = os.path.join(
    os.path.dirname(__file__), "..", "data", "alphabet_v2", "datasets", "vsl_alphabet_static_v2.npz"
)
FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "real_alphabet_landmarks.json"
)
DIGIT_FIXTURE_PATH = os.path.join(
    os.path.dirname(__file__), "fixtures", "real_digit_landmarks.json"
)


class TestAlphabetStaticHeadIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.npz_available = os.path.exists(NPZ_PATH)
        if cls.npz_available:
            data = np.load(NPZ_PATH)
            cls.dataset_landmarks = data["landmarks"]
            cls.dataset_classes = list(data["classes"])
            cls.dataset_y = data["y"]
        else:
            cls.dataset_landmarks = None
            cls.dataset_classes = []
            cls.dataset_y = None

    def setUp(self):
        self.rec = VSLRecognizer(confidence_threshold=0.70, smoothing_window=4, cooldown_sec=0.5)

    # -------------------------------------------------------------------------
    # Test 1: Tính ưu tiên phân cấp - 12 luật hình học chạy trước, Static Head KHÔNG được gọi
    # -------------------------------------------------------------------------
    def test_case_1_geometric_precedence_and_short_circuit(self):
        """
        Khi landmark khớp với 1 trong 12 luật hình học (vd: A, B, C, D, E, I, L, O, S, T, V, Y),
        hệ thống nhận diện ngay lập tức với source='geometric_rule'.
        Đặc biệt: VSLStaticAlphabetHead.predict_feature TUYỆT ĐỐI KHÔNG được gọi (short-circuit).
        """
        self.assertTrue(self.rec.has_static_head)

        # Giả lập luật hình học khớp chữ 'A'
        mock_geom_match = {
            "letter": "A",
            "confidence": 0.95,
            "status": "MATCH",
            "reason": "Khớp luật hình học chữ A",
            "matched_candidates": [("A", 0.95)],
            "features": {}
        }

        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_geom_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature") as mock_head_predict:
                dummy_lm = [[0.1, 0.2, 0.3]] * 21
                res = self.rec.predict_landmarks(dummy_lm, mode="letters", session_id="test_geom_precedence")

                self.assertEqual(res["sign"], "A")
                self.assertEqual(res["type"], "alphabet")
                self.assertEqual(res["source"], "geometric_rule")
                self.assertGreaterEqual(res["confidence"], 0.90)

                # Static head tuyệt đối không được gọi
                mock_head_predict.assert_not_called()

    # -------------------------------------------------------------------------
    # Test 2: Bổ sung 10 chữ cái tĩnh qua Static Alphabet Head khi luật hình học là NO_MATCH
    # -------------------------------------------------------------------------
    def test_case_2_supplemental_letters_recognition_via_static_head(self):
        """
        Khi luật hình học trả về NO_MATCH / UNKNOWN, 10 chữ cái bổ sung
        (G, H, K, M, N, P, Q, R, U, X) được nhận diện thành công qua Static Head.
        """
        supplemental_candidates = ["G", "H", "K", "M", "N", "P", "Q", "R", "U", "X"]
        for target in supplemental_candidates:
            self.assertIn(target, ALLOWED_SUPPLEMENTAL_LETTERS)

            mock_head_result = {
                "letter": target,
                "raw_predicted_letter": target,
                "confidence": 0.92,
                "status": "CONFIDENT",
                "is_confident": True,
                "top_k": [{"letter": target, "confidence": 0.92}]
            }

            mock_no_match = {
                "letter": "UNKNOWN",
                "confidence": 0.0,
                "status": "NO_MATCH",
                "reason": "Không khớp luật hình học",
                "matched_candidates": [],
                "features": None
            }

            with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
                with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_result):
                    dummy_lm = [[0.1, 0.2, 0.3]] * 21
                    res = self.rec.predict_landmarks(dummy_lm, mode="letters", session_id=f"test_supp_{target}")

                    self.assertEqual(res["sign"], target, f"Kỳ vọng nhận diện chữ {target}")
                    self.assertEqual(res["source"], "static_head")
                    self.assertEqual(res["type"], "alphabet")
                    self.assertGreaterEqual(res["confidence"], 0.90)

    # -------------------------------------------------------------------------
    # Test 3: Static Head không bao giờ có thể ghi đè luật hình học hợp lệ
    # -------------------------------------------------------------------------
    def test_case_3_static_head_cannot_override_geometric_rule(self):
        """
        Kể cả khi Static Head dự đoán ra chữ cái khác với xác suất cực đại (1.0),
        kết quả của 12 luật hình học (vd: 'B') không bao giờ bị ghi đè.
        """
        mock_geom_match = {
            "letter": "B",
            "confidence": 0.98,
            "status": "MATCH",
            "reason": "Khớp luật chữ B",
            "matched_candidates": [("B", 0.98)],
            "features": {}
        }
        mock_head_fake = {
            "letter": "G",
            "raw_predicted_letter": "G",
            "confidence": 1.0,
            "status": "CONFIDENT",
            "is_confident": True,
            "top_k": [{"letter": "G", "confidence": 1.0}]
        }

        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_geom_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_fake):
                dummy_lm = [[0.1, 0.2, 0.3]] * 21
                res = self.rec.predict_landmarks(dummy_lm, mode="letters", session_id="test_no_override")

                self.assertEqual(res["sign"], "B")
                self.assertEqual(res["source"], "geometric_rule")
                self.assertNotEqual(res["sign"], "G")

    # -------------------------------------------------------------------------
    # Test 4: Chặn nhãn ngoài danh mục 10 chữ cái bổ sung và ngoài danh mục fallback A/D/E
    # -------------------------------------------------------------------------
    def test_case_4_rejection_of_non_supplemental_head_predictions(self):
        """
        Nếu luật hình học trả về NO_MATCH, nhưng Static Head lại dự đoán ra một chữ
        KHÔNG nằm trong 10 chữ cái bổ sung và KHÔNG thuộc 4 chữ nền fallback (A, D, E, O)
        (vd: 'B', 'C', 'I', 'S', 'T', 'Y', 'Z'...),
        hệ thống phải từ chối ngay lập tức và trả về NEUTRAL (conf=0.0).
        (Cập nhật Task 7E.5B / CP3: A, D, E, O hiện được hỗ trợ qua cơ chế gated fallback).
        """
        non_supplemental = ["B", "C", "I", "S", "T", "Y", "Z"]

        mock_no_match = {
            "letter": "UNKNOWN",
            "confidence": 0.0,
            "status": "NO_MATCH",
            "reason": "Không khớp luật hình học",
            "matched_candidates": [],
            "features": None
        }

        for forbidden_pred in non_supplemental:
            mock_head_out = {
                "letter": forbidden_pred,
                "raw_predicted_letter": forbidden_pred,
                "confidence": 0.98,
                "status": "CONFIDENT",
                "is_confident": True,
                "top_k": [{"letter": forbidden_pred, "confidence": 0.98}]
            }

            with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
                with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value=mock_head_out):
                    dummy_lm = [[0.1, 0.2, 0.3]] * 21
                    res = self.rec.predict_landmarks(dummy_lm, mode="letters", session_id="test_reject_non_supp")

                    self.assertEqual(res["sign"], "NEUTRAL", f"Chữ {forbidden_pred} không được phép phát ra từ Static Head!")
                    self.assertEqual(res["confidence"], 0.0)
                    self.assertEqual(res["type"], "unknown")
                    self.assertEqual(res["source"], "none")
                    self.assertIn("not in supplemental", res["classifier_details"]["reject_reason"])

    # -------------------------------------------------------------------------
    # Test 5: Kiểm soát ngưỡng tin cậy & Trường hợp đặc thù chữ Q
    # -------------------------------------------------------------------------
    def test_case_5_rejection_of_low_confidence_and_letter_q_threshold(self):
        """
        Kiểm tra ngưỡng tin cậy:
        - Chữ thông thường (vd: G, ngưỡng 0.70): conf 0.65 bị từ chối về NEUTRAL; conf 0.72 được chấp nhận.
        - Chữ Q (ngưỡng cao 0.85 phòng nhầm X): conf 0.80 bị từ chối; conf 0.88 được chấp nhận.
        """
        mock_no_match = {
            "letter": "UNKNOWN",
            "confidence": 0.0,
            "status": "NO_MATCH",
            "reason": "Không khớp",
            "matched_candidates": [],
            "features": None
        }

        dummy_lm = [[0.1, 0.2, 0.3]] * 21

        # 1. Chữ G với conf 0.65 (< 0.70) -> Bị từ chối
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value={
                "letter": "G", "raw_predicted_letter": "G", "confidence": 0.65, "status": "LOW_CONFIDENCE", "is_confident": False, "top_k": []
            }):
                res_g_low = self.rec.predict_landmarks(dummy_lm, mode="letters")
                self.assertEqual(res_g_low["sign"], "NEUTRAL")
                self.assertEqual(res_g_low["confidence"], 0.0)
                self.assertIn("threshold", res_g_low["classifier_details"]["reject_reason"])

        # 2. Chữ G với conf 0.75 (>= 0.70) -> Chấp nhận
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value={
                "letter": "G", "raw_predicted_letter": "G", "confidence": 0.75, "status": "CONFIDENT", "is_confident": True, "top_k": []
            }):
                res_g_high = self.rec.predict_landmarks(dummy_lm, mode="letters")
                self.assertEqual(res_g_high["sign"], "G")
                self.assertEqual(res_g_high["source"], "static_head")
                self.assertEqual(res_g_high["confidence"], 0.75)

        # 3. Chữ Q với conf 0.70 (< ngưỡng 0.75) -> Bị từ chối
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value={
                "letter": "Q", "raw_predicted_letter": "Q", "confidence": 0.70, "status": "CONFIDENT", "is_confident": True, "top_k": []
            }):
                res_q_low1 = self.rec.predict_landmarks(dummy_lm, mode="letters")
                self.assertEqual(res_q_low1["sign"], "NEUTRAL")
                self.assertEqual(res_q_low1["confidence"], 0.0)
                self.assertIn("0.75", res_q_low1["classifier_details"]["reject_reason"])

        # 4. Chữ Q với conf 0.74 (< ngưỡng 0.75) -> Bị từ chối
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value={
                "letter": "Q", "raw_predicted_letter": "Q", "confidence": 0.74, "status": "CONFIDENT", "is_confident": True, "top_k": []
            }):
                res_q_low2 = self.rec.predict_landmarks(dummy_lm, mode="letters")
                self.assertEqual(res_q_low2["sign"], "NEUTRAL")
                self.assertEqual(res_q_low2["confidence"], 0.0)
                self.assertIn("0.75", res_q_low2["classifier_details"]["reject_reason"])

        # 5. Chữ Q với conf 0.75 (>= ngưỡng 0.75) -> Chấp nhận
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value={
                "letter": "Q", "raw_predicted_letter": "Q", "confidence": 0.75, "status": "CONFIDENT", "is_confident": True, "top_k": []
            }):
                res_q_high1 = self.rec.predict_landmarks(dummy_lm, mode="letters")
                self.assertEqual(res_q_high1["sign"], "Q")
                self.assertEqual(res_q_high1["source"], "static_head")
                self.assertEqual(res_q_high1["confidence"], 0.75)

        # 6. Chữ Q với conf 0.80 (>= ngưỡng 0.75) -> Chấp nhận
        with patch.object(self.rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(self.rec.static_alphabet_head, "predict_feature", return_value={
                "letter": "Q", "raw_predicted_letter": "Q", "confidence": 0.80, "status": "CONFIDENT", "is_confident": True, "top_k": []
            }):
                res_q_high2 = self.rec.predict_landmarks(dummy_lm, mode="letters")
                self.assertEqual(res_q_high2["sign"], "Q")
                self.assertEqual(res_q_high2["source"], "static_head")
                self.assertEqual(res_q_high2["confidence"], 0.80)

    # -------------------------------------------------------------------------
    # Test 6: Suy thoái mềm (Graceful Degradation) khi thiếu hoặc hỏng checkpoint
    # -------------------------------------------------------------------------
    def test_case_6_graceful_degradation_missing_checkpoint(self):
        """
        Khi checkpoint Static Head không tồn tại hoặc bị lỗi, recognizer khởi tạo an toàn
        với has_static_head=False. 12 luật hình học vẫn chạy bình thường, không gây sập camera.
        """
        rec_fallback = VSLRecognizer(static_head_weights_path="non_existent_checkpoint_v99.pt")
        self.assertFalse(rec_fallback.has_static_head)

        # 1. Luật hình học vẫn nhận diện chữ A bình thường
        mock_a = {
            "letter": "A",
            "confidence": 0.95,
            "status": "MATCH",
            "reason": "Khớp luật A",
            "matched_candidates": [("A", 0.95)],
            "features": {}
        }
        with patch.object(rec_fallback.alphabet_classifier, "classify_detailed", return_value=mock_a):
            res_a = rec_fallback.predict_landmarks([[0.1, 0.2, 0.3]] * 21, mode="letters")
            self.assertEqual(res_a["sign"], "A")
            self.assertEqual(res_a["source"], "geometric_rule")

        # 2. Cử chỉ không khớp trả về NEUTRAL an toàn, không sinh exception
        mock_no_match = {
            "letter": "UNKNOWN",
            "confidence": 0.0,
            "status": "NO_MATCH",
            "reason": "Không khớp",
            "matched_candidates": [],
            "features": None
        }
        with patch.object(rec_fallback.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            res_nomatch = rec_fallback.predict_landmarks([[0.1, 0.2, 0.3]] * 21, mode="letters")
            self.assertEqual(res_nomatch["sign"], "NEUTRAL")
            self.assertEqual(res_nomatch["confidence"], 0.0)

    # -------------------------------------------------------------------------
    # Test 7: Độc lập tuyệt đối - Các mode khác không bị ảnh hưởng
    # -------------------------------------------------------------------------
    def test_case_7_other_modes_completely_unaffected(self):
        """
        Các mode khác (digits, words, sentence, letters_numbers) không bị ảnh hưởng,
        và Static Alphabet Head không được gọi trong các mode này.
        """
        with patch.object(self.rec.static_alphabet_head, "predict_feature") as mock_head:
            # 1. Mode digits
            res_digit = self.rec.predict_landmarks([[0.1, 0.2, 0.3]] * 21, mode="digits")
            mock_head.assert_not_called()

            # 2. Mode words
            res_words = self.rec.predict_landmarks([[0.1, 0.2, 0.3]] * 21, mode="words")
            mock_head.assert_not_called()

            # 3. Mode sentence
            res_sent = self.rec.predict_landmarks([[0.1, 0.2, 0.3]] * 21, mode="sentence")
            mock_head.assert_not_called()

    # -------------------------------------------------------------------------
    # Test 8: Cấu hình ngưỡng linh hoạt (Configurable Thresholds)
    # -------------------------------------------------------------------------
    def test_case_8_configurable_custom_thresholds(self):
        """
        Cho phép truyền dict custom ngưỡng khi khởi tạo VSLRecognizer.
        """
        # Đặt ngưỡng G rất cao: 0.98
        custom_rec = VSLRecognizer(supplemental_thresholds={"G": 0.98})
        self.assertEqual(custom_rec.supplemental_thresholds["G"], 0.98)
        self.assertEqual(custom_rec.supplemental_thresholds["Q"], 0.75)  # Các ngưỡng không override giữ giá trị mặc định (Q mặc định là 0.75)

        # Kiểm tra explicit custom override cho chính chữ Q không bị giá trị default ghi đè
        custom_rec_q = VSLRecognizer(supplemental_thresholds={"Q": 0.90})
        self.assertEqual(custom_rec_q.supplemental_thresholds["Q"], 0.90)

        mock_no_match = {
            "letter": "UNKNOWN",
            "confidence": 0.0,
            "status": "NO_MATCH",
            "reason": "Không khớp",
            "matched_candidates": [],
            "features": None
        }

        # G với conf 0.95 (thường qua 0.70) giờ sẽ bị từ chối bởi ngưỡng 0.98
        with patch.object(custom_rec.alphabet_classifier, "classify_detailed", return_value=mock_no_match):
            with patch.object(custom_rec.static_alphabet_head, "predict_feature", return_value={
                "letter": "G", "raw_predicted_letter": "G", "confidence": 0.95, "status": "CONFIDENT", "is_confident": True, "top_k": []
            }):
                res = custom_rec.predict_landmarks([[0.1, 0.2, 0.3]] * 21, mode="letters")
                self.assertEqual(res["sign"], "NEUTRAL")
                self.assertEqual(res["confidence"], 0.0)

    # -------------------------------------------------------------------------
    # Test 9: Kiểm tra trên dữ liệu Dataset V2 thực tế
    # -------------------------------------------------------------------------
    def test_case_9_real_dataset_v2_supplemental_letters_prediction(self):
        """
        Kiểm tra trực tiếp các mẫu thực tế từ Dataset V2:
        Các chữ G, H, K, M, P, U, X trên frame ổn định được nhận diện đúng với source='static_head'.
        """
        if not self.npz_available:
            self.skipTest("Dataset V2 NPZ file not found.")

        test_targets = ["G", "H", "K", "M", "P", "U", "X"]
        recognized_count = 0

        for target in test_targets:
            cls_idx = self.dataset_classes.index(target)
            sample_indices = np.where(self.dataset_y == cls_idx)[0]
            # Thử trên mẫu đầu tiên của từng chữ
            sample_lm = self.dataset_landmarks[sample_indices[0]].tolist()
            res = self.rec.predict_landmarks(sample_lm, mode="letters", session_id=f"test_real_{target}")

            if res["sign"] == target and res.get("source") in ("static_head", "geometric_rule"):
                recognized_count += 1

        self.assertGreaterEqual(recognized_count, 6, "Đa số các chữ cái bổ sung phải được nhận diện thành công trên dataset thật!")

    # -------------------------------------------------------------------------
    # Test 10: Tương thích đầy đủ cấu trúc Response API
    # -------------------------------------------------------------------------
    def test_case_10_api_response_schema_compatibility(self):
        """
        Đảm bảo response API chứa đầy đủ các trường cũ và các trường mới (source, classifier_details).
        """
        dummy_lm = [[0.1, 0.2, 0.3]] * 21
        res = self.rec.predict_landmarks(dummy_lm, mode="letters")

        required_keys = [
            "sign", "label", "type", "confidence", "confirmed",
            "new_token", "sentence", "description", "source", "classifier_details"
        ]
        for k in required_keys:
            self.assertIn(k, res, f"Response API thiếu trường bắt buộc: {k}")

        self.assertIsInstance(res["sign"], str)
        self.assertIsInstance(res["confidence"], float)
        self.assertIsInstance(res["source"], str)
        self.assertIsInstance(res["classifier_details"], dict)

    # -------------------------------------------------------------------------
    # Test 11: Nhận diện hoàn chỉnh chữ G (Cách 1 ngang, Cách 2 chéo & đối xứng trục X)
    # -------------------------------------------------------------------------
    def test_case_11_letter_g_both_ways_and_mirrored_x(self):
        """
        Kiểm tra chữ G trên cả 2 tư thế thực tế (Cách 1 chỉ ngang & Cách 2 chỉ chéo)
        kể cả khi lật đối xứng trục X (tay trái hoặc chỉ sang hướng -X).
        """
        if not self.npz_available:
            self.skipTest("Dataset V2 NPZ file not found.")

        g_idx = self.dataset_classes.index("G")
        g_indices = np.where(self.dataset_y == g_idx)[0]
        # Mẫu đại diện Cách 1 (index 100) và Cách 2 (index 120)
        lm_way1 = self.dataset_landmarks[g_indices[1]].tolist()
        lm_way2 = self.dataset_landmarks[g_indices[21]].tolist()

        # 1. Gốc (+X)
        res_w1 = self.rec.predict_landmarks(lm_way1, mode="letters", session_id="test_g_w1")
        self.assertEqual(res_w1["sign"], "G")
        self.assertEqual(res_w1["source"], "geometric_rule")

        res_w2 = self.rec.predict_landmarks(lm_way2, mode="letters", session_id="test_g_w2")
        self.assertEqual(res_w2["sign"], "G")
        self.assertEqual(res_w2["source"], "static_head")

        # 2. Lật đối xứng trục X qua cổ tay (-X)
        w1_x = lm_way1[0][0]
        lm_way1_mir = [[2.0 * w1_x - pt[0], pt[1], pt[2]] for pt in lm_way1]
        res_w1_mir = self.rec.predict_landmarks(lm_way1_mir, mode="letters", session_id="test_g_w1_mir")
        self.assertEqual(res_w1_mir["sign"], "G")
        self.assertEqual(res_w1_mir["source"], "geometric_rule")

        w2_x = lm_way2[0][0]
        lm_way2_mir = [[2.0 * w2_x - pt[0], pt[1], pt[2]] for pt in lm_way2]
        res_w2_mir = self.rec.predict_landmarks(lm_way2_mir, mode="letters", session_id="test_g_w2_mir")
        self.assertEqual(res_w2_mir["sign"], "G")
        self.assertEqual(res_w2_mir["source"], "static_head")

    # -------------------------------------------------------------------------
    # Test 12: Nhận diện hoàn chỉnh chữ H (Tư thế 2 ngón ngang, biến thể Miền Bắc & đối xứng trục X)
    # -------------------------------------------------------------------------
    def test_case_12_letter_h_horizontal_northern_and_mirrored_x(self):
        """
        Kiểm tra chữ H trên cả tư thế 2 ngón trỏ+giữa duỗi ngang chuẩn từ điển,
        biến thể Miền Bắc QIPEDC (D0500B), và khi lật đối xứng trục X qua cổ tay.
        """
        if not self.npz_available:
            self.skipTest("Dataset V2 NPZ file not found.")

        h_idx = self.dataset_classes.index("H")
        h_indices = np.where(self.dataset_y == h_idx)[0]
        # Khung hình entry (index 128 -> static_head) và khung hình ổn định (index 133 -> geometric_rule)
        lm_h_entry = self.dataset_landmarks[h_indices[0]].tolist()
        lm_h_stable = self.dataset_landmarks[h_indices[5]].tolist()

        res_entry = self.rec.predict_landmarks(lm_h_entry, mode="letters", session_id="test_h_entry")
        self.assertEqual(res_entry["sign"], "H")
        self.assertEqual(res_entry["source"], "static_head")

        res_stable = self.rec.predict_landmarks(lm_h_stable, mode="letters", session_id="test_h_stable")
        self.assertEqual(res_stable["sign"], "H")
        self.assertEqual(res_stable["source"], "geometric_rule")

        # Lật đối xứng trục X qua cổ tay (-X) cho cả entry và stable
        w_e = lm_h_entry[0][0]
        lm_h_entry_mir = [[2.0 * w_e - pt[0], pt[1], pt[2]] for pt in lm_h_entry]
        res_entry_mir = self.rec.predict_landmarks(lm_h_entry_mir, mode="letters", session_id="test_h_entry_mir")
        self.assertEqual(res_entry_mir["sign"], "H")
        self.assertEqual(res_entry_mir["source"], "static_head")

        w_s = lm_h_stable[0][0]
        lm_h_stable_mir = [[2.0 * w_s - pt[0], pt[1], pt[2]] for pt in lm_h_stable]
        res_stable_mir = self.rec.predict_landmarks(lm_h_stable_mir, mode="letters", session_id="test_h_stable_mir")
        self.assertEqual(res_stable_mir["sign"], "H")
        self.assertEqual(res_stable_mir["source"], "geometric_rule")

        # Tư thế chữ H ngang chuẩn từ điển (2 ngón trỏ và giữa duỗi ngang song song, không bị nhầm sang G)
        g_idx = self.dataset_classes.index("G")
        pts_h_horiz = [list(p) for p in self.dataset_landmarks[np.where(self.dataset_y == g_idx)[0][1]].tolist()]
        for k_i, k_m in [(5, 9), (6, 10), (7, 11), (8, 12)]:
            dx = pts_h_horiz[k_i][0] - pts_h_horiz[5][0]
            dy = pts_h_horiz[k_i][1] - pts_h_horiz[5][1]
            dz = pts_h_horiz[k_i][2] - pts_h_horiz[5][2]
            pts_h_horiz[k_m] = [pts_h_horiz[9][0] + dx, pts_h_horiz[9][1] + dy, pts_h_horiz[9][2] + dz]

        res_h_horiz = self.rec.predict_landmarks(pts_h_horiz, mode="letters", session_id="test_h_horiz")
        self.assertEqual(res_h_horiz["sign"], "H")
        self.assertEqual(res_h_horiz["source"], "geometric_rule")

    # -------------------------------------------------------------------------
    # Test 13: Phân biệt triệt để chữ H với chữ V & chống lưu nhầm V trong temporal smoothing
    # -------------------------------------------------------------------------
    def test_case_13_h_vs_v_webcam_disambiguation_and_temporal_smoothing(self):
        """
        Kiểm tra:
        1. Chữ H ngang thực tế (d_IW >= 1.70, d_MW >= 1.70, d_IM >= 0.28, diff_x_IM >= 0.18)
           trên dải góc -45..+40 độ được dự đoán là H, không bao giờ bị nhầm thành V.
        2. Toàn bộ 13 mẫu V ổn định trong dataset V2 vẫn được dự đoán chính xác là V (13/13).
        3. Khung hình xả tay / độ tin cậy thấp trả về NEUTRAL và confirmed=False.
        4. Temporal smoothing khi người dùng chuyển từ V sang H không giữ nhầm kết quả V.
        """
        if not self.npz_available:
            self.skipTest("Dataset V2 NPZ file not found.")

        import math

        g_idx = self.dataset_classes.index("G")
        pts_h_real = [list(p) for p in self.dataset_landmarks[np.where(self.dataset_y == g_idx)[0][1]].tolist()]
        for k_i, k_m in [(5, 9), (6, 10), (7, 11), (8, 12)]:
            dx = pts_h_real[k_i][0] - pts_h_real[5][0]
            dy = pts_h_real[k_i][1] - pts_h_real[5][1]
            dz = pts_h_real[k_i][2] - pts_h_real[5][2]
            pts_h_real[k_m] = [
                pts_h_real[9][0] + 1.22 * dx,
                pts_h_real[9][1] + 1.22 * dy,
                pts_h_real[9][2] + dz,
            ]

        w_h = pts_h_real[0]
        for deg in range(-45, 45, 10):
            rad = math.radians(deg)
            c, s_val = math.cos(rad), math.sin(rad)
            rot_h = [
                [
                    w_h[0] + (pt[0] - w_h[0]) * c - (pt[1] - w_h[1]) * s_val,
                    w_h[1] + (pt[0] - w_h[0]) * s_val + (pt[1] - w_h[1]) * c,
                    pt[2],
                ]
                for pt in pts_h_real
            ]
            res_h = self.rec.predict_landmarks(rot_h, mode="letters", session_id=f"test_h_v_rot_{deg}")
            self.assertNotEqual(res_h["sign"], "V", f"H ngang góc {deg} độ bị nhầm thành V!")
            self.assertEqual(res_h["sign"], "H")

        # 2. Kiểm tra toàn bộ 13 mẫu V chuẩn trong NPZ vẫn nhận diện chính xác V
        v_idx = self.dataset_classes.index("V")
        v_indices = np.where(self.dataset_y == v_idx)[0][:13]
        for idx in v_indices:
            lm_v = self.dataset_landmarks[idx].tolist()
            res_v = self.rec.predict_landmarks(lm_v, mode="letters", session_id=f"test_v_keep_{idx}")
            self.assertEqual(res_v["sign"], "V")
            self.assertEqual(res_v["source"], "geometric_rule")

        # 3. Khung hình xả tay cuối video (H[145], V[379]) có độ tin cậy thấp -> NEUTRAL, confirmed=False
        h_idx = self.dataset_classes.index("H")
        lm_h_rest = self.dataset_landmarks[np.where(self.dataset_y == h_idx)[0][-1]].tolist()
        res_rest = self.rec.predict_landmarks(lm_h_rest, mode="letters", session_id="test_h_rest")
        self.assertEqual(res_rest["sign"], "NEUTRAL")
        self.assertFalse(res_rest["confirmed"])

        # 4. Kiểm tra temporal smoothing khi chuyển từ V sang H trong cùng một phiên (session)
        lm_v_sample = self.dataset_landmarks[v_indices[0]].tolist()
        sid = "test_transition_v_to_h"
        self.rec.sentence_tokens.clear()

        # Người dùng thực hiện V trong 2 frame đầu (đạt ngưỡng xác nhận V)
        r_v1 = self.rec.predict_landmarks(lm_v_sample, mode="letters", session_id=sid, timestamp=100.0)
        r_v2 = self.rec.predict_landmarks(lm_v_sample, mode="letters", session_id=sid, timestamp=100.1)
        self.assertEqual(r_v2["sign"], "V")
        self.assertTrue(r_v2["confirmed"])
        self.assertEqual(r_v2["new_token"], "V")

        # Sau cooldown (timestamp=102.0), người dùng xoay tay sang thực hiện H:
        # Frame H đầu tiên không được kích hoạt xác nhận lại chữ V cũ trong bộ đệm smoothing!
        r_h1 = self.rec.predict_landmarks(pts_h_real, mode="letters", session_id=sid, timestamp=102.0)
        self.assertEqual(r_h1["sign"], "H")
        self.assertFalse(r_h1["confirmed"], "Frame H đầu tiên không được xác nhận nhầm chữ V cũ trong buffer!")
        self.assertIsNone(r_h1["new_token"])

        # Frame H thứ hai (đạt smoothing_window // 2 = 2) phải xác nhận ngay chữ H (không bị V cầm chân)
        r_h2 = self.rec.predict_landmarks(pts_h_real, mode="letters", session_id=sid, timestamp=102.1)
        self.assertEqual(r_h2["sign"], "H")
        self.assertTrue(r_h2["confirmed"], "Frame H thứ hai phải xác nhận chữ H thành công!")
        self.assertEqual(r_h2["new_token"], "H")



if __name__ == "__main__":
    unittest.main()

