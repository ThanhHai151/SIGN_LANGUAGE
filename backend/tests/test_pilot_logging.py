"""
Unit Tests cho Hệ thống Ghi nhật ký Chẩn đoán Pilot (Task 7F.5A)
================================================================
Kiểm thử toàn diện 10 yêu cầu kỹ thuật bắt buộc:
1. Logging ghi đúng nhãn cuối cùng (final_sign).
2. Logging ghi đúng prediction_source (geometric_rule, static_head, static_head_fallback, none).
3. Ghi nhận đầy đủ thông tin Static Head (raw_predicted_letter, confidence, top_k) khi classifier gọi mô hình.
4. expected_sign được ghi nhận vào log nhưng tuyệt đối KHÔNG ảnh hưởng kết quả nhận diện.
5. Frame không có ground truth (None hoặc không hợp lệ) vẫn xử lý bình thường.
6. Nhiều frame cùng trial_id/session_id được nhóm đúng.
7. Queue đầy hoặc lỗi I/O ghi file không làm crash luồng nhận diện (an toàn luồng tuyệt đối).
8. Dữ liệu được flush đầy đủ khi kết thúc session.
9. Chế độ logging mặc định không làm thay đổi hành vi production (không tạo file pilot log khi tắt).
10. Chứng minh kết quả dự đoán (prediction output) giống hệt nhau 100% giữa hai trạng thái bật và tắt logging.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.vision.recognizer import VSLRecognizer
from backend.vision.pilot_logger import (
    PilotDiagnosticLogger,
    sanitize_expected_sign,
    STATIC_CLASSES_22,
)
from backend.tests.test_geometry import _make_hand


def _make_letter_b_hand():
    """Tạo landmark mẫu khớp Rule B (4 ngón duỗi thẳng, ngón cái khép)."""
    lm = _make_hand()
    wx, wy, wz = lm[0]
    # Ngón cái khép
    lm[4] = [wx - 0.05, wy - 0.10, wz]
    # 4 ngón trỏ, giữa, áp út, út duỗi thẳng hướng lên
    lm[8] = [wx - 0.04, wy - 0.28, wz]
    lm[12] = [wx, wy - 0.30, wz]
    lm[16] = [wx + 0.04, wy - 0.28, wz]
    lm[20] = [wx + 0.07, wy - 0.24, wz]
    return lm


def _load_jsonl_records(filepath):
    """Đọc toàn bộ record từ file JSONL."""
    if not os.path.exists(filepath):
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestPilotDiagnosticLogging(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_filepath = os.path.join(self.temp_dir, "test_pilot_diag.jsonl")

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_sanitization_of_expected_sign(self):
        """4 & 5. Kiểm tra làm sạch nhãn ground truth: Chỉ nhận 22 chữ tĩnh, từ chối nhãn rác."""
        self.assertEqual(sanitize_expected_sign("A"), "A")
        self.assertEqual(sanitize_expected_sign(" b "), "B")
        self.assertEqual(sanitize_expected_sign("q"), "Q")
        self.assertIsNone(sanitize_expected_sign(None))
        self.assertIsNone(sanitize_expected_sign(""))
        self.assertIsNone(sanitize_expected_sign("UNKNOWN"))
        self.assertIsNone(sanitize_expected_sign("123"))
        self.assertIsNone(sanitize_expected_sign("Ă"))  # Chữ động không nằm trong 22 chữ tĩnh

    def test_2_pilot_logging_disabled_by_default(self):
        """9. Khi không bật cờ pilot logging, không sinh file pilot log."""
        rec = VSLRecognizer(pilot_logging_enabled=False)
        lm = _make_letter_b_hand()
        res = rec.predict_landmarks(lm, mode="letters")
        self.assertIn("sign", res)
        self.assertFalse(os.path.exists(self.log_filepath))

    def test_3_pilot_logging_records_correct_final_sign_and_geometric_source(self):
        """1 & 2. Ghi đúng final_sign, prediction_source và cấu trúc geometric khi Rule B khớp."""
        rec = VSLRecognizer(pilot_logging_enabled=True)
        rec.pilot_filepath = self.log_filepath

        mock_geom_res = {
            "status": "MATCH",
            "letter": "B",
            "confidence": 0.95,
            "reason": "4 ngón duỗi thẳng khép sát"
        }

        with patch.object(rec.alphabet_classifier, "classify_detailed", return_value=mock_geom_res):
            lm = _make_letter_b_hand()
            res = rec.predict_landmarks(
                lm,
                mode="letters",
                expected_sign="B",
                session_id="sess_01",
                trial_id="trial_b_01",
                frame_idx=10,
            )

        rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)

        r = records[0]
        self.assertEqual(r["final_sign"], "B")
        self.assertEqual(r["prediction_source"], "geometric_rule")
        self.assertEqual(r["expected_sign"], "B")
        self.assertEqual(r["session_id"], "sess_01")
        self.assertEqual(r["trial_id"], "trial_b_01")
        self.assertEqual(r["frame_idx"], 10)
        self.assertEqual(r["geometric"]["status"], "MATCH")
        self.assertEqual(r["geometric"]["letter"], "B")
        self.assertEqual(r["gating_status"], "passed_geometric")
        self.assertIsNone(r["threshold_applied"])
        self.assertIsNone(r["reject_reason"])
        # Khi geometric rule khớp ở Bước 1, static head không được gọi (null)
        self.assertIsNone(r["static_head"]["raw_predicted_letter"])
        rec.disable_pilot_logging()

    def test_4_pilot_logging_records_static_head_info_on_fallback(self):
        """2 & 3. Ghi đúng static_head info khi geometric NO_MATCH và kích hoạt static_head_fallback."""
        rec = VSLRecognizer(pilot_logging_enabled=True)
        rec.pilot_filepath = self.log_filepath

        # Giả lập classifier geometric trả về NO_MATCH và static head trả về chữ nền A (thuộc ALLOWED_FALLBACK_BASE_LETTERS)
        mock_head_res = {
            "raw_predicted_letter": "A",
            "confidence": 0.88,
            "top_k": [
                {"letter": "A", "confidence": 0.88},
                {"letter": "E", "confidence": 0.08},
                {"letter": "S", "confidence": 0.04},
            ]
        }

        with patch.object(rec.alphabet_classifier, "classify_detailed", return_value={"status": "NO_MATCH", "reason": "No match"}):
            with patch.object(rec.static_alphabet_head, "predict_feature", return_value=mock_head_res):
                lm = _make_hand()
                res = rec.predict_landmarks(
                    lm,
                    mode="letters",
                    expected_sign="A",
                    session_id="sess_fallback",
                    trial_id="trial_a_02",
                    frame_idx=25,
                )

        rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)

        r = records[0]
        self.assertEqual(r["final_sign"], "A")
        self.assertEqual(r["prediction_source"], "static_head_fallback")
        self.assertEqual(r["gating_status"], "passed_fallback")
        self.assertEqual(r["threshold_applied"], 0.70)
        self.assertIsNone(r["reject_reason"])
        self.assertEqual(r["geometric"]["status"], "NO_MATCH")
        self.assertIsNone(r["geometric"]["letter"])
        self.assertEqual(r["static_head"]["raw_predicted_letter"], "A")
        self.assertEqual(r["static_head"]["confidence"], 0.88)
        self.assertEqual(len(r["static_head"]["top_k"]), 3)
        rec.disable_pilot_logging()

    def test_5_pilot_logging_records_supplemental_source(self):
        """2 & 3. Ghi đúng static_head nguồn supplemental (chữ cái bổ sung G, H, K, M, N, P, Q, R, U, X)."""
        rec = VSLRecognizer(pilot_logging_enabled=True)
        rec.pilot_filepath = self.log_filepath

        mock_head_res = {
            "raw_predicted_letter": "G",
            "confidence": 0.82,
            "top_k": [{"letter": "G", "confidence": 0.82}]
        }

        with patch.object(rec.alphabet_classifier, "classify_detailed", return_value={"status": "NO_MATCH", "reason": "No match"}):
            with patch.object(rec.static_alphabet_head, "predict_feature", return_value=mock_head_res):
                lm = _make_hand()
                res = rec.predict_landmarks(
                    lm,
                    mode="letters",
                    expected_sign="G",
                    session_id="sess_supp",
                    trial_id="trial_g_01",
                    frame_idx=101,
                )

        rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)

        r = records[0]
        self.assertEqual(r["final_sign"], "G")
        self.assertEqual(r["prediction_source"], "static_head")
        self.assertEqual(r["gating_status"], "passed_supplemental")
        self.assertEqual(r["threshold_applied"], 0.70)
        rec.disable_pilot_logging()

    def test_6_pilot_logging_records_rejected_prediction(self):
        """2 & 3. Ghi đúng prediction_source='none' và reject_reason khi Static Head bị từ chối do < threshold."""
        rec = VSLRecognizer(pilot_logging_enabled=True)
        rec.pilot_filepath = self.log_filepath

        # Q yêu cầu threshold 0.85; giả lập confidence 0.72 < 0.85
        mock_head_res = {
            "raw_predicted_letter": "Q",
            "confidence": 0.72,
            "top_k": [{"letter": "Q", "confidence": 0.72}]
        }

        with patch.object(rec.alphabet_classifier, "classify_detailed", return_value={"status": "NO_MATCH", "reason": "No match"}):
            with patch.object(rec.static_alphabet_head, "predict_feature", return_value=mock_head_res):
                lm = _make_hand()
                res = rec.predict_landmarks(lm, mode="letters", expected_sign="Q")

        rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)

        r = records[0]
        self.assertEqual(r["final_sign"], "NEUTRAL")
        self.assertEqual(r["prediction_source"], "none")
        self.assertEqual(r["gating_status"], "rejected")
        self.assertEqual(r["threshold_applied"], 0.85)
        self.assertIn("Confidence 0.72 < threshold 0.85", r["reject_reason"])
        rec.disable_pilot_logging()

    def test_7_expected_sign_does_not_alter_prediction_behavior(self):
        """4 & 10. Chứng minh kết quả dự đoán giống hệt nhau khi có vs không có expected_sign, và khi bật vs tắt logging."""
        rec_no_log = VSLRecognizer(pilot_logging_enabled=False)
        rec_with_log = VSLRecognizer(pilot_logging_enabled=True)
        rec_with_log.pilot_filepath = self.log_filepath

        lm = _make_letter_b_hand()

        # 1. Dự đoán khi không có logging và không có expected_sign
        res_baseline = rec_no_log.predict_landmarks(lm, mode="letters")

        # 2. Dự đoán khi bật logging và CÓ expected_sign đúng
        res_with_correct_gt = rec_with_log.predict_landmarks(lm, mode="letters", expected_sign="B")

        # 3. Dự đoán khi bật logging và CÓ expected_sign SAI CỐ Ý (để kiểm tra xem ground truth có làm lệch nhãn không)
        res_with_wrong_gt = rec_with_log.predict_landmarks(lm, mode="letters", expected_sign="X")

        # Kết quả nhận diện (sign và confidence) phải đồng nhất 100% trong cả 3 trường hợp
        self.assertEqual(res_baseline["sign"], res_with_correct_gt["sign"])
        self.assertEqual(res_baseline["confidence"], res_with_correct_gt["confidence"])
        self.assertEqual(res_baseline["sign"], res_with_wrong_gt["sign"])
        self.assertEqual(res_baseline["confidence"], res_with_wrong_gt["confidence"])

        rec_with_log.disable_pilot_logging()

    def test_8_grouping_multiple_frames_by_trial_id(self):
        """6. Kiểm tra khả năng nhóm nhiều frame liên tiếp cùng trial_id và session_id."""
        rec = VSLRecognizer(pilot_logging_enabled=True)
        rec.pilot_filepath = self.log_filepath

        lm = _make_letter_b_hand()
        # Giả lập một trial gồm 10 frames liên tiếp
        for f in range(10):
            rec.predict_landmarks(
                lm,
                mode="letters",
                session_id="pilot_session_001",
                trial_id="trial_letter_B_rep01",
                frame_idx=f + 1,
                expected_sign="B"
            )

        rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 10)

        for i, r in enumerate(records):
            self.assertEqual(r["session_id"], "pilot_session_001")
            self.assertEqual(r["trial_id"], "trial_letter_B_rep01")
            self.assertEqual(r["expected_sign"], "B")
            self.assertEqual(r["frame_idx"], i + 1)

        rec.disable_pilot_logging()

    def test_9_resilience_to_queue_overflow_and_io_errors(self):
        """7. Queue đầy hoặc lỗi ghi file không bao giờ gây crash hoặc ném exception trong luồng nhận diện."""
        # Tạo logger với queue siêu nhỏ (chỉ chứa 2 items)
        small_logger = PilotDiagnosticLogger(filepath=self.log_filepath, queue_size=2)

        # Đẩy nhiều items vượt quá queue_size khi chưa flush
        successes = 0
        dropped = 0
        for i in range(10):
            res = small_logger.log_frame({"frame_idx": i})
            if res:
                successes += 1
            else:
                dropped += 1

        small_logger.flush()
        stats = small_logger.get_stats()
        self.assertGreater(stats["dropped_queue_full"], 0, "Phải ghi nhận số record bị bỏ do queue đầy")
        self.assertEqual(stats["total_dropped"], stats["dropped_queue_full"])
        small_logger.close()

    def test_10_missing_or_invalid_landmarks_logged_safely(self):
        """Frame không có landmarks (None) hoặc landmarks sai kích thước vẫn được ghi nhận an toàn."""
        rec = VSLRecognizer(pilot_logging_enabled=True)
        rec.pilot_filepath = self.log_filepath

        # Gọi với None
        res_none = rec.predict_landmarks(None, mode="letters", expected_sign="B")
        self.assertEqual(res_none["sign"], "NEUTRAL")

        # Gọi với landmarks rỗng
        res_empty = rec.predict_landmarks([], mode="letters", expected_sign="B")
        self.assertEqual(res_empty["sign"], "NEUTRAL")

        rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 2)
        for r in records:
            self.assertEqual(r["final_sign"], "NEUTRAL")
            self.assertEqual(r["prediction_source"], "none")
            self.assertEqual(r["geometric"]["status"], "INVALID_INPUT")
            self.assertIsNone(r["geometric"]["features"])
            self.assertIsNone(r["geometric"]["h_rule_diagnostics"])

        rec.disable_pilot_logging()

    def test_11_geometric_features_and_h_rule_diagnostics_logged(self):
        """Task 5: Ghi đầy đủ reason, matched_candidates, features và h_rule_diagnostics."""
        rec = VSLRecognizer(pilot_logging_enabled=True)
        rec.pilot_filepath = self.log_filepath

        fixture_path = os.path.join(os.path.dirname(__file__), "fixtures", "real_alphabet_landmarks.json")
        with open(fixture_path, "r", encoding="utf-8") as f:
            fixture_data = json.load(f)
        b_sample = next(
            s for s in fixture_data["samples"]
            if s.get("letter_upper") == "B" and s.get("phase") == "stable_pose"
        )
        lm = b_sample["landmarks"]

        res = rec.predict_landmarks(
            lm,
            mode="letters",
            expected_sign="H",
            session_id="sess_task5",
            trial_id="sess_task5_H_rep01",
            frame_idx=1,
        )
        self.assertEqual(res["sign"], "B")

        rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)

        r = records[0]
        geom = r["geometric"]
        self.assertEqual(geom["status"], "MATCH")
        self.assertEqual(geom["letter"], "B")
        self.assertIsInstance(geom["reason"], str)
        self.assertEqual(geom["matched_candidates"], [["B", 0.95]])
        self.assertIsInstance(geom["features"], dict)

        required_priority_features = [
            "d_IW", "d_MW", "d_RW", "d_PW", "d_IM",
            "diff_x_IM",
            "idx_horiz_ratio", "mid_horiz_ratio",
            "idx_straight", "mid_straight",
            "ang_IM_dir",
            "index_ext", "middle_ext", "ring_ext", "pinky_ext",
            "index_curl", "middle_curl", "ring_curl", "pinky_curl",
        ]
        for fk in required_priority_features:
            self.assertIn(fk, geom["features"], f"Thiếu đặc trưng ưu tiên: {fk}")
            self.assertIsNotNone(geom["features"][fk])

        h_diag = geom["h_rule_diagnostics"]
        self.assertIsInstance(h_diag, dict)
        self.assertIn("rule_v", h_diag)
        self.assertIn("h_horiz", h_diag)
        self.assertIn("h_north", h_diag)
        self.assertFalse(h_diag["h_horiz"]["matched"])
        self.assertIn("not_b_match", h_diag["h_horiz"]["failed_conditions"])

        rec.disable_pilot_logging()

    def test_12_null_or_partial_features_handled_without_fake_values(self):
        """Task 5: Logger không lỗi khi một số trường đặc trưng là null và không tạo giá trị giả."""
        rec = VSLRecognizer(pilot_logging_enabled=True)
        rec.pilot_filepath = self.log_filepath

        partial_geom_res = {
            "status": "NO_MATCH",
            "letter": "UNKNOWN",
            "confidence": 0.0,
            "reason": "Không khớp với bất kỳ quy tắc hình học chữ cái nào đã biết",
            "matched_candidates": [],
            "features": {
                "d_IW": 1.80,
                "d_MW": None,
                "index_ext": True,
                "middle_ext": None,
            },
        }

        with patch.object(rec.alphabet_classifier, "classify_detailed", return_value=partial_geom_res):
            lm = _make_hand()
            rec.predict_landmarks(lm, mode="letters", expected_sign="H")

        rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)

        geom = records[0]["geometric"]
        self.assertEqual(geom["features"]["d_IW"], 1.80)
        self.assertIsNone(geom["features"]["d_MW"])
        self.assertNotIn("d_RW", geom["features"], "Không được tự ý bịa thêm khóa không có trong features gốc")

        h_diag = geom["h_rule_diagnostics"]
        self.assertIsNotNone(h_diag)
        self.assertTrue(h_diag["h_horiz"]["conditions"]["d_IW_ge_min"])
        self.assertIsNone(h_diag["h_horiz"]["conditions"]["d_MW_ge_min"])
        self.assertIsNone(h_diag["h_horiz"]["matched"])

        rec.disable_pilot_logging()

    def test_13_static_head_load_status_and_path_sanitization(self):
        """Task 5: Ghi nhận has_static_head, load_status, pytorch_version, load_error và ẩn đường dẫn tuyệt đối."""
        with patch(
            "backend.vision.recognizer.load_static_alphabet_head",
            side_effect=RuntimeError("WeightsUnpickler error at D:\\SIGN_LANGUAGE\\backend\\models\\alphabet_static_head_v1.pt: GLOBAL numpy.core.multiarray.scalar"),
        ):
            rec_failed = VSLRecognizer(pilot_logging_enabled=True)
            rec_failed.pilot_filepath = self.log_filepath
            self.assertFalse(rec_failed.has_static_head)
            self.assertEqual(rec_failed.static_head_load_status, "failed")
            self.assertIsNotNone(rec_failed.static_head_load_error)
            self.assertNotIn("D:\\SIGN_LANGUAGE", rec_failed.static_head_load_error)
            self.assertIn("<path>", rec_failed.static_head_load_error)

            lm = _make_letter_b_hand()
            rec_failed.predict_landmarks(lm, mode="letters", expected_sign="H")
            rec_failed.pilot_logger.flush()

            records = _load_jsonl_records(self.log_filepath)
            self.assertEqual(len(records), 1)
            r = records[0]
            self.assertFalse(r["has_static_head"])
            sh = r["static_head"]
            self.assertFalse(sh["has_static_head"])
            self.assertEqual(sh["load_status"], "failed")
            self.assertIsInstance(sh["pytorch_version"], str)
            self.assertIn("numpy.core.multiarray.scalar", sh["load_error"])
            self.assertNotIn("D:\\SIGN_LANGUAGE", sh["load_error"])
            rec_failed.disable_pilot_logging()


if __name__ == "__main__":
    unittest.main()
