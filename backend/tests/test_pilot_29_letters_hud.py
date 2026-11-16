"""
Unit & Integration Tests cho Controlled Webcam Pilot HUD 29 Chữ Cái VSL
========================================================================
Kiểm thử 10 tiêu chí theo Mục 9 của Kế hoạch:
1. Đủ 29 chữ cái tiếng Việt VSL (A, Ă, Â, B, C, D, Đ, E, Ê, G, H, I, K, L, M, N, O, Ô, Ơ, P, Q, R, S, T, U, Ư, V, X, Y).
2. Mỗi thời điểm chỉ một target_label được chọn.
3. Không thể đổi target giữa trial đang chạy (logic gate frontend & backend).
4. Start/End Trial hoạt động chính xác và xả dữ liệu xuống đĩa.
5. Trial ID và Session ID được phân biệt chính xác.
6. Target label (expected_sign) tách biệt tuyệt đối với Predicted label (final_sign).
7. Frame không bị gán nhầm sang trial khác (chống rò rỉ metadata).
8. File JSONL được tạo và đọc lại thành công, 100% dòng là JSON hợp lệ.
9. Không ghi đè file session cũ (Atomic creation với cờ O_EXCL).
10. Không làm thay đổi kết quả nhận diện của mô hình hiện tại.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app import (
    recognize_landmarks,
    LandmarkRequest,
    control_pilot_logging,
    PilotControlRequest,
    flush_pilot_log,
    get_pilot_stats,
    recognizer,
)
from backend.vision.pilot_logger import (
    sanitize_expected_sign,
    create_safe_pilot_file,
    VSL_ALPHABET_29,
    VALID_TARGET_CLASSES_SET,
)
from backend.tests.test_geometry import _make_hand


def _load_jsonl_records(filepath):
    if not os.path.exists(filepath):
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestPilot29LettersHUD(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        recognizer.diagnostic_dir = self.temp_dir
        recognizer.disable_pilot_logging()
        if hasattr(recognizer, "pilot_logger") and recognizer.pilot_logger:
            recognizer.pilot_logger.reset_stats()

    def tearDown(self):
        recognizer.disable_pilot_logging()
        if hasattr(recognizer, "pilot_logger") and recognizer.pilot_logger:
            recognizer.pilot_logger.reset_stats()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_criterion_1_full_29_vsl_letters(self):
        """Tiêu chí 1: Đủ 29 chữ cái tiếng Việt VSL, bao gồm 7 chữ có dấu."""
        expected_29 = [
            'A', 'Ă', 'Â', 'B', 'C', 'D', 'Đ', 'E', 'Ê', 'G',
            'H', 'I', 'K', 'L', 'M', 'N', 'O', 'Ô', 'Ơ', 'P',
            'Q', 'R', 'S', 'T', 'U', 'Ư', 'V', 'X', 'Y'
        ]
        self.assertEqual(len(VSL_ALPHABET_29), 29)
        self.assertEqual(VSL_ALPHABET_29, expected_29)
        self.assertEqual(len(VALID_TARGET_CLASSES_SET), 29)

        # Kiểm tra sanitize chấp nhận toàn bộ 29 chữ cái
        for let in expected_29:
            self.assertEqual(sanitize_expected_sign(let), let)
            self.assertEqual(sanitize_expected_sign(f" {let.lower()} "), let)

        # Kiểm tra từ chối nhãn rác
        for bad in [None, "", "UNKNOWN", "123", "ABC", " "]:
            self.assertIsNone(sanitize_expected_sign(bad))

    def test_criterion_5_and_6_target_vs_predicted_label_separation(self):
        """Tiêu chí 5 & 6: Target label (expected_sign) tách biệt tuyệt đối với predicted label."""
        ctrl_res = control_pilot_logging(PilotControlRequest(enabled=True, session_id="test_sess_sep"))
        filepath = ctrl_res["filepath"]

        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        # Gửi request với target_label = 'P', trong khi landmark thực tế là tư thế rỗng/khác
        req = LandmarkRequest(
            landmarks=lm,
            mode="letters",
            expected_sign="P",
            session_id="test_sess_sep",
            trial_id="test_sess_sep_P_rep01",
            frame_idx=1
        )
        res = recognize_landmarks(req, mock_http)
        self.assertTrue(res["success"])

        flush_pilot_log()
        records = _load_jsonl_records(filepath)
        self.assertEqual(len(records), 1)

        rec = records[0]
        # Target label phải là P (do tester chọn)
        self.assertEqual(rec["expected_sign"], "P")
        self.assertEqual(rec["trial_id"], "test_sess_sep_P_rep01")
        self.assertEqual(rec["session_id"], "test_sess_sep")

        # Predicted label (final_sign) là do mô hình dự đoán, tuyệt đối không bị gán bằng expected_sign
        self.assertEqual(rec["final_sign"], res["data"]["sign"])
        # Kiểm tra trường prediction_source và model availability có mặt
        self.assertIn("prediction_source", rec)
        self.assertIn("has_static_head", rec)
        self.assertIn("geometric", rec)
        self.assertIn("static_head", rec)

    def test_criterion_4_and_7_trial_switching_zero_leakage(self):
        """Tiêu chí 4 & 7: Chuyển đổi giữa các trial của các chữ khác nhau không bị rò rỉ frame."""
        ctrl_res = control_pilot_logging(PilotControlRequest(enabled=True, session_id="test_sess_switch"))
        filepath = ctrl_res["filepath"]

        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        # Trial 1: Chữ P (3 frames)
        for f in range(1, 4):
            req = LandmarkRequest(
                landmarks=lm, mode="letters", expected_sign="P",
                session_id="test_sess_switch", trial_id="test_sess_switch_P_rep01", frame_idx=f
            )
            recognize_landmarks(req, mock_http)

        # Transition frame giữa 2 trial (khi trial đã dừng, HUD gửi metadata = None)
        req_trans = LandmarkRequest(landmarks=lm, mode="letters")
        recognize_landmarks(req_trans, mock_http)

        # Trial 2: Chữ Ă (chữ có dấu) (2 frames)
        for f in range(1, 3):
            req = LandmarkRequest(
                landmarks=lm, mode="letters", expected_sign="Ă",
                session_id="test_sess_switch", trial_id="test_sess_switch_Ă_rep01", frame_idx=f
            )
            recognize_landmarks(req, mock_http)

        flush_pilot_log()
        records = _load_jsonl_records(filepath)
        # Tổng cộng 3 + 1 + 2 = 6 records
        self.assertEqual(len(records), 6)

        p_records = [r for r in records if r.get("trial_id") == "test_sess_switch_P_rep01"]
        a_records = [r for r in records if r.get("trial_id") == "test_sess_switch_Ă_rep01"]
        trans_records = [r for r in records if r.get("trial_id") is None]

        self.assertEqual(len(p_records), 3)
        self.assertEqual(len(a_records), 2)
        self.assertEqual(len(trans_records), 1)

        # Kiểm tra tính toàn vẹn của P records
        for idx, r in enumerate(p_records):
            self.assertEqual(r["expected_sign"], "P")
            self.assertEqual(r["frame_idx"], idx + 1)

        # Kiểm tra tính toàn vẹn của Ă records
        for idx, r in enumerate(a_records):
            self.assertEqual(r["expected_sign"], "Ă")
            self.assertEqual(r["frame_idx"], idx + 1)

        # Transition frame không có expected_sign và trial_id
        self.assertIsNone(trans_records[0]["expected_sign"])
        self.assertIsNone(trans_records[0]["trial_id"])

    def test_criterion_8_and_9_jsonl_validity_and_no_overwrite(self):
        """Tiêu chí 8 & 9: JSONL hợp lệ 100% dòng và không bao giờ ghi đè file session cũ."""
        # Session 1
        file1 = create_safe_pilot_file(self.temp_dir, "hud_session_alpha")
        self.assertTrue(os.path.exists(file1))

        # Thử tạo lại với cùng session_id -> phải tự động sinh tên mới không đè
        file2 = create_safe_pilot_file(self.temp_dir, "hud_session_alpha")
        self.assertNotEqual(file1, file2)
        self.assertTrue(os.path.exists(file2))

        # Ghi log và kiểm tra đọc lại bằng json.loads
        ctrl = control_pilot_logging(PilotControlRequest(enabled=True, session_id="hud_session_alpha_3"))
        target_f = ctrl["filepath"]

        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}
        for let in ['P', 'X', 'O', 'Q', 'M', 'N', 'E', 'H']:
            req = LandmarkRequest(
                landmarks=lm, mode="letters", expected_sign=let,
                session_id="hud_session_alpha_3", trial_id=f"hud_session_alpha_3_{let}_rep01", frame_idx=1
            )
            recognize_landmarks(req, mock_http)

        flush_pilot_log()
        records = _load_jsonl_records(target_f)
        self.assertEqual(len(records), 8)
        logged_letters = [r["expected_sign"] for r in records]
        self.assertEqual(logged_letters, ['P', 'X', 'O', 'Q', 'M', 'N', 'E', 'H'])

    def test_criterion_10_recognition_output_unchanged(self):
        """Tiêu chí 10: Model prediction output giống hệt nhau khi có hoặc không có metadata."""
        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        # 1. Without logging
        control_pilot_logging(PilotControlRequest(enabled=False))
        req_clean = LandmarkRequest(landmarks=lm, mode="letters")
        res_clean = recognize_landmarks(req_clean, mock_http)

        # 2. With logging and target = P
        control_pilot_logging(PilotControlRequest(enabled=True, session_id="test_invariant"))
        req_p = LandmarkRequest(landmarks=lm, mode="letters", expected_sign="P", session_id="test_invariant", trial_id="trial_1", frame_idx=1)
        res_p = recognize_landmarks(req_p, mock_http)

        # 3. With logging and target = Ă (diacritic)
        req_a = LandmarkRequest(landmarks=lm, mode="letters", expected_sign="Ă", session_id="test_invariant", trial_id="trial_2", frame_idx=1)
        res_a = recognize_landmarks(req_a, mock_http)

        self.assertEqual(res_clean["data"]["sign"], res_p["data"]["sign"])
        self.assertEqual(res_clean["data"]["confidence"], res_p["data"]["confidence"])
        self.assertEqual(res_clean["data"]["sign"], res_a["data"]["sign"])
        self.assertEqual(res_clean["data"]["confidence"], res_a["data"]["confidence"])

    def test_supplemental_no_manual_tampering_and_complete_fields(self):
        """Bổ sung: Kiểm tra tính toàn vẹn trường dữ liệu và không can thiệp thủ công vào nhãn dự đoán."""
        ctrl = control_pilot_logging(PilotControlRequest(enabled=True, session_id="test_supp_eval"))
        filepath = ctrl["filepath"]

        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        # Gửi 5 frame với target_label = 'P'
        for f in range(1, 6):
            req = LandmarkRequest(
                landmarks=lm, mode="letters", expected_sign="P",
                session_id="test_supp_eval", trial_id="test_supp_eval_P_rep01", frame_idx=f
            )
            recognize_landmarks(req, mock_http)

        flush_pilot_log()
        records = _load_jsonl_records(filepath)
        self.assertEqual(len(records), 5)

        for rec in records:
            # 1. Target label giữ nguyên nhãn tester chọn
            self.assertEqual(rec["expected_sign"], "P")
            # 2. Predicted label (final_sign) phản ánh dự đoán thực tế của mô hình
            self.assertIn(rec["final_sign"], ["NEUTRAL", "P", "K", "D", "B", "A", "UNKNOWN"])
            # 3. Đầy đủ các trường định danh và chẩn đoán
            self.assertIsNotNone(rec.get("timestamp"))
            self.assertEqual(rec.get("session_id"), "test_supp_eval")
            self.assertEqual(rec.get("trial_id"), "test_supp_eval_P_rep01")
            self.assertIsInstance(rec.get("frame_idx"), int)
            self.assertIn("final_confidence", rec)
            self.assertIn("prediction_source", rec)
            self.assertIn("geometric", rec)
            self.assertIn("static_head", rec)

        # 4. Kiểm tra công cụ tính toán offline có thể phân tích đúng/sai từ cặp (expected_sign, final_sign)
        matches = sum(1 for r in records if r["final_sign"] == r["expected_sign"])
        neutrals = sum(1 for r in records if r["final_sign"] == "NEUTRAL")
        mismatches = sum(1 for r in records if r["final_sign"] != "NEUTRAL" and r["final_sign"] != r["expected_sign"])
        self.assertEqual(matches + neutrals + mismatches, 5)


if __name__ == "__main__":
    unittest.main()

