"""
End-to-End Integration Tests cho Controlled Webcam Pilot Testing HUD (Task 7F.5B)
================================================================================
Kiểm thử 10 kịch bản tích hợp bắt buộc:
1. Metadata từ frontend (qua JSON body hoặc HTTP headers) được backend nhận chính xác.
2. expected_sign xuất hiện đúng trong file log JSONL.
3. trial_id thay đổi chính xác khi bắt đầu trial mới (không bao giờ trùng lặp).
4. frame_idx tăng đơn điệu trong trial và không bị reset giữa chừng.
5. Kết thúc trial không làm request tiếp theo mang ground truth cũ (chống rò rỉ metadata).
6. Hai session/trial độc lập hoàn toàn không bị trộn metadata.
7. Endpoint /api/pilot/flush đảm bảo toàn bộ bản ghi được xả xuống đĩa trước khi trả về.
8. Endpoint /api/pilot/stats phản ánh đúng số record ghi thành công và số record thất thoát.
9. Logging mặc định vẫn tắt nếu không gọi enable hoặc /api/pilot/control.
10. Kết quả nhận diện (sign, confidence) không thay đổi dù có hay không có metadata từ HUD.
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
from backend.tests.test_geometry import _make_hand


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


class TestPilotHUDIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_filepath = os.path.join(self.temp_dir, "pilot_hud_session.jsonl")
        recognizer.diagnostic_dir = self.temp_dir
        # Đảm bảo reset trạng thái và stats trước mỗi test
        recognizer.disable_pilot_logging()
        if hasattr(recognizer, "pilot_logger") and recognizer.pilot_logger:
            recognizer.pilot_logger.reset_stats()

    def tearDown(self):
        recognizer.disable_pilot_logging()
        if hasattr(recognizer, "pilot_logger") and recognizer.pilot_logger:
            recognizer.pilot_logger.reset_stats()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_and_2_frontend_metadata_received_and_logged(self):
        """1 & 2. Metadata frontend gửi qua body/headers được backend nhận và ghi đúng expected_sign vào JSONL."""
        ctrl_res = control_pilot_logging(PilotControlRequest(enabled=True, session_id="hud_sess_1"))
        self.assertTrue(ctrl_res["success"])
        self.assertTrue(ctrl_res["enabled"])
        self.log_filepath = ctrl_res["filepath"]

        lm = _make_hand()
        req = LandmarkRequest(
            landmarks=lm,
            mode="letters",
            expected_sign="B",
            session_id="pilot_test_sess",
            trial_id="pilot_test_sess_B_rep01",
            frame_idx=1
        )
        mock_http = MagicMock()
        mock_http.headers = {}

        res = recognize_landmarks(req, mock_http)
        self.assertTrue(res["success"])

        flush_pilot_log()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)

        r = records[0]
        self.assertEqual(r["expected_sign"], "B")
        self.assertEqual(r["session_id"], "pilot_test_sess")
        self.assertEqual(r["trial_id"], "pilot_test_sess_B_rep01")
        self.assertEqual(r["frame_idx"], 1)

    def test_3_trial_id_changes_between_trials(self):
        """3. trial_id thay đổi đúng khi người dùng chuyển sang trial hoặc repetition mới."""
        ctrl_res = control_pilot_logging(PilotControlRequest(enabled=True, session_id="hud_sess_3"))
        self.log_filepath = ctrl_res["filepath"]

        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        # Trial 1: Chữ B lần 1
        req1 = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="B",
            session_id="sess_01", trial_id="sess_01_B_rep01", frame_idx=1
        )
        recognize_landmarks(req1, mock_http)

        # Trial 2: Chữ B lần 2
        req2 = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="B",
            session_id="sess_01", trial_id="sess_01_B_rep02", frame_idx=1
        )
        recognize_landmarks(req2, mock_http)

        # Trial 3: Chữ C lần 1
        req3 = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="C",
            session_id="sess_01", trial_id="sess_01_C_rep01", frame_idx=1
        )
        recognize_landmarks(req3, mock_http)

        flush_pilot_log()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 3)

        self.assertEqual(records[0]["trial_id"], "sess_01_B_rep01")
        self.assertEqual(records[0]["expected_sign"], "B")

        self.assertEqual(records[1]["trial_id"], "sess_01_B_rep02")
        self.assertEqual(records[1]["expected_sign"], "B")

        self.assertEqual(records[2]["trial_id"], "sess_01_C_rep01")
        self.assertEqual(records[2]["expected_sign"], "C")

    def test_4_frame_idx_increments_monotonically(self):
        """4. frame_idx tăng đơn điệu trong trial và không bị reset bất thường giữa chừng."""
        ctrl_res = control_pilot_logging(PilotControlRequest(enabled=True, session_id="hud_sess_4"))
        self.log_filepath = ctrl_res["filepath"]

        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        # Giả lập 5 frames liên tục trong 1 trial
        for i in range(1, 6):
            req = LandmarkRequest(
                landmarks=lm, mode="letters", expected_sign="A",
                session_id="sess_monotonic", trial_id="trial_A_01", frame_idx=i
            )
            recognize_landmarks(req, mock_http)

        flush_pilot_log()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 5)

        for i, r in enumerate(records):
            self.assertEqual(r["frame_idx"], i + 1)
            self.assertEqual(r["trial_id"], "trial_A_01")

    def test_5_no_metadata_leakage_after_trial_ends(self):
        """5. Kết thúc trial không làm request tiếp theo mang ground truth cũ (chống rò rỉ metadata)."""
        ctrl_res = control_pilot_logging(PilotControlRequest(enabled=True, session_id="hud_sess_5"))
        self.log_filepath = ctrl_res["filepath"]

        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        # Frame 1: Trong trial B
        req_in_trial = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="B",
            session_id="sess_leak_check", trial_id="trial_B_01", frame_idx=10
        )
        recognize_landmarks(req_in_trial, mock_http)

        # Frame 2: Ngoài trial (người dùng thả tay, HUD ở trạng thái idle, không gửi metadata)
        req_outside_trial = LandmarkRequest(
            landmarks=lm, mode="letters",
            expected_sign=None, session_id=None, trial_id=None, frame_idx=None
        )
        recognize_landmarks(req_outside_trial, mock_http)

        flush_pilot_log()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 2)

        # Record 1 phải có metadata
        self.assertEqual(records[0]["expected_sign"], "B")
        self.assertEqual(records[0]["trial_id"], "trial_B_01")

        # Record 2 tuyệt đối KHÔNG được mang metadata của Record 1
        self.assertIsNone(records[1]["expected_sign"])
        self.assertIsNone(records[1]["trial_id"])
        self.assertIsNone(records[1]["session_id"])

    def test_6_isolated_sessions_do_not_mix(self):
        """6. Hai session/trial độc lập hoàn toàn không bị trộn lẫn metadata."""
        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        # Session 1: Signer 01
        res1 = control_pilot_logging(PilotControlRequest(enabled=True, session_id="signer01"))
        log_sess1 = res1["filepath"]
        req1 = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="T",
            session_id="signer01", trial_id="signer01_T_rep01", frame_idx=1
        )
        recognize_landmarks(req1, mock_http)
        flush_pilot_log()
        # Đóng session 1 trước khi mở session 2
        control_pilot_logging(PilotControlRequest(enabled=False))

        # Session 2: Signer 02
        res2 = control_pilot_logging(PilotControlRequest(enabled=True, session_id="signer02"))
        log_sess2 = res2["filepath"]
        req2 = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="Y",
            session_id="signer02", trial_id="signer02_Y_rep01", frame_idx=1
        )
        recognize_landmarks(req2, mock_http)
        flush_pilot_log()

        rec1 = _load_jsonl_records(log_sess1)
        rec2 = _load_jsonl_records(log_sess2)

        self.assertEqual(len(rec1), 1)
        self.assertEqual(rec1[0]["session_id"], "signer01")
        self.assertEqual(rec1[0]["expected_sign"], "T")

        self.assertEqual(len(rec2), 1)
        self.assertEqual(rec2[0]["session_id"], "signer02")
        self.assertEqual(rec2[0]["expected_sign"], "Y")

    def test_7_and_8_flush_and_stats_endpoints(self):
        """7 & 8. Flush hoàn tất đẩy dữ liệu và Stats phản ánh đúng số bản ghi."""
        ctrl_res = control_pilot_logging(PilotControlRequest(enabled=True, session_id="hud_sess_7"))
        self.log_filepath = ctrl_res["filepath"]
        if recognizer.pilot_logger:
            recognizer.pilot_logger.reset_stats()

        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        # Gửi 3 requests
        for i in range(3):
            req = LandmarkRequest(
                landmarks=lm, mode="letters", expected_sign="L",
                session_id="sess_stats", trial_id="trial_L_01", frame_idx=i + 1
            )
            recognize_landmarks(req, mock_http)

        # Gọi endpoint flush
        flush_res = flush_pilot_log()
        self.assertTrue(flush_res["success"])
        stats = flush_res["stats"]
        self.assertEqual(stats["queued_count"], 3)
        self.assertEqual(stats["logged_count"], 3)
        self.assertEqual(stats["total_dropped"], 0)
        self.assertEqual(stats["pending_in_queue"], 0)

        # Gọi endpoint stats riêng
        stats_api = get_pilot_stats()
        self.assertTrue(stats_api["success"])
        self.assertEqual(stats_api["stats"]["logged_count"], 3)

    def test_9_default_logging_is_disabled(self):
        """9. Khi tắt logging, hệ thống không tạo file log pilot."""
        control_pilot_logging(PilotControlRequest(enabled=False))

        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        req = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="B",
            session_id="sess_disabled", trial_id="trial_B_01", frame_idx=1
        )
        res = recognize_landmarks(req, mock_http)
        self.assertTrue(res["success"])

        flush_pilot_log()
        # Không được tạo file log_filepath
        self.assertFalse(os.path.exists(self.log_filepath))

    def test_10_recognition_output_unchanged_with_hud_metadata(self):
        """10. Kết quả nhận diện (sign, confidence) không thay đổi dù có hay không có metadata HUD."""
        # Chạy baseline không logging
        control_pilot_logging(PilotControlRequest(enabled=False))
        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        req_clean = LandmarkRequest(landmarks=lm, mode="letters")
        res_baseline = recognize_landmarks(req_clean, mock_http)

        # Bật logging và truyền metadata HUD (kèm ground truth đúng)
        ctrl_res = control_pilot_logging(PilotControlRequest(enabled=True, session_id="test_sess_10"))
        self.log_filepath = ctrl_res["filepath"]
        req_hud = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="B",
            session_id="sess_hud", trial_id="trial_hud_01", frame_idx=1
        )
        res_hud = recognize_landmarks(req_hud, mock_http)

        # Bật logging và truyền metadata HUD với ground truth CỐ Ý SAI
        req_wrong_gt = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="X",
            session_id="sess_hud", trial_id="trial_hud_02", frame_idx=2
        )
        res_wrong = recognize_landmarks(req_wrong_gt, mock_http)

        # Kết quả nhận diện phải đồng nhất 100% trong cả 3 tình huống
        self.assertEqual(res_baseline["data"]["sign"], res_hud["data"]["sign"])
        self.assertEqual(res_baseline["data"]["confidence"], res_hud["data"]["confidence"])
        self.assertEqual(res_baseline["data"]["sign"], res_wrong["data"]["sign"])
        self.assertEqual(res_baseline["data"]["confidence"], res_wrong["data"]["confidence"])


if __name__ == "__main__":
    unittest.main()
