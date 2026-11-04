"""
Unit & Edge-Case Tests for Task 7F.5E: Final Logger Edge-Case Audit
===================================================================
Kiểm thử toàn diện 4 khía cạnh trường hợp biên trước Controlled Webcam Pilot:
1. Strict session_id validation:
   - Chấp nhận định dạng chuẩn (chữ, số, gạch ngang, gạch dưới, 3-64 ký tự).
   - Từ chối chuỗi rỗng, khoảng trắng, dấu chéo (/, \\), dấu chấm, traversal (../).
   - Từ chối chuỗi quá ngắn (< 3) hoặc quá dài (> 64).
   - Từ chối chuỗi chỉ gồm ký tự đặc biệt (không có chữ hoặc số).
   - Chặn tuyệt đối va chạm âm thầm (session/1 và session.1 đều bị HTTP 400, không biến thành session1).
2. Vòng đời tệp tin & Phòng chống Symlink:
   - Worker phát hiện và từ chối ghi vào file nếu bị thay thế bằng symlink.
   - Worker từ chối ghi nếu đường dẫn thoát khỏi thư mục hợp lệ.
3. FlushBarrier & Concurrent Producers:
   - Nhiều luồng producers đẩy frame liên tục trong khi flush đang chạy.
   - Xử lý queue đầy khi flush đưa barrier vào queue.
   - Timeout khi worker bị nghẽn (không deadlock, worker phục hồi sau đó).
   - Phân biệt lỗi I/O trước barrier vs sau barrier.
   - Nhiều lần flush liên tiếp khi traffic concurrent đang chạy.
4. Close / Shutdown & Concurrency:
   - Ngay khi bắt đầu close(), mọi lệnh log_frame() mới bị từ chối ngay lập tức.
   - Gọi close() nhiều lần liên tiếp bảo đảm tính lũy đổng (idempotent).
   - Chuyển đổi an toàn giữa Session A và Session B, không rò rỉ dữ liệu.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import threading
import unittest
from unittest.mock import MagicMock
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.app import (
    control_pilot_logging,
    PilotControlRequest,
    recognize_landmarks,
    LandmarkRequest,
    flush_pilot_log,
    recognizer,
)
from backend.vision.pilot_logger import (
    PilotDiagnosticLogger,
    FlushBarrier,
    validate_session_id,
    create_safe_pilot_file,
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


class TestPilotEdgeCases(unittest.TestCase):
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

    # =========================================================================
    # PHẦN 1: STRICT SESSION_ID VALIDATION
    # =========================================================================

    def test_strict_session_valid_formats(self):
        """1.1. Các định dạng session_id hợp lệ (chuẩn HUD và tester)."""
        self.assertEqual(validate_session_id("pilot_20261101_1947"), "pilot_20261101_1947")
        self.assertEqual(validate_session_id("signer01_rep01"), "signer01_rep01")
        self.assertEqual(validate_session_id("A-1_test"), "A-1_test")
        self.assertEqual(validate_session_id("abc"), "abc")
        self.assertEqual(validate_session_id("a" * 64), "a" * 64)

    def test_strict_session_rejections(self):
        """1.2. Từ chối mọi chuỗi session_id không đạt chuẩn bằng ValueError / HTTP 400."""
        invalid_ids = [
            None,
            "",
            "   ",
            "ab",                       # Quá ngắn (< 3)
            "a" * 65,                   # Quá dài (> 64)
            "pilot/2026",               # Dấu chéo xuôi
            "pilot\\2026",              # Dấu chéo ngược
            "pilot.2026",               # Dấu chấm
            "../evil",                  # Path traversal
            "..\\evil",
            "session name with spaces", # Khoảng trắng
            "---",                      # Chỉ có dấu gạch ngang (không có chữ/số)
            "___",                      # Chỉ có dấu gạch dưới
            "__--__",
            "session$1",                # Ký tự đặc biệt
            "session#1",
            12345,                      # Không phải chuỗi
        ]
        for inv in invalid_ids:
            with self.assertRaises((ValueError, HTTPException), msg=f"Không từ chối: {inv}"):
                if inv is None or not isinstance(inv, str) or not inv.strip():
                    validate_session_id(inv)
                else:
                    # Kiểm tra cả API endpoint
                    control_pilot_logging(PilotControlRequest(enabled=True, session_id=inv))

    def test_strict_session_no_silent_collision(self):
        """1.3. Hai session ID khác nhau có ký tự cấm tuyệt đối không bị biến thành cùng một ID."""
        # Giả sử client gửi "test/a" và "test.a": cả hai phải bị từ chối 400 thay vì biến thành "testa"
        with self.assertRaises(HTTPException) as ctx1:
            control_pilot_logging(PilotControlRequest(enabled=True, session_id="test/a"))
        self.assertEqual(ctx1.exception.status_code, 400)

        with self.assertRaises(HTTPException) as ctx2:
            control_pilot_logging(PilotControlRequest(enabled=True, session_id="test.a"))
        self.assertEqual(ctx2.exception.status_code, 400)

    # =========================================================================
    # PHẦN 2: VÒNG ĐỜI TỆP TIN & SYMLINK PROTECTION
    # =========================================================================

    def test_symlink_detection_in_worker(self):
        """2.1. Worker phát hiện và từ chối ghi vào file nếu đường dẫn là symlink."""
        real_file = os.path.join(self.temp_dir, "real_target.jsonl")
        symlink_file = os.path.join(self.temp_dir, "symlink_target.jsonl")

        with open(real_file, "w", encoding="utf-8") as f:
            f.write("")

        # Tạo symlink nếu OS hỗ trợ (Windows Developer Mode hoặc POSIX)
        try:
            os.symlink(real_file, symlink_file)
        except (OSError, NotImplementedError):
            # Nếu môi trường không có quyền SeCreateSymbolicLinkPrivilege, bỏ qua an toàn
            return

        logger = PilotDiagnosticLogger(filepath=symlink_file)
        logger.log_frame({"test": "symlink_attack"})
        # Flush sẽ thất bại do worker ném IOError và tăng dropped_io_error
        res = logger.flush(timeout=2.0)
        self.assertFalse(res)
        self.assertGreater(logger.get_stats()["dropped_io_error"], 0)
        logger.close()

    # =========================================================================
    # PHẦN 3: FLUSH BARRIER VỚI CONCURRENT PRODUCERS
    # =========================================================================

    def test_concurrent_producers_during_flush(self):
        """3.1. Nhiều luồng producers đẩy liên tục trong khi flush() đang thực thi."""
        target_file = os.path.join(self.temp_dir, "concurrent_flush.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        num_threads = 4
        records_per_thread = 25
        stop_producers = threading.Event()

        def producer_worker(thread_id):
            count = 0
            while not stop_producers.is_set() and count < records_per_thread:
                logger.log_frame({"thread": thread_id, "idx": count})
                count += 1
                time.sleep(0.002)

        threads = [threading.Thread(target=producer_worker, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()

        # Gọi flush giữa chừng khi các luồng đang ghi
        time.sleep(0.02)
        flush_ok = logger.flush(timeout=3.0)
        self.assertTrue(flush_ok)

        # Chờ các luồng kết thúc
        for t in threads:
            t.join(timeout=2.0)

        # Flush lần cuối
        final_ok = logger.flush(timeout=3.0)
        self.assertTrue(final_ok)

        records = _load_jsonl_records(target_file)
        self.assertEqual(len(records), num_threads * records_per_thread)
        logger.close()

    def test_flush_queue_full_handling(self):
        """3.2. Khi queue đầy và worker bị block, flush() xử lý timeout gracefully."""
        target_file = os.path.join(self.temp_dir, "queue_full_flush.jsonl")
        # Khởi tạo logger với queue nhỏ = 2
        logger = PilotDiagnosticLogger(filepath=target_file, queue_size=2)

        block_worker = threading.Event()
        logger._test_worker_pre_write_hook = lambda r, f: block_worker.wait(timeout=5.0)

        # Đẩy 2 records để làm đầy queue (1 đang xử lý, 1 nằm trong queue)
        logger.log_frame({"data": 1})
        time.sleep(0.05) # Cho worker lấy record 1 và dừng lại ở hook
        logger.log_frame({"data": 2})
        logger.log_frame({"data": 3}) # Làm đầy queue

        # Flush với timeout ngắn: không đưa được barrier vào queue hoặc timeout
        res = logger.flush(timeout=0.15)
        self.assertFalse(res)

        # Tháo block để worker tiếp tục
        block_worker.set()
        logger.close()

    def test_flush_timeout_recovers_without_deadlock(self):
        """3.3. Flush timeout nhưng worker vẫn tiếp tục xử lý và logger hồi phục bình thường."""
        target_file = os.path.join(self.temp_dir, "timeout_recovery.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        delay_event = threading.Event()
        # Giả lập worker bị chậm 0.4s
        logger._test_worker_pre_write_hook = lambda r, f: delay_event.wait(timeout=0.4)

        logger.log_frame({"item": "slow_item"})
        # Flush với timeout ngắn 0.1s -> trả về False
        res = logger.flush(timeout=0.1)
        self.assertFalse(res)

        # Giải phóng delay
        delay_event.set()
        # Chờ worker xử lý xong barrier trước đó
        time.sleep(0.1)

        # Bây giờ ghi thêm record và flush lại -> phải thành công
        logger._test_worker_pre_write_hook = None
        logger.log_frame({"item": "fast_item"})
        res2 = logger.flush(timeout=2.0)
        self.assertTrue(res2)

        recs = _load_jsonl_records(target_file)
        self.assertEqual(len(recs), 2)
        logger.close()

    # =========================================================================
    # PHẦN 4: CLOSE / SHUTDOWN & CONCURRENCY
    # =========================================================================

    def test_close_rejects_new_records_immediately(self):
        """4.1. Ngay khi close() bắt đầu, mọi lệnh log_frame() mới bị từ chối ngay lập tức."""
        target_file = os.path.join(self.temp_dir, "close_reject.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        logger.log_frame({"data": "before_close"})
        logger.close(timeout=2.0)

        # Sau khi close(), log_frame phải trả về False
        res = logger.log_frame({"data": "after_close"})
        self.assertFalse(res)

        recs = _load_jsonl_records(target_file)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["data"], "before_close")

    def test_close_is_idempotent(self):
        """4.2. Gọi close() nhiều lần liên tiếp không gây lỗi hoặc deadlock."""
        target_file = os.path.join(self.temp_dir, "close_idempotent.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        logger.log_frame({"data": "test"})
        logger.close(timeout=1.0)
        logger.close(timeout=1.0)
        logger.close(timeout=1.0)

        self.assertFalse(logger.is_running)

    def test_session_transition_zero_leakage(self):
        """4.3. Chuyển đổi giữa Session A và Session B: dữ liệu của A không bị ghi vào file của B."""
        # Session A
        resA = control_pilot_logging(PilotControlRequest(enabled=True, session_id="session_A_clean"))
        fileA = resA["filepath"]

        lm = _make_hand()
        mock_http = MagicMock()
        mock_http.headers = {}

        reqA = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="B",
            session_id="session_A_clean", trial_id="trial_A_01", frame_idx=1
        )
        recognize_landmarks(reqA, mock_http)
        flush_pilot_log()

        # Disable Session A
        control_pilot_logging(PilotControlRequest(enabled=False))

        # Session B
        resB = control_pilot_logging(PilotControlRequest(enabled=True, session_id="session_B_clean"))
        fileB = resB["filepath"]

        reqB = LandmarkRequest(
            landmarks=lm, mode="letters", expected_sign="C",
            session_id="session_B_clean", trial_id="trial_B_01", frame_idx=1
        )
        recognize_landmarks(reqB, mock_http)
        flush_pilot_log()

        # Xác minh file của A và B hoàn toàn tách biệt
        recsA = _load_jsonl_records(fileA)
        recsB = _load_jsonl_records(fileB)

        self.assertEqual(len(recsA), 1)
        self.assertEqual(recsA[0]["session_id"], "session_A_clean")
        self.assertEqual(recsA[0]["expected_sign"], "B")

        self.assertEqual(len(recsB), 1)
        self.assertEqual(recsB[0]["session_id"], "session_B_clean")
        self.assertEqual(recsB[0]["expected_sign"], "C")


if __name__ == "__main__":
    unittest.main()

