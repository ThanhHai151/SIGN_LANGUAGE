"""
Unit & Integration Tests for Task 7F.5D: Fix Pilot Preflight Blockers
====================================================================
Kiểm thử toàn diện các giải pháp sửa lỗi Blocker B-01 và B-02:
1. Path Safety & Containment:
   - Từ chối filepath từ HTTP client (HTTP 400).
   - Từ chối session_id chứa path traversal hoặc ký tự không hợp lệ (HTTP 400).
   - Chặn symlink escape và ghi file ra ngoài thư mục pilot.
   - Ngăn ghi đè hoặc nối dữ liệu vào file cũ khi trùng session_id (tạo file duy nhất qua O_CREAT | O_EXCL).
   - Session ID hợp lệ tạo file thành công trong thư mục pilot.
   - HUD hiện tại bật/tắt logger bình thường.
2. Reliable Flush (FlushBarrier):
   - Worker đang ghi dở dang: flush từ thread khác KHÔNG trả về sớm trước khi worker hoàn tất.
   - Queue có nhiều record: flush tuần tự đảm bảo xử lý hết 100%.
   - Worker gặp lỗi I/O: flush phản ánh thất bại (success == False).
   - Flush timeout: trả về False nếu worker bị block vượt quá thời gian chờ.
   - Flush khi queue rỗng: trả về True ngay lập tức.
   - Flush liên tiếp nhiều lần: hoạt động ổn định và chính xác.
   - Shutdown logger khi còn record: close() flush hết queue trước khi kết thúc worker.
3. Session Safety & Exclusivity:
   - Đang chạy session A mà enable session B: từ chối với HTTP 409 Conflict.
   - File đích của session A không bị thay đổi âm thầm.
   - Disable session A xong thì session B kích hoạt thành công.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import threading
import unittest
from unittest.mock import patch, MagicMock
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


class TestPilotBlockerFixes(unittest.TestCase):
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
    # PHẦN 1: PATH SAFETY & CONTAINMENT (Fix B-01)
    # =========================================================================

    def test_path_1_reject_client_filepath(self):
        """1. API /api/pilot/control từ chối mọi yêu cầu có client chỉ định filepath (HTTP 400)."""
        evil_path = os.path.join(self.temp_dir, "outside.jsonl")
        req = PilotControlRequest(enabled=True, filepath=evil_path)
        with self.assertRaises(HTTPException) as ctx:
            control_pilot_logging(req)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("Chỉ định filepath qua API bị từ chối", ctx.exception.detail)
        self.assertFalse(os.path.exists(evil_path))

    def test_path_2_reject_path_traversal_in_session_id(self):
        """2. API từ chối session_id chỉ chứa ký tự path traversal như ../ hoặc .. (HTTP 400)."""
        req_traversal = PilotControlRequest(enabled=True, session_id="../../../")
        with self.assertRaises(HTTPException) as ctx:
            control_pilot_logging(req_traversal)
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertIn("session_id không hợp lệ", ctx.exception.detail)

    def test_path_3_create_safe_pilot_file_blocks_containment_breach(self):
        """3. create_safe_pilot_file chặn symlink escape hoặc thoát khỏi thư mục pilot."""
        with self.assertRaises(ValueError):
            create_safe_pilot_file(self.temp_dir, "../../../evil")

    def test_path_4_duplicate_session_id_avoids_overwriting_or_appending(self):
        """4. Trùng session_id không được ghi đè hoặc tự động nối vào file cũ (tạo file duy nhất)."""
        # Phiên 1: Session 'signer01' tạo file đầu tiên
        file1 = create_safe_pilot_file(self.temp_dir, "signer01")
        self.assertTrue(os.path.exists(file1))
        with open(file1, "a", encoding="utf-8") as f:
            f.write(json.dumps({"frame": 1, "session": "run_1"}) + "\n")

        # Phiên 2: Lần chạy sau lại dùng 'signer01'
        file2 = create_safe_pilot_file(self.temp_dir, "signer01")
        self.assertNotEqual(file1, file2)
        self.assertTrue(os.path.exists(file2))
        self.assertIn("signer01", os.path.basename(file2))

        # Ghi dữ liệu vào file2
        with open(file2, "a", encoding="utf-8") as f:
            f.write(json.dumps({"frame": 1, "session": "run_2"}) + "\n")

        # Xác minh file1 không bị nối dữ liệu của file2
        recs1 = _load_jsonl_records(file1)
        recs2 = _load_jsonl_records(file2)
        self.assertEqual(len(recs1), 1)
        self.assertEqual(recs1[0]["session"], "run_1")
        self.assertEqual(len(recs2), 1)
        self.assertEqual(recs2[0]["session"], "run_2")

    def test_path_5_valid_session_id_generates_file_successfully(self):
        """5. Session ID hợp lệ tạo file thành công trong thư mục pilot."""
        res = control_pilot_logging(PilotControlRequest(enabled=True, session_id="pilot_20261101_signerA"))
        self.assertTrue(res["success"])
        self.assertTrue(res["enabled"])
        self.assertTrue(os.path.exists(res["filepath"]))
        self.assertEqual(os.path.dirname(res["filepath"]), os.path.realpath(self.temp_dir))
        self.assertIn("pilot_20261101_signerA", os.path.basename(res["filepath"]))

    def test_path_6_hud_normal_invocation_works(self):
        """6. HUD gọi với session_id hợp lệ (không truyền filepath) hoạt động chính xác."""
        res_on = control_pilot_logging(PilotControlRequest(enabled=True, session_id="hud_normal_test"))
        self.assertTrue(res_on["success"])
        self.assertTrue(res_on["enabled"])

        res_off = control_pilot_logging(PilotControlRequest(enabled=False))
        self.assertTrue(res_off["success"])
        self.assertFalse(res_off["enabled"])

    # =========================================================================
    # PHẦN 2: RELIABLE FLUSH & WORKER SYNCHRONIZATION (Fix B-02)
    # =========================================================================

    def test_flush_1_timing_controlled_barrier(self):
        """
        1. Test kiểm soát thời gian bắt buộc:
        - Worker lấy 1 record khỏi queue.
        - Worker bị block trước khi ghi xong file (chưa task_done).
        - Thread khác gọi flush(): xác minh flush CHƯA trả về thành công trong khi worker đang xử lý.
        - Thả worker tiếp tục ghi: xác minh flush chỉ trả về thành công sau khi worker xử lý xong.
        """
        logger_file = os.path.join(self.temp_dir, "test_timing_flush.jsonl")
        logger = PilotDiagnosticLogger(filepath=logger_file)

        worker_reached_hook = threading.Event()
        release_worker_event = threading.Event()

        def block_worker_hook(record, filepath):
            worker_reached_hook.set()
            # Worker dừng lại chờ tín hiệu từ test
            release_worker_event.wait(timeout=5.0)

        logger._test_worker_pre_write_hook = block_worker_hook

        # Đẩy 1 record vào queue
        logger.log_frame({"data": "frame_in_flight"})

        # Đợi worker thực sự lấy record và đi vào hook
        self.assertTrue(worker_reached_hook.wait(timeout=2.0))

        flush_thread_finished = threading.Event()
        flush_result = []

        def call_flush():
            res = logger.flush(timeout=3.0)
            flush_result.append(res)
            flush_thread_finished.set()

        t = threading.Thread(target=call_flush)
        t.start()

        # Cho thread flush thời gian enqueuing barrier và bắt đầu wait
        time.sleep(0.1)

        # XÁC MINH CỐT LÕI: flush CHƯA được phép hoàn thành trong khi worker bị block
        self.assertFalse(flush_thread_finished.is_set())
        self.assertEqual(len(flush_result), 0)

        # Thả worker tiếp tục ghi đĩa
        release_worker_event.set()

        # Đợi thread flush kết thúc
        self.assertTrue(flush_thread_finished.wait(timeout=2.0))
        t.join(timeout=1.0)

        # XÁC MINH CỐT LÕI: Flush thành công sau khi worker xử lý xong
        self.assertEqual(flush_result, [True])

        # Đĩa phải có đúng record đã ghi
        recs = _load_jsonl_records(logger_file)
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["data"], "frame_in_flight")

        logger.close()

    def test_flush_2_multiple_records_processed(self):
        """2. Queue có nhiều records: flush đảm bảo xử lý hết 100% trước khi trả về."""
        logger_file = os.path.join(self.temp_dir, "test_multi_flush.jsonl")
        logger = PilotDiagnosticLogger(filepath=logger_file)

        num_records = 50
        for i in range(num_records):
            logger.log_frame({"idx": i})

        success = logger.flush(timeout=3.0)
        self.assertTrue(success)

        recs = _load_jsonl_records(logger_file)
        self.assertEqual(len(recs), num_records)
        self.assertEqual(recs[-1]["idx"], num_records - 1)

        logger.close()

    def test_flush_3_worker_io_error_reflected_in_flush(self):
        """3. Khi worker gặp lỗi I/O, flush phản ánh thất bại (success == False)."""
        logger_file = os.path.join(self.temp_dir, "test_error_flush.jsonl")
        logger = PilotDiagnosticLogger(filepath=logger_file)

        def faulty_hook(record, filepath):
            raise IOError("Simulated disk write failure")

        logger._test_worker_pre_write_hook = faulty_hook

        logger.log_frame({"data": "will_fail"})
        success = logger.flush(timeout=2.0)

        # Flush phát hiện lỗi I/O trước barrier và trả về False
        self.assertFalse(success)
        self.assertEqual(logger.get_stats()["dropped_io_error"], 1)

        logger.close()

    def test_flush_4_timeout_returns_false(self):
        """4. Flush trả về False khi worker bị treo vượt quá timeout."""
        logger_file = os.path.join(self.temp_dir, "test_timeout_flush.jsonl")
        logger = PilotDiagnosticLogger(filepath=logger_file)

        forever_block = threading.Event()
        logger._test_worker_pre_write_hook = lambda r, f: forever_block.wait(timeout=10.0)

        logger.log_frame({"data": "stuck"})
        # Flush với timeout ngắn 0.15s
        success = logger.flush(timeout=0.15)
        self.assertFalse(success)

        forever_block.set()
        logger.close()

    def test_flush_5_empty_queue_returns_true_immediately(self):
        """5. Flush khi queue rỗng trả về True ngay lập tức."""
        logger = PilotDiagnosticLogger(filepath=os.path.join(self.temp_dir, "empty.jsonl"))
        t0 = time.time()
        res = logger.flush(timeout=1.0)
        dur = time.time() - t0
        self.assertTrue(res)
        self.assertLess(dur, 0.2)
        logger.close()

    def test_flush_6_repeated_consecutive_flushes(self):
        """6. Flush liên tiếp nhiều lần hoạt động ổn định và chính xác."""
        logger_file = os.path.join(self.temp_dir, "test_repeat_flush.jsonl")
        logger = PilotDiagnosticLogger(filepath=logger_file)

        for step in range(5):
            logger.log_frame({"step": step})
            self.assertTrue(logger.flush(timeout=1.0))

        recs = _load_jsonl_records(logger_file)
        self.assertEqual(len(recs), 5)
        logger.close()

    def test_flush_7_shutdown_flushes_remaining_records(self):
        """7. Đóng logger qua close() xả hết queue trước khi dừng worker."""
        logger_file = os.path.join(self.temp_dir, "test_close_flush.jsonl")
        logger = PilotDiagnosticLogger(filepath=logger_file)

        for i in range(10):
            logger.log_frame({"idx": i})

        logger.close(timeout=2.0)
        recs = _load_jsonl_records(logger_file)
        self.assertEqual(len(recs), 10)

    # =========================================================================
    # PHẦN 3: SESSION SAFETY & EXCLUSIVITY
    # =========================================================================

    def test_session_1_prevent_concurrent_session_override(self):
        """1. Đang chạy Session A mà gọi bật Session B: từ chối với HTTP 409 Conflict."""
        resA = control_pilot_logging(PilotControlRequest(enabled=True, session_id="session_A"))
        self.assertTrue(resA["success"])
        target_file_A = resA["filepath"]

        # Thử bật Session B trong khi Session A đang active
        reqB = PilotControlRequest(enabled=True, session_id="session_B")
        with self.assertRaises(HTTPException) as ctx:
            control_pilot_logging(reqB)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertIn("session_A", ctx.exception.detail)

        # XÁC MINH CỐT LÕI: File đích và session đang active của A không hề bị đổi
        self.assertEqual(recognizer.pilot_filepath, target_file_A)
        self.assertEqual(recognizer.active_pilot_session_id, "session_A")

    def test_session_2_disable_A_allows_session_B(self):
        """2. Disable Session A xong thì Session B kích hoạt thành công bình thường."""
        resA = control_pilot_logging(PilotControlRequest(enabled=True, session_id="session_A"))
        self.assertTrue(resA["success"])

        # Disable Session A
        res_dis = control_pilot_logging(PilotControlRequest(enabled=False))
        self.assertTrue(res_dis["success"])
        self.assertIsNone(recognizer.active_pilot_session_id)

        # Bây giờ bật Session B phải thành công
        resB = control_pilot_logging(PilotControlRequest(enabled=True, session_id="session_B"))
        self.assertTrue(resB["success"])
        self.assertEqual(recognizer.active_pilot_session_id, "session_B")


if __name__ == "__main__":
    unittest.main()

