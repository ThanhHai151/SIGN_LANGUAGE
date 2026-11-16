"""
Unit & Race-Condition Tests for Task 7F.5G: Close Timeout / In-Flight Producer Lifecycle
========================================================================================
Kiểm thử chuyên sâu các trường hợp biên vòng đời Logger (OPEN -> CLOSING -> CLOSED):
1. Test 1: Producer blocked before enqueue -> close(timeout=0.2) trả về False, worker vẫn sống,
   không có sentinel trong queue, trạng thái giữ CLOSING.
2. Test 2: Giải phóng producer từ Test 1 -> enqueue hoàn tất trả về True. Gọi close() lần 2 ->
   trả về True, bản ghi được ghi đĩa 100%, worker kết thúc an toàn, trạng thái CLOSED.
3. Test 3: Khi logger ở trạng thái CLOSING, mọi lệnh gọi log_frame() mới bị từ chối (trả về False).
4. Test 4: Đồng bộ hóa concurrent close() từ nhiều luồng -> không duplicate sentinel, không deadlock,
   tất cả đều xử lý hoàn tất sạch sẽ và trả về True.
5. Test 5: Quy trình đóng bình thường với hàng đợi chứa sẵn record -> toàn bộ được ghi đĩa,
   worker dừng, trả về True.
6. Test 6: Lỗi I/O trong quá trình worker ghi đĩa khi shutdown -> ghi nhận lỗi vào stats và close()
   trả về False.
7. Test Stress: 8 luồng concurrent producers đẩy dữ liệu trong lúc close() được kích hoạt ->
   dữ liệu trên đĩa khớp 100% với số lượng báo thành công.
8. Test Timeout Bounds: time.monotonic() đảm bảo timeout có upper bound chặt chẽ, không tự ý cộng 0.5s padding.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import threading
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.vision.pilot_logger import PilotDiagnosticLogger


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


class TestPilotCloseEnqueueRace(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_1_producer_blocked_close_timeout_returns_false_and_keeps_worker_alive(self):
        """
        Test 1: Producer bị chặn trước khi enqueue -> close(timeout=0.2) trả về False,
        worker thread vẫn sống, không gửi sentinel, trạng thái giữ nguyên CLOSING.
        """
        target_file = os.path.join(self.temp_dir, "test_1_blocked.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        in_hook_event = threading.Event()
        release_event = threading.Event()

        def pause_hook(record):
            in_hook_event.set()
            release_event.wait(timeout=5.0)

        logger._test_producer_pre_enqueue_hook = pause_hook

        prod_thread = threading.Thread(target=lambda: logger.log_frame({"data": "t1_payload"}))
        prod_thread.start()

        # Đợi producer đi vào hook (đã vượt qua check trạng thái, _in_flight_producers = 1)
        self.assertTrue(in_hook_event.wait(timeout=2.0))
        self.assertEqual(logger._in_flight_producers, 1)

        # Gọi close với timeout ngắn 0.15s
        t0 = time.monotonic()
        close_res = logger.close(timeout=0.15)
        duration = time.monotonic() - t0

        # XÁC MINH CỐT LÕI:
        # 1. close() trả về False vì timeout khi producer in-flight
        self.assertFalse(close_res)
        # 2. Thời gian chạy giới hạn chặt chẽ theo timeout
        self.assertLess(duration, 0.4)
        # 3. Trạng thái là CLOSING, is_running=False
        self.assertEqual(logger.state, PilotDiagnosticLogger.STATE_CLOSING)
        self.assertFalse(logger.is_running)
        # 4. Worker thread vẫn SỐNG (không bị terminate oan)
        self.assertTrue(logger._worker_thread.is_alive())
        # 5. _in_flight_producers vẫn là 1
        self.assertEqual(logger._in_flight_producers, 1)

        # Dọn dẹp: nhả producer và đóng logger
        release_event.set()
        prod_thread.join(timeout=2.0)
        final_close = logger.close(timeout=2.0)
        self.assertTrue(final_close)

    def test_2_resume_producer_and_reclose_succeeds_with_record_written(self):
        """
        Test 2: Nối tiếp kịch bản Test 1: giải phóng producer -> hoàn tất enqueue trả về True.
        Gọi close() lần 2 -> trả về True, record được ghi đĩa 100%, worker kết thúc an toàn, state CLOSED.
        """
        target_file = os.path.join(self.temp_dir, "test_2_resume.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        in_hook_event = threading.Event()
        release_event = threading.Event()

        logger._test_producer_pre_enqueue_hook = lambda r: (in_hook_event.set(), release_event.wait(timeout=5.0))

        prod_result = []

        def prod_task():
            res = logger.log_frame({"data": "t2_payload"})
            prod_result.append(res)

        prod_thread = threading.Thread(target=prod_task)
        prod_thread.start()

        self.assertTrue(in_hook_event.wait(timeout=2.0))

        # close() lần 1 bị timeout
        close_1 = logger.close(timeout=0.1)
        self.assertFalse(close_1)
        self.assertEqual(logger.state, PilotDiagnosticLogger.STATE_CLOSING)
        self.assertTrue(logger._worker_thread.is_alive())

        # Giải phóng producer để hoàn tất enqueue
        release_event.set()
        prod_thread.join(timeout=2.0)

        # Producer ban đầu phải báo thành công (True)
        self.assertEqual(prod_result, [True])
        self.assertEqual(logger._in_flight_producers, 0)

        # Gọi close() lần 2 để hoàn tất shutdown
        close_2 = logger.close(timeout=2.0)
        self.assertTrue(close_2)

        # XÁC MINH CỐT LÕI:
        # 1. State chuyển thành CLOSED
        self.assertEqual(logger.state, PilotDiagnosticLogger.STATE_CLOSED)
        # 2. Worker thread đã kết thúc hoàn toàn
        self.assertFalse(logger._worker_thread.is_alive())
        # 3. Bản ghi đã được ghi xuống file JSONL
        records = _load_jsonl_records(target_file)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["data"], "t2_payload")

    def test_3_in_closing_state_log_frame_rejected(self):
        """
        Test 3: Khi logger ở trạng thái CLOSING, mọi producer mới gọi log_frame() bị từ chối (trả về False).
        _in_flight_producers không bị tăng cho producer bị từ chối.
        """
        target_file = os.path.join(self.temp_dir, "test_3_rejected.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        in_hook_event = threading.Event()
        release_event = threading.Event()

        logger._test_producer_pre_enqueue_hook = lambda r: (in_hook_event.set(), release_event.wait(timeout=5.0))

        prod_thread = threading.Thread(target=lambda: logger.log_frame({"data": "initial_producer"}))
        prod_thread.start()
        self.assertTrue(in_hook_event.wait(timeout=2.0))

        # Đưa logger vào trạng thái CLOSING thông qua close timeout
        logger.close(timeout=0.1)
        self.assertEqual(logger.state, PilotDiagnosticLogger.STATE_CLOSING)

        # Producer mới gọi log_frame() trong khi logger đang CLOSING
        for i in range(5):
            res = logger.log_frame({"data": f"rejected_new_producer_{i}"})
            self.assertFalse(res)

        # Số in-flight producer vẫn chỉ là 1 (chỉ producer ban đầu)
        self.assertEqual(logger._in_flight_producers, 1)

        # Dọn dẹp
        release_event.set()
        prod_thread.join(timeout=2.0)
        logger.close(timeout=2.0)

        # File chỉ chứa bản ghi của producer ban đầu
        records = _load_jsonl_records(target_file)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["data"], "initial_producer")

    def test_4_concurrent_close_calls_synchronized_and_no_deadlock(self):
        """
        Test 4: Đồng thời gọi close() từ 4 luồng khác nhau.
        Xác minh: _shutdown_lock serialize mượt mà, không deadlock, không lỗi duplicate sentinel,
        tất cả các luồng đều nhận kết quả True, worker dừng an toàn.
        """
        target_file = os.path.join(self.temp_dir, "test_4_concurrent_close.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        # Ghi 10 bản ghi trước
        for i in range(10):
            logger.log_frame({"idx": i})

        results = []
        lock = threading.Lock()

        def closer_task():
            res = logger.close(timeout=3.0)
            with lock:
                results.append(res)

        threads = [threading.Thread(target=closer_task) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=4.0)

        # XÁC MINH CỐT LÕI:
        # 1. Toàn bộ 4 luồng đều hoàn tất (không deadlock) và trả về True
        self.assertEqual(len(results), 4)
        self.assertTrue(all(results))
        # 2. State là CLOSED, worker đã tắt
        self.assertEqual(logger.state, PilotDiagnosticLogger.STATE_CLOSED)
        self.assertFalse(logger._worker_thread.is_alive())
        # 3. Đủ 10 bản ghi trên đĩa
        records = _load_jsonl_records(target_file)
        self.assertEqual(len(records), 10)

    def test_5_normal_shutdown_with_records_in_queue_succeeds(self):
        """
        Test 5: Quy trình đóng bình thường: 50 records được đẩy vào hàng đợi.
        close() flush thành công 100%, sentinel tiêu thụ, worker dừng, close() trả về True.
        """
        target_file = os.path.join(self.temp_dir, "test_5_normal.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        for i in range(50):
            self.assertTrue(logger.log_frame({"item_id": i}))

        res = logger.close(timeout=3.0)
        self.assertTrue(res)
        self.assertEqual(logger.state, PilotDiagnosticLogger.STATE_CLOSED)
        self.assertFalse(logger._worker_thread.is_alive())
        self.assertTrue(logger.queue.empty())

        records = _load_jsonl_records(target_file)
        self.assertEqual(len(records), 50)
        self.assertEqual(records[0]["item_id"], 0)
        self.assertEqual(records[-1]["item_id"], 49)

    def test_6_io_failure_during_shutdown_reflected_in_stats_and_returns_false(self):
        """
        Test 6: Lỗi I/O xảy ra khi worker ghi đĩa trong quá trình shutdown.
        Xác minh: Lỗi được phản ánh vào stats['dropped_io_error'], flush thất bại và close() trả về False.
        """
        target_file = os.path.join(self.temp_dir, "test_6_io_error.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        # Hook gây lỗi I/O có chủ đích trong worker khi ghi record
        def failing_write_hook(record, filepath):
            raise IOError("Simulated disk write failure during shutdown")

        logger._test_worker_pre_write_hook = failing_write_hook

        # Enqueue 1 bản ghi
        self.assertTrue(logger.log_frame({"data": "doomed_record"}))

        # Gọi close(): worker ghi sẽ vấp phải IOError, flush barrier phát hiện lỗi -> close trả về False
        res = logger.close(timeout=2.0)
        self.assertFalse(res)

        # Thống kê ghi nhận lỗi I/O
        stats = logger.get_stats()
        self.assertGreaterEqual(stats["dropped_io_error"], 1)

        # Gỡ hook lỗi và đóng dọn dẹp
        logger._test_worker_pre_write_hook = None
        # Drain worker queue and close
        logger.queue.get_nowait() if not logger.queue.empty() else None
        logger._worker_thread.join(timeout=0.5)

    def test_stress_producers_racing_with_close(self):
        """
        Test 7 (Stress): 8 luồng concurrent producers liên tục đẩy frame trong lúc close() được gọi.
        Xác minh: Số record trên đĩa phải CHÍNH XÁC bằng số record được báo thành công (True).
        """
        target_file = os.path.join(self.temp_dir, "test_race_stress.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        num_threads = 8
        records_per_thread = 40
        successful_enqueues = []
        lock = threading.Lock()

        def stress_producer(t_id):
            for i in range(records_per_thread):
                record = {"t_id": t_id, "seq": i}
                ok = logger.log_frame(record)
                if ok:
                    with lock:
                        successful_enqueues.append(record)
                time.sleep(0.001)

        threads = [threading.Thread(target=stress_producer, args=(t,)) for t in range(num_threads)]
        for t in threads:
            t.start()

        time.sleep(0.02)
        close_ok = logger.close(timeout=4.0)
        self.assertTrue(close_ok)

        for t in threads:
            t.join(timeout=2.0)

        self.assertEqual(logger.state, PilotDiagnosticLogger.STATE_CLOSED)
        self.assertFalse(logger._worker_thread.is_alive())
        self.assertTrue(logger.queue.empty())

        disk_records = _load_jsonl_records(target_file)
        self.assertEqual(len(disk_records), len(successful_enqueues))

    def test_timeout_strict_upper_bound_no_arbitrary_padding(self):
        """
        Test 8: Kiểm tra giới hạn thời gian (strict upper bound) không tự ý cộng thêm 0.5s padding.
        close(timeout=0.15) phải kết thúc dưới 0.35s khi producer bị nghẽn.
        """
        target_file = os.path.join(self.temp_dir, "test_timeout_bound.jsonl")
        logger = PilotDiagnosticLogger(filepath=target_file)

        stuck_event = threading.Event()
        release_event = threading.Event()
        logger._test_producer_pre_enqueue_hook = lambda r: (stuck_event.set(), release_event.wait(timeout=5.0))

        t = threading.Thread(target=lambda: logger.log_frame({"data": "stuck"}))
        t.start()
        self.assertTrue(stuck_event.wait(timeout=2.0))

        t0 = time.monotonic()
        res = logger.close(timeout=0.15)
        duration = time.monotonic() - t0

        self.assertFalse(res)
        # Bắt buộc duration phải nằm sát 0.15s, tuyệt đối không được vượt quá 0.35s (không cộng 0.5s padding)
        self.assertLess(duration, 0.35)

        release_event.set()
        t.join(timeout=1.0)
        logger.close(timeout=1.0)


if __name__ == "__main__":
    unittest.main()
