"""
Mô-đun Ghi nhật ký Chẩn đoán Thử nghiệm Webcam (Pilot Diagnostic Logger)
========================================================================
Dự án: VSL Connect AI (Task 7F.5A)
Mục tiêu:
- Thu thập dữ liệu đánh giá chi tiết từng frame webcam có kiểm soát.
- Cơ chế non-blocking: Đưa bản ghi vào queue đa luồng, ghi đĩa bằng worker nền.
- Không gây nghẽn (zero-latency overhead) cho luồng nhận diện.
- Giữ nguyên độ chính xác, không thay đổi bất kỳ hành vi phân loại nào.
- Theo dõi thống kê ghi log: queued, logged, dropped_queue_full, dropped_io_error.
- Hỗ trợ Flush đồng bộ khi kết thúc phiên thử nghiệm hoặc đóng ứng dụng.
"""

import os
import re
import time
import json
import queue
import atexit
import threading
from typing import Optional, Dict, Any, List

# Danh sách 22 chữ cái tĩnh tiếng Việt chuẩn hóa (VSL Alphabet V2)
STATIC_CLASSES_22 = [
    "A", "B", "C", "D", "E", "G", "H", "I", "K", "L",
    "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "X", "Y"
]
VALID_STATIC_CLASSES_SET = set(STATIC_CLASSES_22)

# Thư mục lưu trữ log pilot mặc định
DEFAULT_PILOT_LOG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "diagnostics", "pilot"
)


def sanitize_expected_sign(expected_sign: Any) -> Optional[str]:
    """
    Chuẩn hóa và kiểm tra nhãn ground truth (expected_sign).
    - Chỉ chấp nhận nếu thuộc danh sách 22 chữ cái tĩnh tiếng Việt.
    - Trả về None nếu không hợp lệ hoặc rỗng.
    - Tuyệt đối không can thiệp vào logic dự đoán của mô hình.
    """
    if expected_sign is None:
        return None
    val = str(expected_sign).strip().upper()
    if val in VALID_STATIC_CLASSES_SET:
        return val
    return None


SESSION_ID_PATTERN = re.compile(r'^[a-zA-Z0-9_-]{3,64}$')


def validate_session_id(session_id: Any) -> str:
    r"""
    Strict validation cho session_id (Task 7F.5E):
    - Toàn bộ chuỗi chỉ được chứa [a-zA-Z0-9_-].
    - Độ dài từ 3 đến 64 ký tự.
    - Phải chứa ít nhất 1 ký tự chữ hoặc số (không chấp nhận chuỗi chỉ toàn dấu gạch/dấu chấm).
    - Tuyệt đối không chứa khoảng trắng, dấu gạch chéo (/ hoặc \), dấu chấm, ký tự traversal.
    - Từ chối ngay lập tức (ValueError) nếu vi phạm, không tự ý sanitize âm thầm làm biến đổi ID.
    """
    if session_id is None:
        raise ValueError("session_id không hợp lệ: không được để trống (None).")

    if not isinstance(session_id, str):
        raise ValueError(f"session_id không hợp lệ: phải là kiểu chuỗi (string), nhận được {type(session_id).__name__}.")

    sess = session_id.strip()
    if not sess or len(sess) < 3:
        raise ValueError(f"session_id không hợp lệ: quá ngắn (tối thiểu 3 ký tự): '{session_id}'.")
    if len(sess) > 64:
        raise ValueError(f"session_id không hợp lệ: quá dài (tối đa 64 ký tự, nhận được {len(sess)} ký tự).")

    if not SESSION_ID_PATTERN.match(sess):
        raise ValueError(
            f"session_id không hợp lệ: chứa ký tự bị cấm: '{session_id}'. "
            f"Chỉ cho phép ký tự chữ (a-z, A-Z), số (0-9), gạch nối (-) và gạch dưới (_)."
        )

    if not re.search(r'[a-zA-Z0-9]', sess):
        raise ValueError(f"session_id không hợp lệ: phải chứa ít nhất một ký tự chữ hoặc số: '{session_id}'.")

    return sess


def create_safe_pilot_file(target_dir: str, clean_session_id: str) -> str:
    """
    Tạo tệp log pilot một cách an toàn và nguyên tử (atomic - Task 7F.5D / 7F.5E):
    - Xác thực nghiêm ngặt session_id qua validate_session_id.
    - Đảm bảo target_dir nằm trong thư mục hợp lệ, loại trừ symlink escape và path traversal.
    - Sử dụng cờ os.O_CREAT | os.O_EXCL | os.O_WRONLY để tạo file nguyên tử ở cấp kernel.
    - Nếu tệp đã tồn tại (trùng session_id từ phiên trước), tự động tạo tên mới duy nhất
      thay vì ghi đè hoặc tự động nối dữ liệu vào tệp của session cũ.
    """
    clean_session_id = validate_session_id(clean_session_id)

    target_dir_real = os.path.realpath(target_dir)
    os.makedirs(target_dir_real, exist_ok=True)

    base_candidate = os.path.join(target_dir_real, f"pilot_{clean_session_id}.jsonl")
    candidate_real = os.path.realpath(base_candidate)

    # Kiểm tra path containment và symlink escape
    if os.path.commonpath([candidate_real, target_dir_real]) != target_dir_real:
        raise ValueError(f"Đường dẫn file không hợp lệ hoặc thoát khỏi thư mục pilot: {base_candidate}")

    # Thử tạo file ban đầu với O_CREAT | O_EXCL
    try:
        fd = os.open(base_candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
        return base_candidate
    except FileExistsError:
        pass

    # Nếu file đã tồn tại (trùng session_id), tạo file mới với timestamp và counter tránh ghi nối/đè
    timestamp_str = time.strftime("%Y%m%d_%H%M%S")
    counter = 1
    while True:
        unique_candidate = os.path.join(
            target_dir_real,
            f"pilot_{clean_session_id}_{timestamp_str}_{counter}.jsonl"
        )
        unique_real = os.path.realpath(unique_candidate)
        if os.path.commonpath([unique_real, target_dir_real]) != target_dir_real:
            raise ValueError(f"Đường dẫn file không hợp lệ hoặc thoát khỏi thư mục pilot: {unique_candidate}")
        try:
            fd = os.open(unique_candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.close(fd)
            return unique_candidate
        except FileExistsError:
            counter += 1


class FlushBarrier:
    """Barrier đồng bộ hóa dùng để bảo đảm worker đã xử lý 100% record trước barrier (Task 7F.5D Fix B-02)."""
    def __init__(self, initial_error_count: int = 0):
        self.completed = threading.Event()
        self.initial_error_count = initial_error_count
        self.success = False
        self.errors_occurred = False
        self.error_count_during_flush = 0


class PilotDiagnosticLogger:
    """
    Bộ ghi nhật ký phi nghẽn (Non-blocking Logger) cho Controlled Webcam Pilot.
    Sử dụng Queue kích thước giới hạn và Worker Daemon Thread để cách ly I/O hoàn toàn.
    Hỗ trợ vòng đời 3 trạng thái: OPEN -> CLOSING -> CLOSED (Task 7F.5G).
    """

    STATE_OPEN = "OPEN"
    STATE_CLOSING = "CLOSING"
    STATE_CLOSED = "CLOSED"

    def __init__(
        self,
        log_dir: Optional[str] = None,
        filepath: Optional[str] = None,
        queue_size: int = 10000,
    ):
        self.log_dir = log_dir or DEFAULT_PILOT_LOG_DIR
        self.default_filepath = filepath
        self.queue_size = queue_size
        self.queue: queue.Queue = queue.Queue(maxsize=queue_size)

        # Thống kê bản ghi (Record Statistics)
        self.lock = threading.Lock()
        self.stats = {
            "queued_count": 0,
            "logged_count": 0,
            "dropped_queue_full": 0,
            "dropped_io_error": 0,
        }

        # Vòng đời 3 trạng thái & đồng bộ hóa đóng logger (Task 7F.5F / 7F.5G)
        self.state = self.STATE_OPEN
        self.is_running = True
        self._shutdown_lock = threading.Lock()
        self._in_flight_producers = 0
        self._all_producers_done = threading.Condition(self.lock)
        self._test_worker_pre_write_hook = None  # Hook kiểm soát thời gian phục vụ unit test barrier
        self._test_producer_pre_enqueue_hook = None  # Hook kiểm soát race condition log_frame vs close (Task 7F.5F)
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            name="VSLPilotLoggerWorker",
            daemon=True
        )
        self._worker_thread.start()

        # Đăng ký tự động flush khi thoát chương trình
        atexit.register(self.close)

    def get_default_filepath(self) -> str:
        """Sinh đường dẫn file mặc định theo ngày nếu chưa chỉ định file cụ thể."""
        if self.default_filepath:
            return self.default_filepath
        os.makedirs(self.log_dir, exist_ok=True)
        date_str = time.strftime("%Y%m%d")
        return os.path.join(self.log_dir, f"pilot_diagnostic_{date_str}.jsonl")

    def log_frame(self, record: Dict[str, Any], filepath: Optional[str] = None) -> bool:
        """
        Đưa 1 bản ghi frame vào queue để worker ghi đĩa bất đồng bộ.
        Trả về True nếu đưa vào queue thành công, False nếu queue đầy hoặc logger đang đóng.
        Đảm bảo đồng bộ tuyệt đối với close(): không record nào bị enqueue sau sentinel (Task 7F.5F / 7F.5G).
        """
        with self.lock:
            if self.state != self.STATE_OPEN or not self.is_running:
                return False
            self._in_flight_producers += 1

        try:
            if self._test_producer_pre_enqueue_hook is not None:
                self._test_producer_pre_enqueue_hook(record)

            target_file = filepath or self.get_default_filepath()
            self.queue.put_nowait((record, target_file))
            with self.lock:
                self.stats["queued_count"] += 1
            return True
        except queue.Full:
            with self.lock:
                self.stats["dropped_queue_full"] += 1
            return False
        except Exception:
            with self.lock:
                self.stats["dropped_io_error"] += 1
            return False
        finally:
            with self.lock:
                self._in_flight_producers -= 1
                if self._in_flight_producers == 0:
                    self._all_producers_done.notify_all()

    def _worker_loop(self):
        """Vòng lặp tiêu thụ bản ghi và ghi file JSONL trong background thread."""
        while True:
            try:
                item = self.queue.get(timeout=0.1)
            except queue.Empty:
                with self.lock:
                    if self.state == self.STATE_CLOSED and self.queue.empty():
                        break
                continue

            if item is None:
                # Sentinel báo hiệu dừng luồng
                self.queue.task_done()
                break

            # Xử lý Barrier đồng bộ hóa flush (Task 7F.5D Fix B-02)
            if isinstance(item, FlushBarrier):
                try:
                    with self.lock:
                        current_errors = self.stats["dropped_io_error"] + self.stats["dropped_queue_full"]
                        errors_delta = current_errors - item.initial_error_count
                        item.error_count_during_flush = errors_delta
                        item.errors_occurred = (errors_delta > 0)
                        item.success = (not item.errors_occurred)
                finally:
                    item.completed.set()
                    self.queue.task_done()
                continue

            record, target_file = item
            try:
                if self._test_worker_pre_write_hook is not None:
                    self._test_worker_pre_write_hook(record, target_file)

                # Kiểm tra an toàn symlink & containment trước khi mở tệp ghi (Task 7F.5E)
                if os.path.islink(target_file):
                    raise IOError(f"Bảo vệ an toàn: phát hiện symlink không hợp lệ: {target_file}")

                target_dir_real = os.path.realpath(os.path.dirname(target_file))
                target_real = os.path.realpath(target_file)
                if os.path.commonpath([target_real, target_dir_real]) != target_dir_real:
                    raise IOError(f"Bảo vệ an toàn: tệp đích thoát khỏi thư mục hợp lệ: {target_file}")

                os.makedirs(target_dir_real, exist_ok=True)
                line = json.dumps(record, ensure_ascii=False) + "\n"
                with open(target_file, "a", encoding="utf-8") as f:
                    f.write(line)
                    f.flush()
                with self.lock:
                    self.stats["logged_count"] += 1
            except Exception:
                with self.lock:
                    self.stats["dropped_io_error"] += 1
            finally:
                self.queue.task_done()

    def flush(self, timeout: float = 5.0) -> bool:
        """
        Đợi worker xử lý xong toàn bộ record đã enqueue trước thời điểm gọi flush (Fix B-02 / Task 7F.5E / 7F.5G).
        Sử dụng FlushBarrier đặt vào queue để xác nhận tuần tự FIFO.
        - Không chỉ kiểm tra queue.empty().
        - Tính cả record worker đang lấy và đang ghi dở dang.
        - Trả về True CHỈ KHI worker đã xử lý qua barrier VÀ không có lỗi I/O trước barrier.
        - Trả về False nếu bị timeout, lỗi I/O, hoặc worker không hoạt động.
        """
        if not self._worker_thread.is_alive():
            return self.queue.empty()

        with self.lock:
            initial_errors = self.stats["dropped_io_error"] + self.stats["dropped_queue_full"]

        barrier = FlushBarrier(initial_error_count=initial_errors)
        start_time = time.monotonic()

        try:
            self.queue.put(barrier, timeout=max(0.0, timeout))
        except queue.Full:
            return False

        elapsed = time.monotonic() - start_time
        remaining_timeout = max(0.0, timeout - elapsed)
        if remaining_timeout <= 0:
            return False

        ok = barrier.completed.wait(timeout=remaining_timeout)
        if not ok:
            return False

        return barrier.success

    def get_stats(self) -> Dict[str, int]:
        """Lấy bản sao thống kê số lượng bản ghi."""
        with self.lock:
            res = dict(self.stats)
            res["total_dropped"] = res["dropped_queue_full"] + res["dropped_io_error"]
            res["pending_in_queue"] = self.queue.qsize()
            return res

    def reset_stats(self):
        """Đặt lại toàn bộ bộ đếm thống kê."""
        with self.lock:
            self.stats = {
                "queued_count": 0,
                "logged_count": 0,
                "dropped_queue_full": 0,
                "dropped_io_error": 0,
            }

    def close(self, timeout: float = 3.0) -> bool:
        """
        Đóng logger an toàn theo vòng đời 3 trạng thái (OPEN -> CLOSING -> CLOSED) (Task 7F.5G):
        - Sử dụng time.monotonic() cho deadline tracking chặt chẽ, không bù thời gian giả lập.
        - Đồng bộ bằng _shutdown_lock ngăn chặn duplicate sentinels hoặc race condition giữa các luồng close.
        - Chuyển state sang CLOSING, is_running=False, từ chối mọi producer mới.
        - Đợi toàn bộ in-flight producers hoàn tất enqueue trong giới hạn timeout.
        - Nếu timeout khi còn in-flight producer: ABORT, không gửi sentinel, không dừng worker,
          giữ trạng thái CLOSING và trả về False. Cho phép lần close() sau thử lại hoàn tất.
        - Nếu in-flight producers đã xong: gọi flush() để worker ghi hết record đã enqueue.
        - Nếu flush thất bại (I/O error hoặc timeout): trả về False.
        - Gửi sentinel None vào queue và join worker thread.
        - Chuyển state sang CLOSED và trả về True.
        """
        start_mono = time.monotonic()
        deadline = start_mono + max(0.0, timeout)

        # 1. Đồng bộ hóa giữa các luồng gọi close đồng thời
        acquired = self._shutdown_lock.acquire(timeout=max(0.0, deadline - time.monotonic()))
        if not acquired:
            return False

        try:
            with self.lock:
                if self.state == self.STATE_CLOSED:
                    return True
                if self.state == self.STATE_OPEN:
                    self.state = self.STATE_CLOSING
                    self.is_running = False

            # 2. Đợi toàn bộ in-flight producers hoàn tất enqueue
            with self.lock:
                while self._in_flight_producers > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        # Timeout khi vẫn còn in-flight producer!
                        # Bắt buộc: KHÔNG gửi sentinel, KHÔNG dừng worker, giữ CLOSING, trả về False
                        return False
                    self._all_producers_done.wait(timeout=remaining)

            # 3. Flush barrier để bảo đảm toàn bộ record đã enqueue được worker xử lý
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False

            flush_ok = self.flush(timeout=remaining)
            if not flush_ok:
                return False

            # 4. Đưa sentinel None vào queue để báo hiệu worker dừng
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False

            try:
                self.queue.put(None, timeout=remaining)
            except (queue.Full, Exception):
                return False

            # 5. Đợi worker thread kết thúc
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False

            if self._worker_thread.is_alive():
                self._worker_thread.join(timeout=remaining)
                if self._worker_thread.is_alive():
                    return False

            # 6. Hoàn tất đóng thành công
            with self.lock:
                self.state = self.STATE_CLOSED
                self.is_running = False
            return True
        finally:
            self._shutdown_lock.release()


# Quản lý đối tượng Logger dùng chung
_global_pilot_logger: Optional[PilotDiagnosticLogger] = None
_global_logger_lock = threading.Lock()


def get_pilot_logger(filepath: Optional[str] = None, log_dir: Optional[str] = None) -> PilotDiagnosticLogger:
    """Lấy hoặc khởi tạo instance PilotDiagnosticLogger dùng chung."""
    global _global_pilot_logger
    with _global_logger_lock:
        if _global_pilot_logger is None or not _global_pilot_logger.is_running:
            _global_pilot_logger = PilotDiagnosticLogger(log_dir=log_dir, filepath=filepath)
        elif filepath and _global_pilot_logger.default_filepath != filepath:
            _global_pilot_logger.default_filepath = filepath
        return _global_pilot_logger


def close_pilot_logger(timeout: float = 3.0) -> bool:
    """Đóng và dọn dẹp global pilot logger (Task 7F.5G)."""
    global _global_pilot_logger
    with _global_logger_lock:
        if _global_pilot_logger is not None:
            ok = _global_pilot_logger.close(timeout=timeout)
            _global_pilot_logger = None
            return ok
        return True

