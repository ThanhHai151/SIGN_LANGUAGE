"""
Quản lý và Tự động Dọn dẹp Frame tạm (Temporary Frame Management & Cleanup).
=============================================================================
Đảm bảo an toàn tuyệt đối theo nguyên tắc:
1. Frame video trích xuất chỉ phục vụ phân tích/xử lý tại chỗ là dữ liệu tạm.
2. Tự động dọn dẹp bằng context manager (try...finally), kể cả khi xảy ra Exception.
3. Tạo thư mục cô lập cho từng phiên (session-isolated directory).
4. Tự động phát hiện và dọn dẹp các frame mồ côi (orphaned sessions) từ các lần chạy trước bị crash.
5. Bảo vệ tuyệt đối: Ngăn chặn xóa nhầm video gốc, dataset gốc, annotation và thư mục hệ thống.
6. Hỗ trợ in-memory streaming: Không cần ghi đĩa trừ khi bật cờ lưu debug rõ ràng.
"""

import os
import shutil
import time
import uuid
from typing import Generator, List, Optional, Tuple, Any
import cv2
import numpy as np


# Thư mục tạm mặc định của project
DEFAULT_TEMP_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "data", "alphabet_v2", "temp_frames")
)

# Danh sách các thư mục được bảo vệ tuyệt đối (Không bao giờ được phép xóa)
PROTECTED_PATHS = [
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "media_cache", "videos")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "alphabet_v2", "datasets")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "alphabet_v2", "static_landmarks")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "tests")),
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "vision")),
]


def _is_safe_temp_dir(path: str, allowed_root: str) -> bool:
    """Kiểm tra đường dẫn có an toàn và nằm trọn vẹn trong allowed_root hay không."""
    abs_path = os.path.abspath(path)
    abs_root = os.path.abspath(allowed_root)
    
    # Không được trùng chính xác với root hoặc các thư mục hệ thống
    if abs_path == abs_root or abs_path == os.path.dirname(abs_root):
        return False
        
    # Phải là thư mục con của allowed_root
    common = os.path.commonpath([abs_path, abs_root])
    if common != abs_root:
        return False
        
    # Không được nằm trong danh sách thư mục bảo vệ
    for prot in PROTECTED_PATHS:
        try:
            if os.path.commonpath([abs_path, prot]) == prot:
                return False
        except ValueError:
            continue
            
    return True


class TemporaryFrameSession:
    """
    Context manager quản lý vòng đời thư mục frame tạm cho một phiên phân tích.
    
    Cách dùng:
        with TemporaryFrameSession() as session:
            # Ghi frame tạm nếu cần thiết
            frame_path = session.save_frame(img, "frame_01.jpg")
            # Hoặc stream trực tiếp in-memory mà không ghi đĩa
            for idx, frame in session.stream_video_frames("path/to/video.mp4"):
                landmarks = process(frame)
        # Thư mục tạm tự động được dọn dẹp sạch sẽ tại đây
    """

    def __init__(
        self,
        base_dir: str = DEFAULT_TEMP_ROOT,
        session_id: Optional[str] = None,
        session_prefix: str = "sess_",
        keep_debug_frames: bool = False,
        auto_purge_orphans: bool = True,
        orphan_max_age_sec: float = 1800.0,
    ):
        self.base_dir = os.path.abspath(base_dir)
        for prot in PROTECTED_PATHS:
            try:
                cp = os.path.commonpath([self.base_dir, prot])
            except ValueError:
                continue
            if cp == prot:
                raise ValueError(f"Đường dẫn base_dir {base_dir} nằm trong thư mục được bảo vệ {prot}!")

        self.session_id = session_id or f"{session_prefix}{int(time.time())}_{uuid.uuid4().hex[:8]}"
        self.keep_debug_frames = keep_debug_frames
        self.auto_purge_orphans = auto_purge_orphans
        self.orphan_max_age_sec = orphan_max_age_sec
        self.session_dir = os.path.join(self.base_dir, self.session_id)
        self._created_files: List[str] = []
        self._is_active = False

    def __enter__(self) -> "TemporaryFrameSession":
        if self.auto_purge_orphans:
            self.purge_orphaned_sessions(self.base_dir, max_age_seconds=self.orphan_max_age_sec)

        os.makedirs(self.session_dir, exist_ok=True)
        self._is_active = True
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False  # Không nuốt ngoại lệ

    def save_frame(self, image: np.ndarray, filename: str) -> str:
        """Ghi frame ảnh tạm vào thư mục phiên an toàn."""
        if not self._is_active:
            raise RuntimeError("Session chưa kích hoạt hoặc đã đóng.")
        filepath = os.path.join(self.session_dir, filename)
        cv2.imwrite(filepath, image)
        self._created_files.append(filepath)
        return filepath

    def stream_video_frames(
        self, video_path: str, frame_indices: Optional[List[int]] = None
    ) -> Generator[Tuple[int, np.ndarray, float], None, None]:
        """
        Trích xuất frame trực tiếp in-memory (numpy array) từ video,
        hoàn toàn KHÔNG ghi ra đĩa nhằm tối ưu hiệu năng và tránh rác đĩa.
        Yields: (frame_index, frame_bgr_image, timestamp_sec)
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video không tồn tại: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Không thể mở video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        target_set = set(frame_indices) if frame_indices is not None else None
        current_idx = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if target_set is None or current_idx in target_set:
                    ts = current_idx / fps
                    yield (current_idx, frame, ts)
                current_idx += 1
        finally:
            cap.release()

    def cleanup(self):
        """Dọn dẹp an toàn thư mục tạm của phiên này."""
        self._is_active = False
        if self.keep_debug_frames:
            return  # Giữ lại phục vụ debug khi có chỉ định rõ ràng

        if os.path.exists(self.session_dir):
            if _is_safe_temp_dir(self.session_dir, self.base_dir):
                shutil.rmtree(self.session_dir, ignore_errors=True)

    @staticmethod
    def purge_orphaned_sessions(
        base_dir: str = DEFAULT_TEMP_ROOT, max_age_seconds: float = 1800.0
    ) -> int:
        """
        Phát hiện và dọn dẹp các thư mục phiên mồ côi bị sót lại từ các tiến trình trước đó.
        Chỉ xóa các thư mục có tiền tố 'sess_' và thời gian sửa đổi cũ hơn max_age_seconds.
        Trả về số lượng thư mục đã dọn dẹp.
        """
        abs_base = os.path.abspath(base_dir)
        if not os.path.exists(abs_base):
            return 0

        now = time.time()
        purged_count = 0

        try:
            for item in os.listdir(abs_base):
                item_path = os.path.join(abs_base, item)
                if os.path.isdir(item_path) and item.startswith("sess_"):
                    if _is_safe_temp_dir(item_path, abs_base):
                        try:
                            mtime = os.path.getmtime(item_path)
                            if (now - mtime) >= max_age_seconds:
                                shutil.rmtree(item_path, ignore_errors=True)
                                purged_count += 1
                        except OSError:
                            continue
        except Exception:
            pass

        return purged_count


purge_orphaned_sessions = TemporaryFrameSession.purge_orphaned_sessions
