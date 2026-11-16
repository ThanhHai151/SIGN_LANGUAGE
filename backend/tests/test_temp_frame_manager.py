"""
Unit tests cho TemporaryFrameSession và Quản lý Frame tạm an toàn.
===================================================================
Kiểm thử các yêu cầu Phần 2.1:
1. Tự động dọn dẹp frame tạm khi hoàn thành (try/finally).
2. Tự động dọn dẹp frame tạm kể cả khi xảy ra Exception.
3. Tạo thư mục riêng biệt cho từng phiên (session-isolated).
4. Bảo vệ chống xóa nhầm thư mục chứa video gốc, dataset gốc, models.
5. In-memory streaming không cần ghi frame xuống đĩa trừ khi bật debug.
6. Dọn dẹp session mồ côi (orphaned sessions).
"""

import os
import shutil
import tempfile
# pyrefly: ignore[missing-import]
import pytest
import numpy as np
import cv2

from backend.tools.temp_frame_manager import (
    TemporaryFrameSession,
    purge_orphaned_sessions,
    DEFAULT_TEMP_ROOT,
    PROTECTED_PATHS,
    _is_safe_temp_dir,
)


class TestTemporaryFrameManager:
    """Kiểm tra chức năng quản lý frame tạm an toàn."""

    def test_session_cleanup_on_normal_exit(self, tmp_path):
        """Khi kết thúc bình thường, thư mục session và mọi file tạm phải được xóa."""
        temp_root = str(tmp_path / "temp_frames")
        os.makedirs(temp_root, exist_ok=True)

        session_dir = None
        with TemporaryFrameSession(base_dir=temp_root, session_prefix="sess_normal_") as session:
            session_dir = session.session_dir
            assert os.path.exists(session_dir)
            assert os.path.isdir(session_dir)

            # Lưu một frame tạm
            dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
            saved_path = session.save_frame(dummy_frame, "frame_001.jpg")
            assert os.path.exists(saved_path)

        # Sau khi thoát context manager, session_dir phải không còn tồn tại
        assert not os.path.exists(session_dir)

    def test_session_cleanup_on_exception(self, tmp_path):
        """Khi xảy ra ngoại lệ, cơ chế try/finally vẫn phải dọn dẹp thư mục tạm."""
        temp_root = str(tmp_path / "temp_frames")
        os.makedirs(temp_root, exist_ok=True)

        session_dir = None
        with pytest.raises(RuntimeError, match="Mô phỏng lỗi tiến trình"):
            with TemporaryFrameSession(base_dir=temp_root, session_prefix="sess_error_") as session:
                session_dir = session.session_dir
                dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                session.save_frame(dummy_frame, "frame_err.jpg")
                assert os.path.exists(session_dir)
                raise RuntimeError("Mô phỏng lỗi tiến trình")

        assert not os.path.exists(session_dir)

    def test_protected_paths_guard(self):
        """Cấm tuyệt đối việc sử dụng các thư mục được bảo vệ làm session tạm."""
        for prot_path in PROTECTED_PATHS:
            assert not _is_safe_temp_dir(prot_path, DEFAULT_TEMP_ROOT)
            with pytest.raises(ValueError, match="bảo vệ|không an toàn"):
                TemporaryFrameSession(base_dir=prot_path)

    def test_session_isolation(self, tmp_path):
        """Hai session chạy song song phải có thư mục riêng biệt, không xung đột."""
        temp_root = str(tmp_path / "temp_frames")
        with TemporaryFrameSession(base_dir=temp_root, session_prefix="sess_a_") as s1:
            with TemporaryFrameSession(base_dir=temp_root, session_prefix="sess_b_") as s2:
                assert s1.session_dir != s2.session_dir
                assert os.path.exists(s1.session_dir)
                assert os.path.exists(s2.session_dir)

    def test_in_memory_streaming(self, tmp_path):
        """Test stream frame in-memory từ video mẫu."""
        video_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "media_cache", "videos", "D0505.mp4"
        )
        if not os.path.exists(video_path):
            pytest.skip(f"Video {video_path} không tồn tại")

        temp_root = str(tmp_path / "temp_frames")
        with TemporaryFrameSession(base_dir=temp_root) as session:
            # Stream frame trực tiếp
            frames = list(session.stream_video_frames(video_path, frame_indices=[0, 1, 2, 3, 4]))
            assert len(frames) == 5
            for idx, frame, ts in frames:
                assert isinstance(idx, int)
                assert isinstance(frame, np.ndarray)
                assert frame.shape[0] > 0 and frame.shape[1] > 0
                assert isinstance(ts, float)
            # Không có file nào bị ghi xuống đĩa khi không bật save_to_disk
            assert len(os.listdir(session.session_dir)) == 0

    def test_purge_orphaned_sessions(self, tmp_path):
        """Dọn dẹp các session mồ côi cũ hơn ngưỡng tuổi max_age_seconds."""
        temp_root = str(tmp_path / "temp_frames")
        os.makedirs(temp_root, exist_ok=True)

        old_session = os.path.join(temp_root, "sess_orphan_01")
        os.makedirs(old_session, exist_ok=True)
        with open(os.path.join(old_session, "dummy.txt"), "w") as f:
            f.write("test")

        # Thiết lập thời gian cũ
        old_time = os.path.getmtime(old_session) - 100
        os.utime(old_session, (old_time, old_time))

        purged = purge_orphaned_sessions(base_dir=temp_root, max_age_seconds=10)
        assert purged >= 1
        assert not os.path.exists(old_session)
