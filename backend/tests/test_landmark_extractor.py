"""
Unit tests cho công cụ trích xuất Landmark từ Video và Bảng đăng ký mốc kiểm thử chữ số.
========================================================================================
Kiểm thử các trường hợp:
1. Video không tồn tại (xử lý FileNotFoundError).
2. Kiểm tra định dạng landmark (21 điểm, 3 tọa độ float).
3. Landmark thiếu điểm hoặc sai định dạng bị phát hiện và từ chối.
4. Bảo vệ chống ghi đè dữ liệu fixture ngoài ý muốn (FileExistsError khi overwrite=False).
5. Bảo toàn đầy đủ siêu dữ liệu nguồn (video_id, frame_idx, timestamp_sec, handedness).
6. Tính toàn vẹn của bảng đăng ký video chữ số (D0529-D0538) và trạng thái Ground Truth.
"""

import os
import json
import pytest
import tempfile
from backend.training.extract_video_landmarks import (
    validate_landmark_format,
    extract_landmarks_from_video,
    save_landmark_fixture,
    HAS_OPENCV,
    HAS_MEDIAPIPE
)


class TestLandmarkFormatValidation:
    """Kiểm tra logic xác thực định dạng 21 mốc MediaPipe."""

    def test_valid_21_points(self):
        """Dữ liệu chuẩn 21 điểm [x, y, z] trong khoảng hợp lệ phải pass."""
        valid_landmarks = [[round(i * 0.04, 3), round(i * 0.04, 3), 0.0] for i in range(21)]
        is_valid, msg = validate_landmark_format(valid_landmarks)
        assert is_valid is True
        assert msg == "Hợp lệ"

    def test_invalid_point_count(self):
        """Dữ liệu có số điểm khác 21 phải bị từ chối."""
        too_few = [[0.5, 0.5, 0.0]] * 20
        is_valid, msg = validate_landmark_format(too_few)
        assert is_valid is False
        assert "21 điểm" in msg

        too_many = [[0.5, 0.5, 0.0]] * 22
        is_valid, msg = validate_landmark_format(too_many)
        assert is_valid is False

    def test_missing_coordinates(self):
        """Điểm mốc chỉ có 2 tọa độ (x, y) thiếu z phải bị từ chối."""
        bad_pts = [[0.5, 0.5]] * 21
        is_valid, msg = validate_landmark_format(bad_pts)
        assert is_valid is False
        assert "không đủ 3 tọa độ" in msg

    def test_none_or_invalid_type(self):
        """Input là None hoặc kiểu dữ liệu không hợp lệ phải bị từ chối an toàn."""
        is_valid, msg = validate_landmark_format(None)
        assert is_valid is False

        is_valid, msg = validate_landmark_format("string_data")
        assert is_valid is False

        is_valid, msg = validate_landmark_format({"0": [0, 0, 0]})
        assert is_valid is False


class TestVideoExtractorEdgeCases:
    """Kiểm tra xử lý ngoại lệ và luồng dữ liệu của bộ trích xuất video."""

    def test_nonexistent_video_raises_filenotfound(self):
        """Gọi hàm với đường dẫn video không tồn tại phải ném FileNotFoundError."""
        with pytest.raises(FileNotFoundError) as exc_info:
            extract_landmarks_from_video("backend/data/non_existent_video_xyz.mp4")
        assert "Không tìm thấy file video" in str(exc_info.value)

    @pytest.mark.skipif(not HAS_OPENCV, reason="Yêu cầu OpenCV (cv2)")
    def test_metadata_preservation_real_video(self):
        """Kiểm tra việc bảo toàn siêu dữ liệu trên video thật D0530.mp4."""
        video_path = "backend/data/media_cache/videos/D0530.mp4"
        if not os.path.exists(video_path):
            pytest.skip(f"Video {video_path} không tồn tại trên máy trạm")

        target_frames = [30, 71]
        records = extract_landmarks_from_video(
            video_path=video_path,
            target_frames=target_frames,
            target_handedness="Right"
        )

        assert len(records) == 2
        for rec in records:
            assert rec["video_id"] == "D0530"
            assert rec["source_file"] == "backend/data/media_cache/videos/D0530.mp4"
            assert rec["frame_idx"] in target_frames
            assert isinstance(rec["timestamp_sec"], float)
            assert rec["timestamp_sec"] > 0
            assert rec["handedness"] == "Right"
            assert rec["coordinate_type"] == "normalized_mediapipe_hands"
            # Kiểm tra trạng thái chất lượng được gán rõ ràng
            assert "quality_status" in rec
            if not HAS_MEDIAPIPE:
                assert rec["quality_status"] == "dependency_missing_mediapipe"
                assert rec["landmarks"] is None


class TestFixtureSafetyAndRegistry:
    """Kiểm tra tính an toàn của fixture và tính toàn vẹn của Registry."""

    def test_save_fixture_overwrite_protection(self):
        """Không cho phép ghi đè file fixture khi overwrite=False."""
        with tempfile.TemporaryDirectory() as tmpdir:
            fpath = os.path.join(tmpdir, "test_fixture.json")
            data = {"version": "1.0", "items": []}
            
            # Ghi lần đầu thành công
            save_landmark_fixture(data, fpath, overwrite=False)
            assert os.path.exists(fpath)

            # Ghi đè khi overwrite=False phải ném FileExistsError
            with pytest.raises(FileExistsError):
                save_landmark_fixture({"version": "2.0"}, fpath, overwrite=False)

            # Ghi đè khi overwrite=True thành công
            save_landmark_fixture({"version": "2.0"}, fpath, overwrite=True)
            with open(fpath, "r", encoding="utf-8") as f:
                saved = json.load(f)
            assert saved["version"] == "2.0"

    def test_digit_video_registry_structure(self):
        """Xác thực bảng đăng ký video chữ số digit_video_registry.json."""
        registry_path = "backend/tests/fixtures/digit_video_registry.json"
        assert os.path.exists(registry_path), "Thiếu file fixture digit_video_registry.json"

        with open(registry_path, "r", encoding="utf-8") as f:
            reg = json.load(f)

        assert "digits" in reg
        digits = reg["digits"]
        assert len(digits) == 10  # Đầy đủ 0-9

        # Số 0: Bắt buộc đánh dấu CHƯA CÓ GROUND TRUTH do D0529 chưa tồn tại cục bộ
        assert digits["0"]["has_ground_truth"] is False
        assert digits["0"]["file_exists_locally"] is False
        assert digits["0"]["status"] == "CHƯA CÓ GROUND TRUTH"

        # Số 1 đến 9: Phải trỏ đúng video thật tồn tại cục bộ
        for d in range(1, 10):
            d_str = str(d)
            entry = digits[d_str]
            assert entry["file_exists_locally"] is True
            assert os.path.exists(entry["source_video_file"]), f"File video số {d_str} không tồn tại: {entry['source_video_file']}"
            assert "dev_frame" in entry
            assert "eval_frame" in entry
            # Dev frame và eval frame phải là hai frame riêng biệt để đảm bảo tính độc lập
            assert entry["dev_frame"]["frame_idx"] != entry["eval_frame"]["frame_idx"]

    def test_real_digit_landmarks_fixture_integrity(self):
        """Kiểm tra tính toàn vẹn của tệp fixture thực tế real_digit_landmarks.json."""
        fixture_path = "backend/tests/fixtures/real_digit_landmarks.json"
        assert os.path.exists(fixture_path), f"Tệp fixture {fixture_path} chưa được tạo"

        with open(fixture_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data.get("schema_version") == "1.0"
        assert data.get("total_samples") == 18
        assert data.get("successful_samples") == 18

        samples = data.get("samples", [])
        assert len(samples) == 18

        seen_combinations = set()
        for s in samples:
            d = s["digit"]
            split = s["split"]
            assert d in [str(i) for i in range(1, 10)], f"Số {d} không nằm trong dải 1-9"
            assert split in ["dev", "eval"], f"Split {split} không hợp lệ"
            
            combo = (d, split)
            assert combo not in seen_combinations, f"Trùng lặp mẫu: {combo}"
            seen_combinations.add(combo)

            assert s["quality_status"] == "ok"
            assert s["handedness"] == "Right"
            assert s["confidence"] >= 0.90, f"Độ tin cậy quá thấp: {s['confidence']}"
            assert s["coordinate_type"] == "normalized_mediapipe_hands"

            # Kiểm tra landmark 21 điểm
            lms = s["landmarks"]
            is_valid, msg = validate_landmark_format(lms)
            assert is_valid is True, f"Mẫu số {d} ({split}) không hợp lệ: {msg}"


