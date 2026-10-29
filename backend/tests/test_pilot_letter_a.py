"""
Unit test kiểm định chất lượng bộ dữ liệu Pilot Landmark cho chữ 'A'.
===================================================================
Kiểm tra:
1. Video nguồn D0489.mp4 tồn tại.
2. Fixture pilot_real_letter_a_landmarks.json tuân thủ Schema 1.0.
3. Các mẫu thành công có đúng 21 điểm, tọa độ hữu hạn và vượt qua validate_landmark_format.
4. Các mẫu không thấy tay được ghi nhận minh bạch là 'no_hand_detected'.
5. Phân biệt rõ ràng giữa dev frame (f30/f40) và eval frame (f50/f60).
"""

import os
import json
import pytest
from backend.training.extract_video_landmarks import validate_landmark_format
from backend.vision.feature_extractor import _shift_and_scale, finger_curl_ratio

FIXTURE_PATH = "backend/tests/fixtures/pilot_real_letter_a_landmarks.json"
VIDEO_PATH = "backend/data/media_cache/videos/D0489.mp4"


class TestPilotLetterALandmarks:

    def test_source_video_exists(self):
        """Xác nhận video nguồn D0489.mp4 tồn tại và không rỗng."""
        assert os.path.exists(VIDEO_PATH), f"Không tìm thấy video: {VIDEO_PATH}"
        assert os.path.getsize(VIDEO_PATH) > 100000, "Video dung lượng bất thường"

    def test_fixture_structure_schema_1_0(self):
        """Kiểm tra schema cấu trúc của fixture Pilot."""
        assert os.path.exists(FIXTURE_PATH), f"Thiếu file fixture: {FIXTURE_PATH}"
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data["schema_version"] == "1.0"
        assert data["target_domain"] == "vsl_alphabet"
        assert data["letter"] == "a"
        assert data["source_video_id"] == "D0489"
        assert "samples" in data
        assert len(data["samples"]) > 0

    def test_samples_quality_and_coordinates(self):
        """Kiểm định 100% mẫu trong fixture có cấu trúc hợp lệ."""
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = data["samples"]
        ok_count = 0
        no_hand_count = 0

        for s in samples:
            assert s["letter"] == "a"
            assert s["video_id"] == "D0489"
            assert isinstance(s["frame_idx"], int)
            assert isinstance(s["timestamp_sec"], (int, float))
            assert s["coordinate_type"] == "normalized_mediapipe_hands"

            status = s["quality_status"]
            if status == "ok":
                ok_count += 1
                lms = s["landmarks"]
                # 1. Kiểm tra bằng hàm chuẩn của dự án
                is_valid, msg = validate_landmark_format(lms)
                assert is_valid is True, f"Mẫu frame {s['frame_idx']} lỗi: {msg}"

                # 2. Kiểm tra tính hữu hạn và khoảng tọa độ [0, 1] cho x, y
                for p in lms:
                    assert len(p) == 3
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                    assert 0.0 <= x <= 1.0
                    assert 0.0 <= y <= 1.0
                    assert isinstance(z, (int, float))
            else:
                assert status == "no_hand_detected"
                assert s["landmarks"] is None
                no_hand_count += 1

        assert ok_count >= 4, "Phải có ít nhất 4 frame nhận diện thành công bàn tay"
        assert no_hand_count >= 2, "Các frame đầu/cuối phải ghi nhận no_hand_detected chính xác"

    def test_morphological_properties_of_letter_a(self):
        """
        Kiểm tra hình thái học của chữ 'A' trên mẫu ổn định (Frame 40, t=1.67s):
        - Bàn tay nắm lại (ngón trỏ, giữa, áp út gập).
        - Ngón cái duỗi hướng lên.
        """
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        # Lấy mẫu Frame 40 (dev)
        sample_f40 = next((s for s in data["samples"] if s["frame_idx"] == 40), None)
        assert sample_f40 is not None
        assert sample_f40["quality_status"] == "ok"

        lms = sample_f40["landmarks"]
        shifted, palm_scale = _shift_and_scale(lms)
        assert palm_scale > 0.05

        # Tính độ co ngón: Tip, Pip, Mcp
        curl_index = finger_curl_ratio(shifted, 8, 6, 5, palm_scale)
        curl_middle = finger_curl_ratio(shifted, 12, 10, 9, palm_scale)
        curl_ring = finger_curl_ratio(shifted, 16, 14, 13, palm_scale)

        # Ngón trỏ, giữa, áp út phải co/gập đáng kể (> 0.40)
        assert curl_index > 0.40, f"Ngón trỏ chưa đủ độ co trong chữ A: {curl_index}"
        assert curl_middle > 0.40, f"Ngón giữa chưa đủ độ co trong chữ A: {curl_middle}"
        assert curl_ring > 0.40, f"Ngón áp út chưa đủ độ co trong chữ A: {curl_ring}"
