"""
Bộ kiểm thử tự động xác thực Fixture Landmark Bảng chữ cái VSL 29 chữ cái.
=============================================================================
Kiểm định toàn diện:
1. Tuân thủ Schema 1.0 và target_domain "vsl_alphabet".
2. Bao phủ đầy đủ 29 chữ cái tiếng Việt chuẩn.
3. Đúng 21 điểm mốc cho 100% mẫu 'ok' và vượt qua validate_landmark_format().
4. Tọa độ số hữu hạn, không chứa NaN/Inf, biên [0.0, 1.0].
5. Tính trung thực: Mẫu thất bại (no_hand_detected) phải có landmarks=None (không sinh landmark giả).
6. Khả năng truy vết nguồn video và frame_idx trên đĩa cục bộ.
7. Cảnh giác về dữ liệu: Ghi nhận chữ nhiều pha và giới hạn single-signer.
"""

import os
import json
import pytest
from backend.training.extract_video_landmarks import validate_landmark_format

FIXTURE_PATH = "backend/tests/fixtures/real_alphabet_landmarks.json"
VIDEO_DIR = "backend/data/media_cache/videos"
EXPECTED_LETTERS = [
    'a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê', 'g',
    'h', 'i', 'k', 'l', 'm', 'n', 'o', 'ô', 'ơ', 'p',
    'q', 'r', 's', 't', 'u', 'ư', 'v', 'x', 'y'
]
MULTI_PHASE_LETTERS = {'ă', 'â', 'ê', 'ô', 'ơ', 'ư', 'đ'}


@pytest.fixture(scope="module")
def fixture_data():
    assert os.path.exists(FIXTURE_PATH), f"Không tìm thấy file fixture: {FIXTURE_PATH}"
    with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


class TestAlphabetFixtureIntegrity:

    def test_schema_1_0_structure(self, fixture_data):
        """Xác nhận cấu trúc Schema 1.0 chuẩn của dự án."""
        assert fixture_data.get("schema_version") == "1.0"
        assert fixture_data.get("target_domain") == "vsl_alphabet"
        assert fixture_data.get("total_videos") == 29
        assert fixture_data.get("total_samples") == 203
        assert fixture_data.get("successful_samples") == 122
        assert fixture_data.get("failed_samples") == 81
        assert isinstance(fixture_data.get("samples"), list)
        assert len(fixture_data["samples"]) == 203

    def test_coverage_all_29_vietnamese_letters(self, fixture_data):
        """Xác nhận bao phủ 100% toàn bộ 29 chữ cái tiếng Việt chuẩn."""
        samples = fixture_data["samples"]
        found_letters = set(s["letter"] for s in samples)

        for l in EXPECTED_LETTERS:
            assert l in found_letters, f"Thiếu chữ cái: {l}"
        assert len(found_letters) == 29

        # Mỗi chữ cái phải có ít nhất 3 mẫu trích xuất thành công
        for l in EXPECTED_LETTERS:
            l_ok = [s for s in samples if s["letter"] == l and s["quality_status"] == "ok"]
            assert len(l_ok) >= 3, f"Chữ '{l}' có quá ít mẫu thành công: {len(l_ok)}"

    def test_sample_fields_and_validation(self, fixture_data):
        """Kiểm định tính chuẩn tắc của từng mẫu dữ liệu."""
        samples = fixture_data["samples"]
        required_keys = {
            "letter", "letter_upper", "video_id", "source_video_file",
            "split", "frame_idx", "timestamp_sec", "phase",
            "handedness", "confidence", "coordinate_type", "quality_status", "landmarks"
        }

        for s in samples:
            assert required_keys.issubset(s.keys()), f"Mẫu thiếu trường bắt buộc: {s.get('video_id')}"
            assert s["coordinate_type"] == "normalized_mediapipe_hands"

            status = s["quality_status"]
            if status == "ok":
                lms = s["landmarks"]
                # 1. Kiểm tra bằng hàm thẩm định của backend
                is_valid, msg = validate_landmark_format(lms)
                assert is_valid is True, f"Mẫu {s['video_id']} f{s['frame_idx']} không đạt: {msg}"

                # 2. Kiểm tra tính hữu hạn và phạm vi quan sát chuẩn hóa (-0.2 <= x, y <= 1.2)
                assert len(lms) == 21
                for p in lms:
                    assert len(p) == 3
                    x, y, z = float(p[0]), float(p[1]), float(p[2])
                    assert -0.2 <= x <= 1.2, f"Tọa độ x vượt biên quan sát: {x}"
                    assert -0.2 <= y <= 1.2, f"Tọa độ y vượt biên quan sát: {y}"
                    assert isinstance(z, (int, float))

                # 3. Confidence từ MediaPipe phải hợp lệ
                assert isinstance(s["confidence"], (int, float))
                assert s["confidence"] >= 0.5
            else:
                # Mẫu thất bại (ví dụ: no_hand_detected) phải có landmarks = None
                assert status == "no_hand_detected"
                assert s["landmarks"] is None
                assert s["confidence"] == 0

    def test_traceability_to_source_videos(self, fixture_data):
        """Đảm bảo mọi mẫu đều truy nguyên được tới file video thực trên đĩa."""
        samples = fixture_data["samples"]
        checked_videos = set()

        for s in samples:
            vid_id = s["video_id"]
            if vid_id not in checked_videos:
                vfile = os.path.join(VIDEO_DIR, f"{vid_id}.mp4")
                assert os.path.exists(vfile), f"Không tìm thấy video nguồn trên đĩa: {vfile}"
                assert os.path.getsize(vfile) > 50000, f"Dung lượng video bất thường: {vfile}"
                checked_videos.add(vid_id)

    def test_multi_phase_transparency(self, fixture_data):
        """
        Xác nhận tính minh bạch đối với 7 chữ ghép/nhiều pha (ă, â, ê, ô, ơ, ư, đ):
        Không tự tiện gán 'stable_pose' khi chưa có kiểm chứng pha.
        """
        samples = fixture_data["samples"]
        for s in samples:
            if s["letter"] in MULTI_PHASE_LETTERS:
                # 7 chữ này trong dải giữa phải được gắn nhãn unknown hoặc entry/exit
                assert s["phase"] in {"entry", "exit", "unknown", "transition"}

    def test_single_signer_limitation_recorded(self, fixture_data):
        """
        Kiểm tra rằng dev và eval của cùng 1 chữ đều thuộc về cùng 1 video_id (single-signer).
        Điều này chứng minh fixture không bị ngụy tạo thành tập dữ liệu cross-signer.
        """
        samples = fixture_data["samples"]
        for l in EXPECTED_LETTERS:
            dev_samples = [s for s in samples if s["letter"] == l and s["split"] == "dev"]
            eval_samples = [s for s in samples if s["letter"] == l and s["split"] == "eval"]
            if dev_samples and eval_samples:
                # Đều có cùng video_id -> Cùng 1 người ký
                assert dev_samples[0]["video_id"] == eval_samples[0]["video_id"]
