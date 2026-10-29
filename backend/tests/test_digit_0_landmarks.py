"""
Bộ kiểm thử tính toàn vẹn, chất lượng dữ liệu và Nhận diện Chữ số 0 (VSL Digit 0).
==================================================================================
Kiểm tra:
1. File video nguồn D0529.mp4 tồn tại và khớp mã băm SHA-256.
2. Cấu trúc và định dạng tệp fixture real_digit_0_landmarks.json (21 điểm, tọa độ hữu hạn).
3. Độc lập mẫu dev (frame 30) và eval (frame 35).
4. Tính chất hình thái học của cử chỉ số 0 (khum tròn, ngón cái tiếp cận các đầu ngón).
5. Phân loại độc lập: Cả 4 mẫu số 0 trong fixture được nhận diện chính xác thành '0' (conf >= 0.90).
6. Confusion check: Không mẫu nào trong 18 mẫu số 1–9 bị nhận nhầm thành '0'.
7. An toàn đầu vào: None, rỗng, sai số điểm, NaN, Inf đều trả về NEUTRAL (conf = 0.0).
8. Tích hợp Recognizer: Chế độ 'digits' nhận diện '0' chuẩn xác, không fallback chữ cái.
9. Bảo vệ chế độ 'letters': Tuyệt đối không trả về số '0'.
10. Tương thích ngược: Chế độ 'words', 'sentence' và alias 'letters_numbers' hoạt động đúng hợp đồng.
"""

import os
import copy
import json
import hashlib
import pytest
from backend.training.extract_video_landmarks import validate_landmark_format
from backend.vision.digit_classifier import digit_classifier, VSLDigitClassifier
from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS, ALLOWED_DIGITS
from backend.vision.feature_extractor import calculate_distance


FIXTURE_PATH = "backend/tests/fixtures/real_digit_0_landmarks.json"
DIGITS_1_9_FIXTURE_PATH = "backend/tests/fixtures/real_digit_landmarks.json"
VIDEO_PATH = "backend/data/media_cache/videos/D0529.mp4"
EXPECTED_SHA256 = "29bb754be325199584cca2edcee23d5fcda3b54f670840b5948fcf8c176f8c3b"


class TestDigit0DataIntegrity:
    """Kiểm tra nguồn video và fixture landmark số 0."""

    def test_digit_0_video_exists_and_checksum(self):
        """Xác nhận video D0529.mp4 tồn tại cục bộ và đúng toàn vẹn mã băm SHA-256."""
        assert os.path.exists(VIDEO_PATH), f"Không tìm thấy video: {VIDEO_PATH}"
        
        with open(VIDEO_PATH, "rb") as f:
            data = f.read()
        
        calc_sha256 = hashlib.sha256(data).hexdigest()
        assert calc_sha256 == EXPECTED_SHA256, f"Sai lệch mã băm: {calc_sha256} != {EXPECTED_SHA256}"
        assert len(data) == 376467, f"Sai lệch kích thước video: {len(data)} != 376467"

    def test_digit_0_fixture_structure(self):
        """Kiểm tra cấu trúc và tính hợp lệ của fixture real_digit_0_landmarks.json."""
        assert os.path.exists(FIXTURE_PATH), f"Không tìm thấy fixture: {FIXTURE_PATH}"

        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert data.get("schema_version") == "1.0"
        assert data.get("digit") == "0"
        assert data.get("source_video_id") == "D0529"
        assert data.get("total_samples") >= 2

        samples = data.get("samples", [])
        assert len(samples) >= 2

        splits_found = set()
        for s in samples:
            assert s["digit"] == "0"
            assert s["video_id"] == "D0529"
            assert s["quality_status"] == "ok"
            assert s["coordinate_type"] == "normalized_mediapipe_hands"
            assert s["confidence"] >= 0.90, f"Confidence quá thấp: {s['confidence']}"
            assert isinstance(s["frame_idx"], int)
            assert isinstance(s["timestamp_sec"], (int, float))

            splits_found.add(s["split"])

            # Kiểm tra 21 điểm landmark
            lms = s["landmarks"]
            is_valid, err_msg = validate_landmark_format(lms)
            assert is_valid is True, f"Landmark không hợp lệ tại frame {s['frame_idx']}: {err_msg}"

        assert "dev" in splits_found, "Thiếu mẫu split 'dev' trong fixture"
        assert "eval" in splits_found, "Thiếu mẫu split 'eval' trong fixture"

    def test_digit_0_dev_eval_distinction(self):
        """Mẫu dev và eval phải thuộc 2 frame riêng biệt để bảo đảm tính khách quan."""
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = {s["split"]: s for s in data["samples"]}
        dev_sample = samples["dev"]
        eval_sample = samples["eval"]

        assert dev_sample["frame_idx"] != eval_sample["frame_idx"]
        assert dev_sample["timestamp_sec"] != eval_sample["timestamp_sec"]
        assert dev_sample["frame_idx"] == 30
        assert eval_sample["frame_idx"] == 35

    def test_digit_0_morphological_properties(self):
        """Kiểm tra đặc tính hình thái khum tròn của cử chỉ số 0 từ video thực tế."""
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        for s in data["samples"]:
            lm = s["landmarks"]
            feat = digit_classifier.extract_features(lm)
            assert feat is not None, "Không thể trích xuất feature từ landmark số 0"

            scale = feat["palm_scale"]
            assert scale > 0.05, f"Palm scale quá nhỏ hoặc suy biến: {scale}"

            # Cử chỉ số 0: các ngón không duỗi thẳng xòe rộng
            assert feat["index_ext"] is False
            assert feat["middle_ext"] is False
            assert feat["ring_ext"] is False
            assert feat["pinky_ext"] is False

            # Khoảng cách giữa đầu ngón trỏ (mốc 8) và đầu ngón cái (mốc 4) phải nhỏ
            d_thumb_index = calculate_distance(lm[4], lm[8]) / scale
            assert d_thumb_index < 0.20, f"Khoảng cách ngón cái - ngón trỏ quá lớn cho số 0: {d_thumb_index}"


class TestDigit0Classification:
    """Kiểm tra khả năng phân loại và ranh giới an toàn của Số 0."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.clf = VSLDigitClassifier()
        self.rec = VSLRecognizer()
        with open(FIXTURE_PATH, "r", encoding="utf-8") as f:
            self.d0_samples = json.load(f)["samples"]
        with open(DIGITS_1_9_FIXTURE_PATH, "r", encoding="utf-8") as f:
            self.d19_samples = json.load(f)["samples"]

    def test_all_d0_fixture_samples_classified_as_zero(self):
        """Tất cả 4 mẫu số 0 trong fixture phải được phân loại chuẩn xác thành '0' (conf >= 0.90)."""
        for s in self.d0_samples:
            res = self.clf.classify_detailed(s["landmarks"])
            assert res["status"] == "MATCH", f"Mẫu số 0 ({s['split']}, f{s['frame_idx']}) không MATCH: {res['reason']}"
            assert res["digit"] == "0", f"Mẫu số 0 ({s['split']}) bị nhận nhầm thành {res['digit']}"
            assert res["confidence"] >= 0.90
            assert res["matched_digits"] == ["0"]

    def test_digits_1_to_9_never_confused_with_zero(self):
        """Tất cả 18 mẫu số 1–9 trong fixture thực tế tuyệt đối không bị nhận nhầm thành số '0'."""
        for s in self.d19_samples:
            res = self.clf.classify_detailed(s["landmarks"])
            assert res["digit"] != "0", f"Số {s['digit']} ({s['split']}) bị nhận nhầm thành số 0!"
            assert "0" not in res["matched_digits"]

    def test_invalid_input_safety(self):
        """Đầu vào không hợp lệ (None, rỗng, sai số điểm, NaN, Inf) phải trả về NEUTRAL (conf = 0.0)."""
        # None
        r1 = self.clf.classify_detailed(None)
        assert r1["status"] == "INVALID_INPUT"
        assert r1["digit"] == "NEUTRAL"
        assert r1["confidence"] == 0.0

        # Rỗng
        r2 = self.clf.classify_detailed([])
        assert r2["status"] == "INVALID_INPUT"
        assert r2["digit"] == "NEUTRAL"

        # Sai số lượng điểm
        r3 = self.clf.classify_detailed([[0.5, 0.5, 0.0]] * 15)
        assert r3["status"] == "INVALID_INPUT"
        assert r3["digit"] == "NEUTRAL"

        # NaN
        bad_lm = [[float("nan"), 0.5, 0.0]] * 21
        r4 = self.clf.classify_detailed(bad_lm)
        assert r4["status"] == "INVALID_INPUT"
        assert r4["digit"] == "NEUTRAL"

        # Inf
        inf_lm = [[float("inf"), 0.5, 0.0]] * 21
        r5 = self.clf.classify_detailed(inf_lm)
        assert r5["status"] == "INVALID_INPUT"
        assert r5["digit"] == "NEUTRAL"

    def test_recognizer_digits_mode_recognizes_zero(self):
        """Trong mode='digits', Recognizer nhận diện số 0 và gán label 'Số 0', type 'digit'."""
        sample_dev = next(s for s in self.d0_samples if s["split"] == "dev")
        res = self.rec.predict_landmarks(sample_dev["landmarks"], mode="digits")
        assert res["sign"] == "0"
        assert res["label"] == "Số 0"
        assert res["type"] == "digit"
        assert res["confidence"] >= 0.90

    def test_recognizer_letters_mode_never_returns_digit_zero(self):
        """Trong mode='letters', Recognizer tuyệt đối không trả về số '0'."""
        sample_dev = next(s for s in self.d0_samples if s["split"] == "dev")
        res = self.rec.predict_landmarks(sample_dev["landmarks"], mode="letters")
        assert res["sign"] != "0", "Chế độ letters bị rò rỉ số 0!"
        # Trong chế độ letters, cử chỉ khum tròn O trả về chữ cái O hoặc NEUTRAL
        assert res["sign"] in ALLOWED_LETTERS | {"NEUTRAL"}

    def test_recognizer_words_and_sentence_unaffected(self):
        """Chế độ 'words' và 'sentence' không bị ảnh hưởng bởi việc thêm số 0."""
        sample_dev = next(s for s in self.d0_samples if s["split"] == "dev")
        res_words = self.rec.predict_landmarks(sample_dev["landmarks"], mode="words")
        assert res_words["sign"] not in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

        res_sent = self.rec.predict_landmarks(sample_dev["landmarks"], mode="sentence")
        assert res_sent["sign"] not in {"0", "1", "2", "3", "4", "5", "6", "7", "8", "9"}

    def test_recognizer_legacy_letters_numbers_mode_preserves_policy(self):
        """Chế độ cũ letters_numbers duy trì chính sách cũ không rò rỉ số 0."""
        sample_dev = next(s for s in self.d0_samples if s["split"] == "dev")
        res = self.rec.predict_landmarks(sample_dev["landmarks"], mode="letters_numbers")
        assert res["sign"] != "0", "Chế độ letters_numbers không được rò rỉ số 0"
    def test_digit_0_boundary_d_tm_relaxed(self):
        """Cử chỉ số 0 tự nhiên với ngón giữa hơi hở (d_TM = 0.25 <= 0.30) được nhận diện thành công."""
        sample = copy.deepcopy(self.d0_samples[0]["landmarks"])
        # Dịch nhẹ ngón giữa (mốc 12) ra xa ngón cái để d_TM đạt ~0.25
        sample[12][0] += 0.015
        res = self.clf.classify_detailed(sample)
        assert res["status"] == "MATCH"
        assert res["digit"] == "0"
        assert res["confidence"] >= 0.90
        assert res["features"]["d_TM"] <= 0.30

    def test_digit_0_boundary_d_tm_exceeded(self):
        """Khi d_TM vượt quá ngưỡng 0.30 (ngón giữa xòe quá xa), số 0 bị loại bỏ an toàn thành NEUTRAL."""
        sample = copy.deepcopy(self.d0_samples[0]["landmarks"])
        sample[12][0] += 0.045
        res = self.clf.classify_detailed(sample)
        assert res["status"] == "NO_MATCH"
        assert res["digit"] == "NEUTRAL"
        assert res["features"]["d_TM"] > 0.30

    def test_digit_0_boundary_d_ti_exceeded(self):
        """Khi ngón cái và ngón trỏ không chạm nhau (d_TI > 0.20), số 0 bị loại bỏ an toàn thành NEUTRAL."""
        sample = copy.deepcopy(self.d0_samples[0]["landmarks"])
        # Dịch ngón trỏ (mốc 8) ra xa ngón cái
        sample[8][0] += 0.03
        res = self.clf.classify_detailed(sample)
        assert res["status"] == "NO_MATCH"
        assert res["digit"] == "NEUTRAL"
        assert res["features"]["d_TI"] > 0.20

    def test_digit_0_boundary_d_iw_fist(self):
        """Cử chỉ nắm đấm co sát cổ tay (d_IW < 0.95) bị loại bỏ an toàn thành NEUTRAL."""
        sample = copy.deepcopy(self.d0_samples[0]["landmarks"])
        # Kéo ngón trỏ sát về cổ tay
        sample[8][1] = sample[0][1] - 0.05
        res = self.clf.classify_detailed(sample)
        assert res["status"] == "NO_MATCH"
        assert res["digit"] == "NEUTRAL"

    def test_digit_0_boundary_finger_extended(self):
        """Khi có bất kỳ ngón nào duỗi thẳng (ví dụ ngón út), số 0 bị loại bỏ an toàn thành NEUTRAL."""
        sample = copy.deepcopy(self.d0_samples[0]["landmarks"])
        # Duỗi ngón út (mốc 20 lên trên mốc 18)
        sample[20][1] = sample[18][1] - 0.05
        res = self.clf.classify_detailed(sample)
        assert res["status"] == "NO_MATCH"
        assert res["digit"] == "NEUTRAL"

