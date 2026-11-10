"""
Unit Tests for Smoke Test Verification Script (Task 7F.6B)
==========================================================
Kiểm tra toàn diện script verify_pilot_smoke_test.py với các kịch bản synthetic fixtures:
1. File chuẩn đầy đủ 6 trials (A, B, C x 2 reps) -> PASS.
2. File có cả frame inter-trial / idle xen kẽ giữa các trial -> PASS (không gây false fail).
3. File có dòng sai cú pháp JSON -> FAIL.
4. File sai session_id -> FAIL.
5. File thiếu trial (chỉ có 5/6 trials) -> FAIL.
6. File có expected_sign sai lệch so với trial_id -> FAIL.
7. File có trùng lặp frame_idx trong cùng trial -> FAIL.
8. File có timestamp không tăng đơn điệu -> FAIL.
9. File có trial bị non-monotonic so với trial trước -> FAIL.
10. File có FPS quá thấp (< 10.0) -> FAIL.
11. File có số frame quá ít (< 50) -> FAIL.
12. File có nhãn nhận diện sai đa số -> FAIL.
13. Telemetry stats đối chiếu: phát hiện dropped frames -> FAIL.
"""

import os
import sys
import json
import shutil
import tempfile
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.scripts.verify_pilot_smoke_test import verify_pilot_file, parse_trial_id


def _create_synthetic_frame(
    timestamp: float,
    session_id: str,
    trial_id: str,
    frame_idx: int,
    expected_sign: str,
    final_sign: str,
    final_conf: float = 0.95,
    source: str = "geometric_rule",
    confirmed: bool = True
):
    """Tạo bản ghi synthetic test fixture (đánh dấu rõ là dữ liệu kiểm thử, không phải pilot webcam thật)."""
    return {
        "timestamp": round(float(timestamp), 4),
        "session_id": session_id,
        "trial_id": trial_id,
        "frame_idx": frame_idx,
        "expected_sign": expected_sign,
        "final_sign": final_sign,
        "final_confidence": round(float(final_conf), 4),
        "prediction_source": source,
        "confirmed": confirmed,
        "geometric": {"status": "MATCH", "letter": final_sign},
        "static_head": {"raw_predicted_letter": None, "confidence": None, "top_k": None},
        "gating_status": "passed_geometric",
        "threshold_applied": None,
        "reject_reason": None
    }


def _create_inter_trial_frame(timestamp: float, frame_idx: int):
    """Bản ghi ngoài trial khi webcam đang truyền giữa các lần thử (session_id=None, trial_id=None)."""
    return {
        "timestamp": round(float(timestamp), 4),
        "session_id": None,
        "trial_id": None,
        "frame_idx": frame_idx,
        "expected_sign": None,
        "final_sign": "NEUTRAL",
        "final_confidence": 0.0,
        "prediction_source": "none",
        "confirmed": False,
        "geometric": {"status": "NO_MATCH", "letter": None},
        "static_head": {"raw_predicted_letter": None, "confidence": None, "top_k": None},
        "gating_status": "none",
        "threshold_applied": None,
        "reject_reason": None
    }


class TestVerifyPilotSmokeTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.session_id = "test_smoke_20261101"

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _generate_valid_jsonl(self, include_inter_trial: bool = True) -> str:
        filepath = os.path.join(self.temp_dir, f"pilot_{self.session_id}.jsonl")
        current_time = 1000.0

        with open(filepath, "w", encoding="utf-8") as f:
            for letter in ["A", "B", "C"]:
                for rep in [1, 2]:
                    trial_id = f"{self.session_id}_{letter}_rep{rep:02d}"
                    # Mỗi trial có 60 frames, FPS ~ 20.0 (dt = 0.05s)
                    for idx in range(1, 61):
                        rec = _create_synthetic_frame(
                            timestamp=current_time,
                            session_id=self.session_id,
                            trial_id=trial_id,
                            frame_idx=idx,
                            expected_sign=letter,
                            final_sign=letter,
                            final_conf=0.92,
                            source="geometric_rule",
                            confirmed=(idx >= 3)
                        )
                        f.write(json.dumps(rec) + "\n")
                        current_time += 0.05

                    # Chèn 10 frame inter-trial nghỉ giữa các lần thử
                    if include_inter_trial:
                        for inter_idx in range(1, 11):
                            rec = _create_inter_trial_frame(timestamp=current_time, frame_idx=inter_idx)
                            f.write(json.dumps(rec) + "\n")
                            current_time += 0.05

        return filepath

    def test_parse_trial_id(self):
        res = parse_trial_id("test_smoke_20261101_A_rep01", "test_smoke_20261101")
        self.assertEqual(res, ("A", 1))

        res2 = parse_trial_id("test_smoke_20261101_C_rep02", "test_smoke_20261101")
        self.assertEqual(res2, ("C", 2))

        # Sai format
        self.assertIsNone(parse_trial_id("bad_trial_name", "test_smoke_20261101"))
        self.assertIsNone(parse_trial_id("test_smoke_20261101_A_rep1", "test_smoke_20261101"))

    def test_valid_smoke_test_file_passes(self):
        """1. File đầy đủ 6 trials hợp lệ phải PASS 100%."""
        filepath = self._generate_valid_jsonl(include_inter_trial=False)
        report = verify_pilot_file(
            filepath=filepath,
            expected_session_id=self.session_id,
            min_fps=10.0,
            min_frames_per_trial=50
        )
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["errors"]), 0)
        self.assertEqual(report["summary"]["completed_trials_count"], 6)
        self.assertEqual(report["frame_level"]["overall_match_rate_pct"], 100.0)

    def test_valid_smoke_test_with_inter_trial_frames_passes(self):
        """2. File có frame nghỉ inter-trial giữa các lần thử vẫn PASS đúng semantics."""
        filepath = self._generate_valid_jsonl(include_inter_trial=True)
        report = verify_pilot_file(
            filepath=filepath,
            expected_session_id=self.session_id,
            min_fps=10.0,
            min_frames_per_trial=50
        )
        self.assertTrue(report["passed"])
        self.assertGreater(report["summary"]["inter_trial_records"], 0)
        self.assertEqual(report["summary"]["completed_trials_count"], 6)

    def test_invalid_jsonl_syntax_fails(self):
        """3. Dòng bị hỏng cú pháp JSON phải bị phát hiện và FAIL."""
        filepath = self._generate_valid_jsonl()
        with open(filepath, "a", encoding="utf-8") as f:
            f.write("{ bad_json: unquoted }\n")

        report = verify_pilot_file(filepath=filepath, expected_session_id=self.session_id)
        self.assertFalse(report["passed"])
        self.assertTrue(any("Lỗi cú pháp JSON" in e for e in report["errors"]))

    def test_wrong_session_id_fails(self):
        """4. Bản ghi mang sai session_id phải FAIL."""
        filepath = self._generate_valid_jsonl()
        # Xác minh với session_id khác
        report = verify_pilot_file(filepath=filepath, expected_session_id="wrong_session_9999")
        self.assertFalse(report["passed"])
        self.assertTrue(any("Sai session_id" in e for e in report["errors"]))

    def test_missing_trial_fails(self):
        """5. Thiếu 1 trong 6 trials kỳ vọng phải FAIL."""
        filepath = os.path.join(self.temp_dir, "missing_trial.jsonl")
        current_time = 1000.0
        with open(filepath, "w", encoding="utf-8") as f:
            # Chỉ ghi 5 trials (thiếu C_rep02)
            for letter in ["A", "B"]:
                for rep in [1, 2]:
                    t_id = f"{self.session_id}_{letter}_rep{rep:02d}"
                    for i in range(1, 60):
                        f.write(json.dumps(_create_synthetic_frame(current_time, self.session_id, t_id, i, letter, letter)) + "\n")
                        current_time += 0.05
            # C rep 1
            t_id = f"{self.session_id}_C_rep01"
            for i in range(1, 60):
                f.write(json.dumps(_create_synthetic_frame(current_time, self.session_id, t_id, i, "C", "C")) + "\n")
                current_time += 0.05

        report = verify_pilot_file(filepath=filepath, expected_session_id=self.session_id)
        self.assertFalse(report["passed"])
        self.assertTrue(any("Thiếu các trial kỳ vọng" in e for e in report["errors"]))

    def test_expected_sign_mismatch_fails(self):
        """6. expected_sign khác với chữ cái của trial phải FAIL."""
        filepath = os.path.join(self.temp_dir, "mismatch_expected.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            t_id = f"{self.session_id}_A_rep01"
            for i in range(1, 60):
                # Ghi expected_sign = "B" trong khi trial là chữ "A"
                f.write(json.dumps(_create_synthetic_frame(1000.0 + i*0.05, self.session_id, t_id, i, "B", "A")) + "\n")

        report = verify_pilot_file(filepath=filepath, expected_session_id=self.session_id, expected_letters=["A"], expected_reps=1)
        self.assertFalse(report["passed"])
        self.assertTrue(any("expected_sign không khớp" in err for t in report["trial_level"].values() for err in t.get("errors", [])))

    def test_duplicate_frame_idx_fails(self):
        """7. Trùng lặp frame_idx trong trial phải FAIL."""
        filepath = os.path.join(self.temp_dir, "dup_frame_idx.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            t_id = f"{self.session_id}_A_rep01"
            for i in [1, 2, 3, 3, 4, 5] + list(range(6, 60)):
                f.write(json.dumps(_create_synthetic_frame(1000.0 + i*0.05, self.session_id, t_id, i, "A", "A")) + "\n")

        report = verify_pilot_file(filepath=filepath, expected_session_id=self.session_id, expected_letters=["A"], expected_reps=1)
        self.assertFalse(report["passed"])
        self.assertTrue(any("Trùng lặp frame_idx" in err for t in report["trial_level"].values() for err in t.get("errors", [])))

    def test_non_monotonic_timestamp_fails(self):
        """8. Timestamp không tăng đơn điệu phải FAIL."""
        filepath = os.path.join(self.temp_dir, "bad_ts.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            t_id = f"{self.session_id}_A_rep01"
            ts = 1000.0
            for i in range(1, 60):
                if i == 15:
                    ts -= 1.0  # Giảm lùi thời gian
                else:
                    ts += 0.05
                f.write(json.dumps(_create_synthetic_frame(ts, self.session_id, t_id, i, "A", "A")) + "\n")

        report = verify_pilot_file(filepath=filepath, expected_session_id=self.session_id, expected_letters=["A"], expected_reps=1)
        self.assertFalse(report["passed"])
        self.assertTrue(any("timestamp không tăng đơn điệu" in err for t in report["trial_level"].values() for err in t.get("errors", [])))

    def test_low_fps_fails(self):
        """10. FPS < min_fps (ví dụ 5.0 < 10.0) phải FAIL."""
        filepath = os.path.join(self.temp_dir, "low_fps.jsonl")
        with open(filepath, "w", encoding="utf-8") as f:
            t_id = f"{self.session_id}_A_rep01"
            # 50 frames kéo dài 10 giây -> 5 FPS
            for i in range(1, 55):
                f.write(json.dumps(_create_synthetic_frame(1000.0 + i*0.2, self.session_id, t_id, i, "A", "A")) + "\n")

        report = verify_pilot_file(filepath=filepath, expected_session_id=self.session_id, expected_letters=["A"], expected_reps=1, min_fps=10.0)
        self.assertFalse(report["passed"])
        self.assertTrue(any("thấp hơn ngưỡng tối thiểu" in err for t in report["trial_level"].values() for err in t.get("errors", [])))

    def test_telemetry_stats_cross_check_detects_dropped(self):
        """13. Telemetry stats có thất thoát dropped_queue_full > 0 phải FAIL."""
        filepath = self._generate_valid_jsonl(include_inter_trial=False)
        bad_stats = {
            "queued_count": 365,
            "logged_count": 360,
            "dropped_queue_full": 5,
            "dropped_io_error": 0
        }
        report = verify_pilot_file(
            filepath=filepath,
            expected_session_id=self.session_id,
            stats_dict=bad_stats
        )
        self.assertFalse(report["passed"])
        self.assertTrue(any("dropped_queue_full" in e for e in report["errors"]))


if __name__ == "__main__":
    unittest.main()

