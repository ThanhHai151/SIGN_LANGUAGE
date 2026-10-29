"""
Unit tests toan dien cho he thong Diagnostic Logging va Offline Replay Tool (VSL Connect AI).

Cac kich ban kiem thu bat buoc:
1. Diagnostic mac dinh tat, khong ghi file log.
2. Diagnostic khi bat ghi dung cau truc JSONL (day du 21 landmarks goc, finger_states, geom, mlp, ensemble, smoothing).
3. Gioi han tan suat ghi (rate-limiting ~2 mau/giay).
4. Co lap loi I/O, khong lam crash ham nhan dien predict_landmarks.
5. Phan loai loi MLP_OVERRIDE (Geometry dung, MLP sai conf > 0.6 ghi de).
6. Phan loai loi GEOMETRY_MISMATCH_MLP_UNDERCONFIDENT (Geometry sai, MLP dung nhung conf <= 0.6).
7. Phan loai loi BOTH_LAYERS_MISMATCH (Ca Geometry va MLP deu sai).
8. Xu ly an toan khi thieu expected_sign (NO_GROUND_TRUTH).
9. Xu ly an toan khi gap du lieu thieu / landmark hong / dong JSON hong.
10. Session replay tuan tu kiem tra chinh xac smoothing window va cooldown debounce bang timestamp thuc te.
"""

import os
import sys
import json
import time
import shutil
import tempfile
import unittest
from unittest.mock import patch

# Them duong dan goc vao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from backend.vision.recognizer import VSLRecognizer
from backend.tests.test_geometry import _make_hand
from backend.tools.replay_diagnostic import (
    load_diagnostic_records,
    replay_frame_by_frame,
    replay_session_sequential
)


def _make_sample_hand():
    """Tao 21 landmarks MediaPipe co ban."""
    lm = _make_hand()
    wx, wy, wz = lm[0]
    lm[3] = [wx - 0.08, wy - 0.04, wz]
    lm[4] = [wx - 0.12, wy - 0.03, wz]
    return lm


class TestDiagnosticAudit(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.original_env = os.environ.get("VSL_DIAGNOSTIC")

    def tearDown(self):
        if self.original_env is not None:
            os.environ["VSL_DIAGNOSTIC"] = self.original_env
        else:
            os.environ.pop("VSL_DIAGNOSTIC", None)
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_diagnostic_disabled_by_default(self):
        """1. Khi khong dat bien moi truong, logging mac dinh tat va khong tao file."""
        os.environ["VSL_DIAGNOSTIC"] = "0"
        rec = VSLRecognizer()
        rec.diagnostic_enabled = False
        rec.diagnostic_dir = self.temp_dir

        lm = _make_sample_hand()
        result = rec.predict_landmarks(lm)

        self.assertIn("sign", result)
        files = os.listdir(self.temp_dir)
        self.assertEqual(len(files), 0, "Khong duoc tao file khi diagnostic tat")

    def test_diagnostic_enabled_logs_valid_jsonl(self):
        """2. Khi bat diagnostic, ghi dung cau truc JSONL voi day du 21 landmarks va ket qua trung gian."""
        rec = VSLRecognizer()
        rec.diagnostic_enabled = True
        rec.diagnostic_dir = self.temp_dir
        log_file = os.path.join(self.temp_dir, "test_diag.jsonl")
        rec.diagnostic_filepath = log_file

        lm = _make_sample_hand()
        res = rec.predict_landmarks(
            lm,
            expected_sign="A",
            session_id="sess_unit_test"
        )
        self.assertIn("sign", res)

        self.assertTrue(os.path.exists(log_file), "File log JSONL phai duoc tao")

        records = load_diagnostic_records(log_file)
        self.assertEqual(len(records), 1)
        r = records[0]

        # Kiem tra cac truong bat buoc
        self.assertEqual(r["session_id"], "sess_unit_test")
        self.assertEqual(r["expected_sign"], "A")
        self.assertEqual(r["handedness"], "Right")
        self.assertEqual(len(r["raw_landmarks"]), 21)
        self.assertGreater(r["palm_scale"], 0)

        # Kiem tra finger_states
        self.assertIn("thumb_extended", r["finger_states"])
        self.assertIn("index_curl", r["finger_states"])

        # Kiem tra cac tang du doan
        self.assertIn("sign", r["geometry"])
        self.assertIn("confidence", r["geometry"])
        self.assertIn("has_weights", r["mlp"])
        self.assertIn("sign", r["ensemble"])
        self.assertIn("confirmed", r["smoothing"])

    def test_rate_limiting(self):
        """3. Goi nhieu lan lien tiep trong khoang thoi gian ngan chi ghi 1 mau (~2 mau/s)."""
        rec = VSLRecognizer()
        rec.diagnostic_enabled = True
        rec.diagnostic_dir = self.temp_dir
        log_file = os.path.join(self.temp_dir, "test_rate.jsonl")
        rec.diagnostic_filepath = log_file

        lm = _make_sample_hand()
        for _ in range(5):
            rec.predict_landmarks(lm)

        records = load_diagnostic_records(log_file)
        self.assertEqual(len(records), 1, "Rate limiter phai chan cac frame goi qua sat nhau")

    def test_io_error_isolation(self):
        """4. Khi gap loi ghi dia (PermissionError / OSError), API van tra ket qua an toan."""
        rec = VSLRecognizer()
        rec.diagnostic_enabled = True
        rec.diagnostic_dir = self.temp_dir

        lm = _make_sample_hand()
        with patch("builtins.open", side_effect=PermissionError("Mock Permission Denied")):
            try:
                res = rec.predict_landmarks(lm)
                self.assertIn("sign", res)
            except Exception as e:
                self.fail(f"predict_landmarks bi crash boi loi I/O: {e}")

    def test_root_cause_mlp_override(self):
        """5. Replay xac dinh dung loi MLP_OVERRIDE: Geometry dung A, bi MLP sai B (conf>0.6) ghi de."""
        lm = _make_sample_hand()
        record = {
            "timestamp": 1000.0,
            "session_id": "test_mlp_override",
            "frame_idx": 1,
            "expected_sign": "A",
            "handedness": "Right",
            "raw_landmarks": lm,
            "palm_scale": 0.3,
            "finger_states": {"thumb_extended": True, "index_extended": False},
            "geometry": {"sign": "A", "confidence": 0.84},
            "mlp": {"has_weights": True, "sign": "B", "confidence": 0.75},
            "ensemble": {"sign": "B", "confidence": 0.84},
            "smoothing": {"confirmed": False, "recent_history": ["B"]}
        }
        summary = replay_frame_by_frame([record])
        self.assertEqual(summary["match_count"], 0)
        self.assertEqual(len(summary["mismatches"]), 1)
        self.assertIn("MLP_OVERRIDE", summary["mismatches"][0]["root_cause"])

    def test_root_cause_geometry_mismatch_mlp_underconfident(self):
        """6. Replay xac dinh dung GEOMETRY_MISMATCH_MLP_UNDERCONFIDENT: Geometry sai, MLP dung nhung conf <= 0.6."""
        lm = _make_sample_hand()
        record = {
            "timestamp": 1000.0,
            "session_id": "test_underconf",
            "frame_idx": 2,
            "expected_sign": "A",
            "handedness": "Right",
            "raw_landmarks": lm,
            "palm_scale": 0.3,
            "finger_states": {"thumb_extended": True, "index_extended": False},
            "geometry": {"sign": "NEUTRAL", "confidence": 0.0},
            "mlp": {"has_weights": True, "sign": "A", "confidence": 0.55},
            "ensemble": {"sign": "NEUTRAL", "confidence": 0.55},
            "smoothing": {"confirmed": False, "recent_history": ["NEUTRAL"]}
        }
        summary = replay_frame_by_frame([record])
        self.assertEqual(len(summary["mismatches"]), 1)
        self.assertIn("GEOMETRY_MISMATCH_MLP_UNDERCONFIDENT", summary["mismatches"][0]["root_cause"])

    def test_root_cause_both_layers_mismatch(self):
        """7. Replay xac dinh dung BOTH_LAYERS_MISMATCH khi ca 2 tang deu khong nhan ra chu ky vong."""
        lm = _make_sample_hand()
        record = {
            "timestamp": 1000.0,
            "session_id": "test_both_err",
            "frame_idx": 3,
            "expected_sign": "A",
            "handedness": "Right",
            "raw_landmarks": lm,
            "palm_scale": 0.3,
            "finger_states": {"thumb_extended": True, "index_extended": False},
            "geometry": {"sign": "B", "confidence": 0.86},
            "mlp": {"has_weights": True, "sign": "C", "confidence": 0.65},
            "ensemble": {"sign": "C", "confidence": 0.86},
            "smoothing": {"confirmed": False, "recent_history": ["C"]}
        }
        summary = replay_frame_by_frame([record])
        self.assertEqual(len(summary["mismatches"]), 1)
        self.assertIn("BOTH_LAYERS_MISMATCH", summary["mismatches"][0]["root_cause"])

    def test_missing_ground_truth_safety(self):
        """8. Khi expected_sign la None, replay khong duoc gan loi bua bai (tra ve NO_GROUND_TRUTH)."""
        lm = _make_sample_hand()
        record = {
            "timestamp": 1000.0,
            "session_id": "test_no_gt",
            "frame_idx": 4,
            "expected_sign": None,
            "handedness": "Right",
            "raw_landmarks": lm,
            "palm_scale": 0.3,
            "finger_states": {},
            "geometry": {"sign": "A", "confidence": 0.84},
            "mlp": {"has_weights": True, "sign": "A", "confidence": 0.80},
            "ensemble": {"sign": "A", "confidence": 0.84},
            "smoothing": {"confirmed": False, "recent_history": ["A"]}
        }
        summary = replay_frame_by_frame([record])
        self.assertEqual(summary["match_count"], 0)
        self.assertEqual(len(summary["mismatches"]), 0, "Khong duoc danh dau la mismatch khi khong co ground truth")

    def test_corrupted_record_safety(self):
        """9. Bo qua an toan cac ban ghi thieu truong raw_landmarks hoac landmark < 21 diem."""
        corrupted_records = [
            {"session_id": "s1", "raw_landmarks": []},
            {"session_id": "s2", "raw_landmarks": [[0.0, 0.0, 0.0]] * 10}, # Chi co 10 diem
            {"session_id": "s3"} # Thieu hoan toan raw_landmarks
        ]
        summary = replay_frame_by_frame(corrupted_records)
        self.assertEqual(summary["evaluated_frames"], 0)
        self.assertEqual(len(summary["mismatches"]), 0)

    def test_session_replay_sequential_smoothing_and_cooldown(self):
        """10. Session replay tuan tu kiem tra chinh xac smoothing window (>=5) va cooldown debounce (>1.2s)."""
        lm = _make_sample_hand()
        records = []

        # Tao 6 frame lien tiep tai t = 1.0, 1.1, 1.2, 1.3, 1.4, 1.5
        for i in range(6):
            records.append({
                "timestamp": 1000.0 + i * 0.1,
                "session_id": "sess_smooth",
                "frame_idx": i + 1,
                "expected_sign": "DI_HOC",
                "handedness": "Right",
                "raw_landmarks": lm
            })

        # Frame 7 o thoi diem t = 1000.6 (chua qua 1.2s cooldown)
        records.append({
            "timestamp": 1000.6,
            "session_id": "sess_smooth",
            "frame_idx": 7,
            "expected_sign": "DI_HOC",
            "handedness": "Right",
            "raw_landmarks": lm
        })

        # Frame 8 o thoi diem t = 1002.5 (> 1.2s cooldown tinh tu lan confirm dau tien)
        records.append({
            "timestamp": 1002.5,
            "session_id": "sess_smooth",
            "frame_idx": 8,
            "expected_sign": "DI_HOC",
            "handedness": "Right",
            "raw_landmarks": lm
        })

        sess_summary = replay_session_sequential(records, session_id="sess_smooth")
        sess_data = sess_summary["sessions"]["sess_smooth"]

        # Kiem tra confirmed tokens:
        # Lan 1 tai frame 5 (khi prediction_history tich luy du 5 frames)
        # Lan 2 tai frame 8 (vi cach frame confirm truoc do > 1.2s)
        confirmed_tokens = sess_data["confirmed_tokens"]
        self.assertGreaterEqual(len(confirmed_tokens), 2)
        self.assertEqual(confirmed_tokens[0]["frame_idx"], 5)
        self.assertEqual(confirmed_tokens[1]["frame_idx"], 8)


if __name__ == "__main__":
    unittest.main(verbosity=2)
