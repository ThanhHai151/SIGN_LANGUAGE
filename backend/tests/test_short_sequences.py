"""
Unit & Integration Tests for Short Sequences (< 30 frames) and End-of-Gesture Resampling in VSL.
"""
import os
import unittest
import numpy as np
from fastapi.testclient import TestClient

from backend.app import app, FEATURE_ALLOW_SHORT_SEQUENCES, MIN_GESTURE_FRAMES
from backend.vision.recognizer import VSLRecognizer
from backend.vision.skeleton_norm import resample_sequence_linear, resample_sequence_by_timestamps


class TestShortSequences(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)
        cls.recognizer = VSLRecognizer()
        
        sample_path = os.path.join(
            os.path.dirname(__file__), "..", "data", "qipedc_reference", "v3_D0228.npy"
        )
        if os.path.exists(sample_path):
            cls.real_seq = np.load(sample_path)  # (N, 75, 3)
        else:
            cls.real_seq = np.random.randn(60, 75, 3).astype(np.float32) * 0.1
            cls.real_seq[:, 11, :] = [-0.2, 0.3, 0.0]
            cls.real_seq[:, 12, :] = [0.2, 0.3, 0.0]
            cls.real_seq[:, 23, :] = [-0.15, 0.8, 0.0]
            cls.real_seq[:, 24, :] = [0.15, 0.8, 0.0]

    def test_resample_linear_dimensions_and_finiteness(self):
        """Kiểm tra nội suy tuyến tính từ 12-29 frame lên 30 frame không phát sinh NaN/Inf."""
        for t in [12, 15, 18, 22, 25, 29]:
            sub = self.real_seq[:t]
            resampled = resample_sequence_linear(sub, target_len=30)
            self.assertEqual(resampled.shape, (30, 75, 3))
            self.assertTrue(np.all(np.isfinite(resampled)))

    def test_resample_timestamps_dimensions(self):
        """Kiểm tra nội suy theo timestamps thực tế từ camera."""
        for t in [12, 18, 25]:
            sub = self.real_seq[:t]
            ts = [1000.0 + i * 33.33 for i in range(t)]
            resampled = resample_sequence_by_timestamps(sub, ts, target_len=30)
            self.assertEqual(resampled.shape, (30, 75, 3))
            self.assertTrue(np.all(np.isfinite(resampled)))

    def test_recognizer_rejects_below_min_length(self):
        """recognizer.predict_sequence phải từ chối chuỗi dưới 12 frame (nhiễu)."""
        seq_8 = self.real_seq[:8].tolist()
        result = self.recognizer.predict_sequence(seq_8, allow_short_sequences=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["sign"], "NEUTRAL")
        self.assertEqual(result["confidence"], 0.0)

        seq_11 = self.real_seq[:11].tolist()
        result = self.recognizer.predict_sequence(seq_11, allow_short_sequences=True)
        self.assertTrue(result["success"])
        self.assertEqual(result["sign"], "NEUTRAL")
        self.assertEqual(result["confidence"], 0.0)

    def test_recognizer_accepts_short_sequence_18_and_25(self):
        """recognizer.predict_sequence chấp nhận chuỗi 18 và 25 frame khi allow_short_sequences=True."""
        for length in [18, 25]:
            seq = self.real_seq[:length].tolist()
            result = self.recognizer.predict_sequence(seq, allow_short_sequences=True, threshold=0.1)
            self.assertTrue(result["success"])
            self.assertIsInstance(result["confidence"], (float, int))

    def test_recognizer_keeps_30_frames_unchanged(self):
        """recognizer.predict_sequence xử lý chuỗi 30 frame bình thường."""
        seq_30 = self.real_seq[:30].tolist()
        result = self.recognizer.predict_sequence(seq_30, threshold=0.1)
        self.assertTrue(result["success"])
        self.assertIsInstance(result["confidence"], (float, int))

    def test_api_recognize_sequence_short_gesture(self):
        """API /api/recognize_sequence chấp nhận cử chỉ 18 frame khi allow_short_sequences=True."""
        seq_18 = self.real_seq[:18].tolist()
        payload = {
            "sequence": seq_18,
            "threshold": 0.5,
            "mode": "words",
            "allow_short_sequences": True
        }
        resp = self.client.post("/api/recognize_sequence", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertTrue(data["success"])

    def test_api_recognize_sequence_rejects_under_12_frames(self):
        """API /api/recognize_sequence từ chối cử chỉ < 12 frames."""
        seq_10 = self.real_seq[:10].tolist()
        payload = {
            "sequence": seq_10,
            "threshold": 0.5,
            "mode": "words",
            "allow_short_sequences": True
        }
        resp = self.client.post("/api/recognize_sequence", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertEqual(data["sign"], "NEUTRAL")
        self.assertEqual(data["confidence"], 0.0)

    def test_api_feature_flag_rollback(self):
        """Khi allow_short_sequences=False, chuỗi 18 frame phải bị từ chối như logic cũ."""
        seq_18 = self.real_seq[:18].tolist()
        payload = {
            "sequence": seq_18,
            "threshold": 0.5,
            "mode": "words",
            "allow_short_sequences": False
        }
        resp = self.client.post("/api/recognize_sequence", json=payload)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()["data"]
        self.assertEqual(data["sign"], "NEUTRAL")
        self.assertEqual(data["confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
