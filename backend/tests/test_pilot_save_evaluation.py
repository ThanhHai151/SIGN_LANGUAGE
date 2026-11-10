import os
import json
import tempfile
import unittest
from backend.app import app, save_pilot_evaluation, PilotEvaluationRequest, list_pilot_evaluations

class TestPilotSaveEvaluation(unittest.TestCase):
    def test_save_and_list_evaluation(self):
        sample_report = {
            "session_id": "test_session_123",
            "config": {
                "total_letters": 29,
                "trials_per_letter": 10,
                "duration_sec": 5.0,
                "mode": "all_29_letters"
            },
            "summary": {
                "total_trials_completed": 20,
                "total_trials_passed": 18,
                "overall_accuracy_pct": 90.0,
                "letters_passed": ["A", "B"],
                "letters_needing_fix": ["O"],
                "letters_unsupported": []
            },
            "letters": {
                "A": {"accuracy_pct": 95.0, "status": "PASS"},
                "O": {"accuracy_pct": 40.0, "status": "NEEDS_FIX", "top_confusions": {"C": 10}}
            }
        }
        sample_frames = [
            {"frame_idx": 1, "expected_sign": "A", "predicted_sign": "A", "confidence": 0.95},
            {"frame_idx": 2, "expected_sign": "A", "predicted_sign": "A", "confidence": 0.92}
        ]
        
        req = PilotEvaluationRequest(
            session_id="test_session_123",
            report=sample_report,
            frames=sample_frames
        )
        
        res = save_pilot_evaluation(req)
        self.assertTrue(res["success"])
        self.assertTrue(os.path.exists(res["filepath"]))
        self.assertTrue(os.path.exists(res["frames_filepath"]))
        
        # Verify JSON content
        with open(res["filepath"], "r", encoding="utf-8") as f:
            saved_json = json.load(f)
        self.assertEqual(saved_json["session_id"], "test_session_123")
        self.assertIn("saved_at", saved_json)
        self.assertEqual(saved_json["summary"]["overall_accuracy_pct"], 90.0)
        
        # Verify listing
        list_res = list_pilot_evaluations()
        self.assertTrue(list_res["success"])
        filenames = [e["filename"] for e in list_res["evaluations"]]
        self.assertIn(res["filename"], filenames)
        
        # Clean up test files
        if os.path.exists(res["filepath"]):
            os.remove(res["filepath"])
        if os.path.exists(res["frames_filepath"]):
            os.remove(res["frames_filepath"])

if __name__ == "__main__":
    unittest.main()

