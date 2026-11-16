import unittest
import json
from pathlib import Path

from backend.app import get_pilot_runtime_config, LandmarkRequest, recognizer
from backend.vision.alphabet_classifier import alphabet_classifier


class TestCandidateDPilotLogging(unittest.TestCase):
    def test_runtime_config_reports_actual_candidate_d_constants(self):
        config = get_pilot_runtime_config()
        self.assertTrue(config["success"])
        self.assertEqual(config["classifier_rule_version"], "Candidate-D")
        self.assertEqual(config["schema_version"], "2.1-kinematic-phase")
        self.assertEqual(config["feature_extractor_version"], "2.0-kinematic")
        self.assertEqual(config["classifier_config"]["O_D_TI_MAX"], 0.23)
        self.assertEqual(config["classifier_config"]["O_IDX_STRAIGHT_MAX"], 0.65)

    def test_phase_is_validated_and_persisted_in_frame_record(self):
        req = LandmarkRequest(landmarks=[], phase="STABLE_HOLD")
        self.assertEqual(req.phase, "STABLE_HOLD")
        with self.assertRaises(Exception):
            LandmarkRequest(landmarks=[], phase="UNKNOWN_PHASE")

        record = recognizer._build_pilot_frame_record(
            1, "O", "O", 0.95, "geometric_rule", False, "VALID", None,
            session_id="pilot_test", trial_id="pilot_test_O_rep01", phase=req.phase,
        )
        self.assertEqual(record["phase"], "STABLE_HOLD")
        self.assertEqual(record["classifier_rule_version"], "Candidate-D")
        self.assertEqual(record["schema_version"], "2.1-kinematic-phase")
        self.assertEqual(record["classifier_config"], {"O_D_TI_MAX": 0.23, "O_IDX_STRAIGHT_MAX": 0.65})

    def test_p_and_x_baseline_rule_matches_remain_present(self):
        fixture_dir = Path(__file__).resolve().parents[1] / "data" / "alphabet_v2" / "static_landmarks"
        for letter, preferred_frame in (("P", 38), ("X", 43)):
            with (fixture_dir / f"{letter}.json").open(encoding="utf-8") as fixture_file:
                samples = json.load(fixture_file)["samples"]
            preferred = next((s for s in samples if s.get("frame_index") == preferred_frame and s.get("landmarks")), None)
            stable = [s for s in samples if s.get("phase") == "stable_pose" and s.get("landmarks")]
            candidates = ([preferred] if preferred else []) + stable
            self.assertTrue(
                any(alphabet_classifier.classify_detailed(sample["landmarks"]).get("letter") == letter for sample in candidates),
                f"Existing {letter} baseline fixture no longer matches its classifier rule",
            )


if __name__ == "__main__":
    unittest.main()
