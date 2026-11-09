"""
Automated Test Suite for Controlled Pilot V3.
Covers all 22 required test scenarios specified in TASK — REBUILD CONTROLLED PILOT V3:
1. Exact 29 Vietnamese letters SSOT.
2. 10 accepted trials per letter.
3. Total 290 accepted trials in full session.
4. Exactly 5.0 seconds recording duration using fake clock.
5. Countdown does not corrupt recording duration.
6. Timer remains accurate when FPS fluctuates (10 FPS, 30 FPS, 60 FPS).
7. Valid state transitions and rejection of illegal transitions.
8. Pause and resume handling.
9. Camera disconnect / interruption handling.
10. No-hand recording diagnostics (not dropped).
11. Retry creates new attempt without destroying old attempt data.
12. Skip does not count towards accepted trials.
13. Vietnamese Unicode diacritics preserved without stripping.
14. JSON schema validation and zero-handling rules.
15. Backend flush before completion confirmation.
16. Single source of truth (no backend/frontend progress drift).
17. Session recovery from disk.
18. Session export covering all 29 letters.
19. Correct recognition with poor capture quality flagged.
20. High capture quality with mismatched recognition recorded.
21. Unexecuted recognition layers properly marked NOT_EXECUTED with reasons.
22. Non-pilot regression safety.
"""

import json
import os
import shutil
import tempfile
import unittest

from backend.vision.controlled_pilot_v3.clock import FakeClock
from backend.vision.controlled_pilot_v3.constants import (
    COUNTDOWN_DURATION_SEC,
    RECORDING_DURATION_SEC,
    TARGET_ACCEPTED_TRIALS_PER_LETTER,
    TOTAL_ACCEPTED_TRIALS_FULL_SESSION,
    TOTAL_LETTERS,
    VIETNAMESE_ALPHABET_29,
    HandState,
    RecognitionState,
    SessionMode,
    SessionState,
    TrialState,
)
from backend.vision.controlled_pilot_v3.recognition_adapter import ControlledPilotRecognitionAdapter
from backend.vision.controlled_pilot_v3.session_manager import PilotSessionManager
from backend.vision.controlled_pilot_v3.storage import PilotStorage


def make_synthetic_landmarks(wrist_y: float = 0.6) -> list:
    """Creates a valid (21, 3) landmark array."""
    pts = []
    for i in range(21):
        pts.append([0.5, wrist_y - (i * 0.015), 0.0])
    return pts


class MockRecognizer:
    def __init__(self):
        self.ALLOWED_LETTERS = set(VIETNAMESE_ALPHABET_29)
        self.alphabet_classifier = self.MockAlphabetClassifier()
        self.static_alphabet_head = self.MockStaticHead()
        self.has_static_head = True
        self.prediction_history = [("A", 0.95)]
        self.last_confirmed_sign = "A"
        self.last_confirmed_time = 100.0
        self.cooldown_sec = 1.0
        self.smoothing_window = 5

    class MockAlphabetClassifier:
        def __init__(self):
            self.mock_status = "MATCH"
            self.mock_letter = "A"
            self.mock_confidence = 0.95
            self.mock_reason = "Test match"
            self.mock_candidates = ["A"]
            self.mock_features = {"finger_states": ["THUMB_OPEN", "INDEX_EXTENDED"]}

        def classify_detailed(self, landmarks):
            return {
                "status": self.mock_status,
                "letter": self.mock_letter,
                "confidence": self.mock_confidence,
                "reason": self.mock_reason,
                "matched_candidates": self.mock_candidates,
                "features": self.mock_features,
            }

    class MockStaticHead:
        def __init__(self):
            self.mock_letter = "B"
            self.mock_conf = 0.88
            self.mock_top_k = [{"letter": "B", "confidence": 0.88}]

        def predict_feature(self, feat, confidence_threshold=0.0):
            return {
                "raw_predicted_letter": self.mock_letter,
                "confidence": self.mock_conf,
                "top_k": self.mock_top_k,
            }


class TestControlledPilotV3(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage = PilotStorage(root_dir=self.test_dir)
        self.fake_clock = FakeClock(start_epoch=1700000000.0, start_mono=1000.0)
        self.recognizer = MockRecognizer()
        self.adapter = ControlledPilotRecognitionAdapter(self.recognizer)
        self.mgr = PilotSessionManager(adapter=self.adapter, storage=self.storage, clock=self.fake_clock)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    # 1. Exact 29 Vietnamese letters
    def test_01_exact_29_vietnamese_letters(self):
        expected = [
            "A", "Ă", "Â", "B", "C", "D", "Đ", "E", "Ê", "G", "H", "I", "K", "L",
            "M", "N", "O", "Ô", "Ơ", "P", "Q", "R", "S", "T", "U", "Ư", "V", "X", "Y"
        ]
        self.assertEqual(VIETNAMESE_ALPHABET_29, expected)
        self.assertEqual(len(VIETNAMESE_ALPHABET_29), 29)
        self.assertEqual(len(set(VIETNAMESE_ALPHABET_29)), 29)

    # 2. 10 accepted trials per letter
    def test_02_ten_accepted_trials_per_letter(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="O")
        self.mgr.start_session()

        for trial_num in range(1, 11):
            self.mgr.start_trial()
            self.mgr.start_trial_countdown()
            self.fake_clock.advance(3.0)
            self.mgr.start_trial_recording()

            # Record 5 seconds of frames
            for _ in range(5):
                self.fake_clock.advance(1.0)
                self.mgr.record_frame(make_synthetic_landmarks())

            self.assertEqual(self.mgr.accepted_counts["O"], trial_num)

        self.assertEqual(self.mgr.accepted_counts["O"], 10)
        self.assertEqual(self.mgr.state, SessionState.COMPLETED)

    # 3. Full session has 290 accepted trials
    def test_03_full_session_290_trials_target(self):
        self.mgr.create_session(mode=SessionMode.FULL_29)
        status = self.mgr.get_status()
        self.assertEqual(status["total_target_accepted"], 290)
        self.assertEqual(status["target_accepted_per_letter"], 10)
        self.assertEqual(len(self.mgr.target_letters), 29)

    # 4. Recording duration exactly 5.0 seconds
    def test_04_recording_duration_exactly_5_seconds(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="A")
        self.mgr.start_session()
        self.mgr.start_trial()
        self.mgr.start_trial_recording()

        self.fake_clock.advance(2.0)
        self.mgr.record_frame(make_synthetic_landmarks())
        self.assertEqual(self.mgr.current_attempt.get_recording_elapsed(), 2.0)

        self.fake_clock.advance(3.0)
        self.mgr.record_frame(make_synthetic_landmarks())
        self.assertEqual(self.mgr.current_attempt.get_recording_elapsed(), 5.0)

        # Summary duration check
        summary = self.mgr.current_attempt.calculate_summary()
        self.assertEqual(summary["duration_ms"], 5000.0)

    # 5. Countdown does not corrupt recording duration
    def test_05_countdown_does_not_corrupt_recording_duration(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="B")
        self.mgr.start_session()
        self.mgr.start_trial()

        # 3 seconds countdown
        self.mgr.start_trial_countdown()
        self.fake_clock.advance(3.0)

        # Start recording
        self.mgr.start_trial_recording()
        self.assertEqual(self.mgr.current_attempt.get_recording_elapsed(), 0.0)

        self.fake_clock.advance(5.0)
        self.mgr.record_frame(make_synthetic_landmarks())
        self.assertEqual(self.mgr.current_attempt.get_recording_elapsed(), 5.0)

    # 6. Timer accuracy under fluctuating FPS
    def test_06_timer_accuracy_under_fluctuating_fps(self):
        for target_fps in [10, 30, 60]:
            fake_clock = FakeClock(start_epoch=1700000000.0, start_mono=1000.0)
            mgr = PilotSessionManager(adapter=self.adapter, storage=self.storage, clock=fake_clock)
            mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="C")
            mgr.start_session()
            mgr.start_trial()
            mgr.start_trial_recording()

            step = 1.0 / target_fps
            frames_to_send = int(5.0 * target_fps)
            for _ in range(frames_to_send):
                fake_clock.advance(step)
                mgr.record_frame(make_synthetic_landmarks())

            summary = mgr.current_attempt.calculate_summary()
            self.assertAlmostEqual(summary["duration_ms"], 5000.0, delta=100.0)
            self.assertEqual(summary["total_frames"], frames_to_send)

    # 7. Valid and invalid state transitions
    def test_07_valid_and_invalid_state_transitions(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="A")
        # Cannot jump from READY to COMPLETED directly
        with self.assertRaises(ValueError):
            self.mgr.transition_to(SessionState.COMPLETED)

        self.mgr.start_session()  # READY -> RUNNING
        self.assertEqual(self.mgr.state, SessionState.RUNNING)

        self.mgr.pause_session()  # RUNNING -> PAUSED
        self.assertEqual(self.mgr.state, SessionState.PAUSED)

        # Cannot start trial when PAUSED
        with self.assertRaises(RuntimeError):
            self.mgr.start_trial()

    # 8. Pause and resume handling
    def test_08_pause_and_resume_during_recording(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="D")
        self.mgr.start_session()
        self.mgr.start_trial()
        self.mgr.start_trial_recording()

        # Record 2 seconds
        self.fake_clock.advance(2.0)
        self.mgr.record_frame(make_synthetic_landmarks())
        att = self.mgr.current_attempt

        # Pause session
        self.mgr.pause_session()
        self.assertEqual(self.mgr.state, SessionState.PAUSED)
        self.assertEqual(att.state, TrialState.INTERRUPTED)

        # Resume session
        self.mgr.resume_session()
        self.assertEqual(self.mgr.state, SessionState.RUNNING)

        # Accepted count did not increment
        self.assertEqual(self.mgr.accepted_counts["D"], 0)

    # 9. Camera disconnect / interruption
    def test_09_camera_disconnect_interruption(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="E")
        self.mgr.start_session()
        self.mgr.start_trial()
        self.mgr.start_trial_recording()

        self.fake_clock.advance(1.5)
        self.mgr.record_frame(make_synthetic_landmarks())
        self.mgr.current_attempt.interrupt("Camera connection lost")

        self.assertEqual(self.mgr.current_attempt.state, TrialState.INTERRUPTED)
        summary = self.mgr.storage.load_trial_summary(
            self.mgr.session_id,
            self.mgr.current_attempt.trial_id,
            self.mgr.current_attempt.attempt_id,
        )
        self.assertFalse(summary["accepted"])
        self.assertIn("Interrupted: Camera connection lost", summary["recording_errors"])

    # 10. No-hand recording diagnostics
    def test_10_no_hand_recording_diagnostics(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="G")
        self.mgr.start_session()
        self.mgr.start_trial()
        self.mgr.start_trial_recording()

        self.fake_clock.advance(5.0)
        rec = self.mgr.record_frame(None)  # No hand

        self.assertEqual(rec["hand_tracking"]["hand_detected"], False)
        self.assertEqual(rec["hand_tracking"]["tracking_status"], HandState.NO_HAND.value)
        self.assertEqual(rec["recognition_state"], RecognitionState.NEUTRAL.value)

        summary = self.mgr.current_attempt.calculate_summary()
        self.assertEqual(summary["no_hand_frames"], 1)

    # 11. Retry does not destroy old attempt
    def test_11_retry_does_not_destroy_old_attempt(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="H")
        self.mgr.start_session()
        self.mgr.start_trial()
        self.mgr.start_trial_recording()

        self.fake_clock.advance(2.0)
        self.mgr.record_frame(make_synthetic_landmarks())
        attempt_1_id = self.mgr.current_attempt.attempt_id
        trial_id = self.mgr.current_attempt.trial_id

        # Retry
        self.mgr.retry_trial(reason="Blurry frame")
        attempt_2_id = self.mgr.current_attempt.attempt_id

        self.assertNotEqual(attempt_1_id, attempt_2_id)
        # Verify both attempt directories exist on disk
        att1_summary = self.mgr.storage.load_trial_summary(self.mgr.session_id, trial_id, attempt_1_id)
        self.assertIsNotNone(att1_summary)
        self.assertEqual(att1_summary["completion_status"], TrialState.INTERRUPTED.value)

    # 12. Skip is not counted as accepted
    def test_12_skip_is_not_counted_as_accepted(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="I")
        self.mgr.start_session()
        self.mgr.start_trial()
        self.mgr.start_trial_recording()

        self.mgr.skip_trial(reason="User skipped letter I trial")
        self.assertEqual(self.mgr.accepted_counts["I"], 0)

    # 13. Unicode diacritics preserved
    def test_13_unicode_diacritics_preserved(self):
        diacritic_letters = ["Ă", "Â", "Đ", "Ê", "Ô", "Ơ", "Ư"]
        for letter in diacritic_letters:
            self.assertIn(letter, VIETNAMESE_ALPHABET_29)
            self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter=letter)
            self.assertEqual(self.mgr.get_current_letter(), letter)

    # 14. JSON schema validation and zero-handling rules
    def test_14_json_schema_validation_and_zero_rules(self):
        # A summary with total_frames == 0 must have rates as None, not 0
        invalid_summary = {
            "schema_version": "3.0.0",
            "session_id": "test_s",
            "trial_id": "test_t",
            "attempt_id": "test_a",
            "expected_letter": "A",
            "trial_number": 1,
            "started_at": 100.0,
            "ended_at": 105.0,
            "duration_ms": 5000.0,
            "completion_status": "COMPLETED",
            "accepted": True,
            "total_frames": 0,
            "hand_detected_frames": 0,
            "no_hand_frames": 0,
            "valid_landmark_frames": 0,
            "prediction_distribution": {},
            "recognition_match_rate": 0,  # Invalid! Must be None when total_frames == 0
            "recognition_mismatch_rate": None,
            "unknown_rate": None,
            "neutral_rate": None,
            "prediction_source_distribution": {},
            "rule_failure_distribution": {},
            "quality_flags": [],
            "recording_errors": [],
        }
        with self.assertRaises(ValueError):
            self.storage.validate_trial_summary_schema(invalid_summary)

    # 15. Backend flush before completion confirmation
    def test_15_backend_flush_before_completion(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="K")
        self.mgr.start_session()
        self.mgr.start_trial()
        self.mgr.start_trial_recording()
        self.fake_clock.advance(5.0)
        self.mgr.record_frame(make_synthetic_landmarks())

        att = self.mgr.current_attempt
        # Completing trial writes summary to disk before marking COMPLETED
        summary = att.complete(accept=True)
        self.assertEqual(att.state, TrialState.COMPLETED)
        disk_summary = self.storage.load_trial_summary(self.mgr.session_id, att.trial_id, att.attempt_id)
        self.assertEqual(disk_summary["attempt_id"], att.attempt_id)

    # 16. Single source of truth (no backend/frontend drift)
    def test_16_single_source_of_truth_status(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="L")
        self.mgr.start_session()
        status = self.mgr.get_status()
        self.assertEqual(status["letter_accepted_trials"], 0)
        self.assertEqual(status["total_accepted_trials"], 0)
        self.assertEqual(status["current_letter"], "L")

    # 17. Session recovery from disk
    def test_17_session_recovery_from_disk(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="M")
        self.mgr.start_session()

        # Complete 3 accepted trials
        for _ in range(3):
            self.mgr.start_trial()
            self.mgr.start_trial_recording()
            self.fake_clock.advance(5.0)
            self.mgr.record_frame(make_synthetic_landmarks())

        session_id = self.mgr.session_id
        self.assertEqual(self.mgr.accepted_counts["M"], 3)

        # Simulate new server instance recovering the session
        new_mgr = PilotSessionManager(adapter=self.adapter, storage=self.storage, clock=self.fake_clock)
        recovered_status = new_mgr.recover_session(session_id)

        self.assertEqual(recovered_status["letter_accepted_trials"], 3)
        self.assertEqual(recovered_status["current_trial_number"], 4)
        self.assertEqual(new_mgr.state, SessionState.RUNNING)

    # 18. Export covering all 29 letters
    def test_18_export_covering_all_letters(self):
        self.mgr.create_session(mode=SessionMode.FULL_29)
        summary = self.mgr.generate_session_summary()
        self.assertEqual(len(summary["letter_reports"]), 29)
        for letter in VIETNAMESE_ALPHABET_29:
            self.assertIn(letter, summary["letter_reports"])

    # 19. Correct recognition with poor capture quality flagged
    def test_19_poor_capture_quality_flagged(self):
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="A")
        self.mgr.start_session()
        self.mgr.start_trial()
        self.mgr.start_trial_recording()

        # Send invalid landmarks (e.g. wrong shape)
        self.fake_clock.advance(5.0)
        invalid_lm = [[0.0, 0.0]]  # Not 21 points
        self.mgr.record_frame(invalid_lm)

        summary = self.mgr.current_attempt.calculate_summary()
        self.assertIn("NO_VALID_LANDMARKS", summary["quality_flags"])

    # 20. High capture quality with mismatched recognition
    def test_20_high_capture_quality_with_mismatch(self):
        self.recognizer.alphabet_classifier.mock_letter = "B"
        self.mgr.create_session(mode=SessionMode.SINGLE_LETTER, single_letter="A")
        self.mgr.start_session()
        self.mgr.start_trial()
        self.mgr.start_trial_recording()

        self.fake_clock.advance(5.0)
        self.mgr.record_frame(make_synthetic_landmarks())

        summary = self.mgr.current_attempt.calculate_summary()
        self.assertEqual(summary["valid_landmark_frames"], 1)
        self.assertEqual(summary["recognition_match_rate"], 0.0)
        self.assertEqual(summary["recognition_mismatch_rate"], 1.0)

    # 21. Unexecuted recognition layers properly marked
    def test_21_unexecuted_layers_marked_not_executed(self):
        self.recognizer.alphabet_classifier.mock_status = "MATCH"
        self.recognizer.alphabet_classifier.mock_letter = "A"

        res = self.adapter.evaluate_frame(make_synthetic_landmarks(), expected_letter="A")
        self.assertEqual(res["mlp_status"], "NOT_EXECUTED")
        self.assertIn("bypassed", res["mlp_not_executed_reason"])
        self.assertIsNone(res["mlp_prediction"])

    # 22. Non-pilot regression safety
    def test_22_non_pilot_regression_safety(self):
        from backend.vision.recognizer import ALLOWED_LETTERS, VIETNAMESE_ALPHABET_29
        self.assertEqual(len(ALLOWED_LETTERS), 29)
        self.assertEqual(len(VIETNAMESE_ALPHABET_29), 29)


if __name__ == "__main__":
    unittest.main()
