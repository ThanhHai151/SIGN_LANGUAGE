import unittest
from unittest.mock import patch

from backend.vision.recognizer import VSLRecognizer


class TestPilotTrialTemporalIsolation(unittest.TestCase):
    def setUp(self):
        self.rec = VSLRecognizer(confidence_threshold=0.70, smoothing_window=4, cooldown_sec=0.4)
        self.rec.reset()
        self.match_o = {
            "letter": "O",
            "confidence": 0.95,
            "status": "MATCH",
            "reason": "test O match",
            "matched_candidates": [("O", 0.95)],
            "features": {},
        }
        self.classifier_patch = patch.object(
            self.rec.alphabet_classifier,
            "classify_detailed",
            return_value=self.match_o,
        )
        self.classifier_patch.start()
        self.addCleanup(self.classifier_patch.stop)

    def frame(self, session, trial, timestamp):
        return self.rec.predict_landmarks(
            [[0.5, 0.5, 0.0]] * 21,
            mode="letters",
            session_id=session,
            trial_id=trial,
            timestamp=timestamp,
            phase="STABLE_HOLD",
        )

    def test_new_trial_starts_clean_then_confirms_from_its_own_frames(self):
        first = self.frame("pilot-session", "trial-1", 1.0)
        second = self.frame("pilot-session", "trial-1", 1.1)
        self.assertFalse(first["confirmed"])
        self.assertTrue(second["confirmed"])

        next_trial_first = self.frame("pilot-session", "trial-2", 2.0)
        self.assertFalse(next_trial_first["confirmed"])
        next_trial_second = self.frame("pilot-session", "trial-2", 2.1)
        self.assertTrue(next_trial_second["confirmed"])

    def test_late_previous_trial_frame_cannot_vote_in_current_trial(self):
        self.frame("pilot-session", "trial-1", 1.0)
        self.frame("pilot-session", "trial-1", 1.1)

        trial2_first = self.frame("pilot-session", "trial-2", 2.0)
        self.assertFalse(trial2_first["confirmed"])

        # A delayed request remains attached to trial 1's isolated state.
        self.frame("pilot-session", "trial-1", 2.05)
        trial2_second = self.frame("pilot-session", "trial-2", 2.1)
        self.assertTrue(trial2_second["confirmed"])
        self.assertEqual(list(self.rec._pilot_temporal_states[("pilot-session", "trial-2", "letters")]["prediction_history"]), [("O", 0.95), ("O", 0.95)])

    def test_pilot_sessions_have_independent_confirmation_state(self):
        a_first = self.frame("session-a", "trial-1", 1.0)
        b_first = self.frame("session-b", "trial-1", 1.0)
        self.assertFalse(a_first["confirmed"])
        self.assertFalse(b_first["confirmed"])
        self.assertTrue(self.frame("session-a", "trial-1", 1.1)["confirmed"])
        self.assertTrue(self.frame("session-b", "trial-1", 1.1)["confirmed"])

    def test_untagged_non_pilot_smoothing_behavior_is_unchanged(self):
        first = self.rec.predict_landmarks([[0.5, 0.5, 0.0]] * 21, mode="letters", timestamp=1.0)
        second = self.rec.predict_landmarks([[0.5, 0.5, 0.0]] * 21, mode="letters", timestamp=1.1)
        self.assertFalse(first["confirmed"])
        self.assertTrue(second["confirmed"])
        self.assertEqual(list(self.rec.prediction_history), [("O", 0.95), ("O", 0.95)])


if __name__ == "__main__":
    unittest.main()
