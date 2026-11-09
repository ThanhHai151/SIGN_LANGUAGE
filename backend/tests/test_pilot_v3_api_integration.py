"""
Integration tests for Controlled Pilot V3 API Functions.
Verifies all FastAPI endpoint handlers directly:
create_session, start_session, start_trial, start_countdown, start_record,
record_frame, finish_trial, get_session_status, get_session_summary,
pause_session, resume_session, stop_session.
"""

import tempfile
import shutil
import unittest

from backend.vision.controlled_pilot_v3 import (
    set_session_manager,
    ControlledPilotRecognitionAdapter,
    PilotSessionManager,
    PilotStorage,
    FakeClock,
)
from backend.vision.controlled_pilot_v3.api import (
    create_session,
    start_session,
    start_trial,
    start_countdown,
    start_record,
    record_frame,
    finish_trial,
    get_session_status,
    get_session_summary,
    pause_session,
    resume_session,
    stop_session,
    CreateSessionRequest,
    FrameRequest,
    FinishTrialRequest,
)


class MockRecognizer:
    def __init__(self):
        self.ALLOWED_LETTERS = {"A", "B", "C", "D", "E"}
        self.alphabet_classifier = None
        self.static_alphabet_head = None
        self.has_static_head = False
        self.prediction_history = []
        self.last_confirmed_sign = None
        self.last_confirmed_time = 0.0
        self.cooldown_sec = 1.0
        self.smoothing_window = 5


class TestControlledPilotV3API(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.storage = PilotStorage(root_dir=self.test_dir)
        self.fake_clock = FakeClock()
        self.adapter = ControlledPilotRecognitionAdapter(MockRecognizer())
        self.mgr = PilotSessionManager(adapter=self.adapter, storage=self.storage, clock=self.fake_clock)
        set_session_manager(self.mgr)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_session_lifecycle_and_trial_flow_api(self):
        # 1. Create Session
        req = CreateSessionRequest(mode="SINGLE_LETTER", single_letter="A")
        resp = create_session(req)
        self.assertTrue(resp["success"])
        data = resp["data"]
        self.assertEqual(data["state"], "READY")
        self.assertEqual(data["current_letter"], "A")
        self.assertEqual(data["mode"], "SINGLE_LETTER")

        # 2. Start Session
        resp = start_session()
        self.assertTrue(resp["success"])
        self.assertEqual(resp["data"]["state"], "RUNNING")

        # 3. Start Trial (Prepare)
        resp = start_trial()
        self.assertTrue(resp["success"])
        self.assertEqual(resp["data"]["trial_state"], "READY")

        # 4. Countdown
        resp = start_countdown()
        self.assertTrue(resp["success"])
        self.assertEqual(resp["data"]["trial_state"], "COUNTDOWN")

        # 5. Record Start
        resp = start_record()
        self.assertTrue(resp["success"])
        self.assertEqual(resp["data"]["trial_state"], "RECORDING")

        # 6. Post Frame during recording
        self.fake_clock.advance(2.0)
        sample_lm = [[0.5, 0.5 - i * 0.01, 0.0] for i in range(21)]
        frame_req = FrameRequest(landmarks=sample_lm, handedness="Right")
        resp = record_frame(frame_req)
        self.assertTrue(resp["success"])
        frame_res = resp["data"]
        self.assertEqual(frame_res["hand_tracking"]["hand_detected"], True)
        self.assertEqual(frame_res["hand_tracking"]["tracking_status"], "HAND_DETECTED")

        # 7. Finish Trial after 5.0s total elapsed
        self.fake_clock.advance(3.0)
        finish_req = FinishTrialRequest(accept=True)
        resp = finish_trial(finish_req)
        self.assertTrue(resp["success"])
        summary = resp["data"]
        self.assertEqual(summary["completion_status"], "COMPLETED")
        self.assertTrue(summary["accepted"])
        self.assertEqual(summary["total_frames"], 1)

        # 8. Check Status
        resp = get_session_status()
        self.assertTrue(resp["success"])
        status = resp["data"]
        self.assertEqual(status["letter_accepted_trials"], 1)
        self.assertEqual(status["total_accepted_trials"], 1)

        # 9. Get Summary
        resp = get_session_summary()
        self.assertTrue(resp["success"])
        summary_data = resp["data"]
        self.assertIn("letter_reports", summary_data)
        self.assertIn("A", summary_data["letter_reports"])

        # 10. Pause / Resume / Stop
        resp = pause_session()
        self.assertTrue(resp["success"])
        self.assertEqual(resp["data"]["state"], "PAUSED")

        resp = resume_session()
        self.assertTrue(resp["success"])
        self.assertEqual(resp["data"]["state"], "RUNNING")

        resp = stop_session()
        self.assertTrue(resp["success"])
        self.assertEqual(resp["data"]["state"], "STOPPED")


if __name__ == "__main__":
    unittest.main()
