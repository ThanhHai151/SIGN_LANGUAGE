import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from fastapi import HTTPException

from backend.app import PilotControlRequest, control_pilot_logging


class TestPilotSessionControl(unittest.TestCase):
    def test_stop_refuses_to_close_a_different_active_session(self):
        fake_recognizer = SimpleNamespace(
            pilot_logging_enabled=True,
            active_pilot_session_id="pilot_active",
            disable_pilot_logging=Mock(),
        )
        with patch("backend.app.recognizer", fake_recognizer):
            with self.assertRaises(HTTPException) as raised:
                control_pilot_logging(PilotControlRequest(enabled=False, session_id="pilot_stale"))
        self.assertEqual(raised.exception.status_code, 409)
        fake_recognizer.disable_pilot_logging.assert_not_called()

    def test_stop_acknowledges_the_session_that_was_closed(self):
        fake_recognizer = SimpleNamespace(
            pilot_logging_enabled=True,
            active_pilot_session_id="pilot_active",
            disable_pilot_logging=Mock(),
        )
        with patch("backend.app.recognizer", fake_recognizer):
            response = control_pilot_logging(PilotControlRequest(enabled=False, session_id="pilot_active"))
        self.assertEqual(response, {"success": True, "enabled": False, "session_id": "pilot_active"})
        fake_recognizer.disable_pilot_logging.assert_called_once_with()

    def test_enable_returns_requested_session_and_unique_file_path(self):
        logger = Mock()
        logger.get_stats.return_value = {"queued_count": 0, "logged_count": 0, "total_dropped": 0}
        fake_recognizer = SimpleNamespace(
            pilot_logging_enabled=False,
            active_pilot_session_id=None,
            diagnostic_dir="unused-in-test",
            enable_pilot_logging=Mock(return_value=logger),
        )
        with patch("backend.app.recognizer", fake_recognizer), patch(
            "backend.vision.pilot_logger.create_safe_pilot_file",
            return_value="pilot_new_unique.jsonl",
        ):
            response = control_pilot_logging(PilotControlRequest(enabled=True, session_id="pilot_new"))
        self.assertTrue(response["success"])
        self.assertTrue(response["enabled"])
        self.assertEqual(response["session_id"], "pilot_new")
        self.assertEqual(response["filepath"], "pilot_new_unique.jsonl")
        fake_recognizer.enable_pilot_logging.assert_called_once_with(session_id="pilot_new", filepath="pilot_new_unique.jsonl")


if __name__ == "__main__":
    unittest.main()
