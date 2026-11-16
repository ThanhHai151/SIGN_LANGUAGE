import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import backend.app as app_module
from backend.app import PilotFlushRequest, flush_pilot_log
from backend.vision.pilot_logger import PilotDiagnosticLogger, save_sidecar_sync


class TestPilotFlushContract(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=os.path.dirname(__file__))
        self.recognizer = app_module.recognizer
        self.original = {
            "pilot_logger": getattr(self.recognizer, "pilot_logger", None),
            "pilot_logging_enabled": getattr(self.recognizer, "pilot_logging_enabled", False),
            "active_pilot_session_id": getattr(self.recognizer, "active_pilot_session_id", None),
            "pilot_filepath": getattr(self.recognizer, "pilot_filepath", None),
        }
        self.addCleanup(self.temp_dir.cleanup)
        for name, value in self.original.items():
            self.addCleanup(setattr, self.recognizer, name, value)

    def set_recognizer(self, logger, *, enabled=True, session_id="pilot_test_session", filepath="pilot_test_session.jsonl"):
        self.recognizer.pilot_logger = logger
        self.recognizer.pilot_logging_enabled = enabled
        self.recognizer.active_pilot_session_id = session_id
        self.recognizer.pilot_filepath = filepath

    def fake_logger(self, *, result=True, stats=None):
        default_stats = {
            "queued_count": 0,
            "logged_count": 0,
            "pending_in_queue": 0,
            "dropped_queue_full": 0,
            "dropped_io_error": 0,
            "total_dropped": 0,
        }
        current = {**default_stats, **(stats or {})}
        return SimpleNamespace(
            STATE_OPEN="OPEN",
            state="OPEN",
            is_running=True,
            flush=lambda: result,
            get_stats=lambda: dict(current),
        )

    def test_flush_success_preserves_legacy_fields_and_reports_barrier_semantics(self):
        logger = PilotDiagnosticLogger(filepath=os.path.join(self.temp_dir.name, "ok.jsonl"))
        self.addCleanup(logger.close)
        self.set_recognizer(logger, filepath=logger.default_filepath)
        self.assertTrue(logger.log_frame({"frame_idx": 1}, filepath=logger.default_filepath))

        response = flush_pilot_log(PilotFlushRequest(session_id="pilot_test_session"))

        self.assertTrue(response["success"])
        self.assertTrue(response["flush_success"])
        self.assertTrue(response["flush_attempted"])
        self.assertEqual(response["session_id"], "pilot_test_session")
        self.assertEqual(response["requested_session_id"], "pilot_test_session")
        self.assertEqual(response["counter_scope"], "session_cumulative")
        self.assertEqual(response["filepath"], logger.default_filepath)
        self.assertEqual(response["queued_count"], 1)
        self.assertEqual(response["logged_count"], 1)
        self.assertEqual(response["pending_in_queue"], 0)
        self.assertEqual(response["total_dropped"], 0)
        self.assertIn("does not guarantee fsync", response["flush_semantics"])
        self.assertEqual(response["stats"]["logged_count"], 1)

    def test_flush_false_is_not_reported_as_success(self):
        self.set_recognizer(self.fake_logger(result=False))
        response = flush_pilot_log()
        self.assertFalse(response["success"])
        self.assertFalse(response["flush_success"])
        self.assertEqual(response["error_code"], "flush_failed")

    def test_flush_false_with_new_dropped_record_is_classified_as_io_error(self):
        before = {"queued_count": 1, "logged_count": 0, "pending_in_queue": 1, "dropped_queue_full": 0, "dropped_io_error": 0, "total_dropped": 0}
        after = {**before, "dropped_io_error": 1, "total_dropped": 1}
        snapshots = iter([before, after])
        logger = SimpleNamespace(
            STATE_OPEN="OPEN", state="OPEN", is_running=True,
            flush=lambda: False,
            get_stats=lambda: dict(next(snapshots)),
        )
        self.set_recognizer(logger)
        response = flush_pilot_log()
        self.assertFalse(response["success"])
        self.assertFalse(response["flush_success"])
        self.assertEqual(response["error_code"], "io_error")
        self.assertEqual(response["dropped_io_error"], 1)

    def test_missing_logger_is_unknown_failure_with_counters_null(self):
        self.set_recognizer(None, enabled=False, session_id=None, filepath=None)
        response = flush_pilot_log()
        self.assertFalse(response["success"])
        self.assertFalse(response["flush_success"])
        self.assertFalse(response["flush_attempted"])
        self.assertEqual(response["error_code"], "logger_unavailable")
        self.assertIsNone(response["queued_count"])
        self.assertIsNone(response["logged_count"])
        self.assertIsNone(response["pending_in_queue"])

    def test_logging_disabled_is_distinct_from_missing_logger(self):
        self.set_recognizer(self.fake_logger(), enabled=False)
        response = flush_pilot_log()
        self.assertFalse(response["success"])
        self.assertFalse(response["flush_attempted"])
        self.assertEqual(response["error_code"], "logging_disabled")

    def test_requested_session_mismatch_is_rejected_before_flush(self):
        logger = self.fake_logger()
        self.set_recognizer(logger)
        response = flush_pilot_log(PilotFlushRequest(session_id="pilot_other_session"))
        self.assertFalse(response["success"])
        self.assertFalse(response["flush_attempted"])
        self.assertEqual(response["error_code"], "session_mismatch")
        self.assertEqual(response["requested_session_id"], "pilot_other_session")
        self.assertEqual(response["session_id"], "pilot_test_session")

    def test_nonempty_queue_prevents_overall_success_even_when_barrier_returns_true(self):
        self.set_recognizer(self.fake_logger(stats={"pending_in_queue": 1}))
        response = flush_pilot_log()
        self.assertTrue(response["flush_success"])
        self.assertFalse(response["success"])
        self.assertEqual(response["error_code"], "record_accounting_mismatch")

    def test_queued_logged_mismatch_prevents_overall_success(self):
        self.set_recognizer(self.fake_logger(stats={"queued_count": 3, "logged_count": 2}))
        response = flush_pilot_log()
        self.assertTrue(response["flush_success"])
        self.assertFalse(response["success"])
        self.assertEqual(response["error_code"], "record_accounting_mismatch")

    def test_dropped_records_are_reported_as_io_error(self):
        self.set_recognizer(self.fake_logger(stats={"dropped_io_error": 1, "total_dropped": 1}))
        response = flush_pilot_log()
        self.assertTrue(response["flush_success"])
        self.assertFalse(response["success"])
        self.assertEqual(response["error_code"], "io_error")
        self.assertEqual(response["total_dropped"], 1)

    def test_dead_logger_worker_cannot_claim_empty_queue_flush_success(self):
        logger = PilotDiagnosticLogger(filepath=os.path.join(self.temp_dir.name, "dead.jsonl"))
        logger.close()
        self.set_recognizer(logger)
        response = flush_pilot_log()
        self.assertFalse(response["success"])
        self.assertFalse(response["flush_attempted"])
        self.assertEqual(response["error_code"], "logger_unavailable")

    def test_trial_sidecars_are_unique_and_duplicate_trial_cannot_overwrite(self):
        first = {"session_id": "pilot_shared_session", "trial_id": "pilot_shared_session_O_rep01", "diag": {"id": 1}}
        second = {"session_id": "pilot_shared_session", "trial_id": "pilot_shared_session_O_rep02", "diag": {"id": 2}}
        first_path = save_sidecar_sync("pilot_shared_session", first, self.temp_dir.name)
        second_path = save_sidecar_sync("pilot_shared_session", second, self.temp_dir.name)
        self.assertNotEqual(first_path, second_path)
        with open(first_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["diag"]["id"], 1)
        with open(second_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["diag"]["id"], 2)
        with self.assertRaises(FileExistsError):
            save_sidecar_sync("pilot_shared_session", {**first, "diag": {"id": 99}}, self.temp_dir.name)
        with open(first_path, encoding="utf-8") as handle:
            self.assertEqual(json.load(handle)["diag"]["id"], 1)

    def test_invalid_trial_id_is_rejected_and_legacy_session_sidecar_name_remains(self):
        with self.assertRaises(ValueError):
            save_sidecar_sync("pilot_shared_session", {"trial_id": "../outside"}, self.temp_dir.name)
        legacy_path = save_sidecar_sync("pilot_shared_session", {"schema_version": "old"}, self.temp_dir.name)
        self.assertTrue(legacy_path.endswith("pilot_pilot_shared_session_sync.json"))
        self.assertTrue(os.path.isfile(legacy_path))


if __name__ == "__main__":
    unittest.main()
