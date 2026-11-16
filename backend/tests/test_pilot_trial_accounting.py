import os
import sys
import tempfile
import threading
import unittest
from fastapi import HTTPException

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import backend.app as app_module
from backend.app import (
    PilotTrialRequest,
    flush_pilot_trial,
    get_pilot_trial_accounting,
    get_pilot_stats,
    open_pilot_trial,
)
from backend.vision.pilot_logger import PilotDiagnosticLogger


class TestPilotTrialAccounting(unittest.TestCase):
    SESSION = "pilot_accounting_session"
    TRIAL_A = "pilot_accounting_session_O_seq000001_rep01"
    TRIAL_B = "pilot_accounting_session_O_seq000001_rep02"

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
        self.logger = PilotDiagnosticLogger(filepath=os.path.join(self.temp_dir.name, "trial.jsonl"))
        self.addCleanup(self.logger.close)
        self.recognizer.pilot_logger = self.logger
        self.recognizer.pilot_logging_enabled = True
        self.recognizer.active_pilot_session_id = self.SESSION
        self.recognizer.pilot_filepath = self.logger.default_filepath

    def record(self, trial_id=None, frame_idx=1, session_id=None):
        return {"session_id": session_id or self.SESSION, "trial_id": trial_id, "frame_idx": frame_idx}

    def open(self, trial_id=None):
        return open_pilot_trial(PilotTrialRequest(session_id=self.SESSION, trial_id=trial_id or self.TRIAL_A))

    def test_single_trial_is_accounted_by_backend_enqueue_and_write(self):
        self.open()
        self.assertTrue(self.logger.log_frame(self.record(self.TRIAL_A, 1)))
        self.assertTrue(self.logger.log_frame(self.record(self.TRIAL_A, 2)))
        result = flush_pilot_trial(PilotTrialRequest(session_id=self.SESSION, trial_id=self.TRIAL_A))
        self.assertTrue(result["success"])
        self.assertEqual(result["stats"]["queued_count"], 2)
        self.assertEqual(result["stats"]["logged_count"], 2)
        self.assertEqual(result["stats"]["pending_count"], 0)

    def test_out_of_trial_records_interleaved_do_not_change_trial_scope(self):
        self.open()
        self.logger.log_frame(self.record(None, 1))
        self.logger.log_frame(self.record(self.TRIAL_A, 2))
        self.logger.log_frame(self.record(None, 3))
        result = flush_pilot_trial(PilotTrialRequest(session_id=self.SESSION, trial_id=self.TRIAL_A))
        self.assertEqual(result["stats"]["queued_count"], 1)
        self.assertEqual(self.logger.get_stats()["out_of_trial"]["queued_count"], 2)

    def test_two_trials_in_one_session_have_independent_counters(self):
        self.open(self.TRIAL_A)
        self.open(self.TRIAL_B)
        self.logger.log_frame(self.record(self.TRIAL_A, 1))
        self.logger.log_frame(self.record(self.TRIAL_B, 1))
        self.logger.log_frame(self.record(self.TRIAL_A, 2))
        self.logger.flush()
        self.assertEqual(self.logger.get_trial_stats(self.SESSION, self.TRIAL_A)["logged_count"], 2)
        self.assertEqual(self.logger.get_trial_stats(self.SESSION, self.TRIAL_B)["logged_count"], 1)

    def test_distinct_session_trial_key_is_kept_separate(self):
        self.open()
        other_session = "pilot_other_session"
        other_trial = "pilot_other_session_O_rep01"
        self.logger.log_frame(self.record(self.TRIAL_A, 1))
        self.logger.log_frame(self.record(other_trial, 1, other_session))
        self.logger.flush()
        self.assertEqual(self.logger.get_trial_stats(self.SESSION, self.TRIAL_A)["logged_count"], 1)
        self.assertEqual(self.logger.get_trial_stats(other_session, other_trial)["logged_count"], 1)

    def test_unscoped_records_are_explicit_and_excluded_from_trial_counters(self):
        self.open()
        self.logger.log_frame({"frame_idx": 1})
        self.logger.flush()
        self.assertEqual(self.logger.get_stats()["out_of_trial"]["logged_count"], 1)
        self.assertEqual(self.logger.get_trial_stats(self.SESSION, self.TRIAL_A)["logged_count"], 0)

    def test_registration_rejects_duplicate_trial_id(self):
        self.open()
        with self.assertRaises(HTTPException) as ctx:
            self.open()
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.detail["error_code"], "duplicate_trial_id")

    def test_empty_trial_is_distinguished_from_missing_trial(self):
        self.open()
        flush = flush_pilot_trial(PilotTrialRequest(session_id=self.SESSION, trial_id=self.TRIAL_A))
        self.assertFalse(flush["success"])
        self.assertEqual(flush["error_code"], "trial_has_no_records")
        with self.assertRaises(HTTPException) as ctx:
            get_pilot_trial_accounting(self.SESSION, self.TRIAL_B)
        self.assertEqual(ctx.exception.status_code, 404)
        self.assertEqual(ctx.exception.detail["error_code"], "trial_not_found")

    def test_trial_flush_failure_is_not_reported_as_verified(self):
        self.open()
        self.logger.log_frame(self.record(self.TRIAL_A))
        from unittest.mock import patch
        with patch.object(self.logger, "flush", return_value=False):
            result = flush_pilot_trial(PilotTrialRequest(session_id=self.SESSION, trial_id=self.TRIAL_A))
        self.assertFalse(result["success"])
        self.assertFalse(result["flush_success"])
        self.assertEqual(result["error_code"], "flush_failed")

    def test_pending_count_includes_record_currently_being_written(self):
        self.open()
        entered, release = threading.Event(), threading.Event()
        self.logger._test_worker_pre_write_hook = lambda *_: (entered.set(), release.wait(2))
        self.logger.log_frame(self.record(self.TRIAL_A))
        self.assertTrue(entered.wait(1))
        self.assertEqual(self.logger.get_trial_stats(self.SESSION, self.TRIAL_A)["pending_count"], 1)
        release.set()
        self.assertTrue(self.logger.flush())
        self.assertEqual(self.logger.get_trial_stats(self.SESSION, self.TRIAL_A)["pending_count"], 0)

    def test_queue_full_drop_is_counted_in_trial_scope(self):
        self.logger.close()
        self.logger = PilotDiagnosticLogger(filepath=os.path.join(self.temp_dir.name, "small.jsonl"), queue_size=1)
        self.addCleanup(self.logger.close)
        self.recognizer.pilot_logger = self.logger
        self.open()
        entered, release = threading.Event(), threading.Event()
        self.logger._test_worker_pre_write_hook = lambda *_: (entered.set(), release.wait(2))
        self.assertTrue(self.logger.log_frame(self.record(self.TRIAL_A, 1)))
        self.assertTrue(entered.wait(1))
        self.assertTrue(self.logger.log_frame(self.record(self.TRIAL_A, 2)))
        self.assertFalse(self.logger.log_frame(self.record(self.TRIAL_A, 3)))
        release.set()
        self.logger.flush()
        stats = self.logger.get_trial_stats(self.SESSION, self.TRIAL_A)
        self.assertEqual(stats["dropped_queue_full"], 1)
        self.assertEqual(stats["total_dropped"], 1)

    def test_worker_io_error_decrements_pending_and_counts_drop(self):
        self.open()
        self.logger._test_worker_pre_write_hook = lambda *_: (_ for _ in ()).throw(OSError("simulated write failure"))
        self.logger.log_frame(self.record(self.TRIAL_A))
        self.logger.flush()
        stats = self.logger.get_trial_stats(self.SESSION, self.TRIAL_A)
        self.assertEqual(stats["dropped_io_error"], 1)
        self.assertEqual(stats["pending_count"], 0)

    def test_late_record_after_seal_invalidates_previously_flushed_trial(self):
        self.open()
        self.logger.log_frame(self.record(self.TRIAL_A))
        self.assertTrue(flush_pilot_trial(PilotTrialRequest(session_id=self.SESSION, trial_id=self.TRIAL_A))["success"])
        self.logger.log_frame(self.record(self.TRIAL_A, 2))
        self.logger.flush()
        result = get_pilot_trial_accounting(self.SESSION, self.TRIAL_A)
        self.assertFalse(result["trial_verified"])
        self.assertEqual(result["error_code"], "late_record_after_seal")
        self.assertEqual(result["stats"]["late_record_count"], 1)

    def test_session_mismatch_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            get_pilot_trial_accounting("pilot_other_session", self.TRIAL_A)
        self.assertEqual(ctx.exception.status_code, 409)
        self.assertEqual(ctx.exception.detail["error_code"], "session_mismatch")

    def test_invalid_trial_id_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            open_pilot_trial(PilotTrialRequest(session_id=self.SESSION, trial_id="../escape"))
        self.assertEqual(ctx.exception.status_code, 400)
        self.assertEqual(ctx.exception.detail["error_code"], "invalid_trial_id")

    def test_stopped_worker_is_reported_as_logger_unavailable(self):
        self.logger.close()
        with self.assertRaises(HTTPException) as ctx:
            self.open()
        self.assertEqual(ctx.exception.status_code, 503)
        self.assertEqual(ctx.exception.detail["error_code"], "logger_unavailable")

    def test_trial_sidecars_remain_isolated_and_duplicate_write_is_refused(self):
        from backend.vision.pilot_logger import save_sidecar_sync
        one = save_sidecar_sync(self.SESSION, {"session_id": self.SESSION, "trial_id": self.TRIAL_A}, self.temp_dir.name)
        two = save_sidecar_sync(self.SESSION, {"session_id": self.SESSION, "trial_id": self.TRIAL_B}, self.temp_dir.name)
        self.assertNotEqual(one, two)
        with self.assertRaises(FileExistsError):
            save_sidecar_sync(self.SESSION, {"session_id": self.SESSION, "trial_id": self.TRIAL_A}, self.temp_dir.name)

    def test_legacy_session_stats_keep_old_fields_and_add_out_of_trial_scope(self):
        self.logger.log_frame(self.record(None))
        result = get_pilot_stats()
        self.assertTrue(result["success"])
        self.assertEqual(result["stats"]["queued_count"], 1)
        self.assertEqual(result["stats"]["out_of_trial"]["queued_count"], 1)


if __name__ == "__main__":
    unittest.main()
