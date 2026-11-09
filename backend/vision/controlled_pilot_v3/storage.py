"""
Storage engine for Controlled Pilot V3.
Handles directory layout, atomic file writes, JSONL appending,
schema validation, and session recovery.
"""

import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional
from backend.vision.controlled_pilot_v3.constants import (
    SCHEMA_VERSION,
    SessionState,
    TrialState,
)

DEFAULT_STORAGE_ROOT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data",
    "diagnostics",
    "controlled_pilot_v3",
)

REQUIRED_TRIAL_SUMMARY_FIELDS = [
    "schema_version",
    "session_id",
    "trial_id",
    "attempt_id",
    "expected_letter",
    "trial_number",
    "started_at",
    "ended_at",
    "duration_ms",
    "completion_status",
    "accepted",
    "total_frames",
    "hand_detected_frames",
    "no_hand_frames",
    "valid_landmark_frames",
    "prediction_distribution",
    "recognition_match_rate",
    "recognition_mismatch_rate",
    "unknown_rate",
    "neutral_rate",
    "prediction_source_distribution",
    "rule_failure_distribution",
    "quality_flags",
    "recording_errors",
]


class PilotStorage:
    def __init__(self, root_dir: Optional[str] = None):
        self.root_dir = root_dir or DEFAULT_STORAGE_ROOT
        os.makedirs(self.root_dir, exist_ok=True)

    def get_session_dir(self, session_id: str) -> str:
        return os.path.join(self.root_dir, session_id)

    def get_trials_dir(self, session_id: str) -> str:
        return os.path.join(self.get_session_dir(session_id), "trials")

    def get_attempt_dir(self, session_id: str, trial_id: str, attempt_id: str) -> str:
        return os.path.join(self.get_trials_dir(session_id), trial_id, attempt_id)

    def init_session(self, session_data: Dict[str, Any]) -> str:
        session_id = session_data["session_id"]
        sess_dir = self.get_session_dir(session_id)
        os.makedirs(sess_dir, exist_ok=True)
        os.makedirs(self.get_trials_dir(session_id), exist_ok=True)

        session_path = os.path.join(sess_dir, "session.json")
        self.write_json_atomic(session_path, session_data)
        return sess_dir

    def write_json_atomic(self, file_path: str, data: Any) -> None:
        """Writes JSON atomically by saving to a temporary file then renaming."""
        dirname = os.path.dirname(file_path)
        os.makedirs(dirname, exist_ok=True)

        # Write to temp file in same directory for same-filesystem atomic rename
        fd, tmp_path = tempfile.mkstemp(dir=dirname, suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, file_path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise

    def append_event(self, session_id: str, event: Dict[str, Any]) -> None:
        """Appends a timestamped state transition or lifecycle event to events.jsonl."""
        sess_dir = self.get_session_dir(session_id)
        os.makedirs(sess_dir, exist_ok=True)
        events_path = os.path.join(sess_dir, "events.jsonl")
        line = json.dumps(event, ensure_ascii=False) + "\n"
        with open(events_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def open_attempt(self, session_id: str, trial_id: str, attempt_id: str) -> str:
        attempt_dir = self.get_attempt_dir(session_id, trial_id, attempt_id)
        os.makedirs(attempt_dir, exist_ok=True)
        return attempt_dir

    def append_frame(self, session_id: str, trial_id: str, attempt_id: str, frame_data: Dict[str, Any]) -> None:
        attempt_dir = self.open_attempt(session_id, trial_id, attempt_id)
        frames_path = os.path.join(attempt_dir, "frames.jsonl")
        line = json.dumps(frame_data, ensure_ascii=False) + "\n"
        with open(frames_path, "a", encoding="utf-8") as f:
            f.write(line)
            f.flush()

    def save_trial_summary(self, session_id: str, trial_id: str, attempt_id: str, summary: Dict[str, Any]) -> str:
        """Validates summary schema and writes summary.json atomically."""
        self.validate_trial_summary_schema(summary)
        attempt_dir = self.open_attempt(session_id, trial_id, attempt_id)
        summary_path = os.path.join(attempt_dir, "summary.json")
        self.write_json_atomic(summary_path, summary)
        return summary_path

    def validate_trial_summary_schema(self, summary: Dict[str, Any]) -> None:
        """Ensures all required fields exist and values comply with schema rules."""
        missing = [f for f in REQUIRED_TRIAL_SUMMARY_FIELDS if f not in summary]
        if missing:
            raise ValueError(f"Trial summary missing required fields: {missing}")

        # Check that none of the rate fields use 0 when uncomputable
        if summary.get("total_frames") == 0:
            rate_fields = [
                "recognition_match_rate",
                "recognition_mismatch_rate",
                "unknown_rate",
                "neutral_rate",
            ]
            for rf in rate_fields:
                if summary.get(rf) == 0:
                    raise ValueError(
                        f"Field '{rf}' must be None (not 0) when total_frames is 0."
                    )

    def save_session_summary(self, session_id: str, summary: Dict[str, Any]) -> str:
        sess_dir = self.get_session_dir(session_id)
        path = os.path.join(sess_dir, "session_summary.json")
        self.write_json_atomic(path, summary)
        return path

    def save_export(self, session_id: str, export_data: Dict[str, Any]) -> str:
        sess_dir = self.get_session_dir(session_id)
        path = os.path.join(sess_dir, "export.json")
        self.write_json_atomic(path, export_data)
        return path

    def load_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.get_session_dir(session_id), "session.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def load_trial_summary(self, session_id: str, trial_id: str, attempt_id: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(self.get_attempt_dir(session_id, trial_id, attempt_id), "summary.json")
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def read_frames(self, session_id: str, trial_id: str, attempt_id: str) -> List[Dict[str, Any]]:
        path = os.path.join(self.get_attempt_dir(session_id, trial_id, attempt_id), "frames.jsonl")
        if not os.path.exists(path):
            return []
        frames = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    frames.append(json.loads(line))
        return frames

    def list_sessions(self) -> List[str]:
        if not os.path.exists(self.root_dir):
            return []
        return [
            d for d in os.listdir(self.root_dir)
            if os.path.isdir(os.path.join(self.root_dir, d))
            and os.path.exists(os.path.join(self.root_dir, d, "session.json"))
        ]

    def recover_session_progress(self, session_id: str) -> Dict[str, Any]:
        """
        Recovers a session by scanning on-disk trial directories and summaries.
        Returns a dictionary of accepted trials count per letter, total accepted,
        and all completed attempt references.
        """
        session_data = self.load_session(session_id)
        if not session_data:
            raise FileNotFoundError(f"Session '{session_id}' not found.")

        trials_dir = self.get_trials_dir(session_id)
        accepted_counts: Dict[str, int] = {letter: 0 for letter in session_data.get("letters", [])}
        attempts_found: List[Dict[str, Any]] = []

        if os.path.exists(trials_dir):
            for trial_id in os.listdir(trials_dir):
                trial_path = os.path.join(trials_dir, trial_id)
                if not os.path.isdir(trial_path):
                    continue
                for attempt_id in os.listdir(trial_path):
                    attempt_path = os.path.join(trial_path, attempt_id)
                    summary_path = os.path.join(attempt_path, "summary.json")
                    if os.path.exists(summary_path):
                        try:
                            with open(summary_path, "r", encoding="utf-8") as sf:
                                summary = json.load(sf)
                                attempts_found.append(summary)
                                if summary.get("accepted") and summary.get("completion_status") == "COMPLETED":
                                    letter = summary.get("expected_letter")
                                    if letter in accepted_counts:
                                        accepted_counts[letter] += 1
                        except Exception:
                            pass

        return {
            "session_id": session_id,
            "session_metadata": session_data,
            "accepted_counts": accepted_counts,
            "total_accepted": sum(accepted_counts.values()),
            "attempts_found": attempts_found,
        }
