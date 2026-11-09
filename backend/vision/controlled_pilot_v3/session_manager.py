"""
Controlled Pilot V3 Session Manager.
Single Source of Truth for session lifecycle, trial progression,
attempt retry isolation, pause/resume, session recovery, and summary generation.
"""

import collections
import time
from typing import Any, Dict, List, Optional
import uuid

from backend.vision.controlled_pilot_v3.clock import Clock, SystemClock
from backend.vision.controlled_pilot_v3.constants import (
    SCHEMA_VERSION,
    TARGET_ACCEPTED_TRIALS_PER_LETTER,
    TOTAL_ACCEPTED_TRIALS_FULL_SESSION,
    TOTAL_LETTERS,
    VIETNAMESE_ALPHABET_29,
    SessionMode,
    SessionState,
    TrialState,
    VALID_SESSION_TRANSITIONS,
)
from backend.vision.controlled_pilot_v3.recognition_adapter import ControlledPilotRecognitionAdapter
from backend.vision.controlled_pilot_v3.storage import PilotStorage
from backend.vision.controlled_pilot_v3.trial_manager import TrialAttempt


class PilotSessionManager:
    def __init__(
        self,
        adapter: ControlledPilotRecognitionAdapter,
        storage: Optional[PilotStorage] = None,
        clock: Optional[Clock] = None,
    ):
        self.adapter = adapter
        self.storage = storage or PilotStorage()
        self.clock = clock or SystemClock()

        # Session state
        self.session_id: Optional[str] = None
        self.state: SessionState = SessionState.CREATED
        self.mode: SessionMode = SessionMode.FULL_29
        self.target_letters: List[str] = list(VIETNAMESE_ALPHABET_29)
        self.created_at: Optional[float] = None
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None

        # Accounting: accepted trials per letter
        self.accepted_counts: Dict[str, int] = {}
        # Active trial tracking
        self.current_letter_idx: int = 0
        self.current_trial_number: int = 1  # 1 to 10 for the current letter
        self.current_attempt: Optional[TrialAttempt] = None
        # All completed trial attempts in this session
        self.trial_attempts: List[TrialAttempt] = []

    def create_session(
        self,
        mode: SessionMode = SessionMode.FULL_29,
        single_letter: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Creates and initializes a new controlled pilot session."""
        self.session_id = session_id or f"pilot_v3_{int(self.clock.now())}_{uuid.uuid4().hex[:6]}"
        self.mode = mode
        self.created_at = self.clock.now()

        if mode == SessionMode.SINGLE_LETTER:
            if not single_letter or single_letter not in VIETNAMESE_ALPHABET_29:
                raise ValueError(f"Invalid single letter '{single_letter}'. Must be one of 29 Vietnamese letters.")
            self.target_letters = [single_letter]
        else:
            self.target_letters = list(VIETNAMESE_ALPHABET_29)

        self.accepted_counts = {letter: 0 for letter in self.target_letters}
        self.current_letter_idx = 0
        self.current_trial_number = 1
        self.current_attempt = None
        self.trial_attempts = []
        self.state = SessionState.CREATED

        session_meta = self._build_session_metadata()
        self.storage.init_session(session_meta)
        self._append_event("SESSION_CREATED", reason=f"Session created with mode={self.mode.value}")

        self.transition_to(SessionState.READY)
        return self.get_status()

    def transition_to(self, new_state: SessionState, reason: Optional[str] = None) -> None:
        valid_next = VALID_SESSION_TRANSITIONS.get(self.state, set())
        if new_state not in valid_next:
            raise ValueError(
                f"Invalid session transition from '{self.state.value}' to '{new_state.value}'. "
                f"Valid next states: {[s.value for s in valid_next]}"
            )
        old_state = self.state
        self.state = new_state
        self._append_event(
            "SESSION_STATE_CHANGED",
            from_state=old_state.value,
            to_state=new_state.value,
            reason=reason,
        )
        self._save_session_json()

    def _append_event(self, event_type: str, **kwargs) -> None:
        if not self.session_id:
            return
        event = {
            "type": event_type,
            "session_id": self.session_id,
            "timestamp": self.clock.now(),
            **kwargs,
        }
        self.storage.append_event(self.session_id, event)

    def _save_session_json(self) -> None:
        if not self.session_id:
            return
        meta = self._build_session_metadata()
        session_path = f"{self.storage.get_session_dir(self.session_id)}/session.json"
        self.storage.write_json_atomic(session_path, meta)

    def _build_session_metadata(self) -> Dict[str, Any]:
        total_target = (
            TARGET_ACCEPTED_TRIALS_PER_LETTER
            if self.mode == SessionMode.SINGLE_LETTER
            else TOTAL_ACCEPTED_TRIALS_FULL_SESSION
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "mode": self.mode.value,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "letters": self.target_letters,
            "target_accepted_per_letter": TARGET_ACCEPTED_TRIALS_PER_LETTER,
            "total_target_accepted": total_target,
            "total_accepted_current": sum(self.accepted_counts.values()),
            "accepted_counts": dict(self.accepted_counts),
            "current_letter": self.get_current_letter(),
            "current_trial_number": self.current_trial_number,
        }

    def start_session(self) -> Dict[str, Any]:
        """Starts the session (READY -> RUNNING)."""
        if self.state != SessionState.READY:
            raise RuntimeError(f"Cannot start session from state '{self.state.value}'. Must be READY.")
        self.started_at = self.clock.now()
        self.transition_to(SessionState.RUNNING)
        return self.get_status()

    def pause_session(self) -> Dict[str, Any]:
        """
        Pauses the session.
        If a trial is actively recording, interrupts it cleanly, preserving all recorded data.
        """
        if self.state != SessionState.RUNNING:
            raise RuntimeError(f"Cannot pause session from state '{self.state.value}'. Must be RUNNING.")

        if self.current_attempt and self.current_attempt.state in (TrialState.RECORDING, TrialState.COUNTDOWN):
            self.current_attempt.interrupt(reason="Session paused by user")

        self.transition_to(SessionState.PAUSED, reason="User paused session")
        return self.get_status()

    def resume_session(self) -> Dict[str, Any]:
        """Resumes the session (PAUSED -> RUNNING)."""
        if self.state != SessionState.PAUSED:
            raise RuntimeError(f"Cannot resume session from state '{self.state.value}'. Must be PAUSED.")
        self.transition_to(SessionState.RUNNING, reason="User resumed session")
        return self.get_status()

    def stop_session(self, reason: str = "User stopped session") -> Dict[str, Any]:
        """Stops the session cleanly from RUNNING or PAUSED."""
        if self.state in (SessionState.COMPLETED, SessionState.STOPPED):
            return self.get_status()

        if self.current_attempt and self.current_attempt.state in (TrialState.RECORDING, TrialState.COUNTDOWN):
            self.current_attempt.interrupt(reason=reason)

        self.ended_at = self.clock.now()
        self.transition_to(SessionState.STOPPED, reason=reason)
        self.generate_session_summary()
        return self.get_status()

    def get_current_letter(self) -> Optional[str]:
        if 0 <= self.current_letter_idx < len(self.target_letters):
            return self.target_letters[self.current_letter_idx]
        return None

    def start_trial(self) -> Dict[str, Any]:
        """
        Prepares a new trial attempt for the current letter.
        Transitions attempt PENDING -> READY.
        """
        if self.state != SessionState.RUNNING:
            raise RuntimeError(f"Cannot start trial when session is in state '{self.state.value}'. Must be RUNNING.")

        letter = self.get_current_letter()
        if not letter:
            raise RuntimeError("No letter remaining to test; session is complete.")

        # If already at 10 accepted trials for this letter, advance first
        if self.accepted_counts[letter] >= TARGET_ACCEPTED_TRIALS_PER_LETTER:
            self._advance_to_next_letter_or_complete()
            letter = self.get_current_letter()
            if not letter:
                return self.get_status()

        trial_id = f"trial_{letter}_{self.current_trial_number}"
        attempt_id = f"attempt_{uuid.uuid4().hex[:8]}"

        self.current_attempt = TrialAttempt(
            session_id=self.session_id,
            trial_id=trial_id,
            attempt_id=attempt_id,
            expected_letter=letter,
            trial_number=self.current_trial_number,
            storage=self.storage,
            clock=self.clock,
        )
        self.current_attempt.prepare()
        return self.get_status()

    def start_trial_countdown(self) -> Dict[str, Any]:
        if not self.current_attempt:
            raise RuntimeError("No trial attempt prepared. Call start_trial first.")
        self.current_attempt.start_countdown()
        return self.get_status()

    def start_trial_recording(self) -> Dict[str, Any]:
        if not self.current_attempt:
            raise RuntimeError("No trial attempt prepared. Call start_trial first.")
        self.current_attempt.start_recording()
        return self.get_status()

    def record_frame(
        self,
        landmarks: Optional[List[List[float]]],
        handedness: str = "Right",
        tracking_confidence: Optional[float] = None,
        handedness_score: Optional[float] = None,
        client_capture_ts: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        Evaluates one frame through recognition adapter read-only,
        and saves diagnostic data to disk immediately if in RECORDING state.
        """
        letter = self.get_current_letter() or "UNKNOWN"
        frame_diag = self.adapter.evaluate_frame(
            landmarks=landmarks,
            expected_letter=letter,
            handedness=handedness,
            tracking_confidence=tracking_confidence,
            handedness_score=handedness_score,
        )

        if self.current_attempt and self.current_attempt.state == TrialState.RECORDING:
            saved_record = self.current_attempt.record_frame(frame_diag, client_capture_ts=client_capture_ts)
            # Check if duration target (5.0s) has been reached
            if self.current_attempt.is_recording_time_up():
                self.finish_trial(accept=True)
            return saved_record

        # Not recording (e.g. preview mode): return diagnostic without saving
        return {
            "session_id": self.session_id,
            "expected_letter": letter,
            "trial_state": self.current_attempt.state.value if self.current_attempt else TrialState.PENDING.value,
            **frame_diag,
        }

    def finish_trial(self, accept: bool = True) -> Dict[str, Any]:
        """
        Completes the current trial attempt and flushes summary.
        If accepted, increments backend accepted trial counter and advances progression.
        """
        if not self.current_attempt:
            raise RuntimeError("No active trial attempt to finish.")

        summary = self.current_attempt.complete(accept=accept)
        self.trial_attempts.append(self.current_attempt)
        letter = self.current_attempt.expected_letter

        if summary.get("accepted"):
            self.accepted_counts[letter] += 1
            if self.accepted_counts[letter] >= TARGET_ACCEPTED_TRIALS_PER_LETTER:
                # 10 trials completed for this letter!
                self._advance_to_next_letter_or_complete()
            else:
                self.current_trial_number = self.accepted_counts[letter] + 1

        self._save_session_json()
        return summary

    def retry_trial(self, reason: str = "User requested retry") -> Dict[str, Any]:
        """
        Retries the trial by marking the current attempt INTERRUPTED (if not already completed)
        and preparing a new attempt under the same letter and trial number.
        Preserves the previous attempt data cleanly.
        """
        if self.current_attempt and self.current_attempt.state not in (TrialState.COMPLETED, TrialState.SKIPPED):
            self.current_attempt.interrupt(reason=reason)
            self.trial_attempts.append(self.current_attempt)

        return self.start_trial()

    def skip_trial(self, reason: str = "User skipped trial") -> Dict[str, Any]:
        """
        Skips the trial. Does not count towards accepted trials.
        Advances trial number (or moves to next).
        """
        if not self.current_attempt:
            self.start_trial()

        self.current_attempt.skip(reason=reason)
        self.trial_attempts.append(self.current_attempt)
        self._save_session_json()
        return self.get_status()

    def _advance_to_next_letter_or_complete(self) -> None:
        self.current_letter_idx += 1
        self.current_trial_number = 1
        if self.current_letter_idx >= len(self.target_letters):
            # All letters completed!
            self.ended_at = self.clock.now()
            self.transition_to(SessionState.COMPLETED, reason="All target accepted trials completed")
            self.generate_session_summary()

    def generate_session_summary(self) -> Dict[str, Any]:
        """Generates cross-letter diagnostic report separating recognition accuracy and capture quality."""
        all_summaries: List[Dict[str, Any]] = []
        for att in self.trial_attempts:
            try:
                s = att.calculate_summary()
                all_summaries.append(s)
            except Exception:
                pass

        letter_reports = {}
        for letter in self.target_letters:
            letter_attempts = [s for s in all_summaries if s.get("expected_letter") == letter]
            acc_trials = [s for s in letter_attempts if s.get("accepted")]
            total_frames = sum(s.get("total_frames", 0) for s in acc_trials)
            valid_frames = sum(s.get("valid_landmark_frames", 0) for s in acc_trials)

            valid_landmark_rate = round(valid_frames / total_frames, 4) if total_frames > 0 else None

            # Aggregate predictions across accepted trials
            all_preds: List[str] = []
            for s in acc_trials:
                for p, cnt in s.get("prediction_distribution", {}).items():
                    all_preds.extend([p] * cnt)

            match_cnt = sum(1 for p in all_preds if p == letter)
            match_rate = round(match_cnt / len(all_preds), 4) if all_preds else None

            # Confused letters
            confused = collections.Counter([p for p in all_preds if p != letter and p not in ("NEUTRAL", "UNKNOWN", None)])

            # Rule failures
            all_rule_fails: List[str] = []
            for s in acc_trials:
                for rf, cnt in s.get("rule_failure_distribution", {}).items():
                    all_rule_fails.extend([rf] * cnt)
            frequent_fails = dict(collections.Counter(all_rule_fails).most_common(5))

            # Capture quality status
            if total_frames == 0:
                capture_quality = "NO_DATA"
            elif valid_landmark_rate is not None and valid_landmark_rate >= 0.85:
                capture_quality = "EXCELLENT"
            elif valid_landmark_rate is not None and valid_landmark_rate >= 0.60:
                capture_quality = "ADEQUATE"
            else:
                capture_quality = "POOR_TRACKING"

            letter_reports[letter] = {
                "letter": letter,
                "accepted_trials": len(acc_trials),
                "total_frames": total_frames,
                "valid_landmark_rate": valid_landmark_rate,
                "recognition_match_rate": match_rate,
                "most_confused_letters": dict(confused.most_common(3)),
                "frequent_rule_failures": frequent_fails,
                "data_quality_status": capture_quality,
            }

        session_summary = {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "mode": self.mode.value,
            "state": self.state.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "letters_tested": len(self.target_letters),
            "total_accepted_trials": sum(self.accepted_counts.values()),
            "letter_reports": letter_reports,
        }

        if self.session_id:
            self.storage.save_session_summary(self.session_id, session_summary)
            # Also save consolidated export
            export_data = {
                **session_summary,
                "trial_summaries": all_summaries,
            }
            self.storage.save_export(self.session_id, export_data)

        return session_summary

    def recover_session(self, session_id: str) -> Dict[str, Any]:
        """Recovers an interrupted session by inspecting on-disk data."""
        recovery_info = self.storage.recover_session_progress(session_id)
        meta = recovery_info["session_metadata"]

        self.session_id = session_id
        self.mode = SessionMode(meta.get("mode", SessionMode.FULL_29.value))
        self.target_letters = meta.get("letters", list(VIETNAMESE_ALPHABET_29))
        self.created_at = meta.get("created_at")
        self.started_at = meta.get("started_at")
        self.ended_at = meta.get("ended_at")

        # Load accepted counts
        self.accepted_counts = recovery_info["accepted_counts"]

        # Determine current letter to resume from
        self.current_letter_idx = 0
        for idx, letter in enumerate(self.target_letters):
            if self.accepted_counts[letter] < TARGET_ACCEPTED_TRIALS_PER_LETTER:
                self.current_letter_idx = idx
                break
        else:
            self.current_letter_idx = len(self.target_letters)

        letter = self.get_current_letter()
        if letter:
            self.current_trial_number = self.accepted_counts[letter] + 1
            self.state = SessionState.RUNNING
        else:
            self.current_trial_number = 10
            self.state = SessionState.COMPLETED

        self._append_event("SESSION_RECOVERED", total_accepted=recovery_info["total_accepted"])
        self._save_session_json()
        return self.get_status()

    def get_status(self) -> Dict[str, Any]:
        """Provides the Single Source of Truth status for frontend sync."""
        cur_letter = self.get_current_letter()
        cur_letter_accepted = self.accepted_counts.get(cur_letter, 0) if cur_letter else 0
        total_accepted = sum(self.accepted_counts.values())
        total_target = (
            TARGET_ACCEPTED_TRIALS_PER_LETTER
            if self.mode == SessionMode.SINGLE_LETTER
            else TOTAL_ACCEPTED_TRIALS_FULL_SESSION
        )

        recording_elapsed = 0.0
        trial_state = TrialState.PENDING.value
        trial_id = None
        attempt_id = None
        if self.current_attempt:
            trial_state = self.current_attempt.state.value
            recording_elapsed = round(self.current_attempt.get_recording_elapsed(), 2)
            trial_id = self.current_attempt.trial_id
            attempt_id = self.current_attempt.attempt_id

        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "mode": self.mode.value,
            "state": self.state.value,
            "current_letter": cur_letter,
            "current_trial_number": self.current_trial_number,
            "letter_accepted_trials": cur_letter_accepted,
            "target_accepted_per_letter": TARGET_ACCEPTED_TRIALS_PER_LETTER,
            "total_accepted_trials": total_accepted,
            "total_target_accepted": total_target,
            "accepted_counts": dict(self.accepted_counts),
            "trial_id": trial_id,
            "attempt_id": attempt_id,
            "trial_state": trial_state,
            "recording_elapsed_sec": recording_elapsed,
            "recording_duration_sec": 5.0,
            "countdown_duration_sec": 3.0,
        }
