"""
Controlled Pilot V3 Trial Manager.
Manages individual trial lifecycle, attempt isolation, monotonic recording timer,
per-frame recording, flush contracts, and trial summary statistics calculation.
"""

import collections
from typing import Any, Dict, List, Optional
import uuid

from backend.vision.controlled_pilot_v3.clock import Clock, SystemClock
from backend.vision.controlled_pilot_v3.constants import (
    COUNTDOWN_DURATION_SEC,
    RECORDING_DURATION_SEC,
    SCHEMA_VERSION,
    HandState,
    ProtocolPhase,
    RecognitionState,
    TrialState,
    VALID_TRIAL_TRANSITIONS,
)
from backend.vision.controlled_pilot_v3.storage import PilotStorage


class TrialAttempt:
    def __init__(
        self,
        session_id: str,
        trial_id: str,
        attempt_id: str,
        expected_letter: str,
        trial_number: int,
        storage: PilotStorage,
        clock: Optional[Clock] = None,
    ):
        self.session_id = session_id
        self.trial_id = trial_id
        self.attempt_id = attempt_id
        self.expected_letter = expected_letter
        self.trial_number = trial_number
        self.storage = storage
        self.clock = clock or SystemClock()

        self.state = TrialState.PENDING
        self.phase = ProtocolPhase.PRE_PREPARATION

        # Timers (monotonic)
        self.countdown_start_mono: Optional[float] = None
        self.recording_start_mono: Optional[float] = None
        self.recording_end_mono: Optional[float] = None

        # Timestamps (epoch)
        self.started_at: Optional[float] = None
        self.ended_at: Optional[float] = None

        # Recorded frames cache for summary calculation
        self.frames: List[Dict[str, Any]] = []
        self.frame_counter: int = 0
        self.recording_errors: List[str] = []
        self.quality_flags: List[str] = []
        self.accepted: bool = False

    def transition_to(self, new_state: TrialState, reason: Optional[str] = None) -> None:
        valid_next = VALID_TRIAL_TRANSITIONS.get(self.state, set())
        if new_state not in valid_next:
            raise ValueError(
                f"Invalid trial transition from '{self.state.value}' to '{new_state.value}'. "
                f"Valid next states: {[s.value for s in valid_next]}"
            )

        old_state = self.state
        self.state = new_state
        now_epoch = self.clock.now()

        event = {
            "type": "TRIAL_STATE_CHANGED",
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "attempt_id": self.attempt_id,
            "from_state": old_state.value,
            "to_state": new_state.value,
            "reason": reason,
            "timestamp": now_epoch,
        }
        self.storage.append_event(self.session_id, event)

    def prepare(self) -> None:
        """Transitions PENDING -> READY."""
        self.transition_to(TrialState.READY)
        self.phase = ProtocolPhase.PRE_PREPARATION

    def start_countdown(self) -> None:
        """Transitions READY -> COUNTDOWN."""
        self.transition_to(TrialState.COUNTDOWN)
        self.phase = ProtocolPhase.COUNTDOWN
        self.countdown_start_mono = self.clock.monotonic()

    def start_recording(self) -> None:
        """
        Transitions COUNTDOWN (or READY) -> RECORDING.
        Initializes monotonic recording clock. Countdown is strictly excluded.
        """
        self.transition_to(TrialState.RECORDING)
        self.phase = ProtocolPhase.ACTIVE
        self.recording_start_mono = self.clock.monotonic()
        self.started_at = self.clock.now()

    def get_recording_elapsed(self) -> float:
        if self.recording_start_mono is None:
            return 0.0
        if self.recording_end_mono is not None:
            return max(0.0, self.recording_end_mono - self.recording_start_mono)
        return max(0.0, self.clock.monotonic() - self.recording_start_mono)

    def is_recording_time_up(self) -> bool:
        if self.state != TrialState.RECORDING:
            return False
        return self.get_recording_elapsed() >= RECORDING_DURATION_SEC

    def record_frame(self, frame_diag: Dict[str, Any], client_capture_ts: Optional[float] = None) -> Dict[str, Any]:
        """
        Records a frame during RECORDING state.
        Attaches monotonic timing and frame metadata, then writes to disk immediately.
        """
        if self.state != TrialState.RECORDING:
            raise RuntimeError(f"Cannot record frame in trial state '{self.state.value}'. Must be RECORDING.")

        now_mono = self.clock.monotonic()
        elapsed_sec = now_mono - (self.recording_start_mono or now_mono)
        elapsed_ms = round(elapsed_sec * 1000.0, 2)
        self.frame_counter += 1

        frame_record = {
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "attempt_id": self.attempt_id,
            "expected_letter": self.expected_letter,
            "frame_index": self.frame_counter,
            "timestamp": self.clock.now(),
            "elapsed_ms": elapsed_ms,
            "trial_state": self.state.value,
            "capture_phase": self.phase.value,
            "observed_phase": "UNKNOWN",  # SSOT: UNKNOWN unless verified ground truth kinematics algorithm
            "client_capture_ts": client_capture_ts,
            **frame_diag,
        }

        self.frames.append(frame_record)
        self.storage.append_frame(self.session_id, self.trial_id, self.attempt_id, frame_record)

        # Check if duration target reached
        if elapsed_sec >= RECORDING_DURATION_SEC:
            self.recording_end_mono = now_mono

        return frame_record

    def interrupt(self, reason: str) -> None:
        """Interrupts trial (e.g. paused, camera disconnect, user cancel). Preserves recorded data."""
        if self.state in (TrialState.COMPLETED, TrialState.SKIPPED):
            return
        if self.recording_start_mono is not None and self.recording_end_mono is None:
            self.recording_end_mono = self.clock.monotonic()
        self.ended_at = self.clock.now()
        self.recording_errors.append(f"Interrupted: {reason}")
        self.transition_to(TrialState.INTERRUPTED, reason=reason)
        # Save summary marking INTERRUPTED
        self._flush_and_save_summary(accepted=False)

    def skip(self, reason: str = "User skipped") -> None:
        """Skips trial without counting it as accepted."""
        if self.recording_start_mono is not None and self.recording_end_mono is None:
            self.recording_end_mono = self.clock.monotonic()
        self.ended_at = self.clock.now()
        self.transition_to(TrialState.SKIPPED, reason=reason)
        self._flush_and_save_summary(accepted=False)

    def complete(self, accept: bool = True) -> Dict[str, Any]:
        """
        Completes the 5.0s recording trial.
        Transitions RECORDING -> SAVING -> COMPLETED.
        Flushes and saves trial summary atomically.
        """
        if self.state == TrialState.COMPLETED:
            return self.calculate_summary()

        if self.recording_end_mono is None:
            self.recording_end_mono = self.clock.monotonic()
        self.ended_at = self.clock.now()

        # Check exact duration
        duration_sec = self.get_recording_elapsed()
        if duration_sec < (RECORDING_DURATION_SEC - 0.15):  # Tolerates negligible monotonic boundary
            self.quality_flags.append(f"SHORT_RECORDING: {duration_sec:.2f}s < 5.0s")
            accept = False

        self.transition_to(TrialState.SAVING)
        self.transition_to(TrialState.COMPLETED)
        summary = self._flush_and_save_summary(accepted=accept)
        return summary

    def _flush_and_save_summary(self, accepted: bool) -> Dict[str, Any]:
        self.accepted = accepted
        summary = self.calculate_summary()
        self.storage.save_trial_summary(self.session_id, self.trial_id, self.attempt_id, summary)
        return summary

    def calculate_summary(self) -> Dict[str, Any]:
        total_frames = len(self.frames)
        duration_ms = round(self.get_recording_elapsed() * 1000.0, 1)

        hand_detected_frames = sum(
            1 for f in self.frames if f.get("hand_tracking", {}).get("hand_detected", False)
        )
        no_hand_frames = total_frames - hand_detected_frames

        valid_landmark_frames = sum(
            1 for f in self.frames
            if f.get("hand_tracking", {}).get("tracking_status") == HandState.HAND_DETECTED.value
        )

        # Predictions distribution
        predictions = [
            f.get("final_prediction") for f in self.frames
            if f.get("final_prediction") is not None
        ]
        pred_counter = collections.Counter(predictions)
        prediction_distribution = dict(pred_counter)

        # Sources distribution
        sources = [
            f.get("prediction_source") for f in self.frames
            if f.get("prediction_source") is not None
        ]
        source_counter = collections.Counter(sources)
        prediction_source_distribution = dict(source_counter)

        # Rule failures distribution
        rule_fails: List[str] = []
        for f in self.frames:
            reason = f.get("rule_diagnostics", {}).get("reason") if isinstance(f.get("rule_diagnostics"), dict) else None
            if reason:
                rule_fails.append(reason)
        rule_failure_distribution = dict(collections.Counter(rule_fails))

        # Rates: DO NOT use 0 if total_frames == 0 (must be None / null)
        if total_frames > 0:
            match_count = sum(1 for p in predictions if p == self.expected_letter)
            mismatch_count = sum(
                1 for p in predictions
                if p != self.expected_letter and p not in ("NEUTRAL", "UNKNOWN", None)
            )
            unknown_count = sum(1 for p in predictions if p == "UNKNOWN")
            neutral_count = sum(1 for p in predictions if p == "NEUTRAL")

            recognition_match_rate = round(match_count / total_frames, 4)
            recognition_mismatch_rate = round(mismatch_count / total_frames, 4)
            unknown_rate = round(unknown_count / total_frames, 4)
            neutral_rate = round(neutral_count / total_frames, 4)
        else:
            recognition_match_rate = None
            recognition_mismatch_rate = None
            unknown_rate = None
            neutral_rate = None

        # Quality flags
        quality_flags = list(self.quality_flags)
        if total_frames == 0:
            quality_flags.append("NO_FRAMES_CAPTURED")
        elif valid_landmark_frames == 0:
            quality_flags.append("NO_VALID_LANDMARKS")
        elif valid_landmark_frames < (total_frames * 0.5):
            quality_flags.append("POOR_TRACKING_UNDER_50_PCT")

        return {
            "schema_version": SCHEMA_VERSION,
            "session_id": self.session_id,
            "trial_id": self.trial_id,
            "attempt_id": self.attempt_id,
            "expected_letter": self.expected_letter,
            "trial_number": self.trial_number,
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "duration_ms": duration_ms,
            "completion_status": self.state.value,
            "accepted": self.accepted,
            "total_frames": total_frames,
            "hand_detected_frames": hand_detected_frames,
            "no_hand_frames": no_hand_frames,
            "valid_landmark_frames": valid_landmark_frames,
            "prediction_distribution": prediction_distribution,
            "recognition_match_rate": recognition_match_rate,
            "recognition_mismatch_rate": recognition_mismatch_rate,
            "unknown_rate": unknown_rate,
            "neutral_rate": neutral_rate,
            "prediction_source_distribution": prediction_source_distribution,
            "rule_failure_distribution": rule_failure_distribution,
            "quality_flags": quality_flags,
            "recording_errors": list(self.recording_errors),
        }
