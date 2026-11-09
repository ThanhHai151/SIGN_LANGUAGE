"""
Controlled Pilot V3 Constants and States.
Single source of truth for the 29 Vietnamese letters, state machine definitions,
durations, and schema versions.
"""

from enum import Enum
from typing import List, Set

SCHEMA_VERSION: str = "3.0.0"

# Bảng 29 chữ cái tiếng Việt VSL chuẩn hóa (SSOT)
VIETNAMESE_ALPHABET_29: List[str] = [
    "A", "Ă", "Â", "B", "C", "D", "Đ", "E", "Ê", "G", "H", "I", "K", "L",
    "M", "N", "O", "Ô", "Ơ", "P", "Q", "R", "S", "T", "U", "Ư", "V", "X", "Y"
]

VIETNAMESE_ALPHABET_SET: Set[str] = set(VIETNAMESE_ALPHABET_29)

RECORDING_DURATION_SEC: float = 5.0
COUNTDOWN_DURATION_SEC: float = 3.0
TARGET_ACCEPTED_TRIALS_PER_LETTER: int = 10
TOTAL_LETTERS: int = len(VIETNAMESE_ALPHABET_29)  # 29
TOTAL_ACCEPTED_TRIALS_FULL_SESSION: int = TOTAL_LETTERS * TARGET_ACCEPTED_TRIALS_PER_LETTER  # 290


class SessionMode(str, Enum):
    FULL_29 = "FULL_29"
    SINGLE_LETTER = "SINGLE_LETTER"


class SessionState(str, Enum):
    CREATED = "CREATED"
    READY = "READY"
    RUNNING = "RUNNING"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class TrialState(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    COUNTDOWN = "COUNTDOWN"
    RECORDING = "RECORDING"
    SAVING = "SAVING"
    COMPLETED = "COMPLETED"
    INTERRUPTED = "INTERRUPTED"
    FAILED = "FAILED"
    SKIPPED = "SKIPPED"


class HandState(str, Enum):
    NO_HAND = "NO_HAND"
    HAND_DETECTED = "HAND_DETECTED"
    LANDMARKS_INVALID = "LANDMARKS_INVALID"
    TRACKING_LOST = "TRACKING_LOST"


class RecognitionState(str, Enum):
    MATCH = "MATCH"
    MISMATCH = "MISMATCH"
    UNKNOWN = "UNKNOWN"
    NEUTRAL = "NEUTRAL"
    UNSUPPORTED = "UNSUPPORTED"
    ERROR = "ERROR"


class ProtocolPhase(str, Enum):
    PRE_PREPARATION = "PRE_PREPARATION"
    COUNTDOWN = "COUNTDOWN"
    ACTIVE = "ACTIVE"
    POST_ACTIVE = "POST_ACTIVE"


# Valid state transitions for SessionState
VALID_SESSION_TRANSITIONS = {
    SessionState.CREATED: {SessionState.READY, SessionState.STOPPED, SessionState.ERROR},
    SessionState.READY: {SessionState.RUNNING, SessionState.STOPPED, SessionState.ERROR},
    SessionState.RUNNING: {SessionState.PAUSED, SessionState.COMPLETED, SessionState.STOPPED, SessionState.ERROR},
    SessionState.PAUSED: {SessionState.RUNNING, SessionState.STOPPED, SessionState.ERROR},
    SessionState.COMPLETED: set(),
    SessionState.STOPPED: set(),
    SessionState.ERROR: {SessionState.STOPPED},
}

# Valid state transitions for TrialState
VALID_TRIAL_TRANSITIONS = {
    TrialState.PENDING: {TrialState.READY, TrialState.SKIPPED, TrialState.FAILED},
    TrialState.READY: {TrialState.COUNTDOWN, TrialState.RECORDING, TrialState.SKIPPED, TrialState.INTERRUPTED, TrialState.FAILED},
    TrialState.COUNTDOWN: {TrialState.RECORDING, TrialState.INTERRUPTED, TrialState.FAILED, TrialState.SKIPPED},
    TrialState.RECORDING: {TrialState.SAVING, TrialState.INTERRUPTED, TrialState.FAILED, TrialState.SKIPPED},
    TrialState.SAVING: {TrialState.COMPLETED, TrialState.FAILED},
    TrialState.COMPLETED: set(),
    TrialState.INTERRUPTED: {TrialState.READY, TrialState.SKIPPED},
    TrialState.FAILED: {TrialState.READY, TrialState.SKIPPED},
    TrialState.SKIPPED: set(),
}
