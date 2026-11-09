"""
Controlled Pilot V3 Package.
Evaluation and diagnostic testing suite for 29 Vietnamese letters.
"""

from backend.vision.controlled_pilot_v3.constants import (
    VIETNAMESE_ALPHABET_29,
    SessionMode,
    SessionState,
    TrialState,
    HandState,
    RecognitionState,
    RECORDING_DURATION_SEC,
    COUNTDOWN_DURATION_SEC,
    TARGET_ACCEPTED_TRIALS_PER_LETTER,
    TOTAL_ACCEPTED_TRIALS_FULL_SESSION,
    TOTAL_LETTERS,
    SCHEMA_VERSION,
)
from backend.vision.controlled_pilot_v3.clock import Clock, SystemClock, FakeClock
from backend.vision.controlled_pilot_v3.storage import PilotStorage
from backend.vision.controlled_pilot_v3.recognition_adapter import ControlledPilotRecognitionAdapter
from backend.vision.controlled_pilot_v3.trial_manager import TrialAttempt
from backend.vision.controlled_pilot_v3.session_manager import PilotSessionManager
from backend.vision.controlled_pilot_v3.api import router as pilot_v3_router, set_session_manager, get_session_manager

__all__ = [
    "VIETNAMESE_ALPHABET_29",
    "SessionMode",
    "SessionState",
    "TrialState",
    "HandState",
    "RecognitionState",
    "RECORDING_DURATION_SEC",
    "COUNTDOWN_DURATION_SEC",
    "TARGET_ACCEPTED_TRIALS_PER_LETTER",
    "TOTAL_ACCEPTED_TRIALS_FULL_SESSION",
    "TOTAL_LETTERS",
    "SCHEMA_VERSION",
    "Clock",
    "SystemClock",
    "FakeClock",
    "PilotStorage",
    "ControlledPilotRecognitionAdapter",
    "TrialAttempt",
    "PilotSessionManager",
    "pilot_v3_router",
    "set_session_manager",
    "get_session_manager",
]
