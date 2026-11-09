"""
Monotonic clock interface for Controlled Pilot V3.
Guarantees precise wall-clock independent timing resistant to system clock shifts,
with injectable FakeClock for testing.
"""

import time
from typing import Protocol


class Clock(Protocol):
    def monotonic(self) -> float:
        """Returns monotonic seconds."""
        ...

    def now(self) -> float:
        """Returns epoch seconds."""
        ...


class SystemClock:
    def monotonic(self) -> float:
        return time.monotonic()

    def now(self) -> float:
        return time.time()


class FakeClock:
    def __init__(self, start_epoch: float = 1700000000.0, start_mono: float = 1000.0):
        self._epoch = float(start_epoch)
        self._mono = float(start_mono)

    def monotonic(self) -> float:
        return self._mono

    def now(self) -> float:
        return self._epoch

    def advance(self, seconds: float) -> None:
        self._epoch += float(seconds)
        self._mono += float(seconds)

    def set_time(self, epoch: float, mono: float) -> None:
        self._epoch = float(epoch)
        self._mono = float(mono)
