"""
Module VSLSequentialDigitAggregator — Bộ ghép chữ số tuần tự cho VSL Connect AI
==============================================================================
Hỗ trợ nhận diện các số có nhiều chữ số (10, 20, ..., 90 và mở rộng) bằng cách
ghép chuỗi thời gian các chữ số cơ bản (0–9) được nhận diện ổn định qua camera.

Đặc tính kỹ thuật:
- Độc lập hoàn toàn với thuật toán nhận diện hình học (VSLDigitClassifier).
- Sử dụng timestamp thực tế (giây, float) độc lập với FPS camera.
- Phân tách rõ ràng giữa kết quả tạm thời (provisional) và kết quả đã chốt (committed).
- Chống lặp chữ số khi người ký giữ nguyên tư thế lâu.
- Xử lý chuyển tiếp qua NEUTRAL hoặc hạ tay tự nhiên.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Set


class AggregatorState(str, Enum):
    IDLE = "IDLE"                  # Đang chờ cử chỉ số đầu tiên
    HOLDING = "HOLDING"            # Đang giữ một ký hiệu số, tích lũy thời gian tới T_hold
    WAITING_NEXT = "WAITING_NEXT"  # Đã ghi nhận chữ số gần nhất, chờ chữ số tiếp theo hoặc timeout
    COMMITTED = "COMMITTED"        # Đã chốt số hoàn chỉnh (do timeout hoặc đạt giới hạn chữ số)


@dataclass
class DigitAggregatorConfig:
    t_hold: float = 0.4            # Thời gian giữ tối thiểu để ghi nhận 1 chữ số (giây)
    t_timeout: float = 1.5         # Thời gian chờ tối đa giữa các chữ số trước khi chốt kết quả (giây)
    min_confidence: float = 0.70   # Ngưỡng độ tin cậy tối thiểu để chấp nhận cử chỉ
    max_digits: int = 5            # Số lượng chữ số tối đa trong một số ghép


@dataclass
class AggregatorOutput:
    state: AggregatorState
    current_digit: Optional[str]
    accumulated_digits: List[str]
    provisional_number: Optional[str]
    committed_number: Optional[str]
    is_committed: bool
    hold_progress: float
    time_since_last_digit: float
    time_until_timeout: float

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "current_digit": self.current_digit,
            "accumulated_digits": list(self.accumulated_digits),
            "provisional_number": self.provisional_number,
            "committed_number": self.committed_number,
            "is_committed": self.is_committed,
            "hold_progress": round(self.hold_progress, 2),
            "time_since_last_digit": round(self.time_since_last_digit, 2),
            "time_until_timeout": round(self.time_until_timeout, 2),
        }


class VSLSequentialDigitAggregator:
    """
    Bộ ghép chữ số tuần tự thời gian thực cho VSL.
    """

    ALLOWED_DIGITS: Set[str] = {str(i) for i in range(0, 10)}

    def __init__(self, config: Optional[DigitAggregatorConfig] = None):
        self.config = config or DigitAggregatorConfig()
        self.reset()

    def reset(self) -> None:
        """Đặt lại toàn bộ trạng thái và bộ đệm về mặc định."""
        self.state: AggregatorState = AggregatorState.IDLE
        self.accumulated_digits: List[str] = []
        self.current_candidate: Optional[str] = None
        self.candidate_first_seen: Optional[float] = None
        self.last_accepted_digit: Optional[str] = None
        self.last_digit_time: Optional[float] = None
        self.has_released_since_last_accept: bool = True
        self.last_timestamp: Optional[float] = None
        self.last_committed_number: Optional[str] = None

    def to_dict(self) -> dict:
        """Trả về snapshot trạng thái của bộ ghép số."""
        return {
            "state": self.state.value,
            "current_digit": self.current_candidate,
            "accumulated_digits": list(self.accumulated_digits),
            "provisional_number": "".join(self.accumulated_digits) if self.accumulated_digits else None,
            "committed_number": self.last_committed_number,
            "is_committed": (self.state == AggregatorState.COMMITTED),
            "hold_progress": 0.0,
            "time_since_last_digit": 0.0,
            "time_until_timeout": 0.0,
        }

    def update(
        self,
        sign: Optional[str],
        confidence: float = 0.0,
        timestamp: Optional[float] = None,
    ) -> AggregatorOutput:
        """
        Cập nhật trạng thái bộ ghép với một khung hình nhận diện số mới.

        :param sign: Nhãn nhận diện ('0'..'9', 'NEUTRAL', hoặc None)
        :param confidence: Độ tin cậy nhận diện (0.0 .. 1.0)
        :param timestamp: Dấu thời gian thực tế (giây, float). Nếu None, dùng time.time().
        :return: AggregatorOutput chứa trạng thái và số tạm thời / đã chốt.
        """
        now = time.time() if timestamp is None else float(timestamp)

        # Bảo vệ chống dấu thời gian nhảy ngược phi logic
        if self.last_timestamp is not None and now < self.last_timestamp:
            now = self.last_timestamp
        self.last_timestamp = now

        # Xác định tính hợp lệ của chữ số đầu vào
        is_valid_digit = (
            sign is not None
            and sign in self.ALLOWED_DIGITS
            and confidence >= self.config.min_confidence
        )

        committed_number: Optional[str] = None
        is_committed: bool = False

        # Nếu frame trước vừa COMMITTED, dọn sạch để sẵn sàng cho chuỗi mới
        if self.state == AggregatorState.COMMITTED:
            self.accumulated_digits = []
            self.last_digit_time = None
            if is_valid_digit and sign == self.last_accepted_digit and not self.has_released_since_last_accept:
                # Người dùng vẫn đang giữ nguyên tư thế cũ -> không bắt đầu chuỗi mới
                self.state = AggregatorState.IDLE
            else:
                self.state = AggregatorState.IDLE

        # -------------------------------------------------------------
        # 1. KIỂM TRA TIMEOUT CHO TRẠNG THÁI WAITING_NEXT
        # -------------------------------------------------------------
        if self.state in (AggregatorState.WAITING_NEXT, AggregatorState.HOLDING) and len(self.accumulated_digits) > 0:
            elapsed_since_digit = now - (self.last_digit_time if self.last_digit_time is not None else now)
            if elapsed_since_digit >= (self.config.t_timeout - 1e-4):
                # Đã quá thời gian chờ -> Chốt số hiện có
                committed_number = "".join(self.accumulated_digits)
                is_committed = True
                self.last_committed_number = committed_number
                self.state = AggregatorState.COMMITTED
                # Sau khi chốt, người dùng cần thả tay trước khi tạo lại chữ số cuối này
                if is_valid_digit and sign == self.last_accepted_digit:
                    self.has_released_since_last_accept = False

                output = AggregatorOutput(
                    state=self.state,
                    current_digit=sign if is_valid_digit else None,
                    accumulated_digits=list(self.accumulated_digits),
                    provisional_number=committed_number,
                    committed_number=committed_number,
                    is_committed=True,
                    hold_progress=0.0,
                    time_since_last_digit=elapsed_since_digit,
                    time_until_timeout=0.0,
                )
                self.accumulated_digits = []
                self.current_candidate = None
                self.candidate_first_seen = None
                return output

        # -------------------------------------------------------------
        # 2. XỬ LÝ SỰ KIỆN THẢ TAY (RELEASE) HOẶC ĐỔI KÝ HIỆU
        # -------------------------------------------------------------
        if not is_valid_digit:
            # Người dùng hạ tay hoặc đang ở tư thế trung gian (NEUTRAL)
            self.has_released_since_last_accept = True
            self.current_candidate = None
            self.candidate_first_seen = None
            if self.state == AggregatorState.HOLDING:
                if len(self.accumulated_digits) > 0:
                    self.state = AggregatorState.WAITING_NEXT
                else:
                    self.state = AggregatorState.IDLE
        else:
            # Nếu người dùng đổi sang một chữ số khác chữ số vừa được chốt -> coi như đã thả tay
            if sign != self.last_accepted_digit:
                self.has_released_since_last_accept = True

        # -------------------------------------------------------------
        # 3. TÍCH LŨY THỜI GIAN GIỮ (HOLD ACCUMULATION)
        # -------------------------------------------------------------
        hold_progress = 0.0

        if is_valid_digit:
            # Nếu người dùng vẫn giữ nguyên chữ số vừa được chấp nhận mà chưa thả tay -> Bỏ qua, chống lặp
            if sign == self.last_accepted_digit and not self.has_released_since_last_accept:
                pass
            else:
                # Bắt đầu hoặc tiếp tục tích lũy thời gian giữ cho chữ số ứng viên
                if sign != self.current_candidate:
                    self.current_candidate = sign
                    self.candidate_first_seen = now
                    self.state = AggregatorState.HOLDING
                    hold_progress = 0.0
                else:
                    hold_duration = now - (self.candidate_first_seen if self.candidate_first_seen is not None else now)
                    hold_progress = min(1.0, max(0.0, hold_duration / max(1e-5, self.config.t_hold)))

                    if hold_duration >= (self.config.t_hold - 1e-4):
                        # ĐÃ ĐẠT T_HOLD -> Ghi nhận chữ số mới vào chuỗi!
                        self.accumulated_digits.append(sign)
                        self.last_accepted_digit = sign
                        self.last_digit_time = now
                        self.has_released_since_last_accept = False
                        self.current_candidate = None
                        self.candidate_first_seen = None
                        self.state = AggregatorState.WAITING_NEXT
                        hold_progress = 1.0

                        # Nếu đạt giới hạn tối đa số chữ số -> Tự động chốt ngay lập tức
                        if len(self.accumulated_digits) >= self.config.max_digits:
                            committed_number = "".join(self.accumulated_digits)
                            is_committed = True
                            self.last_committed_number = committed_number
                            self.state = AggregatorState.COMMITTED
                            output = AggregatorOutput(
                                state=self.state,
                                current_digit=sign,
                                accumulated_digits=list(self.accumulated_digits),
                                provisional_number=committed_number,
                                committed_number=committed_number,
                                is_committed=True,
                                hold_progress=1.0,
                                time_since_last_digit=0.0,
                                time_until_timeout=0.0,
                            )
                            self.accumulated_digits = []
                            return output

        # -------------------------------------------------------------
        # 4. TÍNH TOÁN KẾT QUẢ ĐẦU RA TẠM THỜI (PROVISIONAL)
        # -------------------------------------------------------------
        if len(self.accumulated_digits) > 0:
            provisional_number = "".join(self.accumulated_digits)
        elif self.current_candidate is not None:
            provisional_number = self.current_candidate
        else:
            provisional_number = None

        time_since_last_digit = (now - self.last_digit_time) if self.last_digit_time is not None else 0.0
        time_until_timeout = (
            max(0.0, self.config.t_timeout - time_since_last_digit)
            if (self.state == AggregatorState.WAITING_NEXT and self.last_digit_time is not None)
            else self.config.t_timeout
        )

        return AggregatorOutput(
            state=self.state,
            current_digit=sign if is_valid_digit else None,
            accumulated_digits=list(self.accumulated_digits),
            provisional_number=provisional_number,
            committed_number=committed_number,
            is_committed=is_committed,
            hold_progress=hold_progress,
            time_since_last_digit=time_since_last_digit,
            time_until_timeout=time_until_timeout,
        )
