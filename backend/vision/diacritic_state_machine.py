"""
Máy trạng thái phân giải dấu cho 7 chữ cái tiếng Việt VSL (DiacriticStateMachine).
==================================================================================
Dự án: VSL Connect AI (Task 7C)
Kiến trúc: Hybrid Static + Kinematic Finite State Machine (FSM).

Phân giải 5 nhóm chữ nền:
- A -> {A, Ă, Â}
- E -> {E, Ê}
- O -> {O, Ô, Ơ}
- U -> {U, Ư}
- D -> {D, Đ}

Độc lập hoàn toàn, không can thiệp vào VSLRecognizer, geometric rules hay static head.
"""

import math
import time
from typing import Dict, Any, List, Optional, Tuple
from collections import deque
from dataclasses import dataclass

from backend.vision.feature_extractor import (
    calculate_distance,
    calculate_angle,
    compute_finger_states,
    _shift_and_scale
)

# Danh sách 5 nhóm chữ cái nền hợp lệ
BASE_GROUPS = {
    "A": {"base": "A", "variants": {"A", "Ă", "Â"}, "diacritics": ["Ă", "Â"]},
    "E": {"base": "E", "variants": {"E", "Ê"}, "diacritics": ["Ê"]},
    "O": {"base": "O", "variants": {"O", "Ô", "Ơ"}, "diacritics": ["Ô", "Ơ"]},
    "U": {"base": "U", "variants": {"U", "Ư"}, "diacritics": ["Ư"]},
    "D": {"base": "D", "variants": {"D", "Đ"}, "diacritics": ["Đ"]},
}

# Các quy tắc hình học tĩnh có nguy cơ cạnh tranh/cướp nhãn khi tạo dấu (Task 7E.4A)
COMPETING_RULES_MAP = {
    "O": {"I", "Y"},
    "U": {"I", "Y"},
    "A": {"V"},
}

# Các trạng thái của FSM
STATE_IDLE = "IDLE"
STATE_BASE_CANDIDATE = "BASE_CANDIDATE"
STATE_BASE_STABLE = "BASE_STABLE"
STATE_DIACRITIC_CANDIDATE = "DIACRITIC_CANDIDATE"
STATE_DIACRITIC_CONFIRMED = "DIACRITIC_CONFIRMED"
STATE_COOLDOWN = "COOLDOWN"

def extract_hand_vertical_metrics(landmarks_raw: List[List[float]]) -> Tuple[float, float, float, float]:
    """Trích xuất các chỉ số dọc toàn bàn tay: (wrist_y, palm_center_y, hand_min_y, hand_mean_y)."""
    ys = [float(pt[1]) for pt in landmarks_raw[:21]]
    wrist_y = ys[0]
    palm_center_y = (ys[0] + ys[5] + ys[9] + ys[13] + ys[17]) / 5.0
    hand_min_y = min(ys)
    hand_mean_y = sum(ys) / 21.0
    return wrist_y, palm_center_y, hand_min_y, hand_mean_y


class WholeHandExitTracker:
    """
    Bộ phát hiện pha hạ tay toàn bàn tay (EXIT Guard) dùng chung cho FSM và Recognizer.
    Phân biệt tay đặt thấp cố định trong khung hình (wrist_y cao nhưng các ngón vẫn dựng signing)
    với hành vi thực sự hạ/rút toàn bộ bàn tay ra khỏi vùng ký.
    """

    def __init__(self):
        self.last_timestamp: Optional[float] = None
        self.last_wrist_y: Optional[float] = None
        self.last_palm_center_y: Optional[float] = None
        self.last_mean_y: Optional[float] = None
        self.downward_streak: int = 0
        self.downward_start_palm_y: Optional[float] = None

    def reset(self):
        self.last_timestamp = None
        self.last_wrist_y = None
        self.last_palm_center_y = None
        self.last_mean_y = None
        self.downward_streak = 0
        self.downward_start_palm_y = None

    def update(self, landmarks_raw: Optional[List[List[float]]], timestamp: float) -> Dict[str, Any]:
        if not landmarks_raw or len(landmarks_raw) < 21:
            self.reset()
            return {
                "is_exit": False,
                "reason": None,
                "wrist_y": 0.0,
                "palm_center_y": 0.0,
                "hand_min_y": 0.0,
                "hand_mean_y": 0.0,
                "wrist_vy": 0.0,
                "palm_vy": 0.0,
                "mean_vy": 0.0,
                "palm_dy": 0.0,
            }

        wrist_y, palm_center_y, hand_min_y, hand_mean_y = extract_hand_vertical_metrics(landmarks_raw)
        wrist_vy = 0.0
        palm_vy = 0.0
        mean_vy = 0.0
        palm_dy = 0.0
        dt = 0.0

        if (
            self.last_timestamp is not None
            and self.last_palm_center_y is not None
            and self.last_mean_y is not None
        ):
            dt = float(timestamp) - self.last_timestamp
            if dt > 1e-4:
                palm_dy = palm_center_y - self.last_palm_center_y
                mean_dy = hand_mean_y - self.last_mean_y
                wrist_dy = wrist_y - (self.last_wrist_y if self.last_wrist_y is not None else wrist_y)
                palm_vy = palm_dy / dt
                mean_vy = mean_dy / dt
                wrist_vy = wrist_dy / dt

        if palm_vy >= 0.55 and mean_vy >= 0.50:
            if self.downward_streak == 0:
                self.downward_start_palm_y = (
                    self.last_palm_center_y if self.last_palm_center_y is not None else palm_center_y
                )
            self.downward_streak += 1
        elif self.last_timestamp is not None and dt <= 1e-4:
            # Bỏ qua duplicate timestamp mà không làm mất streak
            pass
        else:
            self.downward_streak = 0
            self.downward_start_palm_y = None

        cumulative_drop = (
            palm_center_y - self.downward_start_palm_y
            if self.downward_start_palm_y is not None
            else palm_dy
        )

        # 1. Rơi đột ngột toàn bàn tay (rapid whole-hand plunge)
        is_rapid_plunge = (
            palm_vy >= 1.50
            and mean_vy >= 1.20
            and palm_center_y >= 0.60
            and palm_dy >= 0.12
        )
        # 2. Rơi kéo dài >= 2 frame liên tiếp xuống nửa dưới khung hình
        is_sustained_exit = (
            self.downward_streak >= 2
            and palm_center_y >= 0.68
            and cumulative_drop >= 0.10
        )
        # 3. Toàn bộ 21 điểm (kể cả đỉnh cao nhất) tụt sát mép dưới và không đi lên
        is_bottom_edge_exit = (
            hand_min_y >= 0.76
            and palm_center_y >= 0.80
            and mean_vy >= -0.10
        )

        is_exit = bool(is_rapid_plunge or is_sustained_exit or is_bottom_edge_exit)
        reason = None
        if is_rapid_plunge:
            reason = "rapid_whole_hand_plunge"
        elif is_sustained_exit:
            reason = "sustained_whole_hand_exit"
        elif is_bottom_edge_exit:
            reason = "bottom_edge_whole_hand_exit"

        self.last_timestamp = float(timestamp)
        self.last_wrist_y = wrist_y
        self.last_palm_center_y = palm_center_y
        self.last_mean_y = hand_mean_y

        return {
            "is_exit": is_exit,
            "reason": reason,
            "wrist_y": wrist_y,
            "palm_center_y": palm_center_y,
            "hand_min_y": hand_min_y,
            "hand_mean_y": hand_mean_y,
            "wrist_vy": wrist_vy,
            "palm_vy": palm_vy,
            "mean_vy": mean_vy,
            "palm_dy": palm_dy,
        }


@dataclass
class BaselineSnapshot:
    """Ảnh chụp tư thế nền ổn định (không bao giờ bị ghi đè bởi motion buffer)."""
    base_label: str
    start_time: float
    stable_time: float
    wrist_y: float
    palm_center_y: float
    palm_scale: float
    d_index_middle_norm: float
    d_pinky_index_norm: float
    pinky_curl: float
    index_pip_angle: float
    index_curl: float
    d_pinky_wrist_norm: float = 1.0
    middle_curl: float = 0.0
    middle_pip_angle: float = 0.0
    middle_dy: float = 0.0
    d_thumb_wrist_norm: float = 1.0
    d_thumb_index_mcp_norm: float = 0.3
    d_thumb_index_tip_norm: float = 0.2
    d_thumb_middle_tip_norm: float = 0.25
    sample_count: int = 1

@dataclass
class MotionFrame:
    """Một frame trong bộ đệm chuyển động gần nhất."""
    timestamp: float
    raw_landmarks: List[List[float]]
    shifted_landmarks: List[List[float]]
    palm_scale: float
    finger_states: Dict[str, Any]
    wrist_y: float
    palm_center_y: float
    vy: float  # Vận tốc thẳng đứng (âm = đi lên, dương = đi xuống)
    d_index_middle_norm: float
    d_pinky_index_norm: float
    pinky_curl: float
    index_pip_angle: float
    index_curl: float
    d_pinky_wrist_norm: float = 1.0
    d_ring_wrist_norm: float = 1.0
    middle_pip_angle: float = 0.0
    middle_dy: float = 0.0
    d_thumb_wrist_norm: float = 1.0
    d_thumb_index_mcp_norm: float = 0.3
    d_thumb_index_tip_norm: float = 0.5
    d_thumb_middle_tip_norm: float = 0.5
    hand_min_y: float = 0.0
    hand_mean_y: float = 0.0
    palm_vy: float = 0.0
    mean_vy: float = 0.0
    palm_dy_step: float = 0.0
    is_whole_hand_exit: bool = False
    exit_reason: Optional[str] = None

class DiacriticStateMachine:
    """
    Máy trạng thái phân giải dấu độc lập cho 7 chữ cái có dấu VSL.
    """
    def __init__(
        self,
        min_base_stable_frames: int = 4,
        min_base_stable_sec: float = 0.10,
        max_diacritic_search_frames: int = 15,
        max_diacritic_search_sec: float = 0.65,
        cooldown_sec: float = 0.40,
        # Ngưỡng động học đã được kiểm chứng từ dữ liệu video thật
        a_breve_dist_threshold: float = 0.40,
        u_horn_pinky_curl_threshold: float = 0.40,
        o_horn_dist_threshold: float = 0.50,
        circumflex_lift_threshold: float = 0.020,
        circumflex_vy_threshold: float = 0.05,
        d_stroke_angle_threshold: float = 130.0,
        d_stroke_drop_threshold: float = 12.0,
        max_competing_frames: int = 3,
        min_switch_confirm_frames: int = 2,
    ):
        self.min_base_stable_frames = min_base_stable_frames
        self.min_base_stable_sec = min_base_stable_sec
        self.max_diacritic_search_frames = max_diacritic_search_frames
        self.max_diacritic_search_sec = max_diacritic_search_sec
        self.cooldown_sec = cooldown_sec
        self.max_competing_frames = max_competing_frames
        self.min_switch_confirm_frames = min_switch_confirm_frames

        # Ngưỡng nhận diện dấu
        self.a_breve_dist_threshold = a_breve_dist_threshold
        self.u_horn_pinky_curl_threshold = u_horn_pinky_curl_threshold
        self.o_horn_dist_threshold = o_horn_dist_threshold
        self.circumflex_lift_threshold = circumflex_lift_threshold
        self.circumflex_vy_threshold = circumflex_vy_threshold
        self.d_stroke_angle_threshold = d_stroke_angle_threshold
        self.d_stroke_drop_threshold = d_stroke_drop_threshold

        # Trạng thái FSM
        self.state = STATE_IDLE
        self.current_base: Optional[str] = None
        self.candidate_base: Optional[str] = None
        self.candidate_base_count = 0
        self.candidate_base_start_time = 0.0

        # Lưu trữ lịch sử & candidate frames để tạo BaselineSnapshot đa frame
        self.baseline: Optional[BaselineSnapshot] = None
        self.motion_buffer: deque = deque(maxlen=max_diacritic_search_frames)
        self._candidate_frames: List[MotionFrame] = []

        # Quản lý nhãn cạnh tranh (Rule I/V suppression - Task 7E.4A)
        self.competing_frames_count = 0

        # Quản lý đổi chữ nền & dung sai nhãn sai thoáng qua (Switch Candidate Tolerance - Task 7E.4B)
        self.switch_candidate_base: Optional[str] = None
        self.switch_candidate_count = 0
        self.switch_candidate_start_time = 0.0

        # Quản lý xác nhận & cooldown
        self.confirmed_diacritic: Optional[str] = None
        self.confirmed_time = 0.0
        self.diacritic_candidate: Optional[str] = None
        self.diacritic_candidate_count = 0

        # Frame trước đó để tính dy/dt và phát hiện hạ tay toàn bàn tay (EXIT Guard)
        self._baseline_anchor_time: float = 0.0
        self._last_raw_wrist_y: Optional[float] = None
        self._last_timestamp: Optional[float] = None
        self._exit_tracker = WholeHandExitTracker()

    def _resolve_timestamp(self, timestamp: Optional[float]) -> Tuple[float, bool, bool]:
        """
        Chuẩn hóa timestamp đầu vào theo đơn vị giây:
        - Nhảy lùi rõ rệt (> 1ms): trả về reset_needed=True để đặt lại FSM an toàn.
        - Trễ/lệch vi mô ([-1ms, 0]): kẹp về _last_timestamp để tránh dt âm.
        - Duplicate frame tường minh (dt <= 1e-4 khi caller truyền timestamp): đánh dấu để không đếm đúp switch/candidate.
        """
        curr_time = time.time() if timestamp is None else float(timestamp)
        if self._last_timestamp is None:
            return curr_time, False, False
        delta = curr_time - self._last_timestamp
        if delta < -1e-3:
            return curr_time, True, False
        if delta < 0.0:
            curr_time = self._last_timestamp
            delta = 0.0
        is_duplicate = (timestamp is not None and delta <= 1e-4)
        return curr_time, False, is_duplicate

    def _reset_fsm_state(self):
        """Đặt lại trạng thái nhận diện FSM nhưng giữ bộ theo dõi hạ tay liên tục giữa các frame."""
        self.state = STATE_IDLE
        self.current_base = None
        self.candidate_base = None
        self.candidate_base_count = 0
        self.candidate_base_start_time = 0.0
        self.baseline = None
        self._baseline_anchor_time = 0.0
        self.motion_buffer.clear()
        self._candidate_frames.clear()
        self.competing_frames_count = 0
        self.switch_candidate_base = None
        self.switch_candidate_count = 0
        self.switch_candidate_start_time = 0.0
        self.confirmed_diacritic = None
        self.confirmed_time = 0.0
        self.diacritic_candidate = None
        self.diacritic_candidate_count = 0

    def reset(self):
        """Khôi phục toàn bộ trạng thái về IDLE."""
        self._reset_fsm_state()
        self._last_raw_wrist_y = None
        self._last_timestamp = None
        self._exit_tracker.reset()

    @staticmethod
    def _upper_quantile(values: List[float], quantile: float = 0.75) -> float:
        """Tính phân vị trên có nội suy tuyến tính cho các đặc trưng co/duỗi trước khi khởi phát dấu."""
        if not values:
            return 0.0
        s = sorted(float(v) for v in values)
        if len(s) == 1:
            return s[0]
        pos = max(0.0, min(1.0, quantile)) * (len(s) - 1)
        lo = int(math.floor(pos))
        hi = int(math.ceil(pos))
        frac = pos - lo
        return s[lo] * (1.0 - frac) + s[hi] * frac

    @staticmethod
    def _median_value(values: List[float]) -> float:
        """Tính trung vị bền vững với nhiễu cho các tham chiếu không gian/khoảng cách."""
        if not values:
            return 0.0
        s = sorted(float(v) for v in values)
        n = len(s)
        mid = n // 2
        if n % 2 == 1:
            return s[mid]
        return 0.5 * (s[mid - 1] + s[mid])

    def _build_baseline_from_candidate_frames(
        self,
        base_label: str,
        cand_frames: List[MotionFrame],
        stable_time: float,
    ) -> BaselineSnapshot:
        """
        Tổng hợp BaselineSnapshot đa frame theo ngữ nghĩa từng loại đặc trưng:
        - Các đặc trưng co gập/khép có xu hướng thả lỏng sớm trước khi dấu hình thành
          (index_curl, middle_curl, pinky_curl, index_pip_angle) dùng phân vị tiền chuyển động (~Q75).
        - Tham chiếu không gian (wrist_y, palm_center_y, palm_scale, middle_dy) và độ mở hẹp ban đầu
          dùng giá trị ổn định bền vững (trung vị / cuối pha ổn định).
        """
        frames = cand_frames or []
        latest = frames[-1]
        median_wrist_y = self._median_value([f.wrist_y for f in frames])
        median_palm_y = self._median_value([f.palm_center_y for f in frames])

        # Nếu tay vừa nâng nhẹ lên đến vị trí dừng trong lúc tích lũy candidate,
        # neo ở mức cao/ổn định hơn giữa trung vị và frame cuối để tránh false Â/Ê.
        settled_wrist_y = min(median_wrist_y, latest.wrist_y)
        settled_palm_y = min(median_palm_y, latest.palm_center_y)

        return BaselineSnapshot(
            base_label=base_label,
            start_time=self.candidate_base_start_time,
            stable_time=stable_time,
            wrist_y=settled_wrist_y,
            palm_center_y=settled_palm_y,
            palm_scale=self._median_value([f.palm_scale for f in frames]),
            d_index_middle_norm=self._median_value([f.d_index_middle_norm for f in frames]),
            d_pinky_index_norm=self._median_value([f.d_pinky_index_norm for f in frames]),
            pinky_curl=self._upper_quantile([f.pinky_curl for f in frames], 0.75),
            index_pip_angle=self._upper_quantile([f.index_pip_angle for f in frames], 0.75),
            index_curl=self._upper_quantile([f.index_curl for f in frames], 0.75),
            d_pinky_wrist_norm=self._median_value([f.d_pinky_wrist_norm for f in frames]),
            middle_curl=self._upper_quantile(
                [float(f.finger_states.get("middle_curl", 0.0)) for f in frames], 0.75
            ),
            middle_pip_angle=self._median_value([f.middle_pip_angle for f in frames]),
            middle_dy=self._median_value([f.middle_dy for f in frames]),
            d_thumb_wrist_norm=self._median_value([f.d_thumb_wrist_norm for f in frames]),
            d_thumb_index_mcp_norm=self._median_value([f.d_thumb_index_mcp_norm for f in frames]),
            d_thumb_index_tip_norm=self._median_value([f.d_thumb_index_tip_norm for f in frames]),
            d_thumb_middle_tip_norm=self._median_value([f.d_thumb_middle_tip_norm for f in frames]),
            sample_count=len(frames),
        )

    def is_competing_label(self, label: str) -> bool:
        """Kiểm tra xem một nhãn tĩnh có phải là quy tắc cạnh tranh trực tiếp với chữ nền hiện tại hay không."""
        if not self.current_base:
            return False
        competing = COMPETING_RULES_MAP.get(self.current_base)
        if isinstance(competing, (tuple, list, set)):
            return label in competing
        return competing == label

    def get_state(self) -> Dict[str, Any]:
        """Trả về thông tin trạng thái hiện tại của FSM."""
        return {
            "state": self.state,
            "current_base": self.current_base,
            "candidate_base": self.candidate_base,
            "switch_candidate_base": self.switch_candidate_base,
            "switch_candidate_count": self.switch_candidate_count,
            "has_baseline": self.baseline is not None,
            "buffer_len": len(self.motion_buffer),
            "confirmed_diacritic": self.confirmed_diacritic
        }

    def _extract_frame_features(
        self,
        landmarks_raw: List[List[float]],
        timestamp: float
    ) -> Optional[MotionFrame]:
        """Trích xuất các đặc trưng cần thiết cho 1 frame."""
        if not landmarks_raw or len(landmarks_raw) < 21:
            return None

        # Kiểm tra tính hữu hạn
        for pt in landmarks_raw[:21]:
            if pt is None or len(pt) < 3:
                return None
            for coord in pt:
                if math.isnan(coord) or math.isinf(coord):
                    return None

        exit_info = self._exit_tracker.update(landmarks_raw, timestamp)
        raw_wrist_y = exit_info["wrist_y"]
        raw_palm_center_y = exit_info["palm_center_y"]

        # Tính vận tốc thẳng đứng tức thời (v_y)
        vy = 0.0
        if self._last_timestamp is not None and self._last_raw_wrist_y is not None:
            dt = timestamp - self._last_timestamp
            if dt > 1e-4:
                vy = (raw_wrist_y - self._last_raw_wrist_y) / dt
        self._last_raw_wrist_y = raw_wrist_y
        self._last_timestamp = timestamp

        shifted, palm_scale = _shift_and_scale(landmarks_raw)
        if palm_scale < 1e-4:
            palm_scale = 1.0

        fstates = compute_finger_states(shifted, palm_scale)

        # Khoảng cách chuẩn hóa
        d_im_norm = calculate_distance(shifted[8], shifted[12]) / palm_scale
        d_pi_norm = calculate_distance(shifted[20], shifted[8]) / palm_scale
        d_pw_norm = calculate_distance(shifted[20], shifted[0]) / palm_scale
        d_rw_norm = calculate_distance(shifted[16], shifted[0]) / palm_scale
        d_tw_norm = calculate_distance(shifted[4], shifted[0]) / palm_scale
        d_t5_norm = calculate_distance(shifted[4], shifted[5]) / palm_scale
        d_t8_norm = calculate_distance(shifted[4], shifted[8]) / palm_scale
        d_t12_norm = calculate_distance(shifted[4], shifted[12]) / palm_scale

        # Góc và độ nâng ngón trỏ / giữa
        pip_angle = calculate_angle(shifted[5], shifted[6], shifted[7])
        mid_pip_angle = calculate_angle(shifted[9], shifted[10], shifted[11])
        mid_dy = float(shifted[12][1] - shifted[9][1])

        return MotionFrame(
            timestamp=timestamp,
            raw_landmarks=landmarks_raw,
            shifted_landmarks=shifted,
            palm_scale=palm_scale,
            finger_states=fstates,
            wrist_y=raw_wrist_y,
            palm_center_y=raw_palm_center_y,
            vy=vy,
            d_index_middle_norm=d_im_norm,
            d_pinky_index_norm=d_pi_norm,
            pinky_curl=fstates.get("pinky_curl", 0.0),
            index_pip_angle=pip_angle,
            index_curl=fstates.get("index_curl", 0.0),
            d_pinky_wrist_norm=d_pw_norm,
            d_ring_wrist_norm=d_rw_norm,
            middle_pip_angle=mid_pip_angle,
            middle_dy=mid_dy,
            d_thumb_wrist_norm=d_tw_norm,
            d_thumb_index_mcp_norm=d_t5_norm,
            d_thumb_index_tip_norm=d_t8_norm,
            d_thumb_middle_tip_norm=d_t12_norm,
            hand_min_y=exit_info["hand_min_y"],
            hand_mean_y=exit_info["hand_mean_y"],
            palm_vy=exit_info["palm_vy"],
            mean_vy=exit_info["mean_vy"],
            palm_dy_step=exit_info["palm_dy"],
            is_whole_hand_exit=exit_info["is_exit"],
            exit_reason=exit_info["reason"],
        )

    def _evaluate_diacritic_for_group(
        self,
        base: str,
        curr_frame: MotionFrame,
        baseline: BaselineSnapshot
    ) -> Optional[Tuple[str, str]]:
        """
        Đánh giá dấu cho nhóm chữ nền hiện tại.
        Trả về (tên_chữ_có_dấu, lý_do) hoặc None nếu chưa thỏa mãn.
        """
        is_y_shaka = (
            bool(curr_frame.finger_states.get("thumb_extended", False))
            and curr_frame.d_thumb_index_mcp_norm >= 0.42
            and (curr_frame.d_thumb_index_tip_norm >= 0.40 or curr_frame.d_thumb_middle_tip_norm >= 0.45)
            and (curr_frame.d_thumb_wrist_norm >= 1.15 or curr_frame.d_thumb_index_tip_norm >= 0.50)
        )

        # 1. Nhóm A -> {Ă, Â}
        if base == "A":
            mid_curl = curr_frame.finger_states.get("middle_curl", 0.0)
            d_im_growth = curr_frame.d_index_middle_norm - baseline.d_index_middle_norm
            index_uncurl = baseline.index_curl - curr_frame.index_curl
            middle_uncurl = baseline.middle_curl - mid_curl
            middle_pip_gain = curr_frame.middle_pip_angle - baseline.middle_pip_angle
            middle_dy_lift = baseline.middle_dy - curr_frame.middle_dy
            pinky_folded = (
                curr_frame.pinky_curl >= 0.20
                or curr_frame.d_pinky_wrist_norm <= 1.10
                or not curr_frame.finger_states.get("pinky_extended", False)
            )

            # Ưu tiên 1: Chữ Ă (QIPEDC Miền Nam D0490N: CHỈ bật duỗi 1 ngón trỏ lên, ngón giữa giữ gập chặt)
            middle_stays_curled = (
                mid_curl >= 0.30
                and (curr_frame.middle_pip_angle <= 95.0 or mid_curl >= 0.50)
            )
            is_a_breve = (
                curr_frame.index_curl < 0.35
                and curr_frame.index_pip_angle >= 115.0
                and middle_stays_curled
                and curr_frame.d_index_middle_norm >= 0.45
                and d_im_growth >= 0.20
                and (index_uncurl >= 0.15 or baseline.index_curl >= 0.25)
                and pinky_folded
            )
            if is_a_breve:
                return "Ă", f"Duỗi đơn ngón trỏ chữ Ă (d_IM={curr_frame.d_index_middle_norm:.2f}, ic={curr_frame.index_curl:.2f}, mc={mid_curl:.2f})"

            # Ưu tiên 2: Chữ Â (QIPEDC Miền Nam D0491N: Bật mở ĐỒNG THỜI trỏ+giữa tạo mũ ^ hoặc nâng tay đẩy nón)
            middle_lifted = (
                curr_frame.middle_dy <= -0.005
                or curr_frame.middle_pip_angle >= 65.0
            )
            middle_opened_from_base = (
                middle_dy_lift >= 0.020
                or middle_uncurl >= 0.10
                or middle_pip_gain >= 20.0
            )
            is_a_claw = (
                0.18 <= curr_frame.d_index_middle_norm < 0.55
                and d_im_growth >= 0.03
                and curr_frame.index_curl <= 0.50
                and mid_curl <= 0.60
                and curr_frame.index_pip_angle >= 100.0
                and middle_lifted
                and middle_opened_from_base
                and pinky_folded
            )
            lift = baseline.wrist_y - curr_frame.wrist_y
            is_a_lift = (lift >= self.circumflex_lift_threshold and curr_frame.vy <= self.circumflex_vy_threshold and curr_frame.d_index_middle_norm < 0.35)

            if is_a_claw:
                return "Â", f"Tạo dấu mũ trỏ-giữa chữ Â (d_IM={curr_frame.d_index_middle_norm:.2f}, growth={d_im_growth:.2f}, m_dy={curr_frame.middle_dy:+.2f})"
            elif is_a_lift:
                return "Â", f"Đẩy nón lên đỉnh (lift={lift:.3f} >= {self.circumflex_lift_threshold:.3f})"

        # 2. Nhóm E -> {Ê}
        elif base == "E":
            # Ưu tiên: Dấu mũ circumflex claw Miền Nam hoặc đẩy nón lên đỉnh
            mid_curl = curr_frame.finger_states.get("middle_curl", 0.0)
            d_im_growth = curr_frame.d_index_middle_norm - baseline.d_index_middle_norm
            pinky_folded = (
                curr_frame.pinky_curl >= 0.20
                or curr_frame.d_pinky_wrist_norm <= 1.10
                or not curr_frame.finger_states.get("pinky_extended", False)
            )
            is_e_claw = (
                curr_frame.d_index_middle_norm >= 0.18 and
                d_im_growth >= 0.05 and
                curr_frame.index_curl <= 0.65 and
                mid_curl <= 0.65 and
                pinky_folded
            )
            lift = baseline.wrist_y - curr_frame.wrist_y
            is_e_lift = (lift >= self.circumflex_lift_threshold and curr_frame.vy <= self.circumflex_vy_threshold)

            if is_e_claw:
                return "Ê", f"Tạo dấu mũ trỏ-giữa chữ Ê (d_IM={curr_frame.d_index_middle_norm:.2f} >= 0.18, growth={d_im_growth:.2f})"
            elif is_e_lift:
                return "Ê", f"Đẩy nón lên đỉnh (lift={lift:.3f} >= {self.circumflex_lift_threshold:.3f})"

        # 3. Nhóm O -> {Ô, Ơ}
        elif base == "O":
            d_im_growth = curr_frame.d_index_middle_norm - baseline.d_index_middle_norm
            # Ưu tiên 1: Chữ Ô (Tạo nón trỏ-giữa theo chuẩn QIPEDC D0507N)
            if (curr_frame.d_index_middle_norm >= 0.15 and
                d_im_growth >= 0.04 and
                curr_frame.index_pip_angle <= 125.0 and
                curr_frame.d_pinky_wrist_norm <= 0.90):
                return "Ô", f"Tạo nón trỏ-giữa (d_IM={curr_frame.d_index_middle_norm:.2f} >= 0.15, PIP={curr_frame.index_pip_angle:.1f}° <= 125.0°)"

            # Ưu tiên 2: Chữ Ơ (QIPEDC D0508N: Móc râu ngón út, bảo toàn vòng O và chặn mở cả bàn tay / Shaka Y)
            d_pi_gain = curr_frame.d_pinky_index_norm - baseline.d_pinky_index_norm
            d_pw_gain = curr_frame.d_pinky_wrist_norm - baseline.d_pinky_wrist_norm
            pinky_uncurl = baseline.pinky_curl - curr_frame.pinky_curl
            pinky_ring_sep = curr_frame.d_pinky_wrist_norm - curr_frame.d_ring_wrist_norm
            max_allowed_d_ti = max(0.38, baseline.d_thumb_index_tip_norm + 0.05)
            max_allowed_d_tm = max(0.34, baseline.d_thumb_middle_tip_norm + 0.08)

            has_o_horn_motion = (
                d_pi_gain >= 0.30
                or (d_pw_gain >= 0.25 and pinky_uncurl >= 0.15)
            )
            has_pinky_horn_separation = (
                curr_frame.d_pinky_wrist_norm >= 1.15
                and pinky_ring_sep >= 0.25
                and (
                    curr_frame.d_pinky_index_norm >= self.o_horn_dist_threshold
                    or curr_frame.pinky_curl <= 0.30
                )
            )
            is_base_o_loop_preserved = (
                curr_frame.d_thumb_index_tip_norm <= max_allowed_d_ti
                and curr_frame.d_thumb_middle_tip_norm <= max_allowed_d_tm
                and curr_frame.d_ring_wrist_norm >= 0.66
                and d_im_growth <= 0.35
            )
            is_o_horn = (
                not is_y_shaka
                and has_o_horn_motion
                and has_pinky_horn_separation
                and is_base_o_loop_preserved
            )
            if is_o_horn:
                return "Ơ", (
                    f"Bật râu ngón út (dist={curr_frame.d_pinky_index_norm:.2f}, "
                    f"d_PI_gain={d_pi_gain:+.2f}, d_PW={curr_frame.d_pinky_wrist_norm:.2f}, "
                    f"d_PW_RW={pinky_ring_sep:+.2f}, d_TI={curr_frame.d_thumb_index_tip_norm:.2f})"
                )

        # 4. Nhóm U -> {Ư}
        elif base == "U":
            # Chữ Ư (QIPEDC D0515N): Ngón út bật vươn khum móc râu từ thế U, chặn nhầm sang Shaka Y
            pinky_drop = baseline.pinky_curl - curr_frame.pinky_curl
            if (
                not is_y_shaka
                and curr_frame.pinky_curl <= self.u_horn_pinky_curl_threshold
                and (pinky_drop >= 0.20 or (baseline.pinky_curl >= 0.40 and curr_frame.pinky_curl <= 0.25))
            ):
                return "Ư", f"Ngón út bật móc râu chữ Ư (curl={curr_frame.pinky_curl:.2f} <= {self.u_horn_pinky_curl_threshold:.2f}, drop={pinky_drop:.2f})"

        # 5. Nhóm D -> {Đ}
        elif base == "D":
            # Chữ Đ: Xung gập ngón trỏ vạch ngang
            angle_drop = baseline.index_pip_angle - curr_frame.index_pip_angle
            if angle_drop >= self.d_stroke_drop_threshold and curr_frame.index_pip_angle <= self.d_stroke_angle_threshold:
                return "Đ", f"Xung gập ngón trỏ (angle={curr_frame.index_pip_angle:.1f}deg, drop={angle_drop:.1f}deg)"

        return None

    def update(
        self,
        base_label: str,
        base_confidence: float,
        landmarks_raw: Optional[List[List[float]]],
        timestamp: Optional[float] = None,
        source: str = "classifier"
    ) -> Dict[str, Any]:
        """
        Cập nhật trạng thái FSM với một frame mới.
        """
        curr_time, need_ts_reset, is_duplicate_ts = self._resolve_timestamp(timestamp)

        # -------------------------------------------------------------
        # Kiểm tra an toàn dữ liệu đầu vào & Timestamp
        # -------------------------------------------------------------
        if need_ts_reset:
            # Timestamp nhảy lùi -> reset an toàn
            self.reset()

        frame = self._extract_frame_features(landmarks_raw, curr_time)

        # Nếu mất landmarks (không thấy bàn tay hoặc tọa độ lỗi)
        if frame is None:
            self.reset()
            return {
                "final_label": "NEUTRAL",
                "base_label": "UNKNOWN",
                "state": STATE_IDLE,
                "confidence": 0.0,
                "source": "none",
                "diacritic_detected": None,
                "reason": "Mất landmarks hoặc đầu vào không hợp lệ"
            }

        # -------------------------------------------------------------
        # Chặn các chữ ngoài 5 nhóm nền (B, C, G, H, I, K, L, M, N, P, Q, R, S, T, V, X, Y)
        # -------------------------------------------------------------
        if base_label not in BASE_GROUPS:
            if base_label in (None, "NEUTRAL", "UNKNOWN") and self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE, STATE_COOLDOWN) and self.baseline is not None:
                return self.update_transient(landmarks_raw=landmarks_raw, timestamp=curr_time, _pre_frame=frame)
            if self.is_competing_label(base_label) and self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE, STATE_COOLDOWN) and self.baseline is not None:
                return self.update_competing(
                    competing_label=base_label,
                    competing_confidence=base_confidence,
                    landmarks_raw=landmarks_raw,
                    timestamp=curr_time,
                    _pre_frame=frame,
                )

            if frame.is_whole_hand_exit:
                if self.state != STATE_IDLE:
                    self._reset_fsm_state()
                return {
                    "final_label": base_label if base_label else "NEUTRAL",
                    "base_label": base_label if base_label else "UNKNOWN",
                    "state": STATE_IDLE,
                    "confidence": base_confidence,
                    "source": source,
                    "diacritic_detected": None,
                    "reason": "Phát hiện pha hạ tay (EXIT Guard)",
                }

            # Dung sai nhãn ngoài nhóm nền sai thoáng qua (Task 7E.4B)
            if self.baseline is not None and self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE, STATE_COOLDOWN):
                search_time = curr_time - self.baseline.stable_time
                if search_time <= self.max_diacritic_search_sec:
                    if self.switch_candidate_base == base_label:
                        if not (is_duplicate_ts and abs(frame.palm_dy_step) <= 1e-6):
                            self.switch_candidate_count += 1
                    else:
                        self.switch_candidate_base = base_label
                        self.switch_candidate_count = 1
                        self.switch_candidate_start_time = curr_time

                    if self.switch_candidate_count < self.min_switch_confirm_frames:
                        return {
                            "final_label": base_label,
                            "base_label": base_label,
                            "state": self.state,
                            "confidence": base_confidence,
                            "source": source,
                            "diacritic_detected": None,
                            "reason": f"Tạm thời ghi nhận chữ ngoài nhóm '{base_label}' ({self.switch_candidate_count}/{self.min_switch_confirm_frames} frames để chuyển từ '{self.current_base}')"
                        }

            self._reset_fsm_state()
            return {
                "final_label": base_label if base_label else "NEUTRAL",
                "base_label": base_label if base_label else "UNKNOWN",
                "state": STATE_IDLE,
                "confidence": base_confidence,
                "source": source,
                "diacritic_detected": None,
                "reason": f"Chữ '{base_label}' không thuộc 5 nhóm chữ nền có dấu"
            }

        # -------------------------------------------------------------
        # Chặn pha hạ tay (EXIT Guard): hạ toàn bàn tay khỏi vùng ký
        # -------------------------------------------------------------
        if frame.is_whole_hand_exit:
            if self.state != STATE_IDLE:
                self._reset_fsm_state()
            return {
                "final_label": base_label,
                "base_label": base_label,
                "state": STATE_IDLE,
                "confidence": base_confidence,
                "source": source,
                "diacritic_detected": None,
                "reason": "Phát hiện pha hạ tay (EXIT Guard)"
            }

        # Đẩy frame vào motion buffer
        self.motion_buffer.append(frame)

        # -------------------------------------------------------------
        # QUẢN LÝ CÁC TRẠNG THÁI (STATE MACHINE LOGIC)
        # -------------------------------------------------------------

        # 1. Trạng thái COOLDOWN
        if self.state == STATE_COOLDOWN:
            elapsed = curr_time - self.confirmed_time
            if elapsed < self.cooldown_sec:
                return {
                    "final_label": self.confirmed_diacritic or base_label,
                    "base_label": self.current_base or base_label,
                    "state": STATE_COOLDOWN,
                    "confidence": base_confidence,
                    "source": "diacritic_fsm",
                    "diacritic_detected": self.confirmed_diacritic,
                    "reason": f"Cooldown ({elapsed:.2f}s / {self.cooldown_sec:.2f}s)"
                }
            else:
                # Hết cooldown: reset FSM để thu thập lại BaselineSnapshot mới sạch sẽ thay vì dùng snapshot cũ
                self._reset_fsm_state()

        # 2. Xử lý đổi chữ nền (Base switch): A đổi sang D
        if self.current_base is not None and base_label != self.current_base:
            # Nếu nhãn này là competing label của current_base -> chuyển qua update_competing
            if self.is_competing_label(base_label) and self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE, STATE_COOLDOWN) and self.baseline is not None:
                return self.update_competing(
                    competing_label=base_label,
                    competing_confidence=base_confidence,
                    landmarks_raw=landmarks_raw,
                    timestamp=curr_time,
                    _pre_frame=frame,
                )

            # Đánh giá xem cử chỉ có thỏa mãn dấu của current_base hay không trước khi đổi chữ nền
            if self.baseline is not None and self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE, STATE_COOLDOWN):
                match_res = self._evaluate_diacritic_for_group(self.current_base, frame, self.baseline)
                if match_res is not None:
                    detected_letter, reason = match_res
                    req_conf_frames = 1 if detected_letter == "Đ" else 2
                    if self.diacritic_candidate == detected_letter:
                        self.diacritic_candidate_count += 1
                    else:
                        self.diacritic_candidate = detected_letter
                        self.diacritic_candidate_count = 1

                    if self.diacritic_candidate_count >= req_conf_frames:
                        self.state = STATE_COOLDOWN
                        self.confirmed_diacritic = detected_letter
                        self.confirmed_time = curr_time
                        return {
                            "final_label": detected_letter,
                            "base_label": self.current_base,
                            "state": STATE_DIACRITIC_CONFIRMED,
                            "confidence": 0.90,
                            "source": "diacritic_fsm",
                            "diacritic_detected": detected_letter,
                            "reason": f"Xác nhận dấu {detected_letter}: {reason} (trong pha transient)"
                        }
                    else:
                        self.state = STATE_DIACRITIC_CANDIDATE
                        return {
                            "final_label": "NEUTRAL",
                            "base_label": self.current_base,
                            "state": STATE_DIACRITIC_CANDIDATE,
                            "confidence": 0.0,
                            "source": "none",
                            "diacritic_detected": detected_letter,
                            "reason": f"Ứng viên {detected_letter} ({self.diacritic_candidate_count}/{req_conf_frames}) trong pha transient"
                        }

            # Dung sai nhãn nền sai thoáng qua (Task 7E.4B - ví dụ E xuất hiện 1 frame trong O)
            if self.baseline is not None and self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE, STATE_COOLDOWN):
                search_time = curr_time - self.baseline.stable_time
                if search_time <= self.max_diacritic_search_sec:
                    if self.switch_candidate_base == base_label:
                        if not (is_duplicate_ts and abs(frame.palm_dy_step) <= 1e-6):
                            self.switch_candidate_count += 1
                    else:
                        self.switch_candidate_base = base_label
                        self.switch_candidate_count = 1
                        self.switch_candidate_start_time = curr_time

                    if self.switch_candidate_count < self.min_switch_confirm_frames:
                        return {
                            "final_label": base_label,
                            "base_label": base_label,
                            "state": self.state,
                            "confidence": base_confidence,
                            "source": source,
                            "diacritic_detected": None,
                            "reason": f"Tạm thời ghi nhận chữ nền khác '{base_label}' ({self.switch_candidate_count}/{self.min_switch_confirm_frames} frames để chuyển từ '{self.current_base}')"
                        }
            self._reset_fsm_state()

        # Nếu frame này quay lại chữ nền hiện tại -> xóa candidate switch
        if self.current_base is not None and base_label == self.current_base:
            self.switch_candidate_base = None
            self.switch_candidate_count = 0

        # 3. Trạng thái IDLE hoặc BASE_CANDIDATE
        if self.state in (STATE_IDLE, STATE_BASE_CANDIDATE):
            if self.state == STATE_IDLE:
                self.state = STATE_BASE_CANDIDATE

            # Chặn pha đưa tay (ENTRY Guard): dùng vận tốc toàn bàn tay thay cho rung đơn điểm cổ tay
            if abs(frame.palm_vy) > 0.85 or abs(frame.vy) > 1.25:
                self.candidate_base = None
                self.candidate_base_count = 0
                self._candidate_frames.clear()
                return {
                    "final_label": base_label,
                    "base_label": base_label,
                    "state": STATE_BASE_CANDIDATE,
                    "confidence": base_confidence,
                    "source": source,
                    "diacritic_detected": None,
                    "reason": "Đang đưa tay lên (ENTRY Guard)"
                }

            if self.candidate_base == base_label:
                self.candidate_base_count += 1
                self._candidate_frames.append(frame)
                max_cand_keep = max(self.min_base_stable_frames + 2, 5)
                if len(self._candidate_frames) > max_cand_keep:
                    self._candidate_frames = self._candidate_frames[-max_cand_keep:]
            else:
                self.candidate_base = base_label
                self.candidate_base_count = 1
                self.candidate_base_start_time = curr_time
                self._candidate_frames = [frame]

            duration = curr_time - self.candidate_base_start_time

            # Không khóa BaselineSnapshot nếu bàn tay vẫn đang được nâng lên từ rìa dưới khung hình
            is_still_rising_entry = (frame.palm_vy < -0.30)

            # Chuyển sang BASE_STABLE khi đủ frame, đủ thời lượng và bàn tay đã ổn định vị trí vào khung
            if (
                self.candidate_base_count >= self.min_base_stable_frames
                and duration >= self.min_base_stable_sec
                and not is_still_rising_entry
            ):
                self.state = STATE_BASE_STABLE
                self.current_base = base_label
                window_frames = self._candidate_frames[-self.min_base_stable_frames:]
                self.baseline = self._build_baseline_from_candidate_frames(
                    base_label=base_label,
                    cand_frames=window_frames,
                    stable_time=curr_time,
                )
                self._baseline_anchor_time = curr_time
                self._candidate_frames = []
            elif (
                self.candidate_base_count >= self.min_base_stable_frames
                and duration >= self.min_base_stable_sec
                and is_still_rising_entry
            ):
                return {
                    "final_label": base_label,
                    "base_label": base_label,
                    "state": STATE_BASE_CANDIDATE,
                    "confidence": base_confidence,
                    "source": source,
                    "diacritic_detected": None,
                    "reason": "Đang ổn định vị trí đưa tay lên (ENTRY Guard)"
                }

            return {
                "final_label": base_label,
                "base_label": base_label,
                "state": self.state,
                "confidence": base_confidence,
                "source": source,
                "diacritic_detected": None,
                "reason": f"Đang ổn định chữ nền ({self.candidate_base_count}/{self.min_base_stable_frames})"
            }

        # 4. Trạng thái BASE_STABLE hoặc DIACRITIC_CANDIDATE
        if self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE):
            assert self.baseline is not None, "Baseline snapshot must be present"

            # Kiểm tra thời gian tìm kiếm tối đa
            search_duration = curr_time - self.baseline.stable_time
            if search_duration > self.max_diacritic_search_sec:
                self.baseline.stable_time = curr_time
                self.baseline.wrist_y = frame.wrist_y
                self.baseline.palm_center_y = frame.palm_center_y
                self._baseline_anchor_time = curr_time
                self.state = STATE_BASE_STABLE
                self.diacritic_candidate = None
                self.diacritic_candidate_count = 0

            # Đánh giá dấu
            match_res = self._evaluate_diacritic_for_group(self.current_base, frame, self.baseline)

            if match_res is not None:
                detected_letter, reason = match_res
                req_conf_frames = 1 if detected_letter == "Đ" else 2

                if self.diacritic_candidate == detected_letter:
                    self.diacritic_candidate_count += 1
                else:
                    self.diacritic_candidate = detected_letter
                    self.diacritic_candidate_count = 1
                    self.state = STATE_DIACRITIC_CANDIDATE

                if self.diacritic_candidate_count >= req_conf_frames:
                    self.state = STATE_COOLDOWN
                    self.confirmed_diacritic = detected_letter
                    self.confirmed_time = curr_time

                    return {
                        "final_label": detected_letter,
                        "base_label": self.current_base,
                        "state": STATE_DIACRITIC_CONFIRMED,
                        "confidence": max(base_confidence, 0.90),
                        "source": "diacritic_fsm",
                        "diacritic_detected": detected_letter,
                        "reason": f"Xác nhận dấu {detected_letter}: {reason}"
                    }
                else:
                    return {
                        "final_label": self.current_base,
                        "base_label": self.current_base,
                        "state": STATE_DIACRITIC_CANDIDATE,
                        "confidence": base_confidence,
                        "source": source,
                        "diacritic_detected": detected_letter,
                        "reason": f"Ứng viên {detected_letter} ({self.diacritic_candidate_count}/{req_conf_frames})"
                    }
            else:
                self.diacritic_candidate = None
                self.diacritic_candidate_count = 0
                if self.state == STATE_BASE_STABLE and self.baseline is not None:
                    self.baseline.stable_time = curr_time
                    # Tránh trôi vị trí nhiều giây khi giữ tĩnh kéo dài: chỉ cập nhật neo dọc khi bàn tay thật sự đứng yên
                    if (
                        self._baseline_anchor_time > 0.0
                        and (curr_time - self._baseline_anchor_time) > self.max_diacritic_search_sec
                        and abs(frame.palm_vy) < 0.25
                        and abs(frame.vy) < 0.35
                    ):
                        self.baseline.wrist_y = frame.wrist_y
                        self.baseline.palm_center_y = frame.palm_center_y
                        self._baseline_anchor_time = curr_time
                return {
                    "final_label": self.current_base,
                    "base_label": self.current_base,
                    "state": STATE_BASE_STABLE,
                    "confidence": base_confidence,
                    "source": source,
                    "diacritic_detected": None,
                    "reason": f"Giữ nguyên chữ nền {self.current_base} (không có dấu)"
                }

        return {
            "final_label": base_label,
            "base_label": base_label,
            "state": self.state,
            "confidence": base_confidence,
            "source": source,
            "diacritic_detected": None,
            "reason": "Trạng thái không xác định"
        }

    def update_transient(
        self,
        landmarks_raw: Optional[List[List[float]]],
        timestamp: Optional[float] = None,
        _pre_frame: Optional[MotionFrame] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý một frame mất nhãn nền tạm thời (transient label loss: NEUTRAL / UNKNOWN)
        khi bàn tay vẫn có landmarks hợp lệ (Task 7E.3 - Tình huống B).
        Bảo toàn baseline snapshot và motion buffer nếu FSM đang ở trạng thái BASE_STABLE,
        DIACRITIC_CANDIDATE hoặc COOLDOWN.
        """
        if _pre_frame is None:
            curr_time, need_ts_reset, _ = self._resolve_timestamp(timestamp)
            if need_ts_reset:
                self.reset()
        else:
            curr_time = time.time() if timestamp is None else float(timestamp)

        frame = _pre_frame if _pre_frame is not None else self._extract_frame_features(landmarks_raw, curr_time)
        if frame is None:
            self.reset()
            return {
                "final_label": "NEUTRAL",
                "base_label": "UNKNOWN",
                "state": STATE_IDLE,
                "confidence": 0.0,
                "source": "none",
                "diacritic_detected": None,
                "reason": "Mất landmarks hoặc đầu vào không hợp lệ trong pha transient"
            }

        # Nếu chưa có baseline snapshot hợp lệ, không thể duy trì FSM trong pha transient
        if self.state not in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE, STATE_COOLDOWN) or self.baseline is None:
            self._reset_fsm_state()
            return {
                "final_label": "NEUTRAL",
                "base_label": self.current_base or "UNKNOWN",
                "state": STATE_IDLE,
                "confidence": 0.0,
                "source": "none",
                "diacritic_detected": None,
                "reason": "Không có baseline snapshot để duy trì trong pha transient"
            }

        # Đã vào pha transient hợp lệ -> xóa bộ đếm chuyển đổi nhãn (Task 7E.4B)
        self.switch_candidate_base = None
        self.switch_candidate_count = 0

        # Chặn pha hạ tay (EXIT Guard): hạ toàn bàn tay khỏi vùng ký
        if frame.is_whole_hand_exit:
            self._reset_fsm_state()
            return {
                "final_label": "NEUTRAL",
                "base_label": self.current_base,
                "state": STATE_IDLE,
                "confidence": 0.0,
                "source": "none",
                "diacritic_detected": None,
                "reason": "Phát hiện pha hạ tay trong transient (EXIT Guard)"
            }

        # Đẩy frame vào motion buffer
        self.motion_buffer.append(frame)

        # 1. Nếu đang ở COOLDOWN
        if self.state == STATE_COOLDOWN:
            elapsed = curr_time - self.confirmed_time
            if elapsed < self.cooldown_sec:
                return {
                    "final_label": self.confirmed_diacritic or "NEUTRAL",
                    "base_label": self.current_base,
                    "state": STATE_COOLDOWN,
                    "confidence": 0.90,
                    "source": "diacritic_fsm",
                    "diacritic_detected": self.confirmed_diacritic,
                    "reason": f"Cooldown ({elapsed:.2f}s / {self.cooldown_sec:.2f}s) trong pha transient"
                }
            else:
                self.reset()
                return {
                    "final_label": "NEUTRAL",
                    "base_label": self.current_base,
                    "state": STATE_IDLE,
                    "confidence": 0.0,
                    "source": "none",
                    "diacritic_detected": None,
                    "reason": "Kết thúc cooldown khi nhãn nền là NEUTRAL"
                }

        # 2. Đang ở BASE_STABLE hoặc DIACRITIC_CANDIDATE
        search_duration = curr_time - self.baseline.stable_time
        if search_duration > self.max_diacritic_search_sec:
            # Hết thời gian tìm kiếm dấu mà nhãn nền vẫn là NEUTRAL -> reset FSM về IDLE (giữ exit tracker)
            self._reset_fsm_state()
            return {
                "final_label": "NEUTRAL",
                "base_label": self.current_base or "UNKNOWN",
                "state": STATE_IDLE,
                "confidence": 0.0,
                "source": "none",
                "diacritic_detected": None,
                "reason": f"Hết thời gian tìm kiếm dấu ({search_duration:.2f}s > {self.max_diacritic_search_sec:.2f}s) trong pha transient"
            }

        # Đánh giá dấu động học với baseline đã lưu
        match_res = self._evaluate_diacritic_for_group(self.current_base, frame, self.baseline)

        if match_res is not None:
            detected_letter, reason = match_res
            req_conf_frames = 1 if detected_letter == "Đ" else 2

            if self.diacritic_candidate == detected_letter:
                self.diacritic_candidate_count += 1
            else:
                self.diacritic_candidate = detected_letter
                self.diacritic_candidate_count = 1
                self.state = STATE_DIACRITIC_CANDIDATE

            if self.diacritic_candidate_count >= req_conf_frames:
                self.state = STATE_COOLDOWN
                self.confirmed_diacritic = detected_letter
                self.confirmed_time = curr_time

                return {
                    "final_label": detected_letter,
                    "base_label": self.current_base,
                    "state": STATE_DIACRITIC_CONFIRMED,
                    "confidence": 0.90,
                    "source": "diacritic_fsm",
                    "diacritic_detected": detected_letter,
                    "reason": f"Xác nhận dấu {detected_letter}: {reason} (trong pha transient)"
                }
            else:
                return {
                    "final_label": "NEUTRAL",
                    "base_label": self.current_base,
                    "state": STATE_DIACRITIC_CANDIDATE,
                    "confidence": 0.0,
                    "source": "none",
                    "diacritic_detected": detected_letter,
                    "reason": f"Ứng viên {detected_letter} ({self.diacritic_candidate_count}/{req_conf_frames}) trong pha transient"
                }
        else:
            self.diacritic_candidate = None
            self.diacritic_candidate_count = 0
            return {
                "final_label": "NEUTRAL",
                "base_label": self.current_base,
                "state": STATE_BASE_STABLE,
                "confidence": 0.0,
                "source": "none",
                "diacritic_detected": None,
                "reason": f"Bảo toàn baseline {self.current_base} trong pha transient ({search_duration:.2f}s/{self.max_diacritic_search_sec:.2f}s)"
            }

    def update_competing(
        self,
        competing_label: str,
        competing_confidence: float,
        landmarks_raw: Optional[List[List[float]]],
        timestamp: Optional[float] = None,
        _pre_frame: Optional[MotionFrame] = None,
    ) -> Dict[str, Any]:
        """
        Xử lý khi phát hiện nhãn cạnh tranh (Rule I/V) trong lúc FSM đang theo dõi cử chỉ tạo dấu (Task 7E.4A).
        - Nếu đang ở COOLDOWN: duy trì bảo vệ dấu đã xác nhận trong suốt thời gian cooldown.
        - Nếu đang ở BASE_STABLE/DIACRITIC_CANDIDATE:
          1. Kiểm tra timeout (max_diacritic_search_sec). Nếu quá hạn, reset và trả về nhãn cạnh tranh.
          2. Đưa frame vào motion buffer và đánh giá xem chuyển động này có thỏa mãn dấu (Ơ, Ư, Ă) không.
          3. Nếu thỏa mãn dấu: xác nhận dấu ngay!
          4. Nếu chưa thỏa mãn dấu: tăng competing_frames_count.
             - Nếu competing_frames_count < max_competing_frames: tạm thời chặn nhãn cạnh tranh, bảo toàn baseline và trả về NEUTRAL.
             - Nếu competing_frames_count >= max_competing_frames: người ký cố ý ký I/V -> reset FSM và trả về nhãn cạnh tranh.
        """
        if _pre_frame is None:
            curr_time, need_ts_reset, _ = self._resolve_timestamp(timestamp)
        else:
            curr_time = time.time() if timestamp is None else float(timestamp)
            need_ts_reset = False

        # 1. Mất landmarks -> reset ngay lập tức
        if landmarks_raw is None or len(landmarks_raw) < 21:
            self.reset()
            return {
                "final_label": "NEUTRAL",
                "base_label": "UNKNOWN",
                "state": STATE_IDLE,
                "confidence": 0.0,
                "source": "none",
                "diacritic_detected": None,
                "is_competing": True,
                "reason": "Mất landmarks trong pha cạnh tranh -> Reset FSM"
            }

        # 2. Kiểm tra tính đơn điệu của timestamp
        if need_ts_reset:
            self.reset()
            return {
                "final_label": "NEUTRAL",
                "base_label": "UNKNOWN",
                "state": STATE_IDLE,
                "confidence": 0.0,
                "source": "none",
                "diacritic_detected": None,
                "is_competing": True,
                "reason": "Lỗi timestamp nhảy lùi trong pha cạnh tranh -> Reset FSM"
            }

        # Nếu không có baseline hợp lệ
        if self.baseline is None or not self.current_base:
            self._reset_fsm_state()
            return {
                "final_label": competing_label,
                "base_label": competing_label,
                "state": STATE_IDLE,
                "confidence": competing_confidence,
                "source": "classifier",
                "diacritic_detected": None,
                "is_competing": False,
                "reason": f"Không có baseline {self.current_base} -> Nhận diện thẳng nhãn {competing_label}"
            }

        frame = _pre_frame if _pre_frame is not None else self._extract_frame_features(landmarks_raw, curr_time)
        if frame is None:
            self.reset()
            return {
                "final_label": "NEUTRAL",
                "base_label": "UNKNOWN",
                "state": STATE_IDLE,
                "confidence": 0.0,
                "source": "none",
                "diacritic_detected": None,
                "is_competing": True,
                "reason": "Lỗi trích xuất landmarks trong pha cạnh tranh"
            }

        # Chặn pha hạ tay toàn bàn tay (EXIT Guard) cả trong pha cạnh tranh
        if frame.is_whole_hand_exit:
            self._reset_fsm_state()
            return {
                "final_label": competing_label,
                "base_label": competing_label,
                "state": STATE_IDLE,
                "confidence": competing_confidence,
                "source": "classifier",
                "diacritic_detected": None,
                "is_competing": False,
                "reason": "Phát hiện pha hạ tay trong pha cạnh tranh (EXIT Guard)"
            }

        # Đã vào pha cạnh tranh hợp lệ -> xóa bộ đếm chuyển đổi nhãn (Task 7E.4B)
        self.switch_candidate_base = None
        self.switch_candidate_count = 0

        # 3. Đang ở trạng thái COOLDOWN (ví dụ vừa xác nhận Ư, frame kế tiếp bị Rule I nhận)
        if self.state == STATE_COOLDOWN:
            elapsed = curr_time - self.confirmed_time
            if elapsed < self.cooldown_sec:
                return {
                    "final_label": self.confirmed_diacritic or self.current_base,
                    "base_label": self.current_base,
                    "state": STATE_COOLDOWN,
                    "confidence": 0.90,
                    "source": "diacritic_fsm",
                    "diacritic_detected": self.confirmed_diacritic,
                    "is_competing": True,
                    "suppressed": True,
                    "reason": f"Bảo lưu dấu {self.confirmed_diacritic} trong cooldown ({elapsed:.2f}s/{self.cooldown_sec:.2f}s), chặn nhãn cạnh tranh {competing_label}"
                }
            else:
                self._reset_fsm_state()
                return {
                    "final_label": competing_label,
                    "base_label": competing_label,
                    "state": STATE_IDLE,
                    "confidence": competing_confidence,
                    "source": "classifier",
                    "diacritic_detected": None,
                    "is_competing": False,
                    "reason": f"Hết cooldown dấu ({elapsed:.2f}s > {self.cooldown_sec:.2f}s), kết thúc ưu tiên FSM để nhận {competing_label}"
                }

        # 4. Đang ở BASE_STABLE hoặc DIACRITIC_CANDIDATE
        search_duration = curr_time - self.baseline.stable_time
        if search_duration > self.max_diacritic_search_sec:
            self._reset_fsm_state()
            return {
                "final_label": competing_label,
                "base_label": competing_label,
                "state": STATE_IDLE,
                "confidence": competing_confidence,
                "source": "classifier",
                "diacritic_detected": None,
                "is_competing": False,
                "reason": f"Hết thời gian tìm kiếm dấu ({search_duration:.2f}s > {self.max_diacritic_search_sec:.2f}s), chuyển sang {competing_label}"
            }

        self.motion_buffer.append(frame)

        match_res = self._evaluate_diacritic_for_group(self.current_base, frame, self.baseline)

        if match_res is not None:
            detected_letter, reason = match_res
            req_conf_frames = 1 if detected_letter == "Đ" else 2

            if self.diacritic_candidate == detected_letter:
                self.diacritic_candidate_count += 1
            else:
                self.diacritic_candidate = detected_letter
                self.diacritic_candidate_count = 1
                self.state = STATE_DIACRITIC_CANDIDATE

            if self.diacritic_candidate_count >= req_conf_frames:
                self.state = STATE_COOLDOWN
                self.confirmed_diacritic = detected_letter
                self.confirmed_time = curr_time
                self.competing_frames_count = 0

                return {
                    "final_label": detected_letter,
                    "base_label": self.current_base,
                    "state": STATE_DIACRITIC_CONFIRMED,
                    "confidence": 0.92,
                    "source": "diacritic_fsm",
                    "diacritic_detected": detected_letter,
                    "is_competing": True,
                    "reason": f"Xác nhận dấu {detected_letter}: {reason} (từ pha cạnh tranh {competing_label})"
                }
            else:
                return {
                    "final_label": "NEUTRAL",
                    "base_label": self.current_base,
                    "state": STATE_DIACRITIC_CANDIDATE,
                    "confidence": 0.0,
                    "source": "diacritic_fsm",
                    "diacritic_detected": detected_letter,
                    "is_competing": True,
                    "suppressed": True,
                    "reason": f"Ứng viên {detected_letter} ({self.diacritic_candidate_count}/{req_conf_frames}) trong pha cạnh tranh {competing_label}"
                }
        else:
            self.competing_frames_count += 1
            if self.competing_frames_count >= self.max_competing_frames:
                self.reset()
                return {
                    "final_label": competing_label,
                    "base_label": competing_label,
                    "state": STATE_IDLE,
                    "confidence": competing_confidence,
                    "source": "classifier",
                    "diacritic_detected": None,
                    "is_competing": False,
                    "reason": f"Người ký duy trì nhãn {competing_label} liên tiếp {self.max_competing_frames} frames, kết thúc ưu tiên FSM"
                }
            else:
                return {
                    "final_label": "NEUTRAL",
                    "base_label": self.current_base,
                    "state": self.state,
                    "confidence": 0.0,
                    "source": "diacritic_fsm",
                    "diacritic_detected": None,
                    "is_competing": True,
                    "suppressed": True,
                    "reason": f"Tạm thời chặn nhãn cạnh tranh {competing_label} ({self.competing_frames_count}/{self.max_competing_frames}) trong cửa sổ dấu {self.current_base}"
                }
