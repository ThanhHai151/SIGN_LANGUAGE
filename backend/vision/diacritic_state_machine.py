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
    "O": "I",
    "U": "I",
    "A": "V",
}

# Các trạng thái của FSM
STATE_IDLE = "IDLE"
STATE_BASE_CANDIDATE = "BASE_CANDIDATE"
STATE_BASE_STABLE = "BASE_STABLE"
STATE_DIACRITIC_CANDIDATE = "DIACRITIC_CANDIDATE"
STATE_DIACRITIC_CONFIRMED = "DIACRITIC_CONFIRMED"
STATE_COOLDOWN = "COOLDOWN"

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
        u_horn_pinky_curl_threshold: float = 0.35,
        o_horn_dist_threshold: float = 0.50,
        circumflex_lift_threshold: float = 0.035,
        circumflex_vy_threshold: float = -0.10,
        d_stroke_angle_threshold: float = 110.0,
        d_stroke_drop_threshold: float = 15.0,
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

        # Lưu trữ lịch sử
        self.baseline: Optional[BaselineSnapshot] = None
        self.motion_buffer: deque = deque(maxlen=max_diacritic_search_frames)

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

        # Frame trước đó để tính dy/dt
        self._last_raw_wrist_y: Optional[float] = None
        self._last_timestamp: Optional[float] = None

    def reset(self):
        """Khôi phục toàn bộ trạng thái về IDLE."""
        self.state = STATE_IDLE
        self.current_base = None
        self.candidate_base = None
        self.candidate_base_count = 0
        self.candidate_base_start_time = 0.0
        self.baseline = None
        self.motion_buffer.clear()
        self.competing_frames_count = 0
        self.switch_candidate_base = None
        self.switch_candidate_count = 0
        self.switch_candidate_start_time = 0.0
        self.confirmed_diacritic = None
        self.confirmed_time = 0.0
        self.diacritic_candidate = None
        self.diacritic_candidate_count = 0
        self._last_raw_wrist_y = None
        self._last_timestamp = None

    def is_competing_label(self, label: str) -> bool:
        """Kiểm tra xem một nhãn tĩnh có phải là quy tắc cạnh tranh trực tiếp với chữ nền hiện tại hay không."""
        if not self.current_base:
            return False
        return COMPETING_RULES_MAP.get(self.current_base) == label

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

        raw_wrist_y = float(landmarks_raw[0][1])
        raw_palm_center_y = float(
            (landmarks_raw[0][1] + landmarks_raw[5][1] + landmarks_raw[9][1] +
             landmarks_raw[13][1] + landmarks_raw[17][1]) / 5.0
        )

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

        # Góc ngón trỏ
        pip_angle = calculate_angle(shifted[5], shifted[6], shifted[7])

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
            index_curl=fstates.get("index_curl", 0.0)
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
        # 1. Nhóm A -> {Ă, Â}
        if base == "A":
            # Ưu tiên 1: Chữ Ă (Mở ngón V xòe: d(8,12) >= threshold và index duỗi thẳng)
            if curr_frame.d_index_middle_norm >= self.a_breve_dist_threshold and curr_frame.index_curl < 0.25:
                return "Ă", f"Mở ngón V (dist={curr_frame.d_index_middle_norm:.2f} >= {self.a_breve_dist_threshold:.2f})"

            # Ưu tiên 2: Chữ Â (Vẽ nón: nâng cổ tay lên đỉnh trong khi ngón vẫn gập dáng A)
            lift = baseline.wrist_y - curr_frame.wrist_y  # Dương khi tay đi lên
            if lift >= self.circumflex_lift_threshold and curr_frame.vy <= self.circumflex_vy_threshold:
                if curr_frame.d_index_middle_norm < 0.35:  # Không mở ngón
                    return "Â", f"Đẩy nón lên đỉnh (lift={lift:.3f} >= {self.circumflex_lift_threshold:.3f})"

        # 2. Nhóm E -> {Ê}
        elif base == "E":
            lift = baseline.wrist_y - curr_frame.wrist_y
            if lift >= self.circumflex_lift_threshold and curr_frame.vy <= self.circumflex_vy_threshold:
                return "Ê", f"Đẩy nón lên đỉnh (lift={lift:.3f} >= {self.circumflex_lift_threshold:.3f})"

        # 3. Nhóm O -> {Ô, Ơ}
        elif base == "O":
            # Ưu tiên 1: Chữ Ơ (Móc râu: ngón út vươn tách xa khỏi khuyên O)
            if curr_frame.d_pinky_index_norm >= self.o_horn_dist_threshold:
                return "Ơ", f"Bật râu ngón út (dist={curr_frame.d_pinky_index_norm:.2f} >= {self.o_horn_dist_threshold:.2f})"

            # Ưu tiên 2: Chữ Ô (Vẽ nón: giữ khuyên tròn O nhưng nâng lên)
            lift = baseline.wrist_y - curr_frame.wrist_y
            if lift >= self.circumflex_lift_threshold and curr_frame.vy <= self.circumflex_vy_threshold:
                if curr_frame.d_pinky_index_norm < 0.35:  # Không mở râu
                    return "Ô", f"Đẩy nón lên đỉnh (lift={lift:.3f} >= {self.circumflex_lift_threshold:.3f})"

        # 4. Nhóm U -> {Ư}
        elif base == "U":
            # Chữ Ư: Ngón út bật duỗi thẳng (pinky_curl <= threshold)
            if curr_frame.pinky_curl <= self.u_horn_pinky_curl_threshold:
                return "Ư", f"Ngón út duỗi thẳng (curl={curr_frame.pinky_curl:.2f} <= {self.u_horn_pinky_curl_threshold:.2f})"

        # 5. Nhóm D -> {Đ}
        elif base == "D":
            # Chữ Đ: Xung gập ngón trỏ vạch ngang
            angle_drop = baseline.index_pip_angle - curr_frame.index_pip_angle
            if curr_frame.index_pip_angle <= self.d_stroke_angle_threshold or angle_drop >= self.d_stroke_drop_threshold:
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
        curr_time = time.time() if timestamp is None else float(timestamp)

        # -------------------------------------------------------------
        # Kiểm tra an toàn dữ liệu đầu vào & Timestamp
        # -------------------------------------------------------------
        if self._last_timestamp is not None and curr_time < self._last_timestamp:
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
                return self.update_transient(landmarks_raw=landmarks_raw, timestamp=curr_time)
            if self.is_competing_label(base_label) and self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE, STATE_COOLDOWN) and self.baseline is not None:
                return self.update_competing(competing_label=base_label, competing_confidence=base_confidence, landmarks_raw=landmarks_raw, timestamp=curr_time)

            # Dung sai nhãn ngoài nhóm nền sai thoáng qua (Task 7E.4B)
            if self.baseline is not None and self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE, STATE_COOLDOWN):
                search_time = curr_time - self.baseline.stable_time
                if search_time <= self.max_diacritic_search_sec:
                    if self.switch_candidate_base == base_label:
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

            self.reset()
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
        # Chặn pha hạ tay (EXIT Guard): cổ tay rơi nhanh về mép dưới
        # -------------------------------------------------------------
        if frame.vy > 0.65 or frame.wrist_y > 0.75:
            if self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE):
                self.reset()
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
                self.state = STATE_BASE_STABLE
                self.confirmed_diacritic = None

        # 2. Xử lý đổi chữ nền (Base switch): A đổi sang D
        if self.current_base is not None and base_label != self.current_base:
            # Dung sai nhãn nền sai thoáng qua (Task 7E.4B - ví dụ E xuất hiện 1 frame trong O)
            if self.baseline is not None and self.state in (STATE_BASE_STABLE, STATE_DIACRITIC_CANDIDATE, STATE_COOLDOWN):
                search_time = curr_time - self.baseline.stable_time
                if search_time <= self.max_diacritic_search_sec:
                    if self.switch_candidate_base == base_label:
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
            self.reset()

        # Nếu frame này quay lại chữ nền hiện tại -> xóa candidate switch
        if self.current_base is not None and base_label == self.current_base:
            self.switch_candidate_base = None
            self.switch_candidate_count = 0

        # 3. Trạng thái IDLE hoặc BASE_CANDIDATE
        if self.state in (STATE_IDLE, STATE_BASE_CANDIDATE):
            if self.state == STATE_IDLE:
                self.state = STATE_BASE_CANDIDATE

            # Chặn pha đưa tay (ENTRY Guard): Nếu cổ tay đang vọt lên quá nhanh
            if abs(frame.vy) > 0.70:
                self.candidate_base = None
                self.candidate_base_count = 0
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
            else:
                self.candidate_base = base_label
                self.candidate_base_count = 1
                self.candidate_base_start_time = curr_time

            duration = curr_time - self.candidate_base_start_time

            # Chuyển sang BASE_STABLE
            if self.candidate_base_count >= self.min_base_stable_frames and duration >= self.min_base_stable_sec:
                self.state = STATE_BASE_STABLE
                self.current_base = base_label

                # Chụp Baseline Snapshot
                self.baseline = BaselineSnapshot(
                    base_label=base_label,
                    start_time=self.candidate_base_start_time,
                    stable_time=curr_time,
                    wrist_y=frame.wrist_y,
                    palm_center_y=frame.palm_center_y,
                    palm_scale=frame.palm_scale,
                    d_index_middle_norm=frame.d_index_middle_norm,
                    d_pinky_index_norm=frame.d_pinky_index_norm,
                    pinky_curl=frame.pinky_curl,
                    index_pip_angle=frame.index_pip_angle,
                    index_curl=frame.index_curl
                )

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
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Xử lý một frame mất nhãn nền tạm thời (transient label loss: NEUTRAL / UNKNOWN)
        khi bàn tay vẫn có landmarks hợp lệ (Task 7E.3 - Tình huống B).
        Bảo toàn baseline snapshot và motion buffer nếu FSM đang ở trạng thái BASE_STABLE,
        DIACRITIC_CANDIDATE hoặc COOLDOWN.
        """
        curr_time = time.time() if timestamp is None else float(timestamp)

        if self._last_timestamp is not None and curr_time < self._last_timestamp:
            self.reset()

        frame = self._extract_frame_features(landmarks_raw, curr_time)
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
            self.reset()
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

        # Chặn pha hạ tay (EXIT Guard): cổ tay rơi nhanh về mép dưới
        if frame.vy > 0.65 or frame.wrist_y > 0.75:
            self.reset()
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
            # Hết thời gian tìm kiếm dấu mà nhãn nền vẫn là NEUTRAL -> reset FSM về IDLE
            self.reset()
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
        timestamp: Optional[float] = None
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
        curr_time = time.time() if timestamp is None else float(timestamp)

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
        if self._last_timestamp is not None and curr_time < self._last_timestamp - 1e-4:
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
            self.reset()
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
                self.reset()
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
            self.reset()
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

        frame = self._extract_frame_features(landmarks_raw, curr_time)
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
