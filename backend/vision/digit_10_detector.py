"""
Module VSL10GestureDetector — Bộ nhận diện cử chỉ động riêng cho Số 10 VSL
========================================================================
Nhận diện trực tiếp cử chỉ Số 10 chuẩn VSL từ video QIPEDC D0539.mp4.

Đặc tính cử chỉ Số 10 (D0539):
- Bàn tay nắm, 3 ngón ngoài (giữa, áp út, út) luôn gập sát lòng bàn tay (tucked).
- Ngón trỏ duỗi đơn và thực hiện chuyển động xoay / nghiêng cổ tay (hoặc gật ngón trỏ)
  tạo ra biên độ quét góc đáng kể (angular sweep >= 40°) trong khoảng 0.35s - 1.2s.
- Phân biệt triệt để với Số 1 tĩnh (ngón trỏ đứng yên, biên độ quét < 20°).
- Độc lập hoàn toàn với FPS camera thông qua timestamp thực tế (giây).
"""

import math
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class Digit10DetectorConfig:
    min_frames: int = 4              # Số frame tối thiểu trong cửa sổ để đánh giá
    min_duration: float = 0.30       # Thời gian chuyển động tối thiểu (giây)
    max_duration: float = 1.20       # Thời gian cửa sổ quan sát tối đa (giây)
    min_angular_span_deg: float = 40.0 # Biên độ quét góc ngón trỏ tối thiểu để xác nhận Số 10
    observing_span_deg: float = 20.0 # Ngưỡng góc bắt đầu ghi nhận chuyển động đang diễn ra
    cooldown_sec: float = 1.5        # Thời gian cooldown chống phát hành lặp
    max_other_finger_dist: float = 0.88 # Ngưỡng gập của 3 ngón giữa, áp út, út (chuẩn hóa palm scale)


class VSL10GestureDetector:
    """
    Bộ phát hiện cử chỉ động Số 10 dựa trên chuỗi thời gian 21 landmarks.
    """

    def __init__(self, config: Optional[Digit10DetectorConfig] = None):
        self.config = config or Digit10DetectorConfig()
        self.buffer = deque(maxlen=40)
        self.last_detected_time: float = 0.0
        self.last_timestamp: Optional[float] = None

    def reset(self) -> None:
        """Đặt lại bộ đệm và trạng thái phát hiện."""
        self.buffer.clear()
        self.last_detected_time = 0.0
        self.last_timestamp = None

    def _extract_frame_features(self, landmarks: List[List[float]]) -> Optional[Dict[str, Any]]:
        """Trích xuất góc ngón trỏ và trạng thái gập của các ngón khác từ 21 landmarks."""
        if not landmarks or len(landmarks) < 21:
            return None

        try:
            wrist = landmarks[0]
            index_mcp = landmarks[5]
            index_tip = landmarks[8]
            middle_tip = landmarks[12]
            ring_tip = landmarks[16]
            pinky_tip = landmarks[20]

            dx = index_mcp[0] - wrist[0]
            dy = index_mcp[1] - wrist[1]
            scale = math.hypot(dx, dy)
            if scale < 1e-5:
                return None

            # Khoảng cách chuẩn hóa từ các đầu ngón tới cổ tay
            d_iw = math.hypot(index_tip[0] - wrist[0], index_tip[1] - wrist[1]) / scale
            d_mw = math.hypot(middle_tip[0] - wrist[0], middle_tip[1] - wrist[1]) / scale
            d_rw = math.hypot(ring_tip[0] - wrist[0], ring_tip[1] - wrist[1]) / scale
            d_pw = math.hypot(pinky_tip[0] - wrist[0], pinky_tip[1] - wrist[1]) / scale

            # 3 ngón giữa, áp út, út phải gập sát lòng bàn tay
            other_tucked = (
                d_mw <= self.config.max_other_finger_dist
                and d_rw <= self.config.max_other_finger_dist
                and d_pw <= self.config.max_other_finger_dist
            )

            # Vector hướng ngón trỏ: từ MCP (khớp gốc) đến TIP (đầu ngón)
            v_idx_x = index_tip[0] - index_mcp[0]
            v_idx_y = index_tip[1] - index_mcp[1]

            # Góc ngón trỏ so với trục thẳng đứng hướng lên (tính bằng độ từ -180 đến 180)
            angle_deg = math.degrees(math.atan2(v_idx_x, -v_idx_y))

            return {
                "scale": scale,
                "d_iw": d_iw,
                "other_tucked": other_tucked,
                "angle_deg": angle_deg,
                "is_index_active": (d_iw >= 1.0 or abs(angle_deg) < 60.0)
            }
        except Exception:
            return None

    def update(
        self,
        landmarks: Optional[List[List[float]]],
        timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        Cập nhật khung hình mới và kiểm tra cử chỉ Số 10.
        """
        now = time.time() if timestamp is None else float(timestamp)
        if self.last_timestamp is not None and now < self.last_timestamp:
            now = self.last_timestamp
        self.last_timestamp = now

        # Kiểm tra cooldown
        in_cooldown = (now - self.last_detected_time < self.config.cooldown_sec)

        # Trích xuất đặc trưng
        feat = self._extract_frame_features(landmarks) if landmarks else None

        if feat is None:
            # Mất tracking hoặc bàn tay không hợp lệ: dọn các frame quá cũ
            while self.buffer and (now - self.buffer[0]["ts"] > self.config.max_duration):
                self.buffer.popleft()
            return {
                "detected": False,
                "confidence": 0.0,
                "is_observing": False,
                "angular_span_deg": 0.0,
                "num_frames": len(self.buffer)
            }

        # Nếu 3 ngón ngoài không gập (ví dụ xòe 5 ngón hoặc mở ngón giữa) -> Không phải cử chỉ số 10
        if not feat["other_tucked"]:
            # Cử chỉ của bàn tay khác loại -> xóa bộ đệm
            self.buffer.clear()
            return {
                "detected": False,
                "confidence": 0.0,
                "is_observing": False,
                "angular_span_deg": 0.0,
                "num_frames": 0
            }

        # Đưa frame vào bộ đệm trượt
        self.buffer.append({
            "ts": now,
            "angle": feat["angle_deg"],
            "d_iw": feat["d_iw"]
        })

        # Loại bỏ các frame vượt quá max_duration
        while self.buffer and (now - self.buffer[0]["ts"] > self.config.max_duration):
            self.buffer.popleft()

        if in_cooldown or len(self.buffer) < self.config.min_frames:
            return {
                "detected": False,
                "confidence": 0.0,
                "is_observing": False,
                "angular_span_deg": 0.0,
                "num_frames": len(self.buffer)
            }

        # Tính toán đặc trưng chuỗi thời gian trong bộ đệm
        angles = [f["angle"] for f in self.buffer]
        duration = self.buffer[-1]["ts"] - self.buffer[0]["ts"]

        # Tính biên độ góc quét (angular span) có tính đến vòng tròn 360 độ
        # Chuyển đổi sang vector đơn vị trung bình để đo độ phân tán hoặc cực trị góc
        rads = [math.radians(a) for a in angles]
        xs = [math.cos(r) for r in rads]
        ys = [math.sin(r) for r in rads]

        # Khoảng biến thiên góc đơn giản (max - min)
        min_ang = min(angles)
        max_ang = max(angles)
        span_deg = max_ang - min_ang

        # Phát hiện trạng thái đang quan sát chuyển động (chưa đủ span hoặc duration)
        is_observing = (span_deg >= self.config.observing_span_deg)

        # Điều kiện xác nhận Số 10:
        # 1. Thời lượng chuyển động nằm trong khoảng cho phép [min_duration, max_duration]
        # 2. Biên độ góc quét >= min_angular_span_deg (ít nhất 40 độ xoay)
        # 3. Phải có ít nhất 1 frame ngón trỏ duỗi thẳng hướng lên (abs(angle) <= 30°)
        has_upright_frame = any(abs(a) <= 30.0 for a in angles)

        if (
            duration >= self.config.min_duration
            and span_deg >= self.config.min_angular_span_deg
            and has_upright_frame
            and not in_cooldown
        ):
            self.last_detected_time = now
            self.buffer.clear()
            conf = min(0.98, 0.85 + (span_deg / 180.0) * 0.1)
            return {
                "detected": True,
                "confidence": round(conf, 2),
                "is_observing": False,
                "angular_span_deg": round(span_deg, 1),
                "num_frames": len(angles)
            }

        return {
            "detected": False,
            "confidence": 0.0,
            "is_observing": is_observing,
            "angular_span_deg": round(span_deg, 1),
            "num_frames": len(angles)
        }

