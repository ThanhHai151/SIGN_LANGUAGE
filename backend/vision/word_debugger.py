"""
word_debugger.py: Công cụ debug/logging CÔ LẬP cho Word Mode (mode == "words").

- CHỈ ghi log khi có phiên test đang chạy (người test bấm "Start Test" và chọn target).
  Không có phiên => mọi hàm record_* là no-op, không tạo file/thư mục nào.
- Không thay đổi kết quả nhận diện: module này chỉ ĐỌC dữ liệu đầu vào / kết quả dự đoán.
- Ground truth (target_word, target_class, trial) lấy từ mapping chính thức SIGN_DICT_102.

Cấu trúc log:
    debug/words/<Nam>_trial_01_<YYYY-MM-DD_HHMMSS>/
        events.jsonl
        session.json
        sequences/sequence_001.npy ...
"""
import os
import re
import json
import threading
import unicodedata
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np

from backend.models.vsl_102_dictionary import SIGN_DICT_102

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEBUG_BASE_DIR = os.path.join(PROJECT_ROOT, "debug", "words")

# Word Mode: 63 class cuối của SSTCN (không đổi mapping, chỉ đọc)
WORD_CLASS_MIN = 39
WORD_CLASS_MAX = 101

CHECKPOINT_NAME = "sign_sstcn_attention_model.pth"
SEQ_LEN = 30

# Các event mà frontend được phép gửi lên (frontend là nơi duy nhất biết buffer frame)
FRONTEND_EVENT_TYPES = {"SEQUENCE_RESET"}


def get_word_targets() -> List[Dict[str, Any]]:
    """63 từ Word Mode (class 39..101) lấy trực tiếp từ SIGN_DICT_102."""
    return [
        {"class": idx, "word": SIGN_DICT_102[idx]}
        for idx in range(WORD_CLASS_MIN, WORD_CLASS_MAX + 1)
        if idx in SIGN_DICT_102
    ]


def sanitize_filename(text: str) -> str:
    """'Năm' -> 'Nam', 'Đêm' -> 'Dem', 'Con gái' -> 'Con_gai' (chỉ dùng cho tên thư mục)."""
    text = text.replace("đ", "d").replace("Đ", "D")
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")
    return text or "word"


class WordDebugger:
    def __init__(self):
        self._lock = threading.RLock()
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self):
        self.active: bool = False
        self.session_id: str = ""
        self.session_dir: str = ""
        self.events_file: str = ""
        self.sequences_dir: str = ""
        self.target_word: Optional[str] = None
        self.target_class: Optional[int] = None
        self.trial: Optional[int] = None
        self.start_time: str = ""
        self.seq_counter: int = 0
        self.last_top1_label: Optional[str] = None

    # ---------------------------------------------------------------- session
    def start_session(self, target_class: Optional[int] = None, target_word: Optional[str] = None,
                      trial: int = 1) -> Dict[str, Any]:
        with self._lock:
            # Resolve ground truth từ mapping chính thức, không tự đoán
            if target_class is None and target_word:
                matches = [t["class"] for t in get_word_targets() if t["word"] == target_word]
                if len(matches) != 1:
                    return {"success": False, "message": f"Không xác định được class cho từ '{target_word}'"}
                target_class = matches[0]
            if target_class is None or not (WORD_CLASS_MIN <= int(target_class) <= WORD_CLASS_MAX) \
                    or int(target_class) not in SIGN_DICT_102:
                return {"success": False,
                        "message": f"target_class phải nằm trong {WORD_CLASS_MIN}..{WORD_CLASS_MAX}"}
            target_class = int(target_class)
            word = SIGN_DICT_102[target_class]
            if target_word and target_word != word:
                return {"success": False,
                        "message": f"target_word '{target_word}' không khớp class {target_class} ('{word}')"}
            try:
                trial = int(trial)
            except (TypeError, ValueError):
                return {"success": False, "message": "trial phải là số nguyên"}
            if trial < 1:
                return {"success": False, "message": "trial phải >= 1"}

            # Phiên cũ còn mở => đóng trước, không ghi lẫn vào phiên mới
            if self.active:
                self._end_session_locked(reason="replaced_by_new_session")

            now = datetime.now()
            base = f"{sanitize_filename(word)}_trial_{trial:02d}_{now.strftime('%Y-%m-%d_%H%M%S')}"
            session_id, n = base, 1
            while os.path.exists(os.path.join(DEBUG_BASE_DIR, session_id)):  # không overwrite
                n += 1
                session_id = f"{base}_{n}"

            self._reset_state()
            self.active = True
            self.session_id = session_id
            self.session_dir = os.path.join(DEBUG_BASE_DIR, session_id)
            self.sequences_dir = os.path.join(self.session_dir, "sequences")
            os.makedirs(self.sequences_dir, exist_ok=True)
            self.events_file = os.path.join(self.session_dir, "events.jsonl")
            self.target_word = word
            self.target_class = target_class
            self.trial = trial
            self.start_time = now.isoformat()

            self._write_session_json(end_time=None)
            self.log_event("SESSION_START", {"start_time": self.start_time})
            print(f"\n[WordDebug] START  target='{word}' (class {target_class}) trial={trial}\n"
                  f"            -> {self.session_dir}")
            return {"success": True, **self.get_status()}

    def stop_session(self) -> Dict[str, Any]:
        with self._lock:
            if not self.active:
                return {"success": True, "message": "Không có phiên nào đang chạy", **self.get_status()}
            info = self._end_session_locked(reason="user_stop")
            return {"success": True, "message": "Đã kết thúc phiên", **info}

    def _end_session_locked(self, reason: str) -> Dict[str, Any]:
        end_time = datetime.now().isoformat()
        self.log_event("SESSION_END", {
            "end_time": end_time,
            "total_sequences": self.seq_counter,
            "reason": reason,
        })
        self._write_session_json(end_time=end_time)
        info = {"session_id": self.session_id, "session_dir": self.session_dir,
                "target_word": self.target_word, "target_class": self.target_class,
                "trial": self.trial, "total_sequences": self.seq_counter}
        print(f"[WordDebug] STOP   {self.session_id} ({self.seq_counter} sequences)")
        self._reset_state()  # sau Stop: không còn gì được ghi vào phiên này
        return info

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active": self.active,
                "session_id": self.session_id or None,
                "target_word": self.target_word,
                "target_class": self.target_class,
                "trial": self.trial,
                "seq_count": self.seq_counter,
            }

    def _write_session_json(self, end_time: Optional[str]):
        data = {
            "session_id": self.session_id,
            "target_word": self.target_word,
            "target_class": self.target_class,
            "trial": self.trial,
            "mode": "words",
            "checkpoint": CHECKPOINT_NAME,
            "word_class_range": [WORD_CLASS_MIN, WORD_CLASS_MAX],
            "start_time": self.start_time,
            "end_time": end_time,
        }
        try:
            with open(os.path.join(self.session_dir, "session.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WordDebug Error] write session.json failed: {e}")

    # ----------------------------------------------------------------- events
    def log_event(self, event_type: str, data: Optional[Dict[str, Any]] = None):
        """Ghi 1 event vào phiên đang chạy. Không có phiên => bỏ qua."""
        with self._lock:
            if not self.active:
                return
            event: Dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "session_id": self.session_id,
                "target_word": self.target_word,
                "target_class": self.target_class,
                "trial": self.trial,
                "event_type": event_type,
            }
            if data:
                event.update(data)
            try:
                with open(self.events_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(event, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[WordDebug Error] write event failed: {e}")

    def log_frontend_event(self, event_type: str, data: Dict[str, Any]) -> bool:
        """Event do frontend gửi (chỉ whitelist). Trả False nếu bị bỏ qua."""
        if event_type not in FRONTEND_EVENT_TYPES:
            return False
        with self._lock:
            if not self.active:
                return False
            self.log_event(event_type, data)
            return True

    def record_prediction(self, arr_input: np.ndarray, top_k_list: list, max_prob: float,
                          pred_term: str, threshold: Optional[float] = None):
        """Ghi 1 lần inference của Word Mode. No-op nếu không có phiên đang chạy."""
        with self._lock:
            if not self.active:
                return
            try:
                self.seq_counter += 1
                seq_idx = self.seq_counter
                filename = f"sequence_{seq_idx:03d}.npy"
                np.save(os.path.join(self.sequences_dir, filename), arr_input.astype(np.float32))

                # Trạng thái tay/pose: tính thẳng từ tensor thật (frame toàn 0 = không phát hiện)
                n_frames = int(arr_input.shape[0])
                pose_zero = int(np.all(arr_input[:, :33, :] == 0, axis=(1, 2)).sum())
                lh_zero = int(np.all(arr_input[:, 33:54, :] == 0, axis=(1, 2)).sum())
                rh_zero = int(np.all(arr_input[:, 54:75, :] == 0, axis=(1, 2)).sum())

                def top(i: int) -> Dict[str, Any]:
                    return top_k_list[i] if len(top_k_list) > i else {}

                t1, t2, t3 = top(0), top(1), top(2)
                top1_class = t1.get("index")
                current_label = t1.get("term", "UNKNOWN")

                event = {
                    "sequence_id": seq_idx,
                    "sequence_file": f"sequences/{filename}",
                    "frame_count": n_frames,
                    "sequence_length": SEQ_LEN,
                    "sequence_complete": n_frames == SEQ_LEN,
                    "shape": list(arr_input.shape),
                    "pose_detected": pose_zero < n_frames,
                    "left_hand_detected": lh_zero < n_frames,
                    "right_hand_detected": rh_zero < n_frames,
                    "pose_zero_frames": pose_zero,
                    "left_hand_zero_frames": lh_zero,
                    "right_hand_zero_frames": rh_zero,
                    "input_min": float(np.min(arr_input)),
                    "input_max": float(np.max(arr_input)),
                    "top1_class": top1_class, "top1_label": t1.get("term"), "top1_confidence": t1.get("confidence"),
                    "top2_class": t2.get("index"), "top2_label": t2.get("term"), "top2_confidence": t2.get("confidence"),
                    "top3_class": t3.get("index"), "top3_label": t3.get("term"), "top3_confidence": t3.get("confidence"),
                    # Chỉ đúng/sai TẠI THỜI ĐIỂM này, không kết luận cả trial
                    "prediction_correct": (top1_class == self.target_class) if top1_class is not None else False,
                    "displayed_label": pred_term,   # NEUTRAL nếu dưới ngưỡng
                    "threshold": threshold,
                }
                self.log_event("PREDICTION", event)

                mark = "OK " if event["prediction_correct"] else "SAI"
                print(f"[WordDebug] #{seq_idx:03d} target={self.target_word} -> top1={t1.get('term')} "
                      f"({t1.get('confidence')}) {mark} | top2={t2.get('term')} ({t2.get('confidence')}) "
                      f"| zero L/R/pose={lh_zero}/{rh_zero}/{pose_zero}")

                if self.last_top1_label is not None and self.last_top1_label != current_label:
                    self.log_event("PREDICTION_CHANGED", {
                        "sequence_id": seq_idx,
                        "previous": self.last_top1_label,
                        "current": current_label,
                        "top1_confidence": t1.get("confidence"),
                    })
                self.last_top1_label = current_label
            except Exception as e:
                print(f"[WordDebug Error] record prediction failed: {e}")


word_debugger = WordDebugger()
