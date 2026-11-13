"""
word_debugger.py: Công cụ debug/logging CÔ LẬP cho Word Mode (mode == "words").

- CHỈ ghi log khi có phiên test đang chạy (người test bấm "Start Test" và chọn target).
  Không có phiên => mọi hàm record_* là no-op, không tạo file/thư mục nào.
- Không thay đổi kết quả nhận diện: module này chỉ ĐỌC dữ liệu đầu vào / kết quả dự đoán.
- Ground truth (target_word, target_class, trial) lấy từ mapping chính thức SIGN_DICT_102.
- Ghi nhận thông tin định danh checkpoint thực tế từ backend (VSL-P008).

Cấu trúc log:
    debug/words/<Nam>_trial_01_<YYYY-MM-DD_HHMMSS>/
        events.jsonl
        session.json
        trial_summary.csv
        sequences/sequence_001.npy ...
    debug/words/trials_summary.csv (Master CSV tổng hợp tất cả các trial)
"""
import os
import re
import csv
import json
import threading
import unicodedata
from collections import Counter
from datetime import datetime
from typing import Optional, Dict, Any, List

import numpy as np

from backend.models.vsl_102_dictionary import SIGN_DICT_102

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEBUG_BASE_DIR = os.path.join(PROJECT_ROOT, "debug", "words")
MASTER_CSV_PATH = os.path.join(DEBUG_BASE_DIR, "trials_summary.csv")

# Word Mode: 63 class cuối của SSTCN (không đổi mapping, chỉ đọc)
WORD_CLASS_MIN = 39
WORD_CLASS_MAX = 101

SEQ_LEN = 30

# Các event mà frontend được phép gửi lên (frontend là nơi duy nhất biết buffer frame & trạng thái tracker)
FRONTEND_EVENT_TYPES = {
    "SEQUENCE_RESET",
    "HOLISTIC_UNAVAILABLE",
    "ENGINE_FALLBACK_BLOCKED",
    "POSE_TRACKING_LOST",
}

CSV_FIELDNAMES = [
    "session_id",
    "trial",
    "target_word",
    "target_class",
    "verdict",
    "checkpoint_type",
    "checkpoint_filename",
    "checkpoint_sha256",
    "checkpoint_epoch",
    "model_device",
    "total_windows",
    "correct_windows",
    "accuracy",
    "dominant_predicted_label",
    "target_max_confidence",
    "total_resets",
    "landmarks_source",
    "is_static_shoulders",
    "avg_client_fps",
    "avg_inference_ms",
    "avg_window_duration_ms",
    "start_time",
    "end_time"
]


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


def _calc_max_consecutive_zeros(zero_mask: np.ndarray) -> int:
    """Đếm chuỗi frame zero liên tiếp dài nhất."""
    max_len = 0
    curr_len = 0
    for is_zero in zero_mask:
        if is_zero:
            curr_len += 1
            if curr_len > max_len:
                max_len = curr_len
        else:
            curr_len = 0
    return max_len


def analyze_sequence_landmarks(arr_input: np.ndarray, debug_meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Phân tích nguồn gốc và chất lượng 75 landmarks của chuỗi (30, 75, 3).
    Phát hiện chính xác:
      - 'hands_fallback_pose': Pose giả tĩnh từ DEFAULT_POSE (fixed_pose_landmarks_count >= 25 và pose không phải toàn 0,
        hoặc tracker_engine/active_instance_engine == 'hands').
      - 'holistic_pose_missing': Mất toàn bộ hoặc đa số Pose thân trên trong cửa sổ (> 15/30 frames bằng 0).
      - 'holistic_raw': Khung xương 75 khớp thật từ MediaPipe Holistic.
    """
    debug_meta = debug_meta or {}
    n_frames = int(arr_input.shape[0])

    # A. Trạng thái thiếu landmarks (Zeros)
    pose_zero_mask = np.all(arr_input[:, :33, :] == 0, axis=(1, 2))
    lh_zero_mask = np.all(arr_input[:, 33:54, :] == 0, axis=(1, 2))
    rh_zero_mask = np.all(arr_input[:, 54:75, :] == 0, axis=(1, 2))
    both_hands_zero_mask = np.logical_and(lh_zero_mask, rh_zero_mask)

    pose_zero = int(pose_zero_mask.sum())
    lh_zero = int(lh_zero_mask.sum())
    rh_zero = int(rh_zero_mask.sum())
    both_hands_zero = int(both_hands_zero_mask.sum())

    max_consec_missing_pose = _calc_max_consecutive_zeros(pose_zero_mask)
    max_consec_missing_lh = _calc_max_consecutive_zeros(lh_zero_mask)
    max_consec_missing_rh = _calc_max_consecutive_zeros(rh_zero_mask)

    total_landmarks = int(arr_input.shape[0] * arr_input.shape[1])  # 30 * 75 = 2250
    zero_landmarks_count = int(np.all(arr_input == 0, axis=2).sum())
    total_missing_landmarks_ratio = round(zero_landmarks_count / total_landmarks, 4) if total_landmarks > 0 else 1.0

    # B. Biến thiên khớp vai, khuỷu tay, cổ tay & nguồn landmarks
    shoulder_l_std = float(np.mean(np.std(arr_input[:, 11, :2], axis=0)))
    shoulder_r_std = float(np.mean(np.std(arr_input[:, 12, :2], axis=0)))
    elbow_l_std = float(np.mean(np.std(arr_input[:, 13, :2], axis=0)))
    elbow_r_std = float(np.mean(np.std(arr_input[:, 14, :2], axis=0)))
    wrist_l_std = float(np.mean(np.std(arr_input[:, 15, :2], axis=0)))
    wrist_r_std = float(np.mean(np.std(arr_input[:, 16, :2], axis=0)))

    pose_active_mask = np.any(arr_input[:, :33, :] != 0, axis=(0, 2))  # (33,)
    pose_mean_stds = np.mean(np.std(arr_input[:, :33, :2], axis=0), axis=1)
    # Khớp được coi là fixed fake pose nếu nó có dữ liệu (không phải 0) nhưng độ lệch chuẩn gần như bằng 0
    fixed_pose_landmarks_count = int(np.sum((pose_mean_stds < 1e-4) & pose_active_mask))
    is_static_shoulders = bool(shoulder_l_std < 1e-4 and shoulder_r_std < 1e-4 and pose_active_mask[11] and pose_active_mask[12])

    reported_engine = str(debug_meta.get("active_instance_engine") or debug_meta.get("tracker_engine") or "unknown").lower()
    reported_source = str(debug_meta.get("landmarks_source") or "").lower()

    # Nhận diện chữ ký DEFAULT_POSE (vai cố định [0.65, 0.45] & [0.35, 0.45] hoặc >= 25 khớp pose bất động khác 0)
    non_zero_pose_frames = n_frames - pose_zero
    has_default_pose_shoulders = bool(
        non_zero_pose_frames > 0
        and np.allclose(arr_input[:, 11, :2], [0.65, 0.45], atol=1e-3)
        and np.allclose(arr_input[:, 12, :2], [0.35, 0.45], atol=1e-3)
    )
    is_synthetic_pose = bool(
        has_default_pose_shoulders
        or reported_engine == "hands"
        or reported_source == "hands_fallback_pose"
        or (non_zero_pose_frames > 0 and fixed_pose_landmarks_count >= 25)
    )

    if is_synthetic_pose:
        landmarks_source = "hands_fallback_pose"
    elif pose_zero >= n_frames:
        landmarks_source = "holistic_pose_missing"
    else:
        landmarks_source = "holistic_raw"

    # C. Đánh giá tính hợp lệ cho SSTCN ở mode 'words' (A01.1: chặn pose giả & chuỗi mất pose/mất cả 2 tay)
    blocked_reason = None
    if is_synthetic_pose or landmarks_source == "hands_fallback_pose":
        blocked_reason = "hands_fallback_pose_blocked"
    elif pose_zero > (n_frames // 2):
        blocked_reason = "holistic_pose_missing"
    elif both_hands_zero >= n_frames:
        blocked_reason = "both_hands_missing"

    return {
        "n_frames": n_frames,
        "landmarks_source": landmarks_source,
        "is_synthetic_pose": is_synthetic_pose,
        "is_valid_for_words_sstcn": blocked_reason is None,
        "blocked_reason": blocked_reason,
        "is_static_shoulders": is_static_shoulders,
        "fixed_pose_landmarks_count": fixed_pose_landmarks_count,
        "shoulder_left_std": round(shoulder_l_std, 6),
        "shoulder_right_std": round(shoulder_r_std, 6),
        "elbow_left_std": round(elbow_l_std, 6),
        "elbow_right_std": round(elbow_r_std, 6),
        "wrist_left_std": round(wrist_l_std, 6),
        "wrist_right_std": round(wrist_r_std, 6),
        "pose_zero_frames": pose_zero,
        "left_hand_zero_frames": lh_zero,
        "right_hand_zero_frames": rh_zero,
        "both_hands_zero_frames": both_hands_zero,
        "pose_missing_ratio": round(pose_zero / n_frames, 4) if n_frames > 0 else 1.0,
        "left_hand_missing_ratio": round(lh_zero / n_frames, 4) if n_frames > 0 else 1.0,
        "right_hand_missing_ratio": round(rh_zero / n_frames, 4) if n_frames > 0 else 1.0,
        "both_hands_missing_ratio": round(both_hands_zero / n_frames, 4) if n_frames > 0 else 1.0,
        "max_consecutive_missing_pose": max_consec_missing_pose,
        "max_consecutive_missing_lh": max_consec_missing_lh,
        "max_consecutive_missing_rh": max_consec_missing_rh,
        "total_missing_landmarks_ratio": total_missing_landmarks_ratio,
    }


class WordDebugger:
    def __init__(self):
        self._lock = threading.RLock()
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _reset_state(self):
        self.active: bool = False
        self.session_prefix: Optional[str] = None
        self.session_id: str = ""
        self.session_dir: str = ""
        self.events_file: str = ""
        self.sequences_dir: str = ""
        self.target_word: Optional[str] = None
        self.target_class: Optional[int] = None
        self.trial: Optional[int] = None
        self.start_time: str = ""
        self.seq_counter: int = 0
        self.reset_counter: int = 0
        self.blocked_fallback_counter: int = 0
        self.last_top1_label: Optional[str] = None
        self.session_windows: List[Dict[str, Any]] = []
        self.checkpoint_info: Dict[str, Any] = {
            "checkpoint_type": "unknown",
            "checkpoint_filename": "unknown",
            "checkpoint_sha256": "unknown",
            "checkpoint_epoch": None,
            "model_device": "unknown"
        }
        self.checkpoint_mismatch: bool = False

    # ---------------------------------------------------------------- session
    def start_session(self, target_class: Optional[int] = None, target_word: Optional[str] = None,
                      trial: int = 1, checkpoint_info: Optional[Dict[str, Any]] = None,
                      session_prefix: Optional[str] = "A01_1_T01") -> Dict[str, Any]:
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
            pfx = f"{session_prefix.strip('_')}_" if session_prefix and session_prefix.strip('_') else ""
            base = f"{pfx}{sanitize_filename(word)}_trial_{trial:02d}_{now.strftime('%Y-%m-%d_%H%M%S')}"
            session_id, n = base, 1
            os.makedirs(DEBUG_BASE_DIR, exist_ok=True)
            while os.path.exists(os.path.join(DEBUG_BASE_DIR, session_id)):  # không overwrite
                n += 1
                session_id = f"{base}_{n}"

            self._reset_state()
            self.active = True
            self.session_prefix = session_prefix.strip('_') if session_prefix else None
            self.session_id = session_id
            self.session_dir = os.path.join(DEBUG_BASE_DIR, session_id)
            self.sequences_dir = os.path.join(self.session_dir, "sequences")
            os.makedirs(self.sequences_dir, exist_ok=True)
            self.events_file = os.path.join(self.session_dir, "events.jsonl")
            self.target_word = word
            self.target_class = target_class
            self.trial = trial
            self.start_time = now.isoformat()

            # Lưu metadata checkpoint định danh thực tế (VSL-P008)
            if checkpoint_info and isinstance(checkpoint_info, dict):
                self.checkpoint_info = {
                    "checkpoint_type": str(checkpoint_info.get("checkpoint_type", "unknown")),
                    "checkpoint_filename": str(checkpoint_info.get("checkpoint_filename", "unknown")),
                    "checkpoint_sha256": str(checkpoint_info.get("checkpoint_sha256", "unknown")),
                    "checkpoint_epoch": checkpoint_info.get("checkpoint_epoch"),
                    "model_device": str(checkpoint_info.get("model_device", "unknown"))
                }
            else:
                self.checkpoint_info = {
                    "checkpoint_type": "unknown",
                    "checkpoint_filename": "unknown",
                    "checkpoint_sha256": "unknown",
                    "checkpoint_epoch": None,
                    "model_device": "unknown"
                }
            self.checkpoint_mismatch = False

            self._write_session_json(end_time=None, summary=None)
            self.log_event("SESSION_START", {
                "start_time": self.start_time,
                "checkpoint": self.checkpoint_info
            })
            print(f"\n[WordDebug] START  target='{word}' (class {target_class}) trial={trial}\n"
                  f"            checkpoint='{self.checkpoint_info['checkpoint_filename']}' ({self.checkpoint_info['checkpoint_type']})\n"
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

        # Tổng kết Trial
        total_windows = len(self.session_windows)
        correct_windows = sum(1 for w in self.session_windows if w.get("prediction_correct"))
        accuracy = round(correct_windows / total_windows, 4) if total_windows > 0 else 0.0

        target_max_conf = 0.0
        if self.session_windows:
            target_max_conf = max((w.get("target_confidence", 0.0) for w in self.session_windows), default=0.0)

        labels = [w.get("top1_label") for w in self.session_windows if w.get("top1_label")]
        dominant_label = Counter(labels).most_common(1)[0][0] if labels else "NONE"

        # Đánh giá Verdict theo trial (kèm kiểm tra tính nhất quán checkpoint VSL-P008)
        if self.checkpoint_mismatch:
            verdict = "INVALID_CHECKPOINT_MISMATCH"
        elif dominant_label == self.target_word and (correct_windows > 0):
            verdict = "CORRECT"
        elif correct_windows > 0:
            verdict = "PARTIAL"
        else:
            verdict = "FAIL"

        # Tính trung bình FPS, inference ms, window duration ms
        fps_list = [w.get("client_fps") for w in self.session_windows if w.get("client_fps") is not None]
        avg_client_fps = round(float(np.mean(fps_list)), 1) if fps_list else None

        infer_list = [w.get("inference_latency_ms") for w in self.session_windows if w.get("inference_latency_ms") is not None]
        avg_inference_ms = round(float(np.mean(infer_list)), 2) if infer_list else None

        dur_list = [w.get("window_duration_ms") for w in self.session_windows if w.get("window_duration_ms") is not None]
        avg_window_dur = round(float(np.mean(dur_list)), 1) if dur_list else None

        # Nguồn landmarks phổ biến nhất trong session
        sources = [w.get("landmarks_source") for w in self.session_windows if w.get("landmarks_source")]
        dominant_source = Counter(sources).most_common(1)[0][0] if sources else "unknown"

        static_shoulders = any(w.get("is_static_shoulders") for w in self.session_windows) if self.session_windows else False

        summary = {
            "session_id": self.session_id,
            "trial": self.trial,
            "target_word": self.target_word,
            "target_class": self.target_class,
            "verdict": verdict,
            "checkpoint_type": self.checkpoint_info.get("checkpoint_type", "unknown"),
            "checkpoint_filename": self.checkpoint_info.get("checkpoint_filename", "unknown"),
            "checkpoint_sha256": self.checkpoint_info.get("checkpoint_sha256", "unknown"),
            "checkpoint_epoch": self.checkpoint_info.get("checkpoint_epoch"),
            "model_device": self.checkpoint_info.get("model_device", "unknown"),
            "checkpoint_mismatch": self.checkpoint_mismatch,
            "total_windows": total_windows,
            "correct_windows": correct_windows,
            "accuracy": accuracy,
            "dominant_predicted_label": dominant_label,
            "target_max_confidence": round(float(target_max_conf), 3),
            "total_resets": self.reset_counter,
            "landmarks_source": dominant_source,
            "is_static_shoulders": static_shoulders,
            "avg_client_fps": avg_client_fps,
            "avg_inference_ms": avg_inference_ms,
            "avg_window_duration_ms": avg_window_dur,
            "start_time": self.start_time,
            "end_time": end_time
        }

        # Ghi sự kiện kết thúc
        self.log_event("SESSION_END", {
            "end_time": end_time,
            "total_sequences": self.seq_counter,
            "reason": reason,
            "summary": summary
        })

        # Ghi session.json với summary
        self._write_session_json(end_time=end_time, summary=summary)

        # Ghi CSV summary cho phiên và master CSV
        self._write_trial_csv(summary)

        info = {
            "session_id": self.session_id,
            "session_dir": self.session_dir,
            "target_word": self.target_word,
            "target_class": self.target_class,
            "trial": self.trial,
            "total_sequences": self.seq_counter,
            "summary": summary
        }
        print(f"[WordDebug] STOP   {self.session_id} ({self.seq_counter} sequences) -> Verdict: {verdict}")
        self._reset_state()  # sau Stop: không còn gì được ghi vào phiên này
        return info

    def get_status(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "active": self.active,
                "session_id": self.session_id or None,
                "campaign": self.session_prefix,
                "target_word": self.target_word,
                "target_class": self.target_class,
                "trial": self.trial,
                "seq_count": self.seq_counter,
                "reset_count": self.reset_counter,
                "blocked_fallback_count": self.blocked_fallback_counter,
                "checkpoint": self.checkpoint_info
            }

    def _write_session_json(self, end_time: Optional[str], summary: Optional[Dict[str, Any]] = None):
        data = {
            "session_id": self.session_id,
            "campaign": self.session_prefix,
            "target_word": self.target_word,
            "target_class": self.target_class,
            "trial": self.trial,
            "mode": "words",
            "checkpoint": {
                "checkpoint_type": self.checkpoint_info.get("checkpoint_type", "unknown"),
                "checkpoint_filename": self.checkpoint_info.get("checkpoint_filename", "unknown"),
                "checkpoint_sha256": self.checkpoint_info.get("checkpoint_sha256", "unknown"),
                "checkpoint_epoch": self.checkpoint_info.get("checkpoint_epoch"),
                "model_device": self.checkpoint_info.get("model_device", "unknown")
            },
            "word_class_range": [WORD_CLASS_MIN, WORD_CLASS_MAX],
            "start_time": self.start_time,
            "end_time": end_time,
            "summary": summary
        }
        try:
            with open(os.path.join(self.session_dir, "session.json"), "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[WordDebug Error] write session.json failed: {e}")

    def _write_trial_csv(self, summary: Dict[str, Any]):
        """Ghi CSV tổng hợp phiên test và append vào master CSV."""
        try:
            # 1. Ghi file trial_summary.csv trong session_dir
            session_csv = os.path.join(self.session_dir, "trial_summary.csv")
            with open(session_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
                writer.writerow({k: summary.get(k) for k in CSV_FIELDNAMES})

            # 2. Append vào master CSV debug/words/trials_summary.csv
            file_exists = os.path.exists(MASTER_CSV_PATH)
            with open(MASTER_CSV_PATH, "a", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                if not file_exists:
                    writer.writeheader()
                writer.writerow({k: summary.get(k) for k in CSV_FIELDNAMES})
        except Exception as e:
            print(f"[WordDebug Error] write trial csv failed: {e}")

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
            if event_type == "SEQUENCE_RESET":
                self.reset_counter += 1
            elif event_type in ("HOLISTIC_UNAVAILABLE", "ENGINE_FALLBACK_BLOCKED", "POSE_TRACKING_LOST"):
                self.blocked_fallback_counter += 1
            self.log_event(event_type, data)
            return True

    def record_blocked_sequence(self, arr_input: np.ndarray, blocked_reason: str,
                                debug_meta: Optional[Dict[str, Any]] = None,
                                checkpoint_info: Optional[Dict[str, Any]] = None,
                                landmark_stats: Optional[Dict[str, Any]] = None):
        """
        Ghi nhận sự kiện chặn chuỗi không hợp lệ (ví dụ: hands_fallback_pose, holistic_pose_missing,
        both_hands_missing) trước khi đi vào SSTCN ở chế độ 'words'. No-op nếu không có phiên đang chạy.
        """
        with self._lock:
            if not self.active:
                return
            try:
                self.blocked_fallback_counter += 1
                debug_meta = debug_meta or {}
                stats = landmark_stats or analyze_sequence_landmarks(arr_input, debug_meta=debug_meta)
                event = {
                    "blocked_id": self.blocked_fallback_counter,
                    "blocked_reason": blocked_reason,
                    "shape": list(arr_input.shape),
                    "frame_count": int(arr_input.shape[0]),
                    "tracker_engine": debug_meta.get("tracker_engine", "unknown"),
                    "active_instance_engine": debug_meta.get("active_instance_engine", "unknown"),
                    "landmarks_source": stats.get("landmarks_source", "unknown"),
                    "is_synthetic_pose": stats.get("is_synthetic_pose", False),
                    "is_static_shoulders": stats.get("is_static_shoulders", False),
                    "fixed_pose_landmarks_count": stats.get("fixed_pose_landmarks_count", 0),
                    "shoulder_left_std": stats.get("shoulder_left_std", 0.0),
                    "shoulder_right_std": stats.get("shoulder_right_std", 0.0),
                    "pose_zero_frames": stats.get("pose_zero_frames", 0),
                    "left_hand_zero_frames": stats.get("left_hand_zero_frames", 0),
                    "right_hand_zero_frames": stats.get("right_hand_zero_frames", 0),
                    "both_hands_zero_frames": stats.get("both_hands_zero_frames", 0),
                    "total_missing_landmarks_ratio": stats.get("total_missing_landmarks_ratio", 0.0),
                    "checkpoint_filename": (checkpoint_info or self.checkpoint_info).get("checkpoint_filename", "unknown"),
                }
                self.log_event("FALLBACK_BLOCKED", event)
                print(f"[WordDebug] BLOCKED #{self.blocked_fallback_counter:03d} reason={blocked_reason} "
                      f"source={stats.get('landmarks_source')} fixed_pose={stats.get('fixed_pose_landmarks_count')}/33")
            except Exception as e:
                print(f"[WordDebug Error] record blocked sequence failed: {e}")

    def record_prediction(self, arr_input: np.ndarray, top_k_list: list, max_prob: float,
                          pred_term: str, threshold: Optional[float] = None,
                          inference_latency_ms: Optional[float] = None,
                          debug_meta: Optional[Dict[str, Any]] = None,
                          checkpoint_info: Optional[Dict[str, Any]] = None):
        """Ghi 1 lần inference của Word Mode. No-op nếu không có phiên đang chạy."""
        with self._lock:
            if not self.active:
                return
            try:
                self.seq_counter += 1
                seq_idx = self.seq_counter
                filename = f"sequence_{seq_idx:03d}.npy"
                np.save(os.path.join(self.sequences_dir, filename), arr_input.astype(np.float32))

                n_frames = int(arr_input.shape[0])  # 30

                # ------------------------------------------------------------------
                # VSL-P008: Kiểm tra tính nhất quán của Checkpoint trong Trial
                # ------------------------------------------------------------------
                if checkpoint_info and isinstance(checkpoint_info, dict):
                    curr_sha = checkpoint_info.get("checkpoint_sha256", "unknown")
                    expected_sha = self.checkpoint_info.get("checkpoint_sha256", "unknown")
                    if curr_sha != expected_sha and expected_sha != "unknown":
                        if not self.checkpoint_mismatch:
                            self.checkpoint_mismatch = True
                            self.log_event("CHECKPOINT_MISMATCH", {
                                "expected_sha256": expected_sha,
                                "current_sha256": curr_sha,
                                "warning": "Checkpoint thay đổi hoặc backend restart giữa trial! Không tổng hợp lẫn lộn."
                            })
                            print(f"[WordDebug Warning] CHECKPOINT MISMATCH! Expected {expected_sha[:8]} but got {curr_sha[:8]}")

                # ------------------------------------------------------------------
                # A. Phân tích trạng thái thiếu landmarks (Zeros)
                # ------------------------------------------------------------------
                pose_zero_mask = np.all(arr_input[:, :33, :] == 0, axis=(1, 2))
                lh_zero_mask = np.all(arr_input[:, 33:54, :] == 0, axis=(1, 2))
                rh_zero_mask = np.all(arr_input[:, 54:75, :] == 0, axis=(1, 2))
                both_hands_zero_mask = np.logical_and(lh_zero_mask, rh_zero_mask)

                pose_zero = int(pose_zero_mask.sum())
                lh_zero = int(lh_zero_mask.sum())
                rh_zero = int(rh_zero_mask.sum())
                both_hands_zero = int(both_hands_zero_mask.sum())

                max_consec_missing_lh = _calc_max_consecutive_zeros(lh_zero_mask)
                max_consec_missing_rh = _calc_max_consecutive_zeros(rh_zero_mask)

                total_landmarks = int(arr_input.shape[0] * arr_input.shape[1])  # 30 * 75 = 2250
                zero_landmarks_count = int(np.all(arr_input == 0, axis=2).sum())
                total_missing_landmarks_ratio = round(zero_landmarks_count / total_landmarks, 4)

                # ------------------------------------------------------------------
                # B. Phân tích biến thiên khớp vai, khuỷu tay, cổ tay & nguồn landmarks
                # ------------------------------------------------------------------
                shoulder_l_std = float(np.mean(np.std(arr_input[:, 11, :2], axis=0)))
                shoulder_r_std = float(np.mean(np.std(arr_input[:, 12, :2], axis=0)))
                elbow_l_std = float(np.mean(np.std(arr_input[:, 13, :2], axis=0)))
                elbow_r_std = float(np.mean(np.std(arr_input[:, 14, :2], axis=0)))
                wrist_l_std = float(np.mean(np.std(arr_input[:, 15, :2], axis=0)))
                wrist_r_std = float(np.mean(np.std(arr_input[:, 16, :2], axis=0)))

                pose_mean_stds = np.mean(np.std(arr_input[:, :33, :2], axis=0), axis=1)
                fixed_pose_landmarks_count = int(np.sum(pose_mean_stds < 1e-4))

                is_static_shoulders = bool(shoulder_l_std < 1e-4 and shoulder_r_std < 1e-4)
                if fixed_pose_landmarks_count >= 25:
                    landmarks_source = "hands_fallback_pose"
                else:
                    landmarks_source = "holistic_raw"

                # ------------------------------------------------------------------
                # C. Top-K predictions & Ground Truth Matching
                # ------------------------------------------------------------------
                def top(i: int) -> Dict[str, Any]:
                    return top_k_list[i] if len(top_k_list) > i else {}

                t1 = top(0)
                top1_class = t1.get("index")
                current_label = t1.get("term", "UNKNOWN")

                top_5 = [
                    {
                        "rank": idx + 1,
                        "class": item.get("index"),
                        "term": item.get("term"),
                        "confidence": item.get("confidence")
                    }
                    for idx, item in enumerate(top_k_list[:5])
                ]

                target_confidence = 0.0
                for item in top_k_list:
                    if item.get("index") == self.target_class:
                        target_confidence = float(item.get("confidence", 0.0))
                        break

                prediction_correct = bool(top1_class == self.target_class) if top1_class is not None else False

                # ------------------------------------------------------------------
                # D. Đóng gói Event
                # ------------------------------------------------------------------
                debug_meta = debug_meta or {}

                event = {
                    "window_id": seq_idx,
                    "sequence_file": f"sequences/{filename}",
                    "shape": list(arr_input.shape),
                    "frame_count": n_frames,
                    "sequence_complete": n_frames == SEQ_LEN,

                    # VSL-P008: Checkpoint Identity (ở cấp prediction event)
                    "checkpoint_type": self.checkpoint_info.get("checkpoint_type", "unknown"),
                    "checkpoint_filename": self.checkpoint_info.get("checkpoint_filename", "unknown"),
                    "checkpoint_sha256": self.checkpoint_info.get("checkpoint_sha256", "unknown"),
                    "checkpoint_epoch": self.checkpoint_info.get("checkpoint_epoch"),
                    "model_device": self.checkpoint_info.get("model_device", "unknown"),

                    # Tracking & Frame Stats
                    "tracker_engine": debug_meta.get("tracker_engine", "unknown"),
                    "client_fps": debug_meta.get("client_fps"),
                    "avg_frame_process_ms": debug_meta.get("avg_frame_process_ms"),
                    "max_frame_process_ms": debug_meta.get("max_frame_process_ms"),
                    "window_duration_ms": debug_meta.get("window_duration_ms"),
                    "consecutive_missing_hands_client": debug_meta.get("consecutive_missing_hands"),
                    "inference_latency_ms": inference_latency_ms,

                    # Landmarks Variation & Origin
                    "landmarks_source": landmarks_source,
                    "is_static_shoulders": is_static_shoulders,
                    "fixed_pose_landmarks_count": fixed_pose_landmarks_count,
                    "shoulder_left_std": round(shoulder_l_std, 6),
                    "shoulder_right_std": round(shoulder_r_std, 6),
                    "elbow_left_std": round(elbow_l_std, 6),
                    "elbow_right_std": round(elbow_r_std, 6),
                    "wrist_left_std": round(wrist_l_std, 6),
                    "wrist_right_std": round(wrist_r_std, 6),

                    # Missing landmarks stats
                    "pose_zero_frames": pose_zero,
                    "left_hand_zero_frames": lh_zero,
                    "right_hand_zero_frames": rh_zero,
                    "both_hands_zero_frames": both_hands_zero,
                    "pose_missing_ratio": round(pose_zero / n_frames, 4),
                    "left_hand_missing_ratio": round(lh_zero / n_frames, 4),
                    "right_hand_missing_ratio": round(rh_zero / n_frames, 4),
                    "both_hands_missing_ratio": round(both_hands_zero / n_frames, 4),
                    "max_consecutive_missing_lh": max_consec_missing_lh,
                    "max_consecutive_missing_rh": max_consec_missing_rh,
                    "total_missing_landmarks_ratio": total_missing_landmarks_ratio,

                    # Prediction Details
                    "top1_class": top1_class,
                    "top1_label": t1.get("term"),
                    "top1_confidence": t1.get("confidence"),
                    "target_class": self.target_class,
                    "target_word": self.target_word,
                    "target_confidence": round(target_confidence, 4),
                    "prediction_correct": prediction_correct,
                    "displayed_label": pred_term,   # NEUTRAL nếu dưới ngưỡng
                    "threshold": threshold,
                    "top_5": top_5
                }
                self.log_event("PREDICTION", event)

                # Lưu vào danh sách các window của session để tổng kết trial
                self.session_windows.append({
                    "window_id": seq_idx,
                    "prediction_correct": prediction_correct,
                    "top1_label": t1.get("term"),
                    "top1_confidence": t1.get("confidence"),
                    "target_confidence": target_confidence,
                    "client_fps": debug_meta.get("client_fps"),
                    "inference_latency_ms": inference_latency_ms,
                    "window_duration_ms": debug_meta.get("window_duration_ms"),
                    "landmarks_source": landmarks_source,
                    "is_static_shoulders": is_static_shoulders,
                    "total_missing_landmarks_ratio": total_missing_landmarks_ratio
                })

                mark = "OK " if prediction_correct else "SAI"
                print(f"[WordDebug] Window #{seq_idx:03d} target={self.target_word} -> top1={t1.get('term')} "
                      f"({t1.get('confidence')}) [{mark}] | ckpt={self.checkpoint_info['checkpoint_filename']} | "
                      f"zero L/R/pose={lh_zero}/{rh_zero}/{pose_zero} | infer={inference_latency_ms}ms")

                if self.last_top1_label is not None and self.last_top1_label != current_label:
                    self.log_event("PREDICTION_CHANGED", {
                        "window_id": seq_idx,
                        "previous": self.last_top1_label,
                        "current": current_label,
                        "top1_confidence": t1.get("confidence"),
                    })
                self.last_top1_label = current_label
            except Exception as e:
                print(f"[WordDebug Error] record prediction failed: {e}")


word_debugger = WordDebugger()
