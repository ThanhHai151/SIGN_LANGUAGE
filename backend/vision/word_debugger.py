"""
word_debugger.py: Isolated black-box logging module exclusively for VSL Word Mode.
Tự động ghi nhận thông số kỹ thuật (Top 1/2/3, confidence, tensors) mỗi khi inference mode words.
Hoàn toàn không can thiệp hay thay đổi UI/Frontend.
"""
import os
import json
from datetime import datetime
from typing import Optional, Dict, Any
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEBUG_BASE_DIR = os.path.join(PROJECT_ROOT, "debug", "word_sessions")


class WordDebugger:
    def __init__(self):
        self.session_dir: Optional[str] = None
        self.events_file: Optional[str] = None
        self.sequences_dir: Optional[str] = None
        self.seq_counter: int = 0
        self.last_top1_label: Optional[str] = None
        self._ensure_session()

    def _ensure_session(self):
        if not self.session_dir or not os.path.exists(self.session_dir):
            now_str = datetime.now().strftime("%Y-%m-%d_%H%M%S")
            session_id = f"word_debug_{now_str}"
            self.session_dir = os.path.join(DEBUG_BASE_DIR, session_id)
            self.sequences_dir = os.path.join(self.session_dir, "sequences")
            os.makedirs(self.sequences_dir, exist_ok=True)
            self.events_file = os.path.join(self.session_dir, "events.jsonl")
            self.seq_counter = 0
            self.log_event({
                "event": "SESSION_START",
                "session_id": session_id,
                "created_at": datetime.now().isoformat()
            })
            print(f"\n[WordDebugger] Đã tạo phiên debug mới tại: {self.session_dir}")

    def log_event(self, event_data: Dict[str, Any]):
        self._ensure_session()
        if "time" not in event_data:
            event_data["time"] = datetime.now().strftime("%H:%M:%S.%f")[:-3]

        try:
            with open(self.events_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event_data, ensure_ascii=False) + "\n")
        except Exception as e:
            print(f"[WordDebugger Error] write event failed: {e}")

    def record_prediction(self, arr_input: np.ndarray, top_k_list: list, max_prob: float, pred_term: str):
        self._ensure_session()
        self.seq_counter += 1
        seq_idx = self.seq_counter
        filename = f"seq_{seq_idx:03d}.npy"
        filepath = os.path.join(self.sequences_dir, filename)

        try:
            # 1. Lưu chính xác mảng 30x75x3
            np.save(filepath, arr_input.astype(np.float32))

            # 2. Thống kê đặc tính khớp
            pose_zeros = int(np.all(arr_input[:, :33, :] == 0, axis=(1, 2)).sum())
            lh_zeros = int(np.all(arr_input[:, 33:54, :] == 0, axis=(1, 2)).sum())
            rh_zeros = int(np.all(arr_input[:, 54:75, :] == 0, axis=(1, 2)).sum())

            top1 = top_k_list[0] if len(top_k_list) > 0 else {}
            top2 = top_k_list[1] if len(top_k_list) > 1 else {}
            top3 = top_k_list[2] if len(top_k_list) > 2 else {}

            current_label = top1.get("term", "UNKNOWN")

            # 3. In trực tiếp ra console terminal để xem ngay tức thì
            print(f"\n[WORD PREDICTION #{seq_idx:03d}] {datetime.now().strftime('%H:%M:%S.%f')[:-3]}")
            print(f"  👉 Top 1: {top1.get('term')} (index: {top1.get('index')}, conf: {top1.get('confidence'):.3f})")
            print(f"     Top 2: {top2.get('term')} (index: {top2.get('index')}, conf: {top2.get('confidence'):.3f})")
            print(f"     Top 3: {top3.get('term')} (index: {top3.get('index')}, conf: {top3.get('confidence'):.3f})")
            print(f"  📊 Đã lưu tensor: {filepath}")

            # 4. Ghi log sự kiện vào events.jsonl
            pred_event = {
                "event": "PREDICTION",
                "sequence_id": seq_idx,
                "sequence_file": filename,
                "shape": list(arr_input.shape),
                "stats": {
                    "min": float(np.min(arr_input)),
                    "max": float(np.max(arr_input)),
                    "frames_with_pose_zero": pose_zeros,
                    "frames_with_left_hand_zero": lh_zeros,
                    "frames_with_right_hand_zero": rh_zeros,
                },
                "top1": {"index": top1.get("index"), "label": top1.get("term"), "conf": top1.get("confidence")},
                "top2": {"index": top2.get("index"), "label": top2.get("term"), "conf": top2.get("confidence")},
                "top3": {"index": top3.get("index"), "label": top3.get("term"), "conf": top3.get("confidence")},
            }
            self.log_event(pred_event)

            # 5. Ghi log PREDICTION_CHANGED nếu Top-1 nhảy nhãn
            if self.last_top1_label is not None and self.last_top1_label != current_label:
                print(f"  ⚡ NHẢY NHÃN: {self.last_top1_label} ➔ {current_label}")
                self.log_event({
                    "event": "PREDICTION_CHANGED",
                    "sequence_id": seq_idx,
                    "previous": self.last_top1_label,
                    "current": current_label,
                    "top1_conf": top1.get("confidence")
                })
            self.last_top1_label = current_label

        except Exception as e:
            print(f"[WordDebugger Error] record prediction failed: {e}")


word_debugger = WordDebugger()
