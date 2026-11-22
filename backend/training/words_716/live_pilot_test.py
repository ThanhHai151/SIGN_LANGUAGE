"""
Standalone Live Webcam Test Tool for WordRecognizer716 Pilot 40 (TASK WORDS 3.2I).

Enables interactive real-time validation of WordRecognizer716 on live webcam streams
comparing 3 distinct configurations on the exact same captured sequence:
  A. Baseline Raw: Baseline Pilot Checkpoint without Body-Centric Normalization.
  B. Baseline + Norm: Baseline Pilot Checkpoint with Body-Centric Normalization.
  C. Retrained + Norm: Body-Centric Retrained Checkpoint with Body-Centric Normalization.

Features:
- OpenCV HighGUI interactive UI with 5-second countdown and auto-stop.
- Key M: Mirror preview toggle (display only; AI input remains strictly unmirrored).
- High-fidelity MediaPipe Holistic extraction & active segment trimming via .venv-sstcn bridge.
- Free-Form mode (free testing anytime) or Guided Protocol mode (12 words x 2 distances = 24 trials).
- Blind Evaluation mode: hides predictions between trials to prevent observer bias.
- Zero disk write by default: captured frames are processed in-memory / temporary buffer.
- Optional --save-videos flag if user explicitly requests saving raw MP4s.
- Detailed JSON / CSV reporting for evaluation sessions.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.training.words_716.model import WordRecognizer716
from backend.training.words_716.normalization import normalize_body_centric

# Default paths
DEFAULT_BASE_CKPT = ROOT / "backend" / "models" / "words_716_checkpoints" / "pilot_40" / "best.pth"
DEFAULT_RETRAIN_CKPT = ROOT / "backend" / "models" / "words_716_checkpoints" / "experiments_body_centric" / "pilot_40" / "best.pth"
DEFAULT_CLASSES_FILE = ROOT / "backend" / "data" / "words_716" / "pilot_40_classes.json"
DEFAULT_PROTOCOL_FILE = ROOT / "backend" / "data" / "words_716" / "holdout_protocol.json"
DEFAULT_SSTCN_PYTHON = ROOT / ".venv-sstcn" / "Scripts" / "python.exe"
DEFAULT_RESULTS_DIR = ROOT / "backend" / "evaluations" / "live_pilot_results"


class PilotModelManager:
    """
    Manages loading and simultaneous inference across the 3 model configurations:
      A: Baseline Raw
      B: Baseline + Norm
      C: Retrained + Norm
    """

    def __init__(
        self,
        base_ckpt_path: Union[str, Path] = DEFAULT_BASE_CKPT,
        retrain_ckpt_path: Union[str, Path] = DEFAULT_RETRAIN_CKPT,
        classes_file: Union[str, Path] = DEFAULT_CLASSES_FILE,
        device: str = "cpu",
    ):
        self.base_ckpt_path = Path(base_ckpt_path)
        self.retrain_ckpt_path = Path(retrain_ckpt_path)
        self.classes_file = Path(classes_file)
        self.device = torch.device(device)

        if not self.classes_file.is_file():
            raise FileNotFoundError(f"Classes config not found: {self.classes_file}")
        if not self.base_ckpt_path.is_file():
            raise FileNotFoundError(f"Baseline checkpoint not found: {self.base_ckpt_path}")
        if not self.retrain_ckpt_path.is_file():
            raise FileNotFoundError(f"Retrained checkpoint not found: {self.retrain_ckpt_path}")

        # Load 40 Pilot classes
        pilot_data = json.loads(self.classes_file.read_text(encoding="utf-8"))
        self.classes = pilot_data.get("classes", pilot_data)
        self.active_indices = [c["class_index"] for c in self.classes]
        self.idx_to_meta = {c["class_index"]: c for c in self.classes}
        self.idx_to_word = {c["class_index"]: c["word"] for c in self.classes}

        # Initialize Model A/B (Baseline)
        self.model_base = WordRecognizer716(
            num_classes=716, active_indices=self.active_indices, dropout=0.0
        )
        ckpt_a = torch.load(self.base_ckpt_path, map_location=self.device, weights_only=False)
        self.model_base.load_state_dict(ckpt_a["model_state_dict"])
        self.model_base.to(self.device).eval()

        # Initialize Model C (Retrained)
        self.model_retrain = WordRecognizer716(
            num_classes=716, active_indices=self.active_indices, dropout=0.0
        )
        ckpt_c = torch.load(self.retrain_ckpt_path, map_location=self.device, weights_only=False)
        self.model_retrain.load_state_dict(ckpt_c["model_state_dict"])
        self.model_retrain.to(self.device).eval()

    def _infer_single(self, model: WordRecognizer716, seq: np.ndarray) -> Dict[str, Any]:
        """Run masked inference on a [3, 30, 75] sequence and return top-3."""
        t = torch.from_numpy(seq).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            masked_logits = model(t, apply_class_mask=True)
            active_logits = masked_logits[:, self.active_indices]  # [1, 40]
            probs = F.softmax(active_logits, dim=-1)[0]

        top3_vals, top3_sub = torch.topk(probs, k=min(3, len(self.active_indices)))
        top3_list = []
        for r in range(len(top3_vals)):
            cls_idx = self.active_indices[top3_sub[r].item()]
            meta = self.idx_to_meta[cls_idx]
            top3_list.append({
                "rank": r + 1,
                "class_index": cls_idx,
                "word": meta["word"],
                "probability": round(top3_vals[r].item(), 4),
            })

        conf = top3_list[0]["probability"]
        margin = round(top3_vals[0].item() - top3_vals[1].item(), 4) if len(top3_vals) > 1 else 1.0
        return {
            "top1_word": top3_list[0]["word"],
            "top1_class_index": top3_list[0]["class_index"],
            "confidence": conf,
            "margin": margin,
            "top3": top3_list,
        }

    def predict_all_three(self, raw_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate raw_sequence on all 3 configurations:
          A: Baseline Raw
          B: Baseline + Norm
          C: Retrained + Norm
        """
        t0 = time.time()
        # Normalization
        norm_seq, norm_meta = normalize_body_centric(raw_sequence)

        # Infer A: Baseline on Raw
        res_a = self._infer_single(self.model_base, raw_sequence)

        # Infer B: Baseline on Normalized
        res_b = self._infer_single(self.model_base, norm_seq)

        # Infer C: Retrained on Normalized
        res_c = self._infer_single(self.model_retrain, norm_seq)
        inference_time_ms = round((time.time() - t0) * 1000, 2)

        return {
            "model_a_raw": res_a,
            "model_b_norm": res_b,
            "model_c_retrain_norm": res_c,
            "normalization_meta": norm_meta,
            "inference_time_ms": inference_time_ms,
        }


class FrameSkeletonExtractor:
    """
    Extracts [3, 30, 75] skeletal sequence from buffered video frames using the
    MediaPipe Holistic pipeline via .venv-sstcn bridge.
    """

    def __init__(self, sstcn_python: Path = DEFAULT_SSTCN_PYTHON):
        self.sstcn_python = Path(sstcn_python)
        if not self.sstcn_python.is_file():
            raise FileNotFoundError(f"MediaPipe Python environment not found: {self.sstcn_python}")

    def extract_from_frames(
        self, frames: List[np.ndarray], fps: float = 30.0
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Write frames to temporary MP4, call extraction bridge, return (tensor_3x30x75, meta).
        Guarantees that all temporary files are purged from disk.
        """
        if not frames:
            return None, {"error": "Empty frame buffer"}

        h, w = frames[0].shape[:2]
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tf:
            temp_mp4 = Path(tf.name)

        out_npy = temp_mp4.with_suffix(".tmp_live.npy")

        try:
            # Write frames to temp MP4
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_mp4), fourcc, float(fps), (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

            # Bridge script for .venv-sstcn
            bridge_script = (
                "import sys, json, numpy as np\n"
                "from pathlib import Path\n"
                "sys.path.insert(0, r'" + str(ROOT) + "')\n"
                "from backend.training.extract_words_716_skeleton import (\n"
                "  DEFAULT_CONFIG, _create_holistic_detector, extract_raw_holistic_from_video,\n"
                "  compute_active_segment, handle_missing_landmarks, resample_sequence_by_timestamps\n"
                ")\n"
                "video_path = Path(sys.argv[1])\n"
                "out_npy_path = Path(sys.argv[2])\n"
                "_, detector = _create_holistic_detector(DEFAULT_CONFIG)\n"
                "try:\n"
                "    raw_seq, timestamps, _ = extract_raw_holistic_from_video(video_path, detector, frame_stride=DEFAULT_CONFIG['mediapipe']['frame_stride'])\n"
                "finally:\n"
                "    detector.close()\n"
                "s_idx, e_idx, trim_meta = compute_active_segment(raw_seq, timestamps, DEFAULT_CONFIG)\n"
                "act_seq = raw_seq[s_idx : e_idx + 1]\n"
                "act_ts = timestamps[s_idx : e_idx + 1]\n"
                "cleaned_seq, valid, missing_meta = handle_missing_landmarks(act_seq, act_ts, DEFAULT_CONFIG)\n"
                "tensor_3x30x75 = resample_sequence_by_timestamps(cleaned_seq, act_ts, target_frames=DEFAULT_CONFIG['target_frames'])\n"
                "meta = {**trim_meta, **missing_meta, 'valid': bool(valid)}\n"
                "np.save(out_npy_path, tensor_3x30x75)\n"
                "print(json.dumps({'meta': meta, 'npy': str(out_npy_path)}, ensure_ascii=False))\n"
            )

            res = subprocess.run(
                [str(self.sstcn_python), "-c", bridge_script, str(temp_mp4), str(out_npy)],
                capture_output=True,
                text=True,
                check=True,
            )

            lines = [l.strip() for l in res.stdout.splitlines() if l.strip().startswith("{")]
            if not lines or not out_npy.is_file():
                return None, {"error": f"Extraction bridge failed: {res.stderr}"}

            meta_data = json.loads(lines[-1])["meta"]
            skeleton = np.load(out_npy).astype(np.float32)
            return skeleton, meta_data

        except Exception as e:
            return None, {"error": str(e)}

        finally:
            # Purge all temporary extraction files
            if temp_mp4.is_file():
                try:
                    temp_mp4.unlink()
                except Exception:
                    pass
            if out_npy.is_file():
                try:
                    out_npy.unlink()
                except Exception:
                    pass


class LivePilotTestApp:
    """
    Main HighGUI OpenCV interactive application.
    Supports Free-Form and Protocol-based testing.
    """

    def __init__(
        self,
        mode: str = "free",
        blind: bool = False,
        signer_id: str = "signer_01",
        device_index: int = 0,
        record_duration_sec: float = 5.0,
        save_videos: bool = False,
        output_dir: Path = DEFAULT_RESULTS_DIR,
    ):
        self.mode = mode.lower()
        self.blind = bool(blind)
        self.signer_id = signer_id
        self.device_index = device_index
        self.record_duration_sec = float(record_duration_sec)
        self.save_videos = bool(save_videos)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_manager = PilotModelManager()
        self.extractor = FrameSkeletonExtractor()

        # Build protocol queue if in protocol mode
        self.protocol_queue: List[Dict[str, Any]] = []
        if self.mode == "protocol":
            self.protocol_queue = self._build_protocol_queue()

        # Trial and attempt tracking
        self.official_records: List[Dict[str, Any]] = []  # Strictly accepted records (1 per planned trial)
        self.all_attempts_log: List[Dict[str, Any]] = []  # Full audit log including retried/rejected attempts
        self.pending_candidate_record: Optional[Dict[str, Any]] = None
        self.report_exported: bool = False
        self._last_report_payload: Optional[Dict[str, Any]] = None

    @property
    def session_records(self) -> List[Dict[str, Any]]:
        """Alias for backward compatibility with existing tests/callers."""
        return self.official_records

    @session_records.setter
    def session_records(self, val: List[Dict[str, Any]]):
        self.official_records = val

    def _build_protocol_queue(self) -> List[Dict[str, Any]]:
        """
        Build 24 trials: 12 words x 2 distances:
          Round 1 (Trials 1..12): Normal distance (~0.5 - 0.7m)
          Round 2 (Trials 13..24): Far distance (~1.0 - 1.5m)
        """
        proto_data = json.loads(DEFAULT_PROTOCOL_FILE.read_text(encoding="utf-8"))
        words = proto_data.get("words", [])

        queue = []
        trial_id = 1
        # Round 1: Normal distance
        for w in words:
            queue.append({
                "trial_id": trial_id,
                "word": w["word"],
                "class_index": w["class_index"],
                "entry_id": w["entry_id"],
                "region": w.get("region", ""),
                "hands": w.get("hands", ""),
                "gesture_hint": w.get("gesture_hint", ""),
                "distance": "NORMAL (~0.5 - 0.7m, ngang ngực)",
                "distance_code": "normal",
            })
            trial_id += 1

        # Round 2: Far distance
        for w in words:
            queue.append({
                "trial_id": trial_id,
                "word": w["word"],
                "class_index": w["class_index"],
                "entry_id": w["entry_id"],
                "region": w.get("region", ""),
                "hands": w.get("hands", ""),
                "gesture_hint": w.get("gesture_hint", ""),
                "distance": "FAR (~1.0 - 1.5m, thấy cả thân người)",
                "distance_code": "far",
            })
            trial_id += 1

        return queue

    def preflight_check(self) -> Dict[str, Any]:
        """Test webcam availability, resolution, and basic capture."""
        cap = None
        for backend_id in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(self.device_index, backend_id)
                if c.isOpened():
                    cap = c
                    break
                c.release()
            except Exception:
                pass

        if cap is None or not cap.isOpened():
            return {
                "status": "FAIL",
                "message": f"Cannot open camera device {self.device_index}",
            }

        ret, frame = cap.read()
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.release()

        return {
            "status": "PASS" if ret and frame is not None else "FAIL",
            "resolution": f"{w}x{h}",
            "fps": fps,
            "device_index": self.device_index,
        }

    def draw_ui_overlay(
        self,
        frame: np.ndarray,
        state: str,
        is_mirrored: bool,
        current_item: Optional[Dict[str, Any]] = None,
        elapsed_sec: float = 0.0,
        predictions: Optional[Dict[str, Any]] = None,
        extraction_meta: Optional[Dict[str, Any]] = None,
        fps_display: float = 30.0,
        attempt_id: int = 1,
    ) -> np.ndarray:
        """
        Render beautiful PIL-based Vietnamese overlay onto frame.
        """
        h, w = frame.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(16, int(h * 0.040)))
            font_sub = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", max(12, int(h * 0.026)))
            font_bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(13, int(h * 0.030)))
            font_large = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(20, int(h * 0.055)))
        except Exception:
            font_title = font_sub = font_bold = font_large = ImageFont.load_default()

        # 1. Top Header Bar
        top_h = int(h * 0.20)
        draw.rectangle([(0, 0), (w, top_h)], fill=(15, 20, 28))

        mode_desc = f"KIỂM THỬ TỰ DO" if self.mode == "free" else f"GIAO THỨC 24 LƯỢT {'(ĐÁNH GIÁ MÙ)' if self.blind else ''}"
        mirror_txt = "GƯƠNG [M]" if is_mirrored else "CHUẨN [M]"
        draw.text((15, 6), f"VSL PILOT 40 LIVE | Người ký: {self.signer_id} | Chế độ: {mode_desc} | {mirror_txt} | FPS: {fps_display:.1f}", font=font_sub, fill=(180, 200, 220))

        if current_item:
            attempt_suffix = f" (Lần thử #{attempt_id})" if attempt_id > 1 else ""
            title_text = f"Lượt {current_item['trial_id']:02d}/24{attempt_suffix}: {current_item['word'].upper()} ({current_item['region']})"
            dist_text = f"Vị trí: {current_item['distance']} | Gợi ý: {current_item['gesture_hint']}"
            draw.text((15, int(top_h * 0.38)), title_text, font=font_title, fill=(255, 215, 0))
            draw.text((15, int(top_h * 0.70)), dist_text, font=font_sub, fill=(170, 220, 255))
        else:
            draw.text((15, int(top_h * 0.45)), "THỰC HIỆN KÝ HIỆU BẤT KỲ TRONG 40 TỪ PILOT", font=font_title, fill=(0, 230, 180))
            draw.text((15, int(top_h * 0.75)), "Nhấn [SPACE] để bắt đầu thu 5s | AI so sánh 3 mô hình cùng lúc", font=font_sub, fill=(180, 200, 220))

        # 2. State-specific Overlays
        bot_h = int(h * 0.28) if (state == "RESULTS" and not (self.blind and self.mode == "protocol")) else int(h * 0.12)
        bot_y = h - bot_h
        draw.rectangle([(0, bot_y), (w, h)], fill=(15, 20, 28))

        if state == "IDLE":
            msg = "SẴN SÀNG | Nhấn [SPACE] BẮT ĐẦU THU (5s tự dừng) | [M] Đổi gương | [Q/ESC] Thoát"
            draw.text((15, bot_y + int(bot_h * 0.30)), msg, font=font_bold, fill=(0, 255, 128))

        elif state == "RECORDING":
            rem = max(0.0, self.record_duration_sec - elapsed_sec)
            pct = min(1.0, elapsed_sec / self.record_duration_sec)

            # Red recording indicator circle
            draw.ellipse([(15, bot_y + 12), (35, bot_y + 32)], fill=(255, 50, 50))
            attempt_str = f" [Lần #{attempt_id}]" if attempt_id > 1 else ""
            rec_msg = f"ĐANG THU{attempt_str}: {elapsed_sec:.1f}s / {self.record_duration_sec:.1f}s (Còn {rem:.1f}s) — Thực hiện cử chỉ rõ ràng"
            draw.text((45, bot_y + 12), rec_msg, font=font_bold, fill=(255, 255, 255))

            # Progress bar
            bar_w = int((w - 30) * pct)
            draw.rectangle([(15, bot_y + 38), (w - 15, bot_y + 46)], fill=(40, 50, 65))
            draw.rectangle([(15, bot_y + 38), (15 + bar_w, bot_y + 46)], fill=(255, 60, 60))

        elif state == "PROCESSING":
            msg = "ĐANG TRÍCH XUẤT HOLISTIC & SUY LUẬN 3 MÔ HÌNH VỚI DỮ LIỆU RAM... Vui lòng đợi"
            draw.text((15, bot_y + int(bot_h * 0.30)), msg, font=font_bold, fill=(255, 200, 50))

        elif state == "RESULTS":
            if self.blind and self.mode == "protocol":
                # Blind Mode: hide predictions to prevent observation bias
                msg = f"ĐÃ THU XONG LƯỢT {current_item['trial_id'] if current_item else 0}/24 (Lần thử #{attempt_id}) [Ẩn kết quả]."
                prompt = "Nhấn [SPACE/ENTER] CHẤP NHẬN & sang lượt kế tiếp | [R] KÝ LẠI lượt này | [Q] Kết thúc"
                draw.text((15, bot_y + 12), msg, font=font_bold, fill=(100, 220, 255))
                draw.text((15, bot_y + 35), prompt, font=font_sub, fill=(220, 230, 240))
            elif predictions:
                col_w = (w - 30) // 3
                # Model A: Baseline Raw
                pa = predictions["model_a_raw"]
                draw.rectangle([(15, bot_y + 8), (15 + col_w - 5, h - 35)], fill=(25, 30, 42), outline=(60, 70, 90))
                draw.text((22, bot_y + 12), "[A] BASELINE RAW", font=font_bold, fill=(200, 200, 210))
                draw.text((22, bot_y + 30), f"Top-1: {pa['top1_word'].upper()} ({pa['confidence']*100:.1f}%)", font=font_bold, fill=(255, 215, 0))
                top3_a_str = ", ".join([f"{t['word']}" for t in pa['top3'][1:]])
                draw.text((22, bot_y + 48), f"Top-2/3: {top3_a_str}", font=font_sub, fill=(180, 190, 205))

                # Model B: Baseline + Norm
                pb = predictions["model_b_norm"]
                x_b = 15 + col_w
                draw.rectangle([(x_b, bot_y + 8), (x_b + col_w - 5, h - 35)], fill=(25, 35, 45), outline=(50, 90, 110))
                draw.text((x_b + 7, bot_y + 12), "[B] BASELINE + NORM", font=font_bold, fill=(100, 220, 255))
                draw.text((x_b + 7, bot_y + 30), f"Top-1: {pb['top1_word'].upper()} ({pb['confidence']*100:.1f}%)", font=font_bold, fill=(0, 255, 180))
                top3_b_str = ", ".join([f"{t['word']}" for t in pb['top3'][1:]])
                draw.text((x_b + 7, bot_y + 48), f"Top-2/3: {top3_b_str}", font=font_sub, fill=(180, 190, 205))

                # Model C: Retrained + Norm
                pc = predictions["model_c_retrain_norm"]
                x_c = 15 + col_w * 2
                draw.rectangle([(x_c, bot_y + 8), (w - 15, h - 35)], fill=(25, 40, 35), outline=(50, 110, 80))
                draw.text((x_c + 7, bot_y + 12), "[C] RETRAINED + NORM", font=font_bold, fill=(150, 255, 150))
                draw.text((x_c + 7, bot_y + 30), f"Top-1: {pc['top1_word'].upper()} ({pc['confidence']*100:.1f}%)", font=font_bold, fill=(100, 255, 100))
                top3_c_str = ", ".join([f"{t['word']}" for t in pc['top3'][1:]])
                draw.text((x_c + 7, bot_y + 48), f"Top-2/3: {top3_c_str}", font=font_sub, fill=(180, 190, 205))

                # Footer action & stats
                n_meta = predictions.get("normalization_meta", {})
                stat_str = f"Vai: {n_meta.get('body_shoulder_width', 0):.3f} (Scale: {n_meta.get('scale_factor', 1):.2f}x) | Suy luận: {predictions.get('inference_time_ms', 0):.0f}ms | [SPACE] Chấp nhận | [R] Ký lại | [Q] Thoát"
                draw.text((15, h - 25), stat_str, font=font_sub, fill=(200, 220, 240))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def run_interactive_session(self) -> int:
        """
        Interactive OpenCV live capture and testing loop.
        """
        print("\n================== STANDALONE PILOT WEBCAM LIVE TEST ==================")
        print(f"Người ký      : {self.signer_id}")
        print(f"Chế độ        : {'KIỂM THỬ TỰ DO' if self.mode == 'free' else 'GIAO THỨC 24 LƯỢT (12 từ x 2 khoảng cách)'}")
        print(f"Đánh giá mù   : {'BẬT (Ẩn kết quả giữa các lượt)' if self.blind else 'TẮT (Hiển thị ngay)'}")
        print(f"Lưu video thô : {'CÓ (Lưu vào disk)' if self.save_videos else 'KHÔNG (Chỉ xử lý in-memory RAM)'}")
        print("Đang kiểm tra thiết bị camera...")

        pre = self.preflight_check()
        if pre["status"] != "PASS":
            print(f"[LỖI] Không thể mở webcam thiết bị {self.device_index}.")
            print("Vui lòng chạy lệnh kiểm thử ngoài Antigravity trong PowerShell:")
            print(f"  .\\.venv\\Scripts\\python.exe backend/training/words_716/live_pilot_test.py --mode {self.mode} --device {self.device_index}")
            return 1

        cap = None
        for backend_id in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(self.device_index, backend_id)
                if c.isOpened():
                    cap = c
                    break
                c.release()
            except Exception:
                pass

        if cap is None or not cap.isOpened():
            print("[LỖI] Không thể khởi tạo luồng camera.")
            return 1

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if actual_fps <= 0 or actual_fps > 120:
            actual_fps = 30.0

        window_name = "VSL Pilot 40 - Live Test HUD"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            print(f"[LỖI] Không thể mở cửa sổ đồ họa OpenCV: {e}")
            cap.release()
            return 1

        is_mirrored = True
        state = "IDLE"  # IDLE -> RECORDING -> PROCESSING -> RESULTS
        raw_buffer: List[np.ndarray] = []
        rec_start_time = 0.0
        current_trial_idx = 0
        last_predictions: Optional[Dict[str, Any]] = None
        last_meta: Optional[Dict[str, Any]] = None

        fps_timer = time.time()
        fps_counter = 0
        current_fps = actual_fps

        print("\n[HỆ THỐNG ĐÃ SẴN SÀNG]")
        print("  - Nhấn [SPACE] để bắt đầu thu mẫu (5 giây)")
        print("  - Nhấn [M] để bật/tắt chế độ lật gương preview")
        print("  - Nhấn [Q] hoặc [ESC] để kết thúc phiên")

        try:
            while True:
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    current_fps = fps_counter / (time.time() - fps_timer)
                    fps_timer = time.time()
                    fps_counter = 0

                current_item = self.protocol_queue[current_trial_idx] if (self.mode == "protocol" and current_trial_idx < len(self.protocol_queue)) else None

                elapsed = 0.0
                if state == "RECORDING":
                    # Invariant: AI input receives strictly unmirrored raw sensor frames
                    raw_buffer.append(frame.copy())
                    elapsed = time.monotonic() - rec_start_time
                    if elapsed >= self.record_duration_sec:
                        state = "PROCESSING"

                # Mirror only for display
                disp_frame = cv2.flip(frame, 1) if is_mirrored else frame.copy()
                disp = self.draw_ui_overlay(
                    disp_frame, state, is_mirrored, current_item,
                    elapsed_sec=elapsed, predictions=last_predictions,
                    extraction_meta=last_meta, fps_display=current_fps,
                )
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('m'), ord('M')):
                    is_mirrored = not is_mirrored
                elif key in (27, ord('q'), ord('Q')):
                    print("\n[KẾT THÚC] Người dùng thoát phiên kiểm thử.")
                    break

                if state == "IDLE":
                    if key in (32, ord('r'), ord('R')):  # SPACE
                        raw_buffer.clear()
                        rec_start_time = time.monotonic()
                        state = "RECORDING"
                        last_predictions = None

                elif state == "RECORDING":
                    if key in (32, ord('r'), ord('R')) and len(raw_buffer) >= 15:  # Manual stop early
                        state = "PROCESSING"

                elif state == "PROCESSING":
                    # Show processing message on screen
                    disp_p = self.draw_ui_overlay(disp_frame, "PROCESSING", is_mirrored, current_item, fps_display=current_fps)
                    cv2.imshow(window_name, disp_p)
                    cv2.waitKey(1)

                    # Extract landmarks from in-memory frames
                    skel, ext_meta = self.extractor.extract_from_frames(raw_buffer, fps=actual_fps)
                    last_meta = ext_meta

                    if skel is not None and ext_meta.get("valid", False):
                        preds = self.model_manager.predict_all_three(skel)
                        last_predictions = preds

                        # Optional video saving
                        saved_path = None
                        if self.save_videos and current_item:
                            saved_name = f"{self.signer_id}_{current_item['class_index']:03d}_{current_item['entry_id']}_live_{current_item['distance_code']}.mp4"
                            saved_path = self.output_dir / "videos" / saved_name
                            saved_path.parent.mkdir(parents=True, exist_ok=True)
                            vw = cv2.VideoWriter(str(saved_path), cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (frame.shape[1], frame.shape[0]))
                            for f in raw_buffer:
                                vw.write(f)
                            vw.release()

                        # Build candidate record (held in pending_candidate_record until user confirms)
                        candidate_record = {
                            "trial_index": current_trial_idx + 1 if current_item else len(self.official_records) + 1,
                            "attempt_id": current_attempt_idx,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "signer_id": self.signer_id,
                            "mode": self.mode,
                            "target_word": current_item["word"] if current_item else "FREE_PLAY",
                            "target_class_index": current_item["class_index"] if current_item else -1,
                            "distance": current_item.get("distance_code", "standard") if current_item else "standard",
                            "model_a_pred": preds["model_a_raw"]["top1_word"],
                            "model_a_conf": preds["model_a_raw"]["confidence"],
                            "model_a_correct": (preds["model_a_raw"]["top1_class_index"] == current_item["class_index"]) if current_item else None,
                            "model_b_pred": preds["model_b_norm"]["top1_word"],
                            "model_b_conf": preds["model_b_norm"]["confidence"],
                            "model_b_correct": (preds["model_b_norm"]["top1_class_index"] == current_item["class_index"]) if current_item else None,
                            "model_c_pred": preds["model_c_retrain_norm"]["top1_word"],
                            "model_c_conf": preds["model_c_retrain_norm"]["confidence"],
                            "model_c_correct": (preds["model_c_retrain_norm"]["top1_class_index"] == current_item["class_index"]) if current_item else None,
                            "shoulder_width": preds["normalization_meta"]["body_shoulder_width"],
                            "scale_factor": preds["normalization_meta"]["scale_factor"],
                            "inference_time_ms": preds["inference_time_ms"],
                            "is_accepted": False,
                            "saved_video_path": str(saved_path) if saved_path else None,
                        }
                        self.pending_candidate_record = candidate_record
                        state = "RESULTS"
                    else:
                        print(f"  [LỖI TRÍCH XUẤT] Không nhận diện đủ khớp cơ thể: {ext_meta.get('error', 'Pose/Hands missing')}")
                        state = "IDLE"

                elif state == "RESULTS":
                    if key in (32, 13, 10):  # SPACE or ENTER to accept and advance
                        cv2.waitKey(200)  # Debounce keypress
                        raw_buffer.clear()
                        if self.pending_candidate_record is not None:
                            accepted = dict(self.pending_candidate_record)
                            accepted["is_accepted"] = True
                            accepted["accepted_at"] = datetime.now(timezone.utc).isoformat()
                            self.official_records.append(accepted)
                            self.all_attempts_log.append(accepted)
                            self.pending_candidate_record = None

                            if not self.blind:
                                print(f"\n[LƯỢT {accepted['trial_index']:02d} - CHẤP NHẬN] Mục tiêu: {accepted['target_word']}")
                                print(f"  - Model A (Baseline Raw)  : {accepted['model_a_pred']:15s} (conf: {accepted['model_a_conf']:.2f}) {'[ĐÚNG]' if accepted['model_a_correct'] else ''}")
                                print(f"  - Model B (Baseline+Norm) : {accepted['model_b_pred']:15s} (conf: {accepted['model_b_conf']:.2f}) {'[ĐÚNG]' if accepted['model_b_correct'] else ''}")
                                print(f"  - Model C (Retrained+Norm): {accepted['model_c_pred']:15s} (conf: {accepted['model_c_conf']:.2f}) {'[ĐÚNG]' if accepted['model_c_correct'] else ''}")
                            else:
                                print(f"  [ĐÃ LƯU] Ghi nhận xong lượt {accepted['trial_index']:02d}/24 (Lần thử #{current_attempt_idx}).")

                        if self.mode == "protocol":
                            current_trial_idx += 1
                            current_attempt_idx = 1
                            if current_trial_idx >= len(self.protocol_queue):
                                print("\n[HOÀN THÀNH GIAO THỨC] Toàn bộ 24 lượt thử đã hoàn tất!")
                                self.export_session_report()
                                break
                        else:
                            current_attempt_idx = 1
                        state = "IDLE"

                    elif key in (ord('r'), ord('R')):  # Redo current trial
                        cv2.waitKey(200)  # Debounce keypress
                        if self.pending_candidate_record is not None:
                            rejected = dict(self.pending_candidate_record)
                            rejected["is_accepted"] = False
                            rejected["retry_reason"] = "USER_RETRY"
                            rejected["rejected_at"] = datetime.now(timezone.utc).isoformat()
                            self.all_attempts_log.append(rejected)
                            print(f"  [KÝ LẠI] Yêu cầu ký lại lượt {rejected['trial_index']:02d} (đã ghi nhận lần thử #{current_attempt_idx}).")
                            self.pending_candidate_record = None

                        current_attempt_idx += 1
                        raw_buffer.clear()
                        state = "IDLE"

        finally:
            cap.release()
            cv2.destroyAllWindows()
            if self.official_records or self.all_attempts_log:
                self.export_session_report()

        return 0

    def export_session_report(self) -> Dict[str, Any]:
        """
        Export session records to JSON and CSV and print a comparison summary table.
        Guarded against multiple calls.
        """
        if self.report_exported and self._last_report_payload is not None:
            return self._last_report_payload

        self.report_exported = True
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.output_dir / f"live_session_{self.signer_id}_{ts}.json"
        csv_file = self.output_dir / f"live_session_{self.signer_id}_{ts}.csv"
        csv_all_attempts_file = self.output_dir / f"live_session_{self.signer_id}_{ts}_all_attempts.csv"

        # Calculate statistics strictly from official accepted records
        official_eval = [r for r in self.official_records if r.get("target_class_index", -1) != -1]
        eval_total = len(official_eval)
        total_attempts = len(self.all_attempts_log)
        retries_count = max(0, total_attempts - len(self.official_records))

        summary = {
            "session_timestamp": ts,
            "signer_id": self.signer_id,
            "mode": self.mode,
            "blind_mode": self.blind,
            "total_official_trials": len(self.official_records),
            "total_trials": len(self.official_records),  # backward compatibility
            "total_attempts": total_attempts,
            "retries_count": retries_count,
            "evaluated_trials": eval_total,
        }

        if eval_total > 0:
            a_correct = sum(1 for r in official_eval if r.get("model_a_correct"))
            b_correct = sum(1 for r in official_eval if r.get("model_b_correct"))
            c_correct = sum(1 for r in official_eval if r.get("model_c_correct"))

            summary["model_a_raw_accuracy"] = round(a_correct / eval_total, 4)
            summary["model_b_norm_accuracy"] = round(b_correct / eval_total, 4)
            summary["model_c_retrain_accuracy"] = round(c_correct / eval_total, 4)

            # Distance breakdown
            norm_recs = [r for r in official_eval if r.get("distance") == "normal"]
            far_recs = [r for r in official_eval if r.get("distance") == "far"]

            summary["distance_breakdown"] = {
                "normal": {
                    "total": len(norm_recs),
                    "model_a_correct": sum(1 for r in norm_recs if r.get("model_a_correct")),
                    "model_b_correct": sum(1 for r in norm_recs if r.get("model_b_correct")),
                    "model_c_correct": sum(1 for r in norm_recs if r.get("model_c_correct")),
                    "model_a_accuracy": round(sum(1 for r in norm_recs if r.get("model_a_correct")) / len(norm_recs), 4) if norm_recs else 0.0,
                    "model_b_accuracy": round(sum(1 for r in norm_recs if r.get("model_b_correct")) / len(norm_recs), 4) if norm_recs else 0.0,
                    "model_c_accuracy": round(sum(1 for r in norm_recs if r.get("model_c_correct")) / len(norm_recs), 4) if norm_recs else 0.0,
                },
                "far": {
                    "total": len(far_recs),
                    "model_a_correct": sum(1 for r in far_recs if r.get("model_a_correct")),
                    "model_b_correct": sum(1 for r in far_recs if r.get("model_b_correct")),
                    "model_c_correct": sum(1 for r in far_recs if r.get("model_c_correct")),
                    "model_a_accuracy": round(sum(1 for r in far_recs if r.get("model_a_correct")) / len(far_recs), 4) if far_recs else 0.0,
                    "model_b_accuracy": round(sum(1 for r in far_recs if r.get("model_b_correct")) / len(far_recs), 4) if far_recs else 0.0,
                    "model_c_accuracy": round(sum(1 for r in far_recs if r.get("model_c_correct")) / len(far_recs), 4) if far_recs else 0.0,
                }
            }

            print("\n================== TỔNG KẾT PHIÊN KIỂM THỬ LIVE ==================")
            print(f"Người ký: {self.signer_id} | Tổng số lượt chính thức: {eval_total} (Tổng lượt thu: {total_attempts}, Ký lại: {retries_count})")
            print(f"  Model A (Baseline Raw)       : {a_correct}/{eval_total} ({summary['model_a_raw_accuracy']*100:.1f}%)")
            print(f"  Model B (Baseline + Norm)    : {b_correct}/{eval_total} ({summary['model_b_norm_accuracy']*100:.1f}%)")
            print(f"  Model C (Retrained + Norm)   : {c_correct}/{eval_total} ({summary['model_c_retrain_accuracy']*100:.1f}%)")

            if norm_recs and far_recs:
                print("\n--- Phân bổ theo khoảng cách camera ---")
                nb = summary["distance_breakdown"]["normal"]
                fb = summary["distance_breakdown"]["far"]
                print(f"  Khoảng cách thông thường (~0.5m) | A: {nb['model_a_correct']}/{nb['total']} ({nb['model_a_accuracy']*100:.1f}%) | B: {nb['model_b_correct']}/{nb['total']} ({nb['model_b_accuracy']*100:.1f}%) | C: {nb['model_c_correct']}/{nb['total']} ({nb['model_c_accuracy']*100:.1f}%)")
                print(f"  Khoảng cách xa hơn (~1.2m)      | A: {fb['model_a_correct']}/{fb['total']} ({fb['model_a_accuracy']*100:.1f}%) | B: {fb['model_b_correct']}/{fb['total']} ({fb['model_b_accuracy']*100:.1f}%) | C: {fb['model_c_correct']}/{fb['total']} ({fb['model_c_accuracy']*100:.1f}%)")

        full_payload = {
            "summary": summary,
            "official_records": self.official_records,
            "all_attempts_log": self.all_attempts_log,
            "records": self.official_records,  # backward compatibility
        }
        json_file.write_text(json.dumps(full_payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\n[XUẤT BÁO CÁO] Đã lưu kết quả tại:")
        print(f"  JSON: {json_file}")

        # Save official CSV
        if self.official_records:
            keys = []
            seen = set()
            for r in self.official_records:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            with open(csv_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.official_records)
            print(f"  CSV : {csv_file}")

        # Save all attempts CSV if retries occurred
        if self.all_attempts_log and len(self.all_attempts_log) > len(self.official_records):
            keys = []
            seen = set()
            for r in self.all_attempts_log:
                for k in r.keys():
                    if k not in seen:
                        seen.add(k)
                        keys.append(k)
            with open(csv_all_attempts_file, "w", newline="", encoding="utf-8-sig") as f:
                writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(self.all_attempts_log)
            print(f"  CSV (Toàn bộ lần thử): {csv_all_attempts_file}")

        self._last_report_payload = full_payload
        return full_payload


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Standalone Live Webcam Test Tool for WordRecognizer716 Pilot 40")
    p.add_argument("--mode", type=str, default="free", choices=["free", "protocol"], help="Testing mode: 'free' (free testing) or 'protocol' (12 words x 2 distances = 24 trials)")
    p.add_argument("--blind", action="store_true", help="Blind Evaluation mode: hide AI predictions between trials")
    p.add_argument("--signer", type=str, default="signer_01", help="Signer ID (default: signer_01)")
    p.add_argument("--device", type=int, default=0, help="Camera device index (default: 0)")
    p.add_argument("--duration", type=float, default=5.0, help="Recording duration per trial (default: 5.0 seconds)")
    p.add_argument("--save-videos", action="store_true", help="Explicitly save captured MP4 files to disk")
    p.add_argument("--output-dir", type=str, default=str(DEFAULT_RESULTS_DIR), help="Output directory for reports")
    p.add_argument("--preflight", action="store_true", help="Run webcam preflight test and exit")
    return p


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    app = LivePilotTestApp(
        mode=args.mode,
        blind=args.blind,
        signer_id=args.signer,
        device_index=args.device,
        record_duration_sec=args.duration,
        save_videos=args.save_videos,
        output_dir=Path(args.output_dir),
    )

    if args.preflight:
        diag = app.preflight_check()
        print(json.dumps(diag, indent=2))
        sys.exit(0 if diag["status"] == "PASS" else 1)

    sys.exit(app.run_interactive_session())


if __name__ == "__main__":
    main()
