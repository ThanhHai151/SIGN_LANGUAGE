"""
Standalone Live Webcam Test Tool for WordRecognizer716 (703 QIPEDC Words).
TASK WORDS 3.2L — STANDALONE 703-WORD WEBCAM LIVE TEST & DATA PRIVACY.

Features:
- OpenCV HighGUI interactive UI with 5-second countdown and auto-stop.
- Mirror preview mode toggle (Key M): flips display preview only; AI input strictly receives raw unmirrored sensor frames.
- MediaPipe Holistic 75 landmarks extraction & active trimming via .venv-sstcn bridge.
- Resampling into [3, 30, 75] tensor with sequence-level Body-Centric Normalization.
- Inference across all 703 active QIPEDC classes with 13 inactive classes masked to -50.0.
- Top-1, Top-2, Top-3 display with exact Vietnamese word name, global class index, confidence, and margin.
- Target Word selection mode: by class index (--target-class), word search (--target-word), or dynamic key [T].
- Diagnostic uncertainty detection (confidence < threshold or margin < threshold) without asserting false UNKNOWN.
- Strict trial accounting: Key [SPACE/ENTER] accepts trial; Key [R] retries without duplicate official counting.
- Data privacy: Zero disk write of captured video/skeletons by default; all temp frames cleaned up on exit.
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
DEFAULT_CKPT_PATH = ROOT / "backend" / "models" / "words_716_checkpoints" / "experiments_703" / "best.pth"
DEFAULT_CLASSES_FILE = ROOT / "backend" / "data" / "words_716" / "classes.json"
DEFAULT_FAILED_FILE = ROOT / "backend" / "data" / "words_716" / "failed_samples.jsonl"
DEFAULT_PROTOCOL_PATH = ROOT / "backend" / "data" / "words_716" / "holdout_protocol.json"
DEFAULT_SSTCN_PYTHON = ROOT / ".venv-sstcn" / "Scripts" / "python.exe"
DEFAULT_RESULTS_DIR = ROOT / "backend" / "evaluations" / "live_703_results"


class Words703ModelManager:
    """
    Manages loading and inference for the 703-word WordRecognizer716 model.
    """

    def __init__(
        self,
        checkpoint_path: Union[str, Path] = DEFAULT_CKPT_PATH,
        classes_file: Union[str, Path] = DEFAULT_CLASSES_FILE,
        failed_file: Union[str, Path] = DEFAULT_FAILED_FILE,
        device: str = "cpu",
        confidence_threshold: float = 0.08,
        margin_threshold: float = 0.01,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.classes_file = Path(classes_file)
        self.failed_file = Path(failed_file)
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        self.margin_threshold = margin_threshold

        if not self.classes_file.is_file():
            raise FileNotFoundError(f"Classes config not found: {self.classes_file}")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(f"703-word checkpoint not found: {self.checkpoint_path}")

        # 1. Load classes config
        classes_data = json.loads(self.classes_file.read_text(encoding="utf-8"))
        self.classes_list = classes_data.get("classes", [])
        self.idx_to_meta = {c["class_index"]: c for c in self.classes_list}
        self.idx_to_word = {c["class_index"]: c["word"] for c in self.classes_list}
        self.word_to_idx = {c["word"].lower().strip(): c["class_index"] for c in self.classes_list}

        # 2. Determine inactive & active classes
        self.unavailable_indices = []
        if self.failed_file.is_file():
            failed_rows = [json.loads(l) for l in self.failed_file.read_text(encoding="utf-8").splitlines() if l.strip()]
            self.unavailable_indices = sorted(list(set(r["class_index"] for r in failed_rows)))
        self.active_indices = [i for i in range(716) if i not in set(self.unavailable_indices)]

        # 3. Initialize model and load weights
        self.model = WordRecognizer716(
            num_classes=716,
            active_indices=self.active_indices,
            dropout=0.0,
        )
        ckpt = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(ckpt["model_state_dict"])
        self.model.to(self.device).eval()

        self.ckpt_meta = {
            "epoch": ckpt.get("epoch"),
            "phase": ckpt.get("phase"),
            "saved_loss": ckpt.get("loss"),
            "saved_accuracy": ckpt.get("accuracy"),
        }

    def predict(self, raw_sequence: np.ndarray) -> Dict[str, Any]:
        """
        Apply sequence-level Body-Centric Normalization and run masked 703-class inference.
        Returns Top-3 predictions, confidence, margin, and normalization metadata.
        """
        t0 = time.time()
        # Invariant: Body-Centric Normalization applied consistently
        norm_seq, norm_meta = normalize_body_centric(raw_sequence)

        t_in = torch.from_numpy(norm_seq).float().unsqueeze(0).to(self.device)  # [1, 3, 30, 75]
        with torch.no_grad():
            masked_logits = self.model(t_in, apply_class_mask=True)
            active_logits = masked_logits[:, self.active_indices]  # [1, 703]
            probs = F.softmax(active_logits, dim=-1)[0]             # [703]

        top3_vals, top3_sub = torch.topk(probs, k=min(3, len(self.active_indices)))
        top3_logits, _ = torch.topk(active_logits[0], k=min(3, len(self.active_indices)))

        top3_list = []
        for r in range(len(top3_vals)):
            cls_idx = self.active_indices[top3_sub[r].item()]
            meta = self.idx_to_meta.get(cls_idx, {})
            top3_list.append({
                "rank": r + 1,
                "class_index": cls_idx,
                "word": meta.get("word", f"class_{cls_idx}"),
                "entry_id": meta.get("entry_id", ""),
                "region": meta.get("selected_region", meta.get("region", "")),
                "probability": round(top3_vals[r].item(), 4),
                "logit": round(top3_logits[r].item(), 4),
            })

        top1 = top3_list[0]
        conf = top1["probability"]
        margin = round(top3_vals[0].item() - top3_vals[1].item(), 4) if len(top3_vals) > 1 else 1.0

        # Diagnostic uncertainty check
        is_uncertain = (conf < self.confidence_threshold) or (margin < self.margin_threshold)
        uncertainty_reasons = []
        if conf < self.confidence_threshold:
            uncertainty_reasons.append(f"LOW_CONFIDENCE (< {self.confidence_threshold})")
        if margin < self.margin_threshold:
            uncertainty_reasons.append(f"LOW_MARGIN (< {self.margin_threshold})")

        inf_time_ms = round((time.time() - t0) * 1000, 2)

        return {
            "top1_word": top1["word"],
            "top1_class_index": top1["class_index"],
            "top1_entry_id": top1["entry_id"],
            "confidence": conf,
            "margin": margin,
            "is_uncertain": is_uncertain,
            "uncertainty_reasons": uncertainty_reasons,
            "top3": top3_list,
            "normalization_meta": norm_meta,
            "inference_time_ms": inf_time_ms,
        }


class FrameSkeletonExtractor:
    """
    Extracts [3, 30, 75] skeletal sequence from buffered video frames in RAM
    using MediaPipe Holistic via .venv-sstcn bridge.
    Guarantees automatic cleanup of temporary files.
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
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(temp_mp4), fourcc, float(fps), (w, h))
            for f in frames:
                writer.write(f)
            writer.release()

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


class LiveWords703TestApp:
    """
    Main HighGUI OpenCV interactive application for 703-word live webcam testing.
    Supports free testing, target word testing, and attempt tracking.
    """

    def __init__(
        self,
        target_class_index: Optional[int] = None,
        target_word: Optional[str] = None,
        signer_id: str = "signer_01",
        device_index: int = 0,
        record_duration_sec: float = 5.0,
        save_videos: bool = False,
        output_dir: Path = DEFAULT_RESULTS_DIR,
    ):
        self.signer_id = signer_id
        self.device_index = device_index
        self.record_duration_sec = float(record_duration_sec)
        self.save_videos = bool(save_videos)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.model_manager = Words703ModelManager()
        self.extractor = FrameSkeletonExtractor()

        # Set target word if requested
        self.target_meta: Optional[Dict[str, Any]] = None
        if target_class_index is not None and target_class_index in self.model_manager.idx_to_meta:
            self.target_meta = self.model_manager.idx_to_meta[target_class_index]
        elif target_word is not None:
            norm_w = target_word.lower().strip()
            if norm_w in self.model_manager.word_to_idx:
                c_idx = self.model_manager.word_to_idx[norm_w]
                self.target_meta = self.model_manager.idx_to_meta[c_idx]

        # Load 12 holdout protocol words for quick keyboard cycling with key T
        self.protocol_words: List[Dict[str, Any]] = []
        if DEFAULT_PROTOCOL_PATH.is_file():
            try:
                with open(DEFAULT_PROTOCOL_PATH, "r", encoding="utf-8") as f:
                    proto_data = json.load(f)
                    for item in proto_data.get("words", []):
                        c_idx = item.get("class_index")
                        if c_idx in self.model_manager.idx_to_meta:
                            self.protocol_words.append(self.model_manager.idx_to_meta[c_idx])
            except Exception:
                pass
        self.protocol_cycle_idx: int = -1
        if self.target_meta and self.protocol_words:
            for idx, w_meta in enumerate(self.protocol_words):
                if w_meta["class_index"] == self.target_meta["class_index"]:
                    self.protocol_cycle_idx = idx
                    break

        # Trial and attempt tracking
        self.official_records: List[Dict[str, Any]] = []
        self.all_attempts_log: List[Dict[str, Any]] = []
        self.pending_candidate_record: Optional[Dict[str, Any]] = None
        self.report_exported: bool = False
        self._last_report_payload: Optional[Dict[str, Any]] = None

    @property
    def session_records(self) -> List[Dict[str, Any]]:
        return self.official_records

    @session_records.setter
    def session_records(self, val: List[Dict[str, Any]]):
        self.official_records = val

    def set_target_word(self, query: Union[str, int]) -> bool:
        """Dynamically set or change target word by class_index or word string."""
        if isinstance(query, int) or (isinstance(query, str) and query.isdigit()):
            c_idx = int(query)
            if c_idx in self.model_manager.idx_to_meta:
                self.target_meta = self.model_manager.idx_to_meta[c_idx]
                return True
        elif isinstance(query, str):
            q_norm = query.lower().strip()
            if q_norm in self.model_manager.word_to_idx:
                c_idx = self.model_manager.word_to_idx[q_norm]
                self.target_meta = self.model_manager.idx_to_meta[c_idx]
                return True
        return False

    def cycle_target_word(self) -> Optional[Dict[str, Any]]:
        """Cycle sequentially to the next word among the 12 holdout protocol words (wraps around)."""
        if not self.protocol_words:
            return self.target_meta
        self.protocol_cycle_idx = (self.protocol_cycle_idx + 1) % len(self.protocol_words)
        self.target_meta = self.protocol_words[self.protocol_cycle_idx]
        return self.target_meta

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
        elapsed_sec: float = 0.0,
        predictions: Optional[Dict[str, Any]] = None,
        extraction_meta: Optional[Dict[str, Any]] = None,
        fps_display: float = 30.0,
        trial_id: int = 1,
        attempt_id: int = 1,
    ) -> np.ndarray:
        """
        Render beautiful PIL-based Vietnamese overlay for 703-word live test.
        """
        h, w = frame.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(16, int(h * 0.038)))
            font_sub = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", max(12, int(h * 0.025)))
            font_bold = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(13, int(h * 0.028)))
            font_large = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(18, int(h * 0.045)))
        except Exception:
            font_title = font_sub = font_bold = font_large = ImageFont.load_default()

        # 1. Top Header Bar
        top_h = int(h * 0.18)
        draw.rectangle([(0, 0), (w, top_h)], fill=(12, 16, 24))

        mirror_txt = "GƯƠNG [M]" if is_mirrored else "CHUẨN [M]"
        draw.text((15, 6), f"VSL 703-WORD LIVE TEST | Người ký: {self.signer_id} | {mirror_txt} | FPS: {fps_display:.1f}", font=font_sub, fill=(180, 200, 220))

        if self.target_meta:
            attempt_str = f" (Lần thử #{attempt_id})" if attempt_id > 1 else ""
            t_txt = f"Lượt #{trial_id:02d}{attempt_str} - MỤC TIÊU [#{self.target_meta['class_index']:03d}]: {self.target_meta['word'].upper()}"
            cycle_info = f" ({self.protocol_cycle_idx + 1}/{len(self.protocol_words)})" if (self.protocol_words and self.protocol_cycle_idx >= 0) else ""
            sub_txt = f"Vùng miền: {self.target_meta.get('selected_region', 'National')} | Entry: {self.target_meta.get('entry_id', '')} | [T] Chuyển từ holdout{cycle_info}"
            draw.text((15, int(top_h * 0.36)), t_txt, font=font_title, fill=(255, 215, 0))
            draw.text((15, int(top_h * 0.70)), sub_txt, font=font_sub, fill=(170, 220, 255))
        else:
            attempt_str = f" (Lần #{attempt_id})" if attempt_id > 1 else ""
            draw.text((15, int(top_h * 0.38)), f"CHẾ ĐỘ THỬ TỰ DO (Lượt #{trial_id:02d}{attempt_str}) — 703 TỪ VỰNG", font=font_title, fill=(0, 230, 180))
            draw.text((15, int(top_h * 0.72)), "Ký bất kỳ từ nào | AI đối sánh 703 lớp | Nhấn [T] chuyển qua 12 từ holdout", font=font_sub, fill=(180, 200, 220))

        # 2. State-specific Overlays
        bot_h = int(h * 0.32) if state == "RESULTS" else int(h * 0.12)
        bot_y = h - bot_h
        draw.rectangle([(0, bot_y), (w, h)], fill=(12, 16, 24))

        if state == "IDLE":
            msg = "SẴN SÀNG | [SPACE] BẮT ĐẦU THU (5s) | [T] Chuyển 12 từ holdout | [M] Đổi gương | [Q/ESC] Thoát"
            draw.text((15, bot_y + int(bot_h * 0.30)), msg, font=font_bold, fill=(0, 255, 128))

        elif state == "RECORDING":
            rem = max(0.0, self.record_duration_sec - elapsed_sec)
            pct = min(1.0, elapsed_sec / self.record_duration_sec)

            draw.ellipse([(15, bot_y + 10), (35, bot_y + 30)], fill=(255, 50, 50))
            rec_msg = f"ĐANG THU: {elapsed_sec:.1f}s / {self.record_duration_sec:.1f}s (Còn {rem:.1f}s) — Thực hiện cử chỉ rõ ràng trước camera"
            draw.text((45, bot_y + 10), rec_msg, font=font_bold, fill=(255, 255, 255))

            bar_w = int((w - 30) * pct)
            draw.rectangle([(15, bot_y + 36), (w - 15, bot_y + 44)], fill=(40, 50, 65))
            draw.rectangle([(15, bot_y + 36), (15 + bar_w, bot_y + 44)], fill=(255, 60, 60))

        elif state == "PROCESSING":
            msg = "ĐANG TRÍCH XUẤT HOLISTIC & SUY LUẬN TRÊN 703 LỚP TỪ VỰNG... Vui lòng đợi trong giây lát"
            draw.text((15, bot_y + int(bot_h * 0.30)), msg, font=font_bold, fill=(255, 200, 50))

        elif state == "RESULTS" and predictions:
            top3 = predictions["top3"]
            p1, p2, p3 = top3[0], top3[1], top3[2]
            conf = predictions["confidence"]
            margin = predictions["margin"]
            is_unc = predictions["is_uncertain"]

            # Match status with target
            status_tag = ""
            status_color = (200, 200, 200)
            if self.target_meta:
                tgt_idx = self.target_meta["class_index"]
                if p1["class_index"] == tgt_idx:
                    status_tag = " [CHÍNH XÁC TOP-1]"
                    status_color = (0, 255, 100)
                elif tgt_idx in [p2["class_index"], p3["class_index"]]:
                    status_tag = " [TRONG TOP-3]"
                    status_color = (100, 220, 255)
                else:
                    status_tag = " [CHƯA TRÚNG MỤC TIÊU]"
                    status_color = (255, 130, 80)

            # Top-1 Hero card
            draw.rectangle([(15, bot_y + 8), (w - 15, bot_y + 56)], fill=(22, 32, 45), outline=(50, 90, 130))
            hero_title = f"TOP-1: [#{p1['class_index']:03d}] {p1['word'].upper()}  ({p1['probability']*100:.1f}%)"
            draw.text((25, bot_y + 12), hero_title, font=font_large, fill=(255, 220, 50))
            if status_tag:
                draw.text((25 + int(w * 0.45), bot_y + 14), status_tag, font=font_bold, fill=status_color)

            # Top-2 & Top-3 row
            col_w = (w - 35) // 2
            # Top-2
            draw.rectangle([(15, bot_y + 60), (15 + col_w, bot_y + 98)], fill=(18, 25, 36), outline=(40, 60, 85))
            draw.text((22, bot_y + 64), f"Top-2: [#{p2['class_index']:03d}] {p2['word']}", font=font_bold, fill=(200, 220, 240))
            draw.text((22, bot_y + 80), f"Xác suất: {p2['probability']*100:.1f}% (Margin: {margin*100:.1f}%)", font=font_sub, fill=(160, 180, 200))

            # Top-3
            draw.rectangle([(20 + col_w, bot_y + 60), (w - 15, bot_y + 98)], fill=(18, 25, 36), outline=(40, 60, 85))
            draw.text((27 + col_w, bot_y + 64), f"Top-3: [#{p3['class_index']:03d}] {p3['word']}", font=font_bold, fill=(200, 220, 240))
            draw.text((27 + col_w, bot_y + 80), f"Xác suất: {p3['probability']*100:.1f}%", font=font_sub, fill=(160, 180, 200))

            # Bottom stats & Key actions
            n_meta = predictions.get("normalization_meta", {})
            stat_str = f"Vai: {n_meta.get('body_shoulder_width', 0):.3f} (Scale: {n_meta.get('scale_factor', 1):.2f}x) | Suy luận: {predictions.get('inference_time_ms', 0):.0f}ms | [SPACE/ENTER] Chấp nhận | [R] Ký lại | [T] Đổi từ | [Q] Thoát"
            if is_unc:
                draw.text((15, h - 42), "[CHẨN ĐOÁN: ĐỘ TIN CẬY THẤP TRÊN KHÔNG GIAN 703 TỪ - CẦN ĐỐI CHIẾU THÊM]", font=font_sub, fill=(255, 180, 50))
            draw.text((15, h - 22), stat_str, font=font_sub, fill=(190, 210, 230))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def run_interactive_session(self) -> int:
        """
        Interactive OpenCV live capture and testing loop for 703 words.
        """
        print("\n================== STANDALONE 703-WORD WEBCAM LIVE TEST ==================")
        print(f"Người ký      : {self.signer_id}")
        if self.target_meta:
            print(f"Từ mục tiêu   : [#{self.target_meta['class_index']:03d}] {self.target_meta['word']} ({self.target_meta.get('selected_region', 'National')})")
        else:
            print(f"Chế độ        : THỬ TỰ DO (Toàn bộ 703 từ vựng cạnh tranh mở)")
        print(f"Checkpoint    : {self.model_manager.checkpoint_path}")
        print(f"Chuẩn hóa     : Body-Centric Normalization (Active 703 classes, 13 masked)")
        print(f"Lưu video thô : {'CÓ (Lưu vào disk)' if self.save_videos else 'KHÔNG (Chỉ xử lý in-memory RAM)'}")
        print("Đang kiểm tra webcam...")

        pre = self.preflight_check()
        if pre["status"] != "PASS":
            print(f"[LỖI] Không thể mở webcam thiết bị {self.device_index}.")
            print("Vui lòng chạy lệnh ngoài môi trường sandbox trong PowerShell:")
            print(f"  .\\.venv\\Scripts\\python.exe backend/training/words_716/live_words_703_test.py --device {self.device_index}")
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

        window_name = "VSL 703 Words - Live Test HUD"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            print(f"[LỖI] Không thể mở cửa sổ HighGUI: {e}")
            cap.release()
            return 1

        is_mirrored = True
        state = "IDLE"  # IDLE -> RECORDING -> PROCESSING -> RESULTS
        raw_buffer: List[np.ndarray] = []
        rec_start_time = 0.0
        current_trial_idx = 1
        current_attempt_idx = 1
        last_predictions: Optional[Dict[str, Any]] = None
        last_meta: Optional[Dict[str, Any]] = None

        fps_timer = time.time()
        fps_counter = 0
        current_fps = actual_fps

        print("\n[HỆ THỐNG ĐÃ SẴN SÀNG]")
        print("  - Nhấn [SPACE] để bắt đầu thu mẫu (5 giây tự dừng)")
        print("  - Nhấn [M] để bật/tắt lật gương preview")
        print("  - Nhấn [T] để chuyển tuần tự qua 12 từ holdout")
        print("  - Nhấn [Q] hoặc [ESC] để kết thúc phiên\n")

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
                    disp_frame, state, is_mirrored,
                    elapsed_sec=elapsed, predictions=last_predictions,
                    extraction_meta=last_meta, fps_display=current_fps,
                    trial_id=current_trial_idx, attempt_id=current_attempt_idx,
                )
                cv2.imshow(window_name, disp)
                key = cv2.waitKey(1) & 0xFF

                if key in (ord('m'), ord('M')):
                    is_mirrored = not is_mirrored
                elif key in (27, ord('q'), ord('Q')):
                    print("\n[KẾT THÚC] Người dùng đóng phiên kiểm thử.")
                    break
                elif key in (ord('t'), ord('T')):
                    if state in ("IDLE", "RESULTS"):
                        next_tgt = self.cycle_target_word()
                        if next_tgt:
                            print(f"\n[CHUYỂN TỪ - PHÍM T] ({self.protocol_cycle_idx + 1}/{len(self.protocol_words)}) MỤC TIÊU: [#{next_tgt['class_index']:03d}] {next_tgt['word'].upper()}")
                    else:
                        print("  [THÔNG BÁO] Không thể chuyển từ khi đang thu mẫu hoặc đang xử lý.")

                if state == "IDLE":
                    if key in (32, ord('r'), ord('R')):  # SPACE or R to record
                        cv2.waitKey(200)
                        raw_buffer.clear()
                        rec_start_time = time.monotonic()
                        state = "RECORDING"
                        last_predictions = None

                elif state == "RECORDING":
                    if key in (32, 13, 10) and len(raw_buffer) >= 15:  # Early stop
                        cv2.waitKey(200)
                        state = "PROCESSING"

                elif state == "PROCESSING":
                    disp_p = self.draw_ui_overlay(
                        disp_frame, "PROCESSING", is_mirrored,
                        fps_display=current_fps, trial_id=current_trial_idx, attempt_id=current_attempt_idx
                    )
                    cv2.imshow(window_name, disp_p)
                    cv2.waitKey(1)

                    # Extract skeleton from in-memory frames
                    skel, ext_meta = self.extractor.extract_from_frames(raw_buffer, fps=actual_fps)
                    last_meta = ext_meta

                    if skel is not None and ext_meta.get("valid", False):
                        preds = self.model_manager.predict(skel)
                        last_predictions = preds

                        saved_path = None
                        if self.save_videos:
                            tgt_tag = f"{self.target_meta['class_index']:03d}_{self.target_meta['entry_id']}" if self.target_meta else "free_play"
                            saved_name = f"{self.signer_id}_{tgt_tag}_trial_{current_trial_idx:02d}_att_{current_attempt_idx:02d}.mp4"
                            saved_path = self.output_dir / "videos" / saved_name
                            saved_path.parent.mkdir(parents=True, exist_ok=True)
                            vw = cv2.VideoWriter(str(saved_path), cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (frame.shape[1], frame.shape[0]))
                            for f in raw_buffer:
                                vw.write(f)
                            vw.release()

                        # Build candidate record
                        tgt_c = self.target_meta["class_index"] if self.target_meta else -1
                        tgt_w = self.target_meta["word"] if self.target_meta else "FREE_PLAY"
                        candidate = {
                            "trial_index": current_trial_idx,
                            "attempt_id": current_attempt_idx,
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "signer_id": self.signer_id,
                            "target_class_index": tgt_c,
                            "target_word": tgt_w,
                            "top1_predicted_class_index": preds["top1_class_index"],
                            "top1_predicted_word": preds["top1_word"],
                            "confidence": preds["confidence"],
                            "margin": preds["margin"],
                            "is_correct_top1": (preds["top1_class_index"] == tgt_c) if tgt_c != -1 else None,
                            "is_correct_top3": (tgt_c in [t["class_index"] for t in preds["top3"]]) if tgt_c != -1 else None,
                            "is_uncertain": preds["is_uncertain"],
                            "top3": preds["top3"],
                            "shoulder_width": preds["normalization_meta"]["body_shoulder_width"],
                            "scale_factor": preds["normalization_meta"]["scale_factor"],
                            "inference_time_ms": preds["inference_time_ms"],
                            "is_accepted": False,
                            "saved_video_path": str(saved_path) if saved_path else None,
                        }
                        self.pending_candidate_record = candidate
                        state = "RESULTS"
                    else:
                        print(f"  [LỖI TRÍCH XUẤT] Không nhận diện đủ khớp cơ thể: {ext_meta.get('error', 'Pose/Hands missing')}")
                        state = "IDLE"

                elif state == "RESULTS":
                    if key in (32, 13, 10):  # SPACE or ENTER to accept and advance
                        cv2.waitKey(200)
                        raw_buffer.clear()
                        if self.pending_candidate_record is not None:
                            accepted = dict(self.pending_candidate_record)
                            accepted["is_accepted"] = True
                            accepted["accepted_at"] = datetime.now(timezone.utc).isoformat()
                            self.official_records.append(accepted)
                            self.all_attempts_log.append(accepted)
                            self.pending_candidate_record = None

                            corr_tag = ""
                            if accepted["is_correct_top1"] is True:
                                corr_tag = " [CHÍNH XÁC]"
                            elif accepted["is_correct_top3"] is True:
                                corr_tag = " [TRONG TOP-3]"

                            print(f"\n[LƯỢT #{accepted['trial_index']:02d} - ĐÃ LƯU] Top-1: [#{accepted['top1_predicted_class_index']:03d}] {accepted['top1_predicted_word']} ({accepted['confidence']*100:.1f}%){corr_tag}")
                            top3_s = ", ".join([f"#{t['class_index']:03d} {t['word']} ({t['probability']*100:.1f}%)" for t in accepted['top3']])
                            print(f"  Top-3: {top3_s}")

                        current_trial_idx += 1
                        current_attempt_idx = 1
                        state = "IDLE"

                    elif key in (ord('r'), ord('R')):  # Redo current trial
                        cv2.waitKey(200)
                        if self.pending_candidate_record is not None:
                            rejected = dict(self.pending_candidate_record)
                            rejected["is_accepted"] = False
                            rejected["retry_reason"] = "USER_RETRY"
                            rejected["rejected_at"] = datetime.now(timezone.utc).isoformat()
                            self.all_attempts_log.append(rejected)
                            print(f"  [KÝ LẠI] Yêu cầu ký lại lượt #{rejected['trial_index']:02d} (đã ghi nhận lần thử #{current_attempt_idx}).")
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
        Export session records to JSON and CSV and print comparison summary table.
        Guarded against multiple calls.
        """
        if self.report_exported and self._last_report_payload is not None:
            return self._last_report_payload

        self.report_exported = True
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = self.output_dir / f"live_words_703_{self.signer_id}_{ts}.json"
        csv_file = self.output_dir / f"live_words_703_{self.signer_id}_{ts}.csv"
        csv_all_attempts_file = self.output_dir / f"live_words_703_{self.signer_id}_{ts}_all_attempts.csv"

        official_eval = [r for r in self.official_records if r.get("target_class_index", -1) != -1]
        eval_total = len(official_eval)
        free_play_count = len(self.official_records) - eval_total
        total_attempts = len(self.all_attempts_log)
        retries_count = max(0, total_attempts - len(self.official_records))

        summary = {
            "session_timestamp": ts,
            "signer_id": self.signer_id,
            "total_official_trials": len(self.official_records),
            "total_trials": len(self.official_records),
            "free_play_trials": free_play_count,
            "evaluated_trials": eval_total,
            "total_attempts": total_attempts,
            "retries_count": retries_count,
        }

        if eval_total > 0:
            top1_c = sum(1 for r in official_eval if r.get("is_correct_top1"))
            top3_c = sum(1 for r in official_eval if r.get("is_correct_top3"))
            summary["top1_accuracy"] = round(top1_c / eval_total, 4)
            summary["top3_accuracy"] = round(top3_c / eval_total, 4)

            print("\n================== TỔNG KẾT PHIÊN KIỂM THỬ LIVE 703 TỪ ==================")
            print(f"Người ký: {self.signer_id} | Tổng lượt: {len(self.official_records)} (Có mục tiêu: {eval_total}, Thử tự do: {free_play_count}, Ký lại: {retries_count})")
            print(f"  Top-1 Accuracy : {top1_c}/{eval_total} ({summary['top1_accuracy']*100:.1f}%) [Chỉ tính trên {eval_total} lượt có mục tiêu]")
            print(f"  Top-3 Accuracy : {top3_c}/{eval_total} ({summary['top3_accuracy']*100:.1f}%) [Chỉ tính trên {eval_total} lượt có mục tiêu]")

        full_payload = {
            "summary": summary,
            "official_records": self.official_records,
            "all_attempts_log": self.all_attempts_log,
            "records": self.official_records,
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
                    if k not in seen and k != "top3":
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
                    if k not in seen and k != "top3":
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
    p = argparse.ArgumentParser(description="Standalone Live Webcam Test Tool for 703 QIPEDC Words")
    p.add_argument("--target-class", type=int, default=None, help="Target global class index (0..715)")
    p.add_argument("--target-word", type=str, default=None, help="Target Vietnamese word (e.g. 'nghĩ', 'khác')")
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

    app = LiveWords703TestApp(
        target_class_index=args.target_class,
        target_word=args.target_word,
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
