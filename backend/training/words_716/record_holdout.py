"""
Independent Offline Webcam Holdout Recording Tool for WordRecognizer716 Pilot.

Features:
- Structured recording queue for 12 selected words x 3 trials + 3 negative controls (39 total).
- Clear visual prompts with Vietnamese word names, gestures, and overall progress.
- Non-destructive resume capability: checks existing files and skips completed trials.
- Preflight diagnostic mode (--preflight): tests webcam opening, resolution, FPS, and video encoding/decoding.
- Status mode (--status): displays completed vs pending matrix.
- Import mode (--import): allows importing user videos recorded via external devices/phone.
- Validation mode (--validate): verifies video integrity, decodability, duration, and metadata.
- Isolated storage in backend/data/words_716/holdout/ with holdout_manifest.json.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WORDS_DIR = ROOT / "backend" / "data" / "words_716"
PROTOCOL_FILE = WORDS_DIR / "holdout_protocol.json"
HOLDOUT_DIR = WORDS_DIR / "holdout"
MANIFEST_FILE = HOLDOUT_DIR / "holdout_manifest.json"


class HoldoutRecorder:
    """
    Manages offline holdout recording, validation, and metadata management.
    """

    def __init__(
        self,
        protocol_path: Path = PROTOCOL_FILE,
        holdout_dir: Path = HOLDOUT_DIR,
        signer_id: str = "signer_01",
    ):
        self.protocol_path = Path(protocol_path)
        self.holdout_dir = Path(holdout_dir)
        self.signer_id = signer_id
        self.holdout_dir.mkdir(parents=True, exist_ok=True)

        if not self.protocol_path.is_file():
            raise FileNotFoundError(f"Protocol file not found: {self.protocol_path}")

        self.protocol = json.loads(self.protocol_path.read_text(encoding="utf-8"))
        self.words = self.protocol.get("words", [])
        self.repetitions = self.protocol.get("protocol", {}).get("repetitions_per_word", 3)
        self.negative_controls = self.protocol.get("protocol", {}).get("idle_negative_controls", [])

    def get_full_queue(self) -> List[Dict[str, Any]]:
        """
        Build the complete ordered queue of 39 target items:
        12 words x 3 trials = 36 items
        + 3 negative control items = 39 total items.
        """
        queue = []
        item_id = 1

        signer_prefix = self.signer_id if self.signer_id.startswith("signer_") else f"signer_{self.signer_id}"

        # 1. 12 words x 3 repetitions
        for w in self.words:
            for rep in range(1, self.repetitions + 1):
                queue.append({
                    "item_id": item_id,
                    "type": "WORD",
                    "class_index": w["class_index"],
                    "entry_id": w["entry_id"],
                    "word": w["word"],
                    "region": w["region"],
                    "hands": w["hands"],
                    "gesture_hint": w.get("gesture_hint", ""),
                    "trial": rep,
                    "target_filename": f"{signer_prefix}_{w['class_index']:03d}_{w['entry_id']}_trial_{rep:02d}.mp4",
                })
                item_id += 1

        # 2. Negative controls
        for c_idx, ctrl_desc in enumerate(self.negative_controls, 1):
            ctrl_key = f"control_{c_idx}"
            queue.append({
                "item_id": item_id,
                "type": "CONTROL",
                "class_index": -1,
                "entry_id": ctrl_key,
                "word": f"ĐỐI CHỨNG: {ctrl_desc}",
                "region": "NONE",
                "hands": "NONE",
                "gesture_hint": ctrl_desc,
                "trial": 1,
                "target_filename": f"{signer_prefix}_ctrl_{c_idx:02d}_trial_01.mp4",
            })
            item_id += 1

        return queue

    def load_manifest(self) -> Dict[str, Any]:
        if MANIFEST_FILE.is_file():
            try:
                return json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {
            "schema_version": "3.2e-holdout-manifest-v1",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "evaluation_mode": "independent_holdout",
            "dataset_isolation_rule": "STRICTLY_ISOLATED_FROM_TRAINING_DATA",
            "signer_id": self.signer_id,
            "target_total_items": len(self.get_full_queue()),
            "items": [],
        }

    def save_manifest(self, manifest: Dict[str, Any]):
        manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
        MANIFEST_FILE.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def get_status(self) -> Dict[str, Any]:
        """
        Scan holdout directory and return completed vs pending status.
        """
        queue = self.get_full_queue()
        manifest = self.load_manifest()
        recorded_filenames = {item["filename"]: item for item in manifest.get("items", [])}

        # Also inspect disk directly
        disk_files = {f.name: f for f in self.holdout_dir.glob("*.mp4")}

        completed = []
        pending = []

        for q in queue:
            fname = q["target_filename"]
            if fname in disk_files and fname in recorded_filenames:
                fpath = disk_files[fname]
                meta = recorded_filenames[fname]
                completed.append({**q, "file_path": str(fpath), "size_bytes": fpath.stat().st_size, **meta})
            elif fname in disk_files:
                # Exists on disk but missing from manifest: repair
                fpath = disk_files[fname]
                v_info = self.probe_video_file(fpath)
                completed.append({**q, "file_path": str(fpath), "size_bytes": fpath.stat().st_size, **v_info})
            else:
                pending.append(q)

        total = len(queue)
        done_n = len(completed)
        pct = (done_n / total) * 100.0 if total > 0 else 0.0

        return {
            "total_planned": total,
            "completed_count": done_n,
            "pending_count": len(pending),
            "percentage_completed": round(pct, 1),
            "signer_id": self.signer_id,
            "completed_items": completed,
            "pending_items": pending,
        }

    def probe_video_file(self, video_path: Path) -> Dict[str, Any]:
        """
        Probe FPS, resolution, frame count, duration of a video.
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return {"readable": False, "error": "Cannot open video file"}

        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        dur_sec = round(n_frames / fps, 3) if fps > 0 else 0.0

        ret, frame = cap.read()
        cap.release()

        return {
            "readable": bool(ret),
            "width": w,
            "height": h,
            "fps": round(fps, 2),
            "frame_count": n_frames,
            "duration_sec": dur_sec,
        }

    def preflight_check(self, device_index: int = 0) -> Dict[str, Any]:
        """
        Diagnostic test of webcam accessibility and video encoding.
        """
        diag = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "opencv_version": cv2.__version__,
            "device_index": device_index,
            "webcam_opened": False,
            "can_read_frame": False,
            "resolution": None,
            "fps": None,
            "encoding_test_pass": False,
            "status": "INITIALIZING",
            "messages": [],
        }

        # 1. Test opening camera
        cap = None
        for backend_id in [cv2.CAP_ANY, cv2.CAP_DSHOW, cv2.CAP_MSMF]:
            try:
                cap = cv2.VideoCapture(device_index, backend_id)
                if cap.isOpened():
                    diag["webcam_opened"] = True
                    break
                else:
                    cap.release()
            except Exception as e:
                diag["messages"].append(f"Backend error: {str(e)}")

        if not diag["webcam_opened"]:
            diag["status"] = "WEBCAM_NOT_FOUND"
            diag["messages"].append(
                f"Webcam device index [{device_index}] cannot be opened by OpenCV. "
                "Ensure a webcam is physically connected, drivers are active, and no other app is locking the camera."
            )
            return diag

        # 2. Test reading frame
        ret, frame = cap.read()
        if ret and frame is not None:
            diag["can_read_frame"] = True
            h, w = frame.shape[:2]
            diag["resolution"] = f"{w}x{h}"
            diag["fps"] = round(cap.get(cv2.CAP_PROP_FPS) or 30.0, 2)
        else:
            diag["status"] = "CANNOT_READ_FRAME"
            diag["messages"].append("Webcam opened but failed to capture video frame.")
            cap.release()
            return diag
        cap.release()

        # 3. Test video encoding and saving
        test_file = self.holdout_dir / "preflight_test.mp4"
        try:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(str(test_file), fourcc, 30.0, (w, h))
            for _ in range(30):
                out.write(frame)
            out.release()

            probe = self.probe_video_file(test_file)
            if probe.get("readable") and probe.get("frame_count", 0) >= 20:
                diag["encoding_test_pass"] = True
                diag["status"] = "PREFLIGHT_PASS"
                diag["messages"].append("Webcam, frame capture, and mp4 video encoding verified successfully.")
            else:
                diag["status"] = "ENCODING_FAILED"
                diag["messages"].append("Video encoding test failed or created corrupted file.")
        finally:
            if test_file.is_file():
                try:
                    test_file.unlink()
                except Exception:
                    pass

        return diag

    def import_external_video(
        self,
        source_video_path: Union[str, Path],
        target_item_id: int,
    ) -> Dict[str, Any]:
        """
        Import an externally recorded video (e.g. from smartphone or camera app) into holdout dataset.
        """
        source_path = Path(source_video_path).resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"Source video not found: {source_path}")

        queue = self.get_full_queue()
        matched = [q for q in queue if q["item_id"] == target_item_id]
        if not matched:
            raise ValueError(f"Invalid item_id {target_item_id}. Must be between 1 and {len(queue)}.")
        target_info = matched[0]

        target_file = self.holdout_dir / target_info["target_filename"]
        if target_file.is_file():
            raise FileExistsError(f"Target video already exists: {target_file}. Delete or move it first.")

        # Probe source
        probe = self.probe_video_file(source_path)
        if not probe.get("readable"):
            raise ValueError(f"Source video cannot be read or is corrupted: {probe.get('error')}")

        if probe.get("duration_sec", 0) < 0.5:
            raise ValueError(f"Source video is too short ({probe.get('duration_sec')}s). Minimum is 0.5s.")

        # Copy to holdout dir
        shutil.copy2(source_path, target_file)

        # Update manifest
        manifest = self.load_manifest()
        record = {
            "item_id": target_info["item_id"],
            "type": target_info["type"],
            "class_index": target_info["class_index"],
            "entry_id": target_info["entry_id"],
            "word": target_info["word"],
            "region": target_info["region"],
            "hands": target_info["hands"],
            "trial": target_info["trial"],
            "filename": target_info["target_filename"],
            "source_import_path": str(source_path),
            "imported_at": datetime.now(timezone.utc).isoformat(),
            **probe,
        }
        manifest["items"] = [i for i in manifest.get("items", []) if i["filename"] != target_info["target_filename"]]
        manifest["items"].append(record)
        self.save_manifest(manifest)

        return record

    def validate_dataset(self) -> Dict[str, Any]:
        """
        Thoroughly validate all videos in the holdout directory against the protocol.
        """
        queue = self.get_full_queue()
        manifest = self.load_manifest()
        status = self.get_status()

        issues = []
        valid_items = []

        for item in status["completed_items"]:
            fpath = Path(item["file_path"])
            probe = self.probe_video_file(fpath)
            if not probe.get("readable"):
                issues.append(f"Corrupted or unreadable file: {fpath.name}")
            elif probe.get("frame_count", 0) < 15:
                issues.append(f"Insufficient frames ({probe.get('frame_count')}) in {fpath.name}")
            elif probe.get("duration_sec", 0) < 0.5:
                issues.append(f"Duration too short ({probe.get('duration_sec')}s) in {fpath.name}")
            else:
                valid_items.append({**item, **probe})

        validation_result = {
            "schema_version": "3.2e-holdout-validation-v1",
            "validated_at": datetime.now(timezone.utc).isoformat(),
            "target_total": len(queue),
            "completed_valid": len(valid_items),
            "pending_count": status["pending_count"],
            "is_complete": bool(len(valid_items) == len(queue)),
            "issues_count": len(issues),
            "issues": issues,
            "completion_percentage": status["percentage_completed"],
        }
        return validation_result

    def draw_overlay(
        self,
        frame: np.ndarray,
        state: str,
        current_item: Dict[str, Any],
        progress_pct: float,
        done_cnt: int,
        total_cnt: int,
        elapsed_sec: float = 0.0,
        frame_count: int = 0,
        review_probe: Optional[Dict[str, Any]] = None,
        is_mirrored: bool = True,
        remaining_sec: Optional[float] = None,
        max_duration_sec: float = 5.0,
    ) -> np.ndarray:
        """
        Render visual guide and state banners on frame using PIL for crisp Vietnamese diacritics.
        Does NOT alter original frame coordinates.
        """
        h, w = frame.shape[:2]
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img)

        # Load fonts safely
        try:
            font_title = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(18, int(h * 0.045)))
            font_sub = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", max(13, int(h * 0.030)))
            font_status = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", max(16, int(h * 0.038)))
        except Exception:
            font_title = font_sub = font_status = ImageFont.load_default()

        # 1. Top status bar (semi-transparent dark background)
        top_h = int(h * 0.22)
        draw.rectangle([(0, 0), (w, top_h)], fill=(15, 18, 24))

        # Progress info & Mirror mode
        mirror_str = "GƯƠNG [M]" if is_mirrored else "CHUẨN [M]"
        p_text = f"VSL HOLDOUT | Người ký: {self.signer_id} | Tiến độ: {done_cnt}/{total_cnt} ({progress_pct:.1f}%) | Chế độ: {mirror_str}"
        draw.text((15, 8), p_text, font=font_sub, fill=(200, 210, 225))

        # Progress bar
        bar_x1, bar_y1, bar_x2, bar_y2 = 15, int(top_h * 0.32), w - 15, int(top_h * 0.32) + 8
        draw.rectangle([(bar_x1, bar_y1), (bar_x2, bar_y2)], fill=(40, 45, 55), outline=(90, 95, 110))
        fill_w = int((bar_x2 - bar_x1) * (progress_pct / 100.0))
        if fill_w > 0:
            draw.rectangle([(bar_x1, bar_y1), (bar_x1 + fill_w, bar_y2)], fill=(0, 204, 102))

        # Item title & word
        item_id = current_item.get("item_id", 0)
        word_label = current_item.get("word", "")
        region = current_item.get("region", "")
        hands = current_item.get("hands", "")
        trial = current_item.get("trial", 1)

        if current_item.get("type") == "CONTROL":
            title_text = f"Mục {item_id:02d}/{total_cnt}: {word_label}"
            sub_text = "Giữ tư thế nghỉ hoặc hành động tự nhiên ngoài từ điển"
        else:
            title_text = f"Mục {item_id:02d}/{total_cnt}: {word_label.upper()} ({region}, {hands}) - Lần {trial}/3"
            hint = current_item.get("gesture_hint", "")
            sub_text = f"Gợi ý: {hint}" if hint else "Thực hiện ký hiệu tự nhiên"

        draw.text((15, int(top_h * 0.45)), title_text, font=font_title, fill=(255, 220, 0))
        draw.text((15, int(top_h * 0.75)), sub_text, font=font_sub, fill=(180, 220, 255))

        # 2. Bottom status action bar
        bot_h = int(h * 0.14)
        bot_y = h - bot_h
        draw.rectangle([(0, bot_y), (w, h)], fill=(15, 18, 24))

        if state == "IDLE":
            msg1 = f"[CHỜ QUAY] Nhấn [SPACE/R] BẮT ĐẦU (Tự dừng {max_duration_sec:.0f}s) | [M] Đổi gương | [Q/ESC] Thoát"
            draw.text((15, bot_y + int(bot_h * 0.28)), msg1, font=font_status, fill=(0, 255, 128))

        elif state == "RECORDING":
            # Blinking red dot
            blink = int(time.time() * 2) % 2 == 0
            dot_color = (255, 50, 50) if blink else (150, 20, 20)
            draw.ellipse([(15, bot_y + int(bot_h * 0.25)), (30, bot_y + int(bot_h * 0.25) + 15)], fill=dot_color)
            rem = max(0.0, max_duration_sec - elapsed_sec) if remaining_sec is None else remaining_sec
            rec_text = f"  ĐANG GHI [{rem:.1f}s / {max_duration_sec:.1f}s] ({frame_count}f) | [SPACE/R] Dừng sớm | [M] Đổi gương"
            draw.text((35, bot_y + int(bot_h * 0.28)), rec_text, font=font_status, fill=(255, 80, 80))

        elif state == "REVIEW":
            dur = review_probe.get("duration_sec", 0.0) if review_probe else (frame_count / 30.0)
            fc = review_probe.get("frame_count", frame_count) if review_probe else frame_count
            draw.text((15, bot_y + 6), f"ĐÃ GHI: {dur:.1f}s ({fc} frames) - Kiểm tra video:", font=font_sub, fill=(255, 230, 100))
            opt_text = "[Y / ENTER] Chấp nhận & Lưu | [N] Quay lại | [P] Xem lại | [M] Đổi gương | [Q / ESC] Thoát"
            draw.text((15, bot_y + int(bot_h * 0.45)), opt_text, font=font_status, fill=(0, 255, 255))

        elif state == "PLAYBACK":
            draw.text((15, bot_y + int(bot_h * 0.28)), "ĐANG XEM LẠI VIDEO... (Nhấn phím bất kỳ để dừng xem | [M] Đổi gương)", font=font_status, fill=(0, 200, 255))

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def save_recorded_trial(
        self,
        frames: List[np.ndarray],
        target_item: Dict[str, Any],
        fps: float = 30.0,
    ) -> Dict[str, Any]:
        """
        Write unaugmented raw BGR frames to target holdout MP4 file and update manifest.
        """
        if not frames:
            raise ValueError("No frames to save")

        target_file = self.holdout_dir / target_item["target_filename"]
        h, w = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")

        # Write to temporary file first
        temp_file = self.holdout_dir / f".tmp_{target_item['target_filename']}"
        out = cv2.VideoWriter(str(temp_file), fourcc, fps, (w, h))
        for f in frames:
            out.write(f)
        out.release()

        probe = self.probe_video_file(temp_file)
        if not probe.get("readable"):
            if temp_file.is_file():
                temp_file.unlink()
            raise RuntimeError(f"Written video failed integrity probe: {probe.get('error')}")

        # Rename to final target
        if target_file.is_file():
            target_file.unlink()
        temp_file.replace(target_file)

        # Update manifest
        manifest = self.load_manifest()
        record = {
            "item_id": target_item["item_id"],
            "type": target_item["type"],
            "class_index": target_item["class_index"],
            "entry_id": target_item["entry_id"],
            "word": target_item["word"],
            "region": target_item["region"],
            "hands": target_item["hands"],
            "trial": target_item["trial"],
            "filename": target_item["target_filename"],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            **probe,
        }
        manifest["items"] = [i for i in manifest.get("items", []) if i["filename"] != target_item["target_filename"]]
        manifest["items"].append(record)
        self.save_manifest(manifest)

        return record

    def run_interactive_session(
        self,
        device_index: int = 0,
        target_fps: float = 30.0,
        max_duration_sec: float = 5.0,
    ) -> int:
        """
        Full HighGUI OpenCV interactive recording loop with live preview, prompts, and controls.
        Supports 5-second auto-stop and mirror display mode toggle (key M).
        Returns exit code 0 on clean exit/completion, non-zero on error.
        """
        print("\n================== VSL HOLDOUT RECORDER (INTERACTIVE) ==================")
        print(f"Người ký: {self.signer_id}")
        print("Đang kiểm tra khả năng mở camera...")

        diag = self.preflight_check(device_index=device_index)
        if diag["status"] != "PREFLIGHT_PASS":
            print(f"\n[LỖI PREFLIGHT] Không thể mở webcam trên thiết bị [{device_index}]: {diag['status']}")
            for msg in diag["messages"]:
                print(f"  - {msg}")
            print("\nHướng dẫn khắc phục:")
            print("  1. Đảm bảo webcam USB đã được cắm hoặc camera tích hợp đã bật.")
            print("  2. Chạy lệnh trực tiếp từ PowerShell ngoài Antigravity:")
            print(f"     .\\.venv\\Scripts\\python.exe backend/training/words_716/record_holdout.py --interactive --device {device_index}")
            print("  3. Nếu có nhiều camera, thử các chỉ số: --device 0, --device 1, --device 2.")
            return 1

        # Check pending queue
        status = self.get_status()
        pending = status["pending_items"]
        if not pending:
            print("\n[THÔNG BÁO] Toàn bộ 39/39 video holdout đã được ghi hình và lưu trữ đầy đủ!")
            val = self.validate_dataset()
            print(f"Kết quả kiểm định dataset: {val['completed_valid']}/39 video hợp lệ ({val['completion_percentage']}%)")
            return 0

        # Open camera stream
        cap = None
        for backend_id in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]:
            try:
                c = cv2.VideoCapture(device_index, backend_id)
                if c.isOpened():
                    cap = c
                    break
                c.release()
            except Exception:
                pass

        if cap is None or not cap.isOpened():
            print(f"[LỖI] Không thể mở luồng video từ thiết bị camera {device_index}.")
            return 1

        # Set resolution to 640x480 by default (or camera default)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        actual_fps = cap.get(cv2.CAP_PROP_FPS) or target_fps
        if actual_fps <= 0 or actual_fps > 120:
            actual_fps = target_fps

        window_name = "VSL Holdout Recorder - HighGUI"
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
        except Exception as e:
            print(f"[LỖI HIGHGUI] Không thể mở cửa sổ hiển thị OpenCV: {e}")
            cap.release()
            return 1

        is_mirrored = True  # Default: MIRROR mode enabled for natural preview
        print(f"Đã mở camera thành công (FPS={actual_fps:.1f})")
        print(f"Chế độ hiển thị : {'GƯƠNG (Phím M: Chuẩn)' if is_mirrored else 'CHUẨN (Phím M: Gương)'}")
        print(f"Thời lượng quay : Tự động dừng sau {max_duration_sec:.1f}s (hoặc nhấn SPACE/R để dừng sớm)")
        print(f"Còn lại {len(pending)}/39 video cần thu thập.")
        print("Bắt đầu giao diện ghi hình...")

        current_idx = 0
        state = "IDLE"  # IDLE -> RECORDING -> REVIEW -> PLAYBACK
        raw_buffer: List[np.ndarray] = []
        rec_start_time = 0.0
        review_probe = None
        session_accepted_count = 0

        try:
            while current_idx < len(pending):
                current_item = pending[current_idx]
                ret, frame = cap.read()
                if not ret or frame is None:
                    # Retry read
                    time.sleep(0.01)
                    continue

                # Compute progress
                st_now = self.get_status()
                done_n = st_now["completed_count"]
                total_n = st_now["total_planned"]
                pct_now = st_now["percentage_completed"]

                if state == "RECORDING":
                    # Invariant: raw_buffer ALWAYS stores unmodified raw sensor frames (never mirrored)
                    raw_buffer.append(frame.copy())
                    now = time.monotonic()
                    elapsed = now - rec_start_time
                    remaining = max(0.0, max_duration_sec - elapsed)

                    # Mirror only for display
                    disp_frame = cv2.flip(frame, 1) if is_mirrored else frame.copy()
                    disp = self.draw_overlay(
                        disp_frame, state, current_item, pct_now, done_n, total_n,
                        elapsed_sec=elapsed, frame_count=len(raw_buffer),
                        is_mirrored=is_mirrored, remaining_sec=remaining,
                        max_duration_sec=max_duration_sec,
                    )
                    cv2.imshow(window_name, disp)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord('m'), ord('M')):
                        is_mirrored = not is_mirrored
                    elif key in (32, ord('r'), ord('R')):
                        if len(raw_buffer) < 15:
                            print("  [Cảnh báo] Video dưới 15 frame (<0.5s), tiếp tục quay...")
                        else:
                            state = "REVIEW"
                            dur_est = len(raw_buffer) / actual_fps
                            review_probe = {"duration_sec": round(dur_est, 2), "frame_count": len(raw_buffer)}
                    elif elapsed >= max_duration_sec:
                        # Auto-stop after max_duration_sec (5.0s) reached
                        state = "REVIEW"
                        dur_est = len(raw_buffer) / actual_fps
                        review_probe = {"duration_sec": round(dur_est, 2), "frame_count": len(raw_buffer)}
                    elif key in (27, ord('q'), ord('Q')):
                        print("\n[TẠM DỪNG] Người dùng đã hủy phiên quay hiện tại.")
                        break

                elif state == "REVIEW":
                    # Display preview with current mirror mode
                    preview_frame = raw_buffer[len(raw_buffer) // 2] if raw_buffer else frame
                    disp_preview = cv2.flip(preview_frame, 1) if is_mirrored else preview_frame.copy()
                    disp = self.draw_overlay(
                        disp_preview, state, current_item, pct_now, done_n, total_n,
                        frame_count=len(raw_buffer), review_probe=review_probe,
                        is_mirrored=is_mirrored, max_duration_sec=max_duration_sec,
                    )
                    cv2.imshow(window_name, disp)
                    key = cv2.waitKey(30) & 0xFF

                    if key in (ord('m'), ord('M')):
                        is_mirrored = not is_mirrored
                    elif key in (ord('y'), ord('Y'), 13, 10):  # Accept (Y or ENTER)
                        saved_rec = self.save_recorded_trial(raw_buffer, current_item, fps=actual_fps)
                        session_accepted_count += 1
                        print(f"  [ĐÃ LƯU] Item {current_item['item_id']:02d}: {saved_rec['filename']} ({saved_rec['duration_sec']}s)")
                        raw_buffer.clear()
                        review_probe = None
                        current_idx += 1
                        state = "IDLE"
                    elif key in (ord('n'), ord('N')):  # Reject / Redo (N)
                        print(f"  [QUAY LẠI] Đã hủy lần quay vừa rồi, bắt đầu quay lại Item {current_item['item_id']:02d}")
                        raw_buffer.clear()
                        review_probe = None
                        state = "IDLE"
                    elif key in (ord('p'), ord('P')):  # Playback (P)
                        state = "PLAYBACK"
                    elif key in (27, ord('q'), ord('Q')):
                        print("\n[TẠM DỪNG] Thoát chế độ xem lại.")
                        break

                elif state == "PLAYBACK":
                    # Play through buffer
                    for p_frame in raw_buffer:
                        p_disp_frame = cv2.flip(p_frame, 1) if is_mirrored else p_frame.copy()
                        p_disp = self.draw_overlay(
                            p_disp_frame, state, current_item, pct_now, done_n, total_n,
                            is_mirrored=is_mirrored, max_duration_sec=max_duration_sec,
                        )
                        cv2.imshow(window_name, p_disp)
                        p_key = cv2.waitKey(int(1000 / actual_fps)) & 0xFF
                        if p_key in (ord('m'), ord('M')):
                            is_mirrored = not is_mirrored
                        elif p_key != 255:
                            break
                    state = "REVIEW"

                else:  # IDLE state
                    disp_frame = cv2.flip(frame, 1) if is_mirrored else frame.copy()
                    disp = self.draw_overlay(
                        disp_frame, state, current_item, pct_now, done_n, total_n,
                        is_mirrored=is_mirrored, max_duration_sec=max_duration_sec,
                    )
                    cv2.imshow(window_name, disp)
                    key = cv2.waitKey(1) & 0xFF

                    if key in (ord('m'), ord('M')):
                        is_mirrored = not is_mirrored
                    elif key in (32, ord('r'), ord('R')):
                        state = "RECORDING"
                        raw_buffer = []
                        rec_start_time = time.monotonic()
                    elif key in (27, ord('q'), ord('Q')):
                        print("\n[THOÁT] Người dùng chọn dừng phiên quay.")
                        break

        finally:
            cap.release()
            cv2.destroyAllWindows()

        final_st = self.get_status()
        print("\n================== TỔNG KẾT PHIÊN THU MẪU ==================")
        print(f"Số video đã lưu trong phiên này : {session_accepted_count}")
        print(f"Tổng số video đã hoàn thành     : {final_st['completed_count']}/{final_st['total_planned']} ({final_st['percentage_completed']}%)")
        print(f"Số video còn lại (Pending)      : {final_st['pending_count']}")
        print(f"Thư mục lưu trữ                 : {self.holdout_dir}")
        if final_st["percentage_completed"] == 100.0:
            print("\n[CHÚC MỪNG] Đã thu thập đủ toàn bộ 39/39 video holdout!")
            val = self.validate_dataset()
            print(f"Kiểm định tính toàn vẹn: {val['completed_valid']}/39 tệp hợp lệ.")
        print("============================================================\n")

        return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true", help="Run webcam diagnostic preflight")
    parser.add_argument("--status", action="store_true", help="Display holdout recording progress and status")
    parser.add_argument("--validate", action="store_true", help="Validate integrity of recorded holdout dataset")
    parser.add_argument("--device", type=int, default=0, help="Webcam device index (default 0)")
    parser.add_argument("--signer", type=str, default="signer_01", help="Signer identifier")
    parser.add_argument("--duration", "--max-duration", dest="max_duration", type=float, default=5.0, help="Thời lượng tự động dừng quay (giây, mặc định 5.0)")
    parser.add_argument("--import-file", type=str, default=None, help="Import external video file")
    parser.add_argument("--item-id", type=int, default=None, help="Target item ID for import (1..39)")
    parser.add_argument("--interactive", action="store_true", help="Run interactive recording session")
    return parser


def main():
    parser = build_arg_parser()
    args = parser.parse_args()

    recorder = HoldoutRecorder(signer_id=args.signer)

    if args.preflight:
        diag = recorder.preflight_check(device_index=args.device)
        print(json.dumps(diag, ensure_ascii=False, indent=2))
        sys.exit(0 if diag["status"] == "PREFLIGHT_PASS" else 1)

    elif args.status:
        st = recorder.get_status()
        print(f"\n=== HOLDOUT RECORDING STATUS ({st['signer_id']}) ===")
        print(f"Total Planned: {st['total_planned']} items (12 words x 3 trials + 3 negative controls)")
        print(f"Completed    : {st['completed_count']} items ({st['percentage_completed']}%)")
        print(f"Pending      : {st['pending_count']} items")
        if st["completed_items"]:
            print("\nCompleted Items:")
            for c in st["completed_items"]:
                print(f"  [Item {c['item_id']:02d}] {c['word']:15s} (Trial {c['trial']}) -> {c['target_filename']}")
        if st["pending_items"]:
            print(f"\nNext Pending Items (showing first 5 of {len(st['pending_items'])}):")
            for p in st["pending_items"][:5]:
                print(f"  [Item {p['item_id']:02d}] {p['word']:15s} (Trial {p['trial']}) -> {p['target_filename']}")
        print("===================================================\n")

    elif args.validate:
        val = recorder.validate_dataset()
        print(json.dumps(val, ensure_ascii=False, indent=2))

    elif args.import_file:
        if args.item_id is None:
            print("Error: --item-id (1..39) is required when using --import-file.")
            sys.exit(1)
        res = recorder.import_external_video(args.import_file, args.item_id)
        print(f"Imported successfully: {res['filename']} (Item {args.item_id}: {res['word']})")

    elif args.interactive:
        exit_code = recorder.run_interactive_session(device_index=args.device, max_duration_sec=args.max_duration)
        sys.exit(exit_code)
        sys.exit(exit_code)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
