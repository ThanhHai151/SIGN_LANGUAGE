"""
Học ký hiệu MỚI ngay khi người dùng cần (từ chưa có chuyển động robot)
=====================================================================

Trình duyệt chạy MediaPipe Holistic trên video QIPEDC của từ đó (src/services/motionLearner.js) và gửi các
điểm khớp lên đây. Backend chuyển sang góc khớp robot (retarget_motion), lưu backend/data/motions/<mã>.json,
viết phần diễn tả ký hiệu và trả về để robot diễn ngay. Lần sau từ đó có sẵn, không phải học lại.
"""
import json
import os
import re
import threading

from backend.models.lexicon import MOTION_DIR, DESC_PATH, motion_clip
from backend.training.retarget_motion import retarget
from backend.training.describe_signs import describe

ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
MIN_HAND_COVERAGE = 0.15
_lock = threading.Lock()


def _valid_frames(frames):
    if not isinstance(frames, list) or not (4 <= len(frames) <= 600):
        return False
    for f in frames:
        if not isinstance(f, dict):
            return False
        for k, n in (("p", 27), ("l", 63), ("r", 63)):
            v = f.get(k)
            if v is None:
                continue
            if not isinstance(v, list) or len(v) != n or not all(isinstance(x, (int, float)) for x in v):
                return False
    return True


def learn(raw: dict) -> dict:
    mid = str(raw.get("id") or "")
    if not ID_RE.match(mid):
        return {"success": False, "reason": "Mã video không hợp lệ"}
    frames = raw.get("frames")
    if not _valid_frames(frames):
        return {"success": False, "reason": "Dữ liệu khung hình không hợp lệ"}
    fps = float(raw.get("fps") or 8)
    raw = {"id": mid, "fps": fps, "dur": float(raw.get("dur") or len(frames) / fps), "frames": frames}

    m = retarget(raw)
    cov = max(m["hand_coverage"].values())
    if cov < MIN_HAND_COVERAGE:
        return {"success": False, "reason": "Không nhận ra tay trong video"}

    os.makedirs(MOTION_DIR, exist_ok=True)
    path = os.path.join(MOTION_DIR, mid + ".json")
    tmp = path + ".part"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(m, f, separators=(",", ":"))
    os.replace(tmp, path)
    motion_clip.cache_clear()

    desc = None
    try:
        desc = describe(raw, m.get("active"))
    except Exception:  # noqa: BLE001 — thiếu phần diễn tả không chặn việc robot diễn
        desc = None
    if desc:
        with _lock:
            data = {}
            if os.path.exists(DESC_PATH):
                try:
                    with open(DESC_PATH, encoding="utf-8") as f:
                        data = json.load(f)
                except (OSError, ValueError):
                    data = {}
            data[mid] = desc
            tmp = DESC_PATH + ".part"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, separators=(",", ":"))
            os.replace(tmp, DESC_PATH)

    return {
        "success": True,
        "motion_id": mid,
        "clip": m.get("active"),
        "sign_description": desc,
        "hand_coverage": m["hand_coverage"],
    }
