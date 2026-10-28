"""
Bộ từ vựng VSL thống nhất (đọc từ backend/data/vsl_lexicon.json).

Đây là nguồn dữ liệu chung cho cả 2 chiều:
  - Camera → Văn bản: mỗi lớp của mô hình SSTCN (0..101) trỏ tới đúng 1 mục từ (sstcn_index).
  - Văn bản → Ký hiệu: tra cứu cụm từ dài nhất trong toàn bộ ~3.300 mục từ QIPEDC + giáo trình.
Tạo lại dữ liệu bằng:  python -m backend.training.build_lexicon
"""

import json
import os
import re
import unicodedata
from functools import lru_cache

LEXICON_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vsl_lexicon.json")
MOTION_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "motions")
_VIDEO_ID_RE = re.compile(r"qipedc\.moet\.gov\.vn/videos/([^./]+)\.mp4")


_motion_cache = {"mtime": None, "ids": set()}


def motion_ids() -> set:
    """Các video đã được 'học' chuyển động (backend/data/motions/<mã video>.json); cache theo mtime thư mục."""
    if not os.path.isdir(MOTION_DIR):
        return set()
    mtime = os.path.getmtime(MOTION_DIR)
    if _motion_cache["mtime"] != mtime:
        _motion_cache["ids"] = {f[:-5] for f in os.listdir(MOTION_DIR) if f.endswith(".json") and not f.startswith("_")}
        _motion_cache["mtime"] = mtime
    return _motion_cache["ids"]


@lru_cache(maxsize=8192)
def motion_clip(motion_id: str):
    """Đoạn có động tác [giây bắt đầu, giây kết thúc] trong video mẫu của chuyển động đã học."""
    try:
        with open(os.path.join(MOTION_DIR, motion_id + ".json"), encoding="utf-8") as f:
            return tuple(json.load(f).get("active") or ())
    except (OSError, ValueError):
        return ()


DESC_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "sign_descriptions.json")
_desc_cache = {"mtime": None, "data": {}}


def sign_description(motion_id: str):
    """Diễn tả cách làm ký hiệu, phân tích tự động từ video (backend/training/describe_signs.py)."""
    if not motion_id or not os.path.exists(DESC_PATH):
        return None
    mtime = os.path.getmtime(DESC_PATH)
    if _desc_cache["mtime"] != mtime:
        try:
            with open(DESC_PATH, encoding="utf-8") as f:
                _desc_cache["data"] = json.load(f)
        except (OSError, ValueError):
            _desc_cache["data"] = {}
        _desc_cache["mtime"] = mtime
    return _desc_cache["data"].get(motion_id)


def motion_id_of(entry: dict, available: set = None):
    m = _VIDEO_ID_RE.search(entry.get("video_url") or "")
    if not m:
        return None
    mid = m.group(1)
    available = motion_ids() if available is None else available
    return mid if mid in available else None


def norm_key(s: str) -> str:
    s = unicodedata.normalize("NFC", s or "").lower().strip()
    s = s.replace("？", "?")
    s = re.sub(r"[?!.,;:\"“”]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def strip_accents(s: str) -> str:
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def gloss_of(entry: dict) -> str:
    """Mã gloss hiển thị/điều khiển robot: ưu tiên khóa tư thế robot, nếu không thì
    viết hoa không dấu của từ (vd. 'bạn bè' → 'BAN_BE')."""
    if entry.get("pose_key"):
        return entry["pose_key"]
    base = re.sub(r"\([^)]*\)", " ", entry["word"])
    return re.sub(r"[^A-Z0-9]+", "_", strip_accents(norm_key(base)).upper()).strip("_") or entry["id"].upper()


def display_label(entry: dict, token: str = None) -> str:
    """Nhãn hiển thị: bỏ chú thích trong ngoặc, ưu tiên đúng từ người dùng đã gõ."""
    w = entry["word"].strip()
    if entry.get("fingerspelling"):
        return w.upper()
    base = re.sub(r"\s+", " ", re.sub(r"\([^)]*\)", " ", w)).strip(" ?") or w
    if token:
        # Người dùng gõ đúng một khóa của mục từ (vd. 'xin chào' → mục 'chào', 'mẹ' → 'má (giống: mẹ)')
        # → hiện đúng từ họ gõ để phụ đề khớp câu gốc
        t = norm_key(token)
        if t in entry.get("keys", []) and t != norm_key(base):
            base = t
    return base[:1].upper() + base[1:]


@lru_cache(maxsize=1)
def load_lexicon() -> dict:
    with open(LEXICON_PATH, encoding="utf-8") as f:
        data = json.load(f)
    entries = data["entries"]
    by_id = {e["id"]: e for e in entries}
    by_sstcn = {e["sstcn_index"]: e for e in entries if e.get("sstcn_index") is not None}
    by_pose = {}
    for e in sorted(entries, key=lambda x: x["rank"]):
        if e.get("pose_key"):
            by_pose.setdefault(e["pose_key"], e)

    # Chỉ mục khóa → mục từ tốt nhất (ưu tiên: từ gốc trùng khớp, có video, thông dụng, hạng cao)
    by_key = {}

    def priority(e, key):
        base = norm_key(re.sub(r"\([^)]*\)", " ", e["word"]))
        return (
            1 if base == key else 0,
            1 if e.get("pose_key") else 0,
            1 if e.get("video_url") else 0,
            1 if e.get("common") else 0,
            -e["rank"],
        )

    for e in entries:
        for k in e.get("keys", []):
            cur = by_key.get(k)
            if cur is None or priority(e, k) > priority(cur, k):
                by_key[k] = e

    max_len = max((len(k.split()) for k in by_key), default=1)
    return {
        "meta": data["meta"],
        "entries": entries,
        "by_id": by_id,
        "by_key": by_key,
        "by_sstcn": by_sstcn,
        "by_pose": by_pose,
        "max_phrase_len": min(max_len, 6),
    }


def get_entry_by_sstcn(index: int):
    return load_lexicon()["by_sstcn"].get(index)


def compact_entry(e: dict) -> dict:
    """Bản rút gọn gửi cho frontend (bỏ trường nội bộ)."""
    return {
        "id": e["id"],
        "word": e["word"],
        "keys": e.get("keys", [])[:4],
        "label": display_label(e),
        "gloss": gloss_of(e),
        "pos": e.get("pos", ""),
        "definition": e.get("definition", ""),
        "how_to": e.get("how_to", ""),
        "category": e.get("category", ""),
        "source": e.get("source", ""),
        "video_url": e.get("video_url"),
        "thumb_url": e.get("thumb_url"),
        "image_url": e.get("image_url"),
        "variants": e.get("variants", []),
        "sstcn_index": e.get("sstcn_index"),
        "pose_key": e.get("pose_key"),
        "motion_id": motion_id_of(e),
        "common": bool(e.get("common")),
        "fingerspelling": bool(e.get("fingerspelling")),
        "proper_noun": bool(e.get("proper_noun")),
        "rank": e.get("rank"),
    }
