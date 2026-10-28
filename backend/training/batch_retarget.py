"""
Chuyển hàng loạt dữ liệu trích từ video (raw_motion/*.json) → backend/data/motions/*.json

    python -m backend.training.batch_retarget <thư_mục_raw | qipedc_motion_raw.json.gz>
"""
import glob
import gzip
import json
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from backend.training.retarget_motion import retarget, fk_check, LAYOUT  # noqa: E402

OUT = os.path.join(ROOT, "backend", "data", "motions")
MIN_HAND_COVERAGE = 0.15  # dưới mức này gần như không thấy tay trong video → bỏ qua (dùng video mẫu)


def iter_raw(src):
    """Đọc từ thư mục (mỗi video 1 file .json) hoặc từ file gói do extract_motion_browser.js tải về."""
    if os.path.isdir(src):
        for f in sorted(glob.glob(os.path.join(src, "*.json"))):
            yield json.load(open(f, encoding="utf-8"))
    else:
        opener = gzip.open if src.endswith(".gz") else open
        with opener(src, "rt", encoding="utf-8") as f:
            data = json.load(f)
        for raw in (data if isinstance(data, list) else [data]):
            yield raw


def main(src_dir):
    os.makedirs(OUT, exist_ok=True)
    stats, skipped = [], []
    for raw in iter_raw(src_dir):
        m = retarget(raw)
        cov = max(m["hand_coverage"].values())
        if cov < MIN_HAND_COVERAGE:
            skipped.append(m["id"])
            continue
        with open(os.path.join(OUT, m["id"] + ".json"), "w", encoding="utf-8") as fo:
            json.dump(m, fo, separators=(",", ":"))
        stats.append((m["id"], cov, fk_check(m, raw)))
    with open(os.path.join(OUT, "_layout.json"), "w", encoding="utf-8") as fo:
        json.dump({"layout_version": 1, "channels": LAYOUT}, fo, indent=1)
    fe = [s[2]["forearm"] for s in stats if s[2]["forearm"] is not None]
    hz = [s[2]["hand_z"] for s in stats if s[2]["hand_z"] is not None]
    print(f"Đã chuyển {len(stats)} ký hiệu, bỏ qua {len(skipped)} (không thấy tay): {skipped[:10]}")
    if fe:
        print(f"Sai lệch hướng cẳng tay TB: {sum(fe) / len(fe):.1f}°, hướng lòng bàn tay TB: {sum(hz) / max(1, len(hz)):.1f}°")
    return stats, skipped


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw_motion")
