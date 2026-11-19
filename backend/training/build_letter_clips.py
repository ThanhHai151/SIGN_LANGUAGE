"""
Đoạn "giữ tay" của từng chữ cái / dấu thanh trong video bảng chữ cái QIPEDC (backend/data/letter_clips.json)
=====================================================================================================
Video chữ cái gồm: đưa tay lên → giữ hình chữ cái (hoặc làm động tác của chữ: Ă, Â, Ơ, Ư, J, Z, dấu thanh)
→ hạ tay. Khi đánh vần một từ, robot chỉ cần đoạn GIỮA (hình chữ cái), không lặp lại đưa tay lên/hạ xuống
cho từng chữ. Script này tìm đoạn đó từ điểm khớp gốc (backend/data/raw_motion):

  • đoạn cổ tay ở gần độ cao lớn nhất và thấy bàn tay phải, liên tục dài nhất trong đoạn có động tác;
  • chữ "tĩnh" (cổ tay đứng yên, ngón tay không đổi) → lấy 0,5 giây giữa đoạn;
  • chữ "động" (có di chuyển / đổi hình tay) → lấy cả đoạn.

Chạy:  python -m backend.training.build_letter_clips
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from backend.training.retarget_motion import LAYOUT, to_robot, torso_frame  # noqa: E402

DATA = os.path.join(ROOT, "backend", "data")
OUT = os.path.join(DATA, "letter_clips.json")

# Chữ cái tiếng Việt + chữ ngoại lai + dấu thanh → mã video QIPEDC
LETTER_VIDEOS = {
    "a": "D0489", "ă": "D0490N", "â": "D0491N", "b": "D0492", "c": "D0493", "d": "D0494", "đ": "D0495",
    "e": "D0496", "ê": "D0497N", "g": "D0498N", "h": "D0500N", "i": "D0501", "k": "D0502", "l": "D0503",
    "m": "D0504", "n": "D0505", "o": "D0506", "ô": "D0507N", "ơ": "D0508N", "p": "D0509N", "q": "D0510",
    "r": "D0511", "s": "D0512", "t": "D0513N", "u": "D0514", "ư": "D0515N", "v": "D0516", "x": "D0517N",
    "y": "D0518N", "f": "D0521", "j": "D0522", "w": "D0523", "z": "D0519",
    "sắc": "D0525N", "huyền": "D0524N", "hỏi": "D0526N", "ngã": "D0527N", "nặng": "D0528",
}
STATIC_HOLD_S = 0.5
# Chữ / dấu mà bản thân là một NÉT VẼ trong không trung (tay đi xuống, lượn...): giữ cả nét, không chỉ đỉnh
STROKE = {"j", "z", "sắc", "huyền", "hỏi", "ngã", "nặng"}
IDX = {k: i for i, k in enumerate(LAYOUT)}
FINGER_COLS = [IDX[k] for k in LAYOUT if k.startswith("right_") and any(f in k for f in ("thumb", "index", "middle", "ring", "pinky"))]


def hold_clip(raw: dict, motion: dict, stroke: bool = False) -> dict:
    fps = float(raw["fps"])
    a0, a1 = motion["active"]
    ys, W = [], []
    for k, fr in enumerate(raw["frames"]):
        t = k / fps
        if fr.get("p") and fr.get("r") and a0 - 1e-6 <= t <= a1 + 1e-6:
            P = to_robot(fr["p"]); T = torso_frame(P); c = (P[1] + P[2]) / 2
            w = T.T @ (P[6] - c)
            ys.append((k, w[1])); W.append((k, w))
    if not ys:
        return {"clip": [a0, a1], "static": False}
    top = max(y for _, y in ys)
    good = {k for k, y in ys if y >= top - (0.15 if stroke else 0.06)}
    runs, cur = [], []
    for k in sorted(good):
        if cur and k != cur[-1] + 1:
            runs.append(cur); cur = []
        cur.append(k)
    runs.append(cur)
    run = max(runs, key=len)
    k0, k1 = run[0], run[-1]
    if k1 - k0 >= 4 and not stroke:          # bỏ 1 khung mỗi đầu (đang dừng lại sau khi đưa tay lên / bắt đầu hạ tay)
        k0, k1 = k0 + 1, k1 - 1
    wpos = np.array([w for k, w in W if k0 <= k <= k1])
    F = np.array(motion["frames"])[k0:k1 + 1][:, FINGER_COLS]
    static = bool(np.ptp(wpos[:, :2], axis=0).max() < 0.03 and (np.ptp(F, axis=0).max() if len(F) > 1 else 0) < 0.45)
    t0, t1 = k0 / fps, k1 / fps
    if static:
        mid = (t0 + t1) / 2
        t0, t1 = max(t0, mid - STATIC_HOLD_S / 2), min(t1, mid + STATIC_HOLD_S / 2)
    return {"clip": [round(t0, 3), round(t1, 3)], "static": static}


def main():
    out = {}
    for ch, vid in LETTER_VIDEOS.items():
        rp = os.path.join(DATA, "raw_motion", vid + ".json")
        mp = os.path.join(DATA, "motions", vid + ".json")
        if not (os.path.exists(rp) and os.path.exists(mp)):
            print("thiếu", ch, vid)
            continue
        with open(rp, encoding="utf-8") as f:
            raw = json.load(f)
        with open(mp, encoding="utf-8") as f:
            motion = json.load(f)
        out[ch] = {"motion_id": vid, **hold_clip(raw, motion, stroke=ch in STROKE)}
        print(f"{ch:6} {vid:8} {out[ch]}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print("Đã ghi", OUT)


if __name__ == "__main__":
    main()
