"""
Kiểm định chất lượng chuyển động robot so với video gốc (toàn bộ ký hiệu đã học)
=============================================================================
Với mỗi ký hiệu, so chuyển động robot (sau khi chuyển đổi + làm mượt) với điểm khớp người thật trích
từ video, CHỈ trong đoạn có động tác (đoạn robot thực sự diễn):

  • wrist_p90_cm : sai lệch vị trí cổ tay trên mặt phẳng nhìn thẳng (bách phân vị 90, cm)
  • hand_p90_deg : sai lệch hướng bàn tay (trục ngón giữa và pháp tuyến lòng bàn tay, bách phân vị 90, độ)
  • finger_p90   : sai lệch độ cong ngón (0 = duỗi, 1 = nắm) do làm mượt
  • coverage     : tỉ lệ khung MediaPipe nhận ra bàn tay của (các) tay đang làm ký hiệu
  • max_gap_s    : khoảng dài nhất liên tiếp không nhận ra bàn tay (robot phải nội suy hình dạng tay)

Chạy:  python -m backend.training.audit_motions            → in tổng hợp, ghi backend/data/motion_quality.json
"""
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from backend.models.rotation import Rotation as Rot  # noqa: E402
from backend.training.retarget_motion import (  # noqa: E402
    retarget, body_scale, to_robot, torso_frame, to_robot_pos, hand_frame_view, damp_z,
    LAYOUT, DOWN, R_SHOULDER_HALF, R_UPPER, R_FORE, R_REACH, KZ_HAND, E_STRAIGHT, E_FIST, FINGERS,
)

DATA = os.path.join(ROOT, "backend", "data")
RAW_DIR = os.path.join(DATA, "raw_motion")
OUT = os.path.join(DATA, "motion_quality.json")
IDX = {k: i for i, k in enumerate(LAYOUT)}

# Ngưỡng xếp loại
LIMITS = {
    "wrist_p90_cm": 4.0,     # cổ tay lệch quá 4 cm ở 10% khung xấu nhất
    "hand_p90_deg": 35.0,    # hướng bàn tay lệch quá 35°
    "finger_p90": 0.35,      # hình dạng ngón lệch quá 1/3 quãng duỗi–nắm
    "coverage": 0.6,         # nhận ra bàn tay dưới 60% số khung
    "max_gap_s": 0.5,        # mất bàn tay liên tục quá 0,5 giây
}


def _curls(H):
    """Độ cong 4 ngón (trỏ..út) từ 21 điểm bàn tay."""
    out = []
    for a, b, c, d in FINGERS.values():
        seg = np.linalg.norm(H[b] - H[a]) + np.linalg.norm(H[c] - H[b]) + np.linalg.norm(H[d] - H[c])
        e = float(np.linalg.norm(H[d] - H[a]) / seg) if seg > 1e-6 else 1.0
        out.append(float(np.clip((E_STRAIGHT - e) / (E_STRAIGHT - E_FIST), 0.0, 1.0)))
    return out


def audit_one(raw: dict, motion: dict = None) -> dict:
    m = motion or retarget(raw)
    fps = float(raw["fps"])
    n = len(raw["frames"])
    a0, a1 = m["active"]
    lo, hi = max(0, int(math.floor(a0 * fps))), min(n, int(math.ceil(a1 * fps)) + 1)
    sc = body_scale(raw)

    # vị trí cổ tay người (khung thân) để biết tay nào đang làm ký hiệu
    wr = {"right": [], "left": []}
    for fr in raw["frames"]:
        if fr.get("p"):
            P = to_robot(fr["p"]); T = torso_frame(P); c = (P[1] + P[2]) / 2
            wr["right"].append(T.T @ (P[6] - c)); wr["left"].append(T.T @ (P[5] - c))
        else:
            wr["right"].append(None); wr["left"].append(None)

    res = {"id": raw["id"], "frames": n, "active_s": round(a1 - a0, 2), "sides": {}}
    werr, herr, ferr = [], [], []
    cov_list, gap_list, used_sides = [], [], []
    for side, (wi, hk) in {"right": (6, "r"), "left": (5, "l")}.items():
        base = [w for w in wr[side][:3] if w is not None]
        act_idx = [i for i in range(lo, hi) if wr[side][i] is not None]
        if not base or not act_idx:
            continue
        b = np.mean(base, axis=0)
        # độ dời trên mặt phẳng nhìn thẳng (độ sâu của MediaPipe quá nhiễu để dùng ở đây)
        # Tay đang làm ký hiệu = cổ tay được NÂNG lên khỏi vị trí buông thõng (> 6 cm) hoặc dời xa (> 18 cm).
        # Tay buông thõng chỉ trôi ngang ~10 cm theo thân người (sai số ước lượng tư thế) không tính.
        moved = [(wr[side][i] - b)[1] > 0.06 or np.linalg.norm((wr[side][i] - b)[:2]) > 0.18 for i in act_idx]
        has_hand = [bool(raw["frames"][i].get(hk)) for i in act_idx]
        # tay đang làm ký hiệu: cổ tay rời vị trí nghỉ ở ≥ 25% số khung của đoạn động tác
        used = float(np.mean(moved)) >= 0.25
        if not used:
            continue
        used_sides.append(side)
        use_idx = [i for i, mv in zip(act_idx, moved) if mv]
        det = [bool(raw["frames"][i].get(hk)) for i in use_idx]
        cov = float(np.mean(det))
        # khoảng mất bàn tay Ở GIỮA động tác (giữa lần thấy đầu và lần thấy cuối): robot phải nội suy hình dạng tay.
        # Mất ở đầu/cuối (lúc tay đang đưa lên / hạ xuống) không ảnh hưởng tới ký hiệu.
        gap = cur = 0
        if any(det):
            f0, f1 = det.index(True), len(det) - 1 - det[::-1].index(True)
            for d in det[f0:f1 + 1]:
                cur = 0 if d else cur + 1
                gap = max(gap, cur)
        else:
            gap = len(det)
        cov_list.append(cov); gap_list.append(gap / fps)
        sgn = -1.0 if side == "right" else 1.0
        clipped = 0
        for i in use_idx:
            fr = raw["frames"][i]; row = m["frames"][i]
            g = lambda k: row[IDX[f"{side}_{k}"]]  # noqa: E731
            P = to_robot(fr["p"]); T = torso_frame(P); c = (P[1] + P[2]) / 2
            Rs = Rot.from_euler("XYZ", [g("arm_x"), g("arm_y"), g("arm_z")]).as_matrix()
            Rf = Rs @ Rot.from_euler("XYZ", [g("fore_x"), g("fore_y"), g("fore_z")]).as_matrix()
            S = np.array([sgn * R_SHOULDER_HALF, 0, 0])
            w_robot = S + Rs @ (DOWN * R_UPPER) + Rf @ (DOWN * R_FORE)
            tgt = to_robot_pos(T.T @ (P[wi] - c), sc)
            if np.linalg.norm(tgt - S) > R_REACH + 0.01:
                clipped += 1
            werr.append(100 * float(np.linalg.norm((w_robot - tgt)[:2])))
            if fr.get(hk):
                Rh = Rf @ Rot.from_euler("Y", g("twist")).as_matrix() @ Rot.from_euler("XYZ", [g("wrist_x"), g("wrist_y"), g("wrist_z")]).as_matrix()
                H = np.stack([T.T @ h for h in to_robot(fr[hk])])
                Hrel = H - H[0]
                Rt = hand_frame_view(Hrel, side)
                ey = math.degrees(math.acos(np.clip(np.dot(Rh[:, 1], Rt[:, 1]), -1, 1)))
                ez = math.degrees(math.acos(np.clip(np.dot(Rh[:, 2], Rt[:, 2]), -1, 1)))
                herr.append(max(ey, ez))
                target = _curls(Hrel)
                for fname, tc in zip(FINGERS, target):
                    ferr.append(abs(g(f"{fname}_pip_x") / 1.75 - tc))
        res["sides"][side] = {"coverage": round(cov, 2), "max_gap_s": round(gap / fps, 2),
                              "clipped": round(clipped / max(1, len(use_idx)), 2)}
    p90 = lambda v: round(float(np.percentile(v, 90)), 2) if v else None  # noqa: E731
    res.update({
        "used": used_sides,
        "wrist_p90_cm": p90(werr),
        "hand_p90_deg": p90(herr),
        "finger_p90": p90(ferr),
        "coverage": round(min(cov_list), 2) if cov_list else 0.0,
        "max_gap_s": round(max(gap_list), 2) if gap_list else None,
    })
    flags = []
    if not used_sides:
        flags.append("no_active_hand")
    else:
        for k, lim in LIMITS.items():
            v = res.get(k)
            if v is None:
                continue
            if (k == "coverage" and v < lim) or (k != "coverage" and v > lim):
                flags.append(k)
    res["flags"] = flags
    return res


def _job(path):
    try:
        with open(path, encoding="utf-8") as f:
            raw = json.load(f)
        return audit_one(raw)
    except Exception as e:  # noqa: BLE001
        return {"id": os.path.basename(path)[:-5], "error": str(e), "flags": ["error"]}


def run(ids=None, workers=None):
    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".json"))
    if ids is not None:
        files = [f for f in files if f[:-5] in ids]
    paths = [os.path.join(RAW_DIR, f) for f in files]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_job, paths, chunksize=16))


BLOCK_FLAGS = ("coverage", "max_gap_s")   # mất dấu bàn tay giữa động tác → hình dạng tay không đáng tin


def main():
    motion_dir = os.path.join(DATA, "motions")
    ids = {f[:-5] for f in os.listdir(motion_dir) if f.endswith(".json") and not f.startswith("_")}
    rows = run(ids=ids)
    ok = [r for r in rows if not r["flags"]]
    blocked = sorted(r["id"] for r in rows if set(r["flags"]) & set(BLOCK_FLAGS) or "error" in r["flags"])
    print(f"Ký hiệu đã kiểm: {len(rows)} | đạt mọi tiêu chí: {len(ok)} ({100 * len(ok) / max(1, len(rows)):.1f}%)")
    for k in ["no_active_hand", *LIMITS, "error"]:
        c = sum(k in r["flags"] for r in rows)
        if c:
            print(f"  {k:<16}{c}")
    summary = {"checked": len(rows), "passed": len(ok), "blocked": len(blocked)}
    for k in ("wrist_p90_cm", "hand_p90_deg", "finger_p90", "coverage"):
        v = [r[k] for r in rows if r.get(k) is not None]
        summary["median_" + k] = round(float(np.median(v)), 2)
        print(f"  trung vị {k}: {np.median(v):.2f}")
    print(f"Không cho robot diễn (mất dấu bàn tay giữa động tác): {len(blocked)} → {blocked}")
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump({"limits": LIMITS, "summary": summary, "blocked": blocked,
                   "flagged": {r["id"]: r for r in rows if r["flags"]}}, f, ensure_ascii=False, indent=1)
    print("Đã ghi", OUT)


if __name__ == "__main__":
    main()
