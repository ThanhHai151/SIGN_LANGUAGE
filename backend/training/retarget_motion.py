"""
Chuyển chuyển động người thật (MediaPipe Holistic) → góc khớp robot RobotSigner3D
====================================================================================

Đầu vào: kết quả trích từ video QIPEDC (xem backend/training/extract_motion_browser.js):
    {"id": "D0002", "fps": 8, "dur": 6.13, "frames": [
        {"p": [9 điểm pose world ×3, mm], "l": [21 điểm bàn tay trái ×3] | null, "r": [...] | null}, ...]}
    p = [mũi, vai T, vai P, khuỷu T, khuỷu P, cổ tay T, cổ tay P, hông T, hông P]
    (T/P = trái/phải CỦA NGƯỜI KÝ HIỆU). Hệ trục MediaPipe world: x → bên trái người, y ↓, z ↑ về camera là âm.

Đầu ra: khung hình góc khớp theo đúng hệ khung xương của robot (Three.js, Euler 'XYZ', robot nhìn +Z,
Y hướng lên, tay phải robot ở −X; mỗi đoạn xương nằm dọc −Y; lòng bàn tay +Z; gập ngón = xoay quanh X):
    rightArm[3], rightForearm[3], rightTwist, rightWrist[3], rightFingers{thumb, index, middle, ring, pinky}, (tương tự left)

Phương pháp:
  1. Đổi hệ trục và đưa về khung thân người (bỏ nghiêng người).
  2. IK cánh tay 2 đoạn: hướng cánh tay trên + mặt phẳng gập khuỷu → xoay vai; góc khuỷu → xoay khuỷu.
  3. Hướng bàn tay (cổ tay→gốc ngón giữa, gốc ngón trỏ→ngón út) → tách swing–twist quanh trục cẳng tay:
     twist → khớp xoay cẳng tay, phần còn lại → cổ tay.
  4. Ngón tay: biểu diễn từng đốt trong khung đốt cha → góc gập (quanh X) và dạng ngón (quanh Z).
  5. Lấp khung thiếu (không thấy tay) bằng nội suy, mở góc liên tục (unwrap) và làm mượt.
"""

import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
from backend.models.rotation import Rotation as Rot  # noqa: E402  (chỉ cần numpy, không cần scipy)
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
DOWN = np.array([0.0, -1.0, 0.0])

# Tư thế nghỉ của robot (khi không thấy tay trong video)
REST = {
    "right": {"arm": [-0.20, 0.05, 0.05], "fore": [-0.35, 0.0, 0.0], "twist": 0.0, "wrist": [0, 0, 0]},
    "left": {"arm": [-0.20, -0.05, -0.05], "fore": [-0.35, 0.0, 0.0], "twist": 0.0, "wrist": [0, 0, 0]},
}
REST_FINGERS = {
    "thumb": {"cmc_x": 0.05, "cmc_y": 0.1, "cmc_z": 0.1, "mcp_x": 0.15, "mcp_y": 0.0, "mcp_z": 0.0, "ip_x": 0.15},
    "index": {"mcp_x": 0.3, "pip_x": 0.3, "dip_x": 0.2, "spread_z": 0.05},
    "middle": {"mcp_x": 0.35, "pip_x": 0.35, "dip_x": 0.2, "spread_z": 0.0},
    "ring": {"mcp_x": 0.4, "pip_x": 0.4, "dip_x": 0.25, "spread_z": -0.05},
    "pinky": {"mcp_x": 0.45, "pip_x": 0.45, "dip_x": 0.3, "spread_z": -0.1},
}
# Thứ tự kênh khi đóng gói 1 khung hình thành mảng phẳng (frontend giải theo đúng thứ tự này)
from backend.models.motion_format import FINGER_CHANNELS, SIDE_CHANNELS, LAYOUT, frames_to_keyframes  # noqa: E402,F401

# Khớp gốc ngón cái của robot (RobotSigner3D: thumbRig.root.rotation = (0.25, -0.35·flip, -0.40·flip))
THUMB_ROOT = {
    "right": Rot.from_euler("XYZ", [0.25, -0.35, -0.40]).as_matrix(),
    "left": Rot.from_euler("XYZ", [0.25, 0.35, 0.40]).as_matrix(),
}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


def to_robot(arr):
    """mm MediaPipe world → mét, hệ trục robot (x: trái người = +X robot, y lên, z về phía camera)."""
    a = np.asarray(arr, dtype=float).reshape(-1, 3) / 1000.0
    return np.stack([a[:, 0], -a[:, 1], -a[:, 2]], axis=1)


def euler(R):
    return Rot.from_matrix(R).as_euler("XYZ")


def rot_between(a, b):
    """Phép quay nhỏ nhất đưa vector a → b."""
    a, b = unit(a), unit(b)
    v = np.cross(a, b)
    c = float(np.dot(a, b))
    if c < -0.9999:
        axis = unit(np.cross(a, [1, 0, 0]) if abs(a[0]) < 0.9 else np.cross(a, [0, 0, 1]))
        return Rot.from_rotvec(axis * math.pi).as_matrix()
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def torso_frame(P):
    lsh, rsh, lhip, rhip = P[1], P[2], P[7], P[8]
    x = unit(lsh - rsh)
    y = unit((lsh + rsh) / 2 - (lhip + rhip) / 2)
    y = unit(y - np.dot(y, x) * x)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)  # cột = trục thân trong hệ world


def arm_ik(S, E, W, prev_bend):
    u = unit(E - S)
    f = unit(W - E)
    bend = f - np.dot(f, u) * u
    if np.linalg.norm(bend) < 0.08:  # tay gần như duỗi thẳng → mặt phẳng gập không xác định
        bend = prev_bend if prev_bend is not None else np.array([0.0, 0.0, 1.0])
        bend = bend - np.dot(bend, u) * u
        if np.linalg.norm(bend) < 1e-6:
            bend = np.array([0.0, 0.0, 1.0]) - u[2] * u
    z = unit(bend)
    y = -u
    x = np.cross(y, z)
    Rs = np.stack([x, y, z], axis=1)
    theta = math.acos(max(-1.0, min(1.0, float(np.dot(u, f)))))
    return Rs, theta, z


def hand_frame(H, side):
    y = unit(H[0] - H[9])
    xr = (H[17] - H[5]) if side == "right" else (H[5] - H[17])
    x = unit(xr - np.dot(xr, y) * y)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def flex_spread(b):
    """Góc gập (quanh X, + về phía lòng bàn tay) và dạng (quanh Z) của một đốt trong khung đốt cha."""
    b = unit(b)
    spread = math.atan2(b[0], math.hypot(b[1], b[2]))
    flex = math.atan2(b[2], -b[1])
    return flex, spread


def bone_angle(a, b):
    return math.acos(max(-1.0, min(1.0, float(np.dot(unit(a), unit(b))))))


def finger_angles(Hl, side):
    """Hl: 21 điểm bàn tay trong khung bàn tay robot."""
    m = -1.0 if side == "left" else 1.0
    out = {}
    for name, (a, b, c, d) in FINGERS.items():
        b1, b2, b3 = Hl[b] - Hl[a], Hl[c] - Hl[b], Hl[d] - Hl[c]
        mcp_x, spread = flex_spread(b1)
        # Khớp PIP/DIP chỉ gập được một chiều → dùng độ lớn góc giữa 2 đốt liền nhau
        pip_x = bone_angle(b1, b2)
        dip_x = bone_angle(b2, b3)
        out[name] = {
            "mcp_x": float(np.clip(mcp_x, -0.4, 1.75)),
            "pip_x": float(np.clip(pip_x, 0.0, 1.95)),
            "dip_x": float(np.clip(dip_x, 0.0, 1.6)),
            "spread_z": float(np.clip(spread, -0.45, 0.45)) * m,
        }
    # Ngón cái: gốc robot có phép quay cố định THUMB_ROOT
    R0 = THUMB_ROOT[side]
    meta, prox, dist = Hl[2] - Hl[1], Hl[3] - Hl[2], Hl[4] - Hl[3]
    Rc = rot_between(DOWN, R0.T @ meta)
    cx, cy, cz = euler(Rc)
    R01 = R0 @ Rc
    Rm = rot_between(DOWN, R01.T @ prox)
    mx, my, mz = euler(Rm)
    R012 = R01 @ Rm
    dl = unit(R012.T @ dist)
    ipx = math.atan2(-dl[2], -dl[1])  # góc quay khớp IP quanh X
    out["thumb"] = {
        "cmc_x": float(cx), "cmc_y": float(cy) * m, "cmc_z": float(cz) * m,
        "mcp_x": float(-mx), "mcp_y": float(my) * m, "mcp_z": float(mz) * m,
        "ip_x": float(np.clip(-ipx, -0.3, 1.4)),
    }
    return out


# MediaPipe ước lượng độ sâu (trục trước–sau) kém hơn nhiều so với 2 trục trên mặt phẳng ảnh,
# nhất là với bàn tay → giảm trọng số độ sâu khi tính HƯỚNG tay để robot nhìn từ phía trước giống video.
KZ_ARM = 0.5
KZ_HAND = 0.35


def damp_z(v, k):
    v = np.array(v, dtype=float)
    v[..., 2] *= k
    return v


def retarget_frame(fr, prev):
    if not fr.get("p"):
        return None
    P = to_robot(fr["p"])
    T = torso_frame(P)
    Tt = T.T
    loc = lambda v: Tt @ (v - (P[1] + P[2]) / 2)  # noqa: E731
    res = {"torso": [0.0, 0.0, 0.0]}
    # Nghiêng thân nhẹ (giới hạn để robot không cúi quá)
    te = euler(T)
    res["torso"] = [float(np.clip(te[0] * 0.6, -0.15, 0.15)), float(np.clip(te[1] * 0.6, -0.2, 0.2)), float(np.clip(te[2] * 0.6, -0.12, 0.12))]
    for side, (si, ei, wi, hk) in {"right": (2, 4, 6, "r"), "left": (1, 3, 5, "l")}.items():
        S, E, W = loc(P[si]), damp_z(loc(P[ei]), KZ_ARM), damp_z(loc(P[wi]), KZ_ARM)
        Rs, theta, bend = arm_ik(S, E, W, (prev or {}).get(side + "_bend"))
        res[side + "_bend"] = bend
        Rfore = Rs @ Rot.from_euler("XYZ", [-theta, 0, 0]).as_matrix()
        entry = {"arm": euler(Rs), "fore": [-theta, 0.0, 0.0], "hand": None}
        if fr.get(hk):
            H = np.stack([loc(h) for h in to_robot(fr[hk])])
            Hrel = H - H[0]
            Rh = hand_frame(damp_z(Hrel, KZ_HAND), side)   # hướng bàn tay: giảm độ sâu
            Rh_shape = hand_frame(Hrel, side)              # hình dạng ngón: dùng 3D đầy đủ
            Rrel = Rfore.T @ Rh
            q = Rot.from_matrix(Rrel).as_quat()  # x, y, z, w
            twist = 2 * math.atan2(q[1], q[3])
            twist = (twist + math.pi) % (2 * math.pi) - math.pi
            Rw = Rot.from_euler("Y", twist).as_matrix().T @ Rrel
            Hl = (Rh_shape.T @ Hrel.T).T
            entry["hand"] = {"twist": twist, "wrist": euler(Rw), "fingers": finger_angles(Hl, side)}
        res[side] = entry
    return res


def flatten_side(entry, side):
    vals = list(entry["arm"]) + list(entry["fore"])
    h = entry.get("hand")
    if h:
        vals += [h["twist"]] + list(h["wrist"])
        F = h["fingers"]
    else:
        vals += [np.nan] * 4
        F = None
    for fname, chans in FINGER_CHANNELS:
        for c in chans:
            vals.append(F[fname][c] if F else np.nan)
    return vals


def retarget(raw, smooth=True):
    frames = []
    prev = None
    for fr in raw["frames"]:
        r = retarget_frame(fr, prev)
        if r:
            prev = r
        frames.append(r)
    n = len(frames)
    M = np.full((n, len(LAYOUT)), np.nan)
    for i, r in enumerate(frames):
        if not r:
            continue
        M[i, :3] = r["torso"]
        M[i, 3:3 + SIDE_CHANNELS] = flatten_side(r["right"], "right")
        M[i, 3 + SIDE_CHANNELS:] = flatten_side(r["left"], "left")

    # Giá trị nghỉ cho phần không thấy tay ở đầu/cuối video
    rest_vec = np.zeros(len(LAYOUT))
    for side, off in (("right", 3), ("left", 3 + SIDE_CHANNELS)):
        R = REST[side]
        rv = list(R["arm"]) + list(R["fore"]) + [R["twist"]] + list(R["wrist"])
        m = -1.0 if side == "left" else 1.0
        for fname, chans in FINGER_CHANNELS:
            for c in chans:
                v = REST_FINGERS[fname][c]
                rv.append(v * (m if c in ("spread_z",) else 1.0))
        rest_vec[off:off + SIDE_CHANNELS] = rv

    t = np.arange(n)
    for j in range(M.shape[1]):
        col = M[:, j]
        ok = ~np.isnan(col)
        if ok.sum() == 0:
            M[:, j] = rest_vec[j]
            continue
        # mở góc liên tục để nội suy/làm mượt không bị nhảy ±π
        col_ok = np.unwrap(col[ok])
        col[ok] = col_ok
        first, last = np.argmax(ok), n - 1 - np.argmax(ok[::-1])
        # nội suy khoảng trống ở giữa, đầu/cuối chuyển dần về tư thế nghỉ
        col = np.interp(t, t[ok], col[ok])
        ramp = 4
        for k in range(first):
            a = min(1.0, (first - k) / ramp)
            col[k] = col[first] * (1 - a) + rest_vec[j] * a
        for k in range(last + 1, n):
            a = min(1.0, (k - last) / ramp)
            col[k] = col[last] * (1 - a) + rest_vec[j] * a
        M[:, j] = col
    if smooth and n >= 3:
        K = np.array([0.25, 0.5, 0.25])
        Mp = np.vstack([M[:1], M, M[-1:]])
        M = K[0] * Mp[:-2] + K[1] * Mp[1:-1] + K[2] * Mp[2:]
    # Đoạn có động tác: từ lúc tay xuất hiện đến lúc tay biến mất (+ đệm), bỏ phần đứng yên đầu/cuối video
    fps = float(raw["fps"])
    # cổ tay rời vị trí nghỉ (vị trí ở các khung đầu video) hơn 7 cm → đang làm động tác
    wr = []
    for fr in raw["frames"]:
        if fr.get("p"):
            P = to_robot(fr["p"])
            T = torso_frame(P)
            c = (P[1] + P[2]) / 2
            wr.append((T.T @ (P[5] - c), T.T @ (P[6] - c)))
        else:
            wr.append(None)
    base = [w for w in wr[:3] if w is not None]
    act = []
    if base:
        bl = np.mean([w[0] for w in base], axis=0)
        br = np.mean([w[1] for w in base], axis=0)
        act = [i for i, w in enumerate(wr) if w is not None and max(np.linalg.norm(w[0] - bl), np.linalg.norm(w[1] - br)) > 0.09]
        # Nhiều video QIPEDC có 2 cách thể hiện liên tiếp ("CÁCH 1", "CÁCH 2") ngăn bởi khoảng nghỉ
        # → chỉ lấy đoạn động tác đầu tiên (đủ dài) để robot diễn gọn, đúng 1 cách như video hiển thị
        segs = []
        for i in act:
            if segs and i - segs[-1][1] <= int(0.6 * fps):
                segs[-1][1] = i
            else:
                segs.append([i, i])
        long_segs = [sg for sg in segs if (sg[1] - sg[0] + 1) / fps >= 0.5]
        segs = long_segs or segs
        if segs:
            act = list(range(segs[0][0], segs[0][1] + 1))
    dur = float(raw.get("dur") or n / fps)
    if act:
        active = [max(0.0, act[0] / fps - 0.15), min(dur, (act[-1] + 1) / fps + 0.15)]
    else:
        active = [0.0, dur]
    hand_cov = {
        "right": float(np.mean([1.0 if (f and f["right"].get("hand")) else 0.0 for f in frames])),
        "left": float(np.mean([1.0 if (f and f["left"].get("hand")) else 0.0 for f in frames])),
    }
    return {
        "id": raw["id"],
        "fps": raw["fps"],
        "dur": raw["dur"],
        "layout_version": 1,
        "active": [round(active[0], 3), round(active[1], 3)],
        "hand_coverage": hand_cov,
        "frames": [[round(float(v), 3) for v in row] for row in M],
    }


def fk_check(motion, raw):
    """Kiểm tra động học thuận: hướng cẳng tay & bàn tay của robot so với video (độ lệch góc trung bình, độ)."""
    errs = {"forearm": [], "hand_y": [], "hand_z": []}
    idx = {k: i for i, k in enumerate(LAYOUT)}
    for row, fr in zip(motion["frames"], raw["frames"]):
        r = retarget_frame(fr, None)
        if not r:
            continue
        P = to_robot(fr["p"])
        T = torso_frame(P)
        for side, (ei, wi, hk) in {"right": (4, 6, "r"), "left": (3, 5, "l")}.items():
            g = lambda k: row[idx[f"{side}_{k}"]]  # noqa: E731
            Rs = Rot.from_euler("XYZ", [g("arm_x"), g("arm_y"), g("arm_z")]).as_matrix()
            Rf = Rs @ Rot.from_euler("XYZ", [g("fore_x"), g("fore_y"), g("fore_z")]).as_matrix()
            fwd = Rf @ DOWN
            true_f = unit(T.T @ (P[wi] - P[ei]))
            errs["forearm"].append(math.degrees(math.acos(np.clip(np.dot(fwd, true_f), -1, 1))))
            if fr.get(hk):
                Rh = Rf @ Rot.from_euler("Y", g("twist")).as_matrix() @ Rot.from_euler("XYZ", [g("wrist_x"), g("wrist_y"), g("wrist_z")]).as_matrix()
                H = np.stack([T.T @ h for h in to_robot(fr[hk])])
                Rt = hand_frame(H, side)
                errs["hand_y"].append(math.degrees(math.acos(np.clip(np.dot(Rh[:, 1], Rt[:, 1]), -1, 1))))
                errs["hand_z"].append(math.degrees(math.acos(np.clip(np.dot(Rh[:, 2], Rt[:, 2]), -1, 1))))
    return {k: (round(float(np.mean(v)), 1) if v else None) for k, v in errs.items()}


if __name__ == "__main__":
    src = sys.argv[1]
    raw = json.load(open(src, encoding="utf-8"))
    m = retarget(raw)
    print(json.dumps({"id": m["id"], "frames": len(m["frames"]), "hand_coverage": m["hand_coverage"], "fk": fk_check(m, raw)}, ensure_ascii=False))
