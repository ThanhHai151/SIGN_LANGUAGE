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


def hand_frame_view(H, side, kz=None):
    """Hướng bàn tay để robot NHÌN TỪ PHÍA TRƯỚC giống video:
    - hướng ngón tay (trục y) lấy từ điểm đã giảm độ sâu (MediaPipe phóng đại trước–sau → ngón tay chĩa
      thẳng vào camera thay vì sang ngang như trong video);
    - hướng LÒNG BÀN TAY lấy từ hình bàn tay 3D đầy đủ rồi chiếu vuông góc với trục ngón. Giảm độ sâu cả
      bàn tay làm lòng bàn tay bị lật ngược (úp thành ngửa) khi ngón tay chĩa về phía camera."""
    kz = KZ_HAND if kz is None else kz
    y = unit(damp_z(H[0] - H[9], kz))
    xr = (H[17] - H[5]) if side == "right" else (H[5] - H[17])
    n = np.cross(unit(xr), unit(H[0] - H[9]))       # pháp tuyến lòng bàn tay 3D
    n = n - np.dot(n, y) * y
    if np.linalg.norm(n) < 1e-6:
        return hand_frame(damp_z(H, kz), side)
    z = unit(n)
    x = np.cross(y, z)
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
    """Hl: 21 điểm bàn tay trong khung bàn tay robot.

    Góc giữa từng cặp đốt ngón tính trực tiếp từ điểm MediaPipe rất nhiễu (một bàn tay duỗi thẳng vẫn
    cho PIP/DIP ≈ 1–1,5 rad vì sai số độ sâu của từng khớp nhỏ). Vì vậy dùng hai đại lượng bền hơn:
      • độ duỗi e = |đầu ngón − gốc ngón| / tổng chiều dài 3 đốt (≈1 khi duỗi, ≈0,4 khi nắm) → độ cong c
        → PIP, DIP gập theo cùng tỉ lệ như ngón tay thật (gập liên động);
      • hướng dây cung gốc→đầu ngón trong khung bàn tay → góc gập MCP và độ dạng ngón.
    """
    m = -1.0 if side == "left" else 1.0
    out = {}
    for name, (a, b, c, d) in FINGERS.items():
        seg = np.linalg.norm(Hl[b] - Hl[a]) + np.linalg.norm(Hl[c] - Hl[b]) + np.linalg.norm(Hl[d] - Hl[c])
        e = float(np.linalg.norm(Hl[d] - Hl[a]) / seg) if seg > 1e-6 else 1.0
        curl = float(np.clip((E_STRAIGHT - e) / (E_STRAIGHT - E_FIST), 0.0, 1.0))
        pip_x = 1.75 * curl
        dip_x = 1.15 * curl
        chord_flex, chord_spread = flex_spread(Hl[d] - Hl[a])
        # dây cung của ngón cong lệch về phía lòng bàn tay ≈ nửa tổng góc PIP+DIP → phần còn lại là MCP
        mcp_x = chord_flex - 0.5 * (pip_x + dip_x) * 0.85
        out[name] = {
            "mcp_x": float(np.clip(mcp_x, -0.25, 1.6)),
            "pip_x": float(pip_x),
            "dip_x": float(dip_x),
            "spread_z": float(np.clip(chord_spread, -0.35, 0.35)) * m,
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


# Tỉ lệ ba đốt và chiều quay từ RobotSigner3D.jsx.
FINGER_LENGTHS = np.array([.038, .026, .021])
THUMB_LENGTHS = np.array([.024, .022, .019])


def finger_vectors(name, angles, side):
    """Tính hướng từng đốt và dây cung chuẩn hóa trong hệ bàn tay robot."""
    mirror = -1 if side == "left" else 1
    if name == "thumb":
        cx, cy, cz, mx, my, mz, ip = angles
        root = THUMB_ROOT[side] @ Rot.from_euler("XYZ", [cx, mirror * cy, mirror * cz]).as_matrix()
        middle = root @ Rot.from_euler("XYZ", [-mx, mirror * my, mirror * mz]).as_matrix()
        end = middle @ Rot.from_euler("X", -ip).as_matrix()
        lengths = THUMB_LENGTHS
    else:
        mcp, pip, dip, spread = angles
        root = Rot.from_euler("XYZ", [-mcp, 0, mirror * spread]).as_matrix()
        middle = root @ Rot.from_euler("X", -pip).as_matrix()
        end = middle @ Rot.from_euler("X", -dip).as_matrix()
        lengths = FINGER_LENGTHS
    directions = np.stack([root @ DOWN, middle @ DOWN, end @ DOWN])
    return directions, (lengths @ directions) / lengths.sum()


def refine_fingers(landmarks, side, estimate):
    """Fit có giới hạn, giữ nghiệm ban đầu khi các điểm đốt suy biến."""
    for name, indices in {**FINGERS, "thumb": (1, 2, 3, 4)}.items():
        segments = np.diff(landmarks[list(indices)], axis=0)
        lengths = np.linalg.norm(segments, axis=1)
        if not np.all(lengths > 1e-6):
            continue
        target_directions = segments / lengths[:, None]
        target_chord = segments.sum(axis=0) / lengths.sum()
        channels = estimate[name]
        seed = np.array(list(channels.values()))
        bounds = (
            [(-np.pi, np.pi)] * 6 + [(-.3, 1.4)]
            if name == "thumb" else [(-.25, 1.6), (0, 1.75), (0, 1.15), (-.35, .35)]
        )

        def cost(angles):
            directions, chord = finger_vectors(name, angles, side)
            errors = np.linalg.norm(directions - target_directions, axis=1)
            # Huber: ảnh hưởng tăng tuyến tính khi một đốt bị nhiễu mạnh.
            direction_cost = np.where(errors < .25, errors ** 2, .5 * errors - .0625)
            return direction_cost.mean() + 2 * np.sum((chord - target_chord) ** 2) + .045 * np.sum((angles - seed) ** 2)

        best, best_cost = seed.copy(), cost(seed)
        for step in (.24, .12, .06, .03):
            for axis, (low, high) in enumerate(bounds):
                center = best.copy()
                for sign in (-1, 1):
                    candidate = center.copy()
                    candidate[axis] = np.clip(center[axis] + sign * step, low, high)
                    candidate_cost = cost(candidate)
                    if candidate_cost < best_cost:
                        best, best_cost = candidate, candidate_cost
        estimate[name] = dict(zip(channels, map(float, best)))
    return estimate


# MediaPipe ước lượng độ sâu (trục trước–sau) kém hơn nhiều so với 2 trục trên mặt phẳng ảnh,
# nhất là với bàn tay → giảm trọng số độ sâu khi tính HƯỚNG tay để robot nhìn từ phía trước giống video.
KZ_ARM = 0.5
KZ_HAND = 0.35
KZ_POS = 0.6          # độ sâu khi đặt VỊ TRÍ cổ tay (MediaPipe phóng đại trước–sau)
E_STRAIGHT, E_FIST = 0.95, 0.45

# Kích thước robot (RobotSigner3D.jsx), toạ độ so với trung điểm hai vai, Y lên, Z ra phía trước
R_SHOULDER_HALF, R_UPPER, R_FORE = 0.20, 0.30, 0.29
R_REACH = R_UPPER + R_FORE - 0.005
R_TORSO = 0.50        # vai → đáy thân robot
R_FACE = 0.215        # vai → tầm mắt robot
PALM_OFFSET = 0.058   # cổ tay → tâm lòng bàn tay (đã nhân tỉ lệ bàn tay 1,3)
HAND_MARGIN = 0.045   # nửa bề dày bàn tay + khe hở
# Hộp va chạm: (xmin, xmax, ymin, ymax, zmin, zmax)
TORSO_BOX = (-0.22, 0.22, -0.50, 0.02, -0.12, 0.12)
HEAD_BOX = (-0.15, 0.15, 0.08, 0.32, -0.12, 0.12)


def damp_z(v, k):
    v = np.array(v, dtype=float)
    v[..., 2] *= k
    return v


def _push_out(p, box, margin, front_only):
    """Độ dời nhỏ nhất đưa điểm p ra khỏi hộp (đã nới thêm margin). front_only: chỉ đẩy ra phía trước."""
    x0, x1, y0, y1, z0, z1 = box
    x0, x1, y0, y1, z0, z1 = x0 - margin, x1 + margin, y0 - margin, y1 + margin, z0 - margin, z1 + margin
    if not (x0 < p[0] < x1 and y0 < p[1] < y1 and z0 < p[2] < z1):
        return np.zeros(3)
    cands = [np.array([0.0, 0.0, z1 - p[2]])]
    if not front_only:
        cands += [np.array([x0 - p[0], 0.0, 0.0]), np.array([x1 - p[0], 0.0, 0.0])]
    return min(cands, key=lambda v: float(np.linalg.norm(v)))


def _avoid_body(W, palm_dir):
    """Dời vị trí cổ tay sao cho cổ tay và lòng bàn tay không lọt vào thân / đầu robot."""
    for _ in range(3):
        moved = False
        for pt in (W, W + palm_dir * PALM_OFFSET):
            for box, front in ((TORSO_BOX, True), (HEAD_BOX, False)):
                d = _push_out(pt, box, HAND_MARGIN, front)
                if np.any(d):
                    W = W + d
                    moved = True
        if not moved:
            break
    return W


def _solve_arm(S, Wt, pole, prev_bend):
    """IK hai đoạn của robot: vai S, đích cổ tay Wt, hướng khuỷu ưu tiên pole → (R vai, góc khuỷu, mặt gập)."""
    d = Wt - S
    D = float(np.linalg.norm(d))
    if D > R_REACH:
        d *= R_REACH / D
        D = R_REACH
    D = max(D, abs(R_UPPER - R_FORE) + 0.02)
    dh = unit(d)
    pole = pole - np.dot(pole, dh) * dh
    if np.linalg.norm(pole) < 1e-4:
        pole = np.array([0.0, -1.0, -0.3])
        pole = pole - np.dot(pole, dh) * dh
    ph = unit(pole)
    ca = (R_UPPER ** 2 + D ** 2 - R_FORE ** 2) / (2 * R_UPPER * D)
    ang = math.acos(max(-1.0, min(1.0, ca)))
    E = S + R_UPPER * (math.cos(ang) * dh + math.sin(ang) * ph)
    return arm_ik(S, E, S + dh * D, prev_bend)


def body_scale(raw):
    """Tỉ lệ quy đổi vị trí người → robot theo từng trục, lấy trung vị qua các khung.

    x: theo bề rộng vai; y dưới vai: theo chiều dài thân (vai → hông); y trên vai: theo độ cao mặt
    (vai → mũi, robot: vai → mắt) để tay chạm mặt/đầu đúng chỗ; z: như trục y dưới vai × KZ_POS."""
    half, torso, nose = [], [], []
    for fr in raw["frames"]:
        if fr.get("p"):
            P = to_robot(fr["p"])
            T = torso_frame(P)
            c = (P[1] + P[2]) / 2
            half.append(np.linalg.norm(P[1] - P[2]) / 2)
            torso.append(np.linalg.norm(c - (P[7] + P[8]) / 2))
            nose.append(float((T.T @ (P[0] - c))[1]))
    med = lambda v, d: float(np.median(v)) if v else d  # noqa: E731
    sx = R_SHOULDER_HALF / max(med(half, 0.17), 0.08)
    sy = R_TORSO / max(med(torso, 0.48), 0.2)
    sup = R_FACE / max(med(nose, 0.14), 0.05)
    return {"x": sx, "y": sy, "up": sup, "z": sy * KZ_POS}


def to_robot_pos(v, sc):
    """Vị trí trong khung thân người (so với trung điểm hai vai) → vị trí tương ứng trên robot."""
    return np.array([v[0] * sc["x"], v[1] * (sc["up"] if v[1] > 0 else sc["y"]), v[2] * sc["z"]])


# Làm mượt NHẸ ở đầu vào (vị trí cổ tay, hướng bàn tay) thay vì làm mượt góc khớp ở đầu ra:
# trung bình các góc Euler làm bàn tay lệch hướng (góc Euler nhảy giữa hai cách biểu diễn tương đương),
# còn làm mượt vị trí / quaternion thì robot vẫn tái hiện chính xác tư thế đã làm mượt.
POS_D0 = 0.02               # cổ tay dịch < 2 cm giữa hai khung → coi như rung, lọc mạnh
ROT_D0 = math.radians(12)   # bàn tay xoay < 12° giữa hai khung → coi như rung, lọc mạnh


def _quat(R):
    return np.asarray(Rot.from_matrix(R).as_quat(), dtype=float)  # x, y, z, w


def _qmat(q):
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _slerp(q0, q1, u):
    d = float(np.dot(q0, q1))
    if d < 0:
        q1, d = -q1, -d
    if d > 0.9995:
        q = q0 + u * (q1 - q0)
        return q / np.linalg.norm(q)
    th = math.acos(min(1.0, d))
    return (math.sin((1 - u) * th) * q0 + math.sin(u * th) * q1) / math.sin(th)


def _smooth3(A, K):
    """Lọc 3 điểm theo thời gian cho mảng (n, d), lặp lại mép."""
    Ap = np.vstack([A[:1], A, A[-1:]])
    return K[0] * Ap[:-2] + K[1] * Ap[1:-1] + K[2] * Ap[2:]


def _smooth_adaptive(A, d0, side=0.2, angular=False):
    """Lọc 3 điểm THÍCH NGHI: khi gần như đứng yên (thay đổi giữa hai khung ≪ d0) lọc mạnh để khử rung của
    MediaPipe; khi đang chuyển động nhanh (thay đổi ≫ d0) gần như giữ nguyên để không làm trễ / thu nhỏ
    động tác. A: (n, d) vị trí (mét) hoặc quaternion (angular=True, d0 tính bằng radian)."""
    n = len(A)
    if n < 3:
        return A
    out = A.copy()
    for i in range(n):
        a, b = A[max(0, i - 1)], A[min(n - 1, i + 1)]
        if angular:
            if np.dot(a, A[i]) < 0:
                a = -a
            if np.dot(b, A[i]) < 0:
                b = -b
            d = max(2 * math.acos(min(1.0, abs(float(np.dot(a, A[i]))))), 2 * math.acos(min(1.0, abs(float(np.dot(b, A[i]))))))
        else:
            d = max(float(np.linalg.norm(a - A[i])), float(np.linalg.norm(b - A[i])))
        w = side / (1.0 + (d / d0) ** 2)
        v = w * a + (1 - 2 * w) * A[i] + w * b
        out[i] = v / np.linalg.norm(v) if angular else v
    return out


SYM_MAX_CM = 7.0        # hai cổ tay đối xứng qua giữa thân (sai khác trung vị ≤ 7 cm)
SYM_JITTER_RATIO = 2.5  # một tay rung/lật hướng nhiều hơn hẳn tay kia
SYM_JITTER_MIN = 45.0   # ... và thực sự lật hướng liên tục (độ lệch trung vị giữa hai khung > 45°)
SYM_AGREE_MIN = 0.3     # ở ≥ 30% số khung, tay rung vẫn trùng hướng với ảnh gương của tay kia (đúng là đối xứng)


def _orient_jitter(frames, hk, side):
    """Độ lệch hướng trung vị giữa hai khung liên tiếp (độ) của một bàn tay; None nếu quá ít khung."""
    qs = []
    for fr in frames:
        if fr.get("p") and fr.get(hk):
            P = to_robot(fr["p"]); T = torso_frame(P); c = (P[1] + P[2]) / 2
            H = np.stack([T.T @ (h - c) for h in to_robot(fr[hk])])
            qs.append(_quat(hand_frame_view(H - H[0], side)))
        else:
            qs.append(None)
    d = [math.degrees(2 * math.acos(min(1.0, abs(float(np.dot(a, b)))))) for a, b in zip(qs, qs[1:]) if a is not None and b is not None]
    return float(np.median(d)) if len(d) >= 4 else None


# Ký hiệu đã ĐỐI CHIẾU VIDEO là hai bàn tay soi gương nhau (hình dạng + hướng). Tự động nhận dạng không đủ tin cậy:
# nhiều ký hiệu hai tay đặt đối xứng nhưng hình bàn tay khác nhau (một tay dựng đứng, tay kia ngửa) — soi gương
# sẽ làm sai. Chỉ thêm mã vào đây sau khi đã xem video và thấy robot diễn đúng hơn.
MIRROR_SIGNS = {
    "W00110N",  # bạn — hai bàn tay úp, các ngón chéo nhau trước ngực
}


def symmetric_fix(frames, sign_id=None):
    """Ký hiệu HAI TAY ĐỐI XỨNG (vd. "bạn"): khi hai bàn tay chồng lên nhau, MediaPipe hay đoán sai hướng
    của một bàn tay (lật úp/ngửa liên tục) trong khi tay kia ổn định. Nếu hai cổ tay chuyển động đối xứng
    qua giữa thân và một tay rung nhiều hơn hẳn → dùng hình bàn tay đối xứng (soi gương) của tay ổn định cho
    tay kia, giữ nguyên vị trí cổ tay của mỗi tay."""
    if sign_id not in MIRROR_SIGNS:
        return frames
    both = [fr for fr in frames if fr.get("p") and fr.get("r") and fr.get("l")]
    if len(both) < 4:
        return frames
    dev = []
    for fr in both:
        P = to_robot(fr["p"]); T = torso_frame(P); c = (P[1] + P[2]) / 2
        wr, wl = T.T @ (P[6] - c), T.T @ (P[5] - c)
        dev.append(100 * float(np.linalg.norm(wr - wl * np.array([-1.0, 1.0, 1.0]))))
    if float(np.median(dev)) > SYM_MAX_CM:
        return frames
    jr, jl = _orient_jitter(frames, "r", "right"), _orient_jitter(frames, "l", "left")
    if jr is None or jl is None:
        return frames
    if jr > SYM_JITTER_RATIO * jl and jr > SYM_JITTER_MIN:
        bad, good, bs = "r", "l", "right"
    elif jl > SYM_JITTER_RATIO * jr and jl > SYM_JITTER_MIN:
        bad, good, bs = "l", "r", "left"
    else:
        return frames
    agree = []
    for fr in both:
        P = to_robot(fr["p"]); T = torso_frame(P); c = (P[1] + P[2]) / 2
        Hb = np.stack([T.T @ (h - c) for h in to_robot(fr[bad])])
        Hg = np.stack([T.T @ (h - c) for h in to_robot(fr[good])])
        Rb = hand_frame_view(Hb - Hb[0], bs)
        Rm = hand_frame_view((Hg - Hg[0]) * np.array([-1.0, 1.0, 1.0]), bs)
        a = max(math.degrees(math.acos(np.clip(Rb[:, 1] @ Rm[:, 1], -1, 1))), math.degrees(math.acos(np.clip(Rb[:, 2] @ Rm[:, 2], -1, 1))))
        agree.append(a < 45)
    if float(np.mean(agree)) < SYM_AGREE_MIN:
        return frames
    out = []
    for fr in frames:
        if fr.get(good) and fr.get(bad):
            fr = dict(fr)
            G = np.asarray(fr[good], dtype=float).reshape(21, 3)
            B = np.asarray(fr[bad], dtype=float).reshape(21, 3)
            M = (G - G[0]) * np.array([-1.0, 1.0, 1.0]) + B[0]   # soi gương qua mặt phẳng giữa thân (trục x)
            fr[bad] = [int(round(v)) for v in M.reshape(-1)]
            fr["sym"] = good
        out.append(fr)
    return out


def _medoid_quats(Q, half=2):
    """Lọc trung vị theo thời gian cho hướng bàn tay (quaternion): tại mỗi khung chọn hướng nằm "giữa" nhất
    trong cửa sổ ±half khung (tổng góc lệch tới các khung khác nhỏ nhất). Loại các khung MediaPipe lật ngược
    hướng bàn tay trong chốc lát (hay gặp khi hai bàn tay chồng lên nhau) mà vẫn giữ nguyên chuyển động xoay
    đều (trung vị của dãy đơn điệu là phần tử giữa)."""
    n = len(Q)
    out = Q.copy()
    ang = lambda a, b: 2 * math.acos(min(1.0, abs(float(np.dot(a, b)))))  # noqa: E731
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        best = min(range(lo, hi), key=lambda j: sum(ang(Q[j], Q[k]) for k in range(lo, hi)))
        q = Q[best]
        out[i] = q if np.dot(q, Q[i]) >= 0 else -q
    return out


def _fill(A):
    """Nội suy tuyến tính các hàng NaN của mảng (n, d); None nếu không có hàng hợp lệ nào."""
    ok = ~np.isnan(A[:, 0])
    if not ok.any():
        return None
    t = np.arange(len(A))
    return np.stack([np.interp(t, t[ok], A[ok, j]) for j in range(A.shape[1])], axis=1)


def _near_euler(e, prev):
    """Trong các bộ góc Euler XYZ biểu diễn cùng một phép quay, chọn bộ gần khung trước nhất
    (frontend nội suy từng góc giữa hai khung → góc phải liên tục)."""
    e = np.asarray(e, dtype=float)
    if prev is None:
        return e
    best = None
    for c in (e, np.array([e[0] + math.pi, math.pi - e[1], e[2] + math.pi])):
        c = prev + (c - prev + math.pi) % (2 * math.pi) - math.pi
        d = float(np.abs(c - prev).sum())
        if best is None or d < best[0]:
            best = (d, c)
    return best[1]


def retarget(raw, smooth=True, *, fit_fingers=False):
    frames_in = raw["frames"]
    n = len(frames_in)
    scale = body_scale(raw)
    nan3 = lambda: np.full((n, 3), np.nan)  # noqa: E731
    torso = nan3()
    sides = {"right": (2, 4, 6, "r"), "left": (1, 3, 5, "l")}
    G = {s: {"W": nan3(), "pole": nan3(), "fdir": nan3(), "Q": np.full((n, 4), np.nan), "F": [None] * n} for s in sides}

    frames_in = symmetric_fix(frames_in, raw.get("id")) if smooth else frames_in

    # 1) Hình học từng khung trong khung thân người
    for i, fr in enumerate(frames_in):
        if not fr.get("p"):
            continue
        P = to_robot(fr["p"])
        T = torso_frame(P)
        Tt = T.T
        mid = (P[1] + P[2]) / 2
        loc = lambda v: Tt @ (v - mid)  # noqa: E731
        te = euler(T)
        torso[i] = [np.clip(te[0] * 0.6, -0.15, 0.15), np.clip(te[1] * 0.6, -0.2, 0.2), np.clip(te[2] * 0.6, -0.12, 0.12)]
        for side, (si, ei, wi, hk) in sides.items():
            g = G[side]
            Sh, Eh, Wh = loc(P[si]), loc(P[ei]), loc(P[wi])
            g["W"][i] = Wh
            g["pole"][i] = Eh - (Sh + Wh) / 2      # khuỷu nằm phía nào của đường vai–cổ tay
            g["fdir"][i] = Wh - Eh
            if fr.get(hk):
                H = np.stack([loc(h) for h in to_robot(fr[hk])])
                Hrel = H - H[0]
                Rh = hand_frame_view(Hrel, side)               # hướng bàn tay (xem hand_frame_view)
                Rh_shape = hand_frame(Hrel, side)              # hình dạng ngón: dùng 3D đầy đủ
                g["Q"][i] = _quat(Rh)
                local_hand = (Rh_shape.T @ Hrel.T).T
                fingers = finger_angles(local_hand, side)
                g["F"][i] = refine_fingers(local_hand, side, fingers) if fit_fingers else fingers

    M = np.full((n, len(LAYOUT)), np.nan)
    M[:, :3] = torso
    for side, off in (("right", 3), ("left", 3 + SIDE_CHANNELS)):
        g = G[side]
        W, pole, fdir = _fill(g["W"]), _fill(g["pole"]), _fill(g["fdir"])
        if W is None:
            continue
        # 2) Hướng bàn tay: lấp khoảng không thấy tay bằng slerp, làm mượt nhẹ trên quaternion
        Q = g["Q"]
        okq = np.where(~np.isnan(Q[:, 0]))[0]
        Rh = [None] * n
        if len(okq):
            for a, b in zip(okq[:-1], okq[1:]):
                if np.dot(Q[a], Q[b]) < 0:
                    Q[b] = -Q[b]
                for k in range(a + 1, b):
                    Q[k] = _slerp(Q[a], Q[b], (k - a) / (b - a))
            lo, hi = okq[0], okq[-1]
            if smooth and hi - lo >= 4:
                Q[lo:hi + 1] = _medoid_quats(Q[lo:hi + 1], half=1)
            Qs = _smooth_adaptive(Q[lo:hi + 1], ROT_D0, angular=True) if (smooth and hi > lo) else Q[lo:hi + 1]
            for k in range(lo, hi + 1):
                Rh[k] = _qmat(Qs[k - lo])
        # 3) Đích cổ tay: giữ VỊ TRÍ so với thân người (quy về tỉ lệ robot), tránh lọt vào thân/đầu, làm mượt nhẹ
        Wt = np.zeros((n, 3))
        for i in range(n):
            palm_dir = -Rh[i][:, 1] if Rh[i] is not None else unit(damp_z(fdir[i], KZ_ARM))
            Wt[i] = _avoid_body(to_robot_pos(W[i], scale), palm_dir)
        if smooth and n >= 3:
            Wt = _smooth_adaptive(Wt, POS_D0)
            pole = _smooth3(pole, (0.25, 0.5, 0.25))
        # 4) IK cánh tay + tách hướng bàn tay thành xoay cẳng tay (twist) và cổ tay
        sgn = -1.0 if side == "right" else 1.0
        Sr = np.array([sgn * R_SHOULDER_HALF, 0.0, 0.0])
        prev_bend = prev_arm = prev_wrist = prev_twist = None
        for i in range(n):
            Rs, theta, prev_bend = _solve_arm(Sr, Wt[i], damp_z(pole[i], KZ_ARM), prev_bend)
            prev_arm = _near_euler(euler(Rs), prev_arm)
            M[i, off:off + 3] = prev_arm
            M[i, off + 3:off + 6] = [-theta, 0.0, 0.0]
            if Rh[i] is None:
                continue
            Rfore = Rs @ Rot.from_euler("XYZ", [-theta, 0, 0]).as_matrix()
            Rrel = Rfore.T @ Rh[i]
            q = _quat(Rrel)
            twist = 2 * math.atan2(q[1], q[3])
            ref = prev_twist if prev_twist is not None else 0.0
            twist = ref + (twist - ref + math.pi) % (2 * math.pi) - math.pi
            Rw = Rot.from_euler("Y", twist).as_matrix().T @ Rrel
            prev_twist = twist
            prev_wrist = _near_euler(euler(Rw), prev_wrist if prev_wrist is not None else np.zeros(3))
            M[i, off + 6] = twist
            M[i, off + 7:off + 10] = prev_wrist
        # 5) Ngón tay (chỉ ở khung thấy tay; khoảng trống được nội suy bên dưới)
        for i in range(n):
            F = g["F"][i]
            if F:
                M[i, off + 10:off + SIDE_CHANNELS] = [F[fname][c] for fname, chans in FINGER_CHANNELS for c in chans]

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
        # Ngón tay: lọc trung vị 3 khung trước để bỏ các khung MediaPipe nhận sai hình dạng tay
        fcols = [j for j, name in enumerate(LAYOUT) if any(f in name for f in ("thumb", "index", "middle", "ring", "pinky"))]
        Mp = np.vstack([M[:1], M, M[-1:]])
        M[:, fcols] = np.median(np.stack([Mp[:-2, fcols], Mp[1:-1, fcols], Mp[2:, fcols]]), axis=0)
        # Cánh tay / cổ tay đã được làm mượt ở đầu vào → chỉ làm mượt thêm ngón tay và thân
        scols = fcols + [0, 1, 2]
        M[:, scols] = _smooth3(M[:, scols], (0.25, 0.5, 0.25))
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
    # Giữ cùng khoảng thời gian đo tư thế nghỉ khi thay đổi tốc độ lấy mẫu.
    rest_frames = max(1, round(fps * 3 / 8))
    base = [w for w in wr[:rest_frames] if w is not None]
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
    hand_cov = {side: float(np.mean([1.0 if F else 0.0 for F in G[side]["F"]])) if n else 0.0 for side in ("right", "left")}
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
    """Kiểm tra động học thuận trên robot:
      • wrist_xy_cm: sai lệch vị trí cổ tay robot so với vị trí người (đã quy về tỉ lệ robot) trên mặt phẳng
        nhìn thẳng (x, y) — điều người xem thấy; cm.
      • hand_y / hand_z: độ lệch hướng bàn tay (trục ngón giữa / pháp tuyến lòng bàn tay) so với video, độ.
    Chỉ tính ở các khung thấy tay."""
    errs = {"wrist_xy_cm": [], "hand_y": [], "hand_z": []}
    idx = {k: i for i, k in enumerate(LAYOUT)}
    sc = body_scale(raw)
    for row, fr in zip(motion["frames"], raw["frames"]):
        if not fr.get("p"):
            continue
        P = to_robot(fr["p"])
        T = torso_frame(P)
        c = (P[1] + P[2]) / 2
        for side, (wi, hk) in {"right": (6, "r"), "left": (5, "l")}.items():
            if not fr.get(hk):
                continue
            g = lambda k: row[idx[f"{side}_{k}"]]  # noqa: E731
            sgn = -1.0 if side == "right" else 1.0
            Rs = Rot.from_euler("XYZ", [g("arm_x"), g("arm_y"), g("arm_z")]).as_matrix()
            Rf = Rs @ Rot.from_euler("XYZ", [g("fore_x"), g("fore_y"), g("fore_z")]).as_matrix()
            wr = np.array([sgn * R_SHOULDER_HALF, 0, 0]) + Rs @ (DOWN * R_UPPER) + Rf @ (DOWN * R_FORE)
            tgt = to_robot_pos(T.T @ (P[wi] - c), sc)
            errs["wrist_xy_cm"].append(100 * float(np.linalg.norm((wr - tgt)[:2])))
            Rh = Rf @ Rot.from_euler("Y", g("twist")).as_matrix() @ Rot.from_euler("XYZ", [g("wrist_x"), g("wrist_y"), g("wrist_z")]).as_matrix()
            H = np.stack([T.T @ h for h in to_robot(fr[hk])])
            Rt = hand_frame_view(H - H[0], side)
            errs["hand_y"].append(math.degrees(math.acos(np.clip(np.dot(Rh[:, 1], Rt[:, 1]), -1, 1))))
            errs["hand_z"].append(math.degrees(math.acos(np.clip(np.dot(Rh[:, 2], Rt[:, 2]), -1, 1))))
    return {k: (round(float(np.median(v)), 1) if v else None) for k, v in errs.items()}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Chuyển điểm khớp MediaPipe sang góc robot.")
    parser.add_argument("source", help="Tệp JSON điểm khớp của một video")
    parser.add_argument("--fit-fingers", action="store_true", help="Thử tinh chỉnh góc ngón bằng động học thuận")
    parser.add_argument("--output", help="Đường dẫn tệp chuyển động để xem thử")
    args = parser.parse_args()
    with open(args.source, encoding="utf-8") as stream:
        raw = json.load(stream)
    m = retarget(raw, fit_fingers=args.fit_fingers)
    if args.output:
        with open(args.output, "x", encoding="utf-8") as stream:
            json.dump(m, stream, separators=(",", ":"))
    print(json.dumps({"id": m["id"], "frames": len(m["frames"]), "hand_coverage": m["hand_coverage"], "fk": fk_check(m, raw)}, ensure_ascii=False))
