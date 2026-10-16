"""
Tự động viết phần DIỄN TẢ KÝ HIỆU (cách làm) từ chuyển động người thật đã trích trong video QIPEDC
=================================================================================================

Từ các điểm MediaPipe (pose + bàn tay) của từng video, mô tả theo đúng các thành phần của ký hiệu:
  • Số tay sử dụng (tay phải / tay trái / hai tay, tay nào làm nền)
  • Hình dạng bàn tay (khép, xòe, nắm, chữ V, chữ O, ngón trỏ duỗi…) và sự thay đổi hình dạng
  • Hướng lòng bàn tay, hướng đầu ngón tay
  • Vị trí đặt tay so với cơ thể (ngang trán, ngang mặt, trước ngực…)
  • Chuyển động (đưa lên/xuống/sang/ra trước, lặp lại, xoay cổ tay, vẽ vòng, hai tay chạm nhau…)

    python -m backend.training.describe_signs qipedc_motion_raw.json.gz   # hoặc thư mục raw_motion/
→ backend/data/sign_descriptions.json  { motion_id: {summary, hands, parts: [...]} }

Chỉ cần numpy. "Trái/phải" luôn là trái/phải CỦA NGƯỜI LÀM KÝ HIỆU.
"""
import glob
import json
import math
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "backend", "data", "sign_descriptions.json")
FINGERS = {"index": (5, 6, 7, 8), "middle": (9, 10, 11, 12), "ring": (13, 14, 15, 16), "pinky": (17, 18, 19, 20)}
FNAME = {"thumb": "ngón cái", "index": "ngón trỏ", "middle": "ngón giữa", "ring": "ngón áp út", "pinky": "ngón út"}
SIDE_VI = {"right": "phải", "left": "trái"}


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v * 0


def to_robot(arr):
    # mm MediaPipe world → mét; x: bên trái người, y: lên, z: về phía trước (phía camera)
    a = np.asarray(arr, dtype=float).reshape(-1, 3) / 1000.0
    return np.stack([a[:, 0], -a[:, 1], -a[:, 2]], axis=1)


def torso_frame(P):
    lsh, rsh, lhip, rhip = P[1], P[2], P[7], P[8]
    x = unit(lsh - rsh)
    y = unit((lsh + rsh) / 2 - (lhip + rhip) / 2)
    y = unit(y - np.dot(y, x) * x)
    z = np.cross(x, y)
    return np.stack([x, y, z], axis=1)


def ang(a, b):
    return math.acos(max(-1.0, min(1.0, float(np.dot(unit(a), unit(b))))))


# ---------------------------------------------------------------- từng khung hình
def frame_info(fr):
    if not fr.get("p"):
        return None
    P = to_robot(fr["p"])
    T = torso_frame(P)
    c = (P[1] + P[2]) / 2
    loc = lambda v: T.T @ (v - c)  # noqa: E731  (toạ độ trong khung thân, gốc giữa hai vai)
    info = {
        "nose": loc(P[0]),
        "hip_y": float(loc((P[7] + P[8]) / 2)[1]),
        "half_sw": float(np.linalg.norm(P[1] - P[2]) / 2),
    }
    for side, wi, hk in (("right", 6, "r"), ("left", 5, "l")):
        wrist = loc(P[wi])
        h = None
        if fr.get(hk):
            H = np.stack([T.T @ v for v in to_robot(fr[hk])])  # chỉ xoay (điểm tay tương đối tâm bàn tay)
            H = H - H[0]
            palm_c = wrist + 0.35 * H[9] + 0.65 * H[[0, 5, 9, 13, 17]].mean(axis=0)  # giữa lòng bàn tay, hơi về phía gốc ngón
            h = {"H": H, "palm": palm_c, **hand_shape(H, side)}
        info[side] = {"wrist": wrist, "hand": h}
    return info


def hand_shape(H, side):
    palm = float(np.linalg.norm(H[9] - H[0])) or 0.08
    st = {}
    # Độ duỗi theo tỉ lệ (đầu ngón→cổ tay) / (gốc ngón giữa→cổ tay): ít bị ảnh hưởng khi ngón chĩa vào camera
    EXT = {"index": 1.45, "middle": 1.55, "ring": 1.45, "pinky": 1.28}
    FOLD = {"index": 1.12, "middle": 1.18, "ring": 1.08, "pinky": 0.98}
    for f, (a, b, c, d) in FINGERS.items():
        tip_d = float(np.linalg.norm(H[d] - H[0])) / palm
        st[f] = "ext" if tip_d > EXT[f] else ("fold" if tip_d < FOLD[f] else "bent")
    # ngón cái: dang ra / áp sát / gập vào lòng
    tt = H[4]
    d_idx = float(np.linalg.norm(tt - H[5])) / palm
    d_mid = float(np.linalg.norm(tt - H[13])) / palm
    if d_mid < 0.45:
        st["thumb"] = "in"
    elif d_idx > 0.85:
        st["thumb"] = "out"
    else:
        st["thumb"] = "side"
    touch = float(np.linalg.norm(H[4] - H[8])) / palm < 0.35
    touch_mid = float(np.linalg.norm(H[4] - H[12])) / palm < 0.35
    spread = ang(H[8] - H[5], H[20] - H[17]) if st["index"] == "ext" and st["pinky"] == "ext" else 0.0
    # pháp tuyến lòng bàn tay & hướng đầu ngón tay
    if side == "right":
        n = unit(np.cross(H[5] - H[0], H[17] - H[0]))
    else:
        n = unit(np.cross(H[17] - H[0], H[5] - H[0]))
    point = unit(H[9] - H[0])
    return {"st": st, "touch": touch, "touch_mid": touch_mid, "spread": spread, "normal": n, "point": point}


# ---------------------------------------------------------------- gọi tên
def shape_name(st, touch, touch_mid, spread):
    ext = [f for f in ("index", "middle", "ring", "pinky") if st[f] == "ext"]
    fold = [f for f in ("index", "middle", "ring", "pinky") if st[f] == "fold"]
    th = st["thumb"]
    if len(ext) == 4:
        if spread > 0.55 or th == "out" and spread > 0.4:
            return "xòe (năm ngón duỗi, dang rộng)"
        return "khép" + (", ngón cái mở ra" if th == "out" else "")
    if touch and set(ext) == {"middle", "ring", "pinky"}:
        return 'giống chữ cái ngón tay "F" (đầu ngón cái chạm đầu ngón trỏ, ba ngón còn lại duỗi)'
    if touch and not ext:
        if fold and len(fold) >= 3:
            return "chụm (các đầu ngón tay chụm vào đầu ngón cái)"
        return 'giống chữ cái ngón tay "O" (các ngón cong, đầu ngón cái chạm đầu ngón trỏ)'
    if not ext:
        if len(fold) >= 3:
            if th == "out":
                return "nắm, ngón cái dựng lên"
            return "nắm"
        return "khum (các ngón cong tự nhiên)"
    if ext == ["index"]:
        if th == "out":
            return 'giống chữ cái ngón tay "L" (ngón cái và ngón trỏ duỗi)'
        return "nắm, ngón trỏ duỗi thẳng"
    if ext == ["index", "middle"]:
        return 'giống chữ cái ngón tay "V" (ngón trỏ và ngón giữa duỗi)'
    if ext == ["index", "middle", "ring"]:
        return 'giống chữ cái ngón tay "W" (ngón trỏ, ngón giữa, ngón áp út duỗi)'
    if ext == ["pinky"]:
        if th == "out":
            return 'giống chữ cái ngón tay "Y" (ngón cái và ngón út duỗi)'
        return 'giống chữ cái ngón tay "I" (ngón út duỗi)'
    if ext == ["index", "pinky"]:
        return "ngón trỏ và ngón út duỗi, các ngón khác nắm"
    if ext == ["middle"]:
        return "ngón giữa duỗi, các ngón khác nắm"
    names = ", ".join(FNAME[f] for f in ext)
    return f"duỗi {names}, các ngón khác gập"


NAMED = ("khép", "xòe", "nắm", "chụm", "khum", "giống chữ", "nắm, ngón trỏ")


def base_shape(name, st):
    """Bỏ chi tiết ngón cái dễ nhiễu; tổ hợp ngón lạ (thường do nhiễu) → hình dạng gần nhất."""
    name = name.replace(", ngón cái mở ra", "")
    if name.startswith("duỗi ") or name.startswith("ngón giữa duỗi"):
        n_ext = sum(st[f] == "ext" for f in ("index", "middle", "ring", "pinky"))
        return "khép" if n_ext >= 3 else "khum (các ngón cong tự nhiên)"
    return name


def dir_name(v, kind):
    """kind = 'palm' | 'point'. v trong khung thân (x: trái người, y: lên, z: trước)."""
    labels = {
        (0, 1): ("hướng sang trái", "hướng sang trái"),
        (0, -1): ("hướng sang phải", "hướng sang phải"),
        (1, 1): ("ngửa (hướng lên trên)", "hướng lên trên"),
        (1, -1): ("úp (hướng xuống dưới)", "hướng xuống dưới"),
        (2, 1): ("hướng ra phía trước", "hướng về phía trước"),
        (2, -1): ("hướng vào người", "hướng vào người"),
    }
    k = 0 if kind == "palm" else 1
    v = np.array(v, dtype=float)
    order = np.argsort(-np.abs(v))
    a0 = int(order[0])
    main = labels[(a0, 1 if v[a0] > 0 else -1)][k]
    a1 = int(order[1])
    if abs(v[a1]) > 0.55 * abs(v[a0]) and abs(v[a1]) > 0.38:
        sec = labels[(a1, 1 if v[a1] > 0 else -1)][1].replace("hướng ", "")
        main = main.replace(" (hướng lên trên)", "").replace(" (hướng xuống dưới)", "")
        return f"{main}, chếch {sec}"
    return main


def palm_change(palm_c, palm0, palm_end):
    """Chỉ báo hướng lòng bàn tay lúc kết thúc khi thật sự xoay khác hẳn."""
    if ang(palm0, palm_end) < 1.0:
        return None
    a, b = dir_name(palm_c, "palm"), dir_name(palm_end, "palm")
    if a.split(" (")[0].split(",")[0] == b.split(" (")[0].split(",")[0]:
        return None
    return b


def where(p, ctx, side=None):
    ny, hip = ctx["nose_y"], ctx["hip_y"]
    y, x = float(p[1]), float(p[0])
    nose = ctx["nose"]
    near_face = math.hypot(x - nose[0], y - nose[1]) < 0.09 and y > ny - 0.13
    if y > ny + 0.13:
        lv = "trên đầu"
    elif y > ny + 0.035:
        lv = "ngang trán"
    elif y > ny - 0.06:
        lv = "ngang mặt"
    elif y > ny - 0.13:
        lv = "ngang miệng – cằm"
    elif y > -0.03:
        lv = "ngang cổ"
    elif y > -0.2:
        lv = "trước ngực"
    elif y > hip + 0.1:
        lv = "trước bụng"
    else:
        lv = "ngang hông"
    sw = ctx["half_sw"]
    if x < -0.85 * sw:
        hs = "bên phải"
    elif x > 0.85 * sw:
        hs = "bên trái"
    else:
        hs = ""
    if near_face and lv in ("ngang trán", "ngang mặt", "ngang miệng – cằm"):
        part = {"ngang trán": "trán", "ngang mặt": "mặt", "ngang miệng – cằm": "miệng – cằm"}[lv]
        return f"sát {part}"
    if lv.startswith("trước") and hs:
        return f"{lv}, lệch {hs}"
    if hs:
        return f"{hs}, {lv}"
    return lv


def movement_desc(pos, normals, shapes, fps):
    """pos: (n,3) vị trí lòng bàn tay theo thời gian (khung thân)."""
    if len(pos) < 3:
        return "giữ yên", 0
    w = np.array([1.0, 1.0, 0.5])  # độ sâu từ video kém tin cậy → giảm trọng số
    P = pos * w
    seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
    L = float(seg.sum())
    D = P[-1] - P[0]
    extent = float(np.linalg.norm(P.max(axis=0) - P.min(axis=0)))
    rot = max((ang(normals[0], n) for n in normals), default=0.0)
    if extent < 0.05:
        if rot > 1.2:
            return "xoay cổ tay", 0
        return "giữ yên", 0
    # trục chuyển động chính
    C = P - P.mean(axis=0)
    _, _, Vt = np.linalg.svd(C, full_matrices=False)
    axis = Vt[0]
    proj = C @ axis
    # đếm số lần đổi chiều có biên độ đáng kể
    rev, last_ext, direction = 0, proj[0], 0
    for v in proj[1:]:
        if direction >= 0 and v < last_ext - 0.03:
            if direction > 0:
                rev += 1
            direction, last_ext = -1, v
        elif direction <= 0 and v > last_ext + 0.03:
            if direction < 0:
                rev += 1
            direction, last_ext = 1, v
        elif (direction > 0 and v > last_ext) or (direction < 0 and v < last_ext):
            last_ext = v
    ax = int(np.argmax(np.abs(axis)))
    # vòng tròn: quãng đường dài, phần bù theo trục thứ 2 lớn
    second = float(np.ptp(C @ Vt[1]))
    if second > 0.06 and second > 0.5 * float(np.ptp(proj)) and L > 2.4 * max(0.05, float(np.linalg.norm(D))) and rev <= 2:
        return "vẽ một vòng tròn", 0
    if rev >= 2:
        reps = (rev + 1) // 2 + (1 if rev % 2 == 0 else 0)
        reps = max(2, min(4, reps))
        kind = {0: "qua lại sang hai bên", 1: "lên xuống", 2: "ra – vào (đẩy ra trước rồi kéo về)"}[ax]
        return f"đưa tay {kind} (lặp lại {reps} lần)", reps
    # chuyển động một chiều: mô tả theo hướng dịch chuyển tổng
    parts = []
    names = {0: ("sang trái", "sang phải"), 1: ("lên trên", "xuống dưới"), 2: ("ra phía trước", "về phía người")}
    Dn = np.abs(D)
    for i in np.argsort(-Dn):
        if Dn[i] > 0.04 and (not parts or Dn[i] > 0.45 * Dn.max()):
            parts.append(names[int(i)][0 if D[i] > 0 else 1])
    if not parts:
        if rev == 1:
            kind = {0: "sang ngang rồi trở lại", 1: "lên rồi trở xuống", 2: "ra trước rồi kéo về"}[ax]
            return f"đưa tay {kind}", 0
        return "cử động nhẹ tại chỗ", 0
    s = "đưa tay " + " và ".join(parts[:2])
    if rev == 1:
        s += " rồi trở lại"
    return s, 0


def majority(xs):
    if not xs:
        return None
    vals, counts = np.unique(np.array(xs, dtype=object), return_counts=True)
    return vals[int(np.argmax(counts))]


# ---------------------------------------------------------------- toàn bộ ký hiệu
def describe(raw, active=None):
    fps = float(raw["fps"])
    infos = [frame_info(f) for f in raw["frames"]]
    ok = [i for i, x in enumerate(infos) if x]
    if len(ok) < 4:
        return None
    base = [infos[i] for i in ok[:3]]
    n = len(infos)
    if active:
        a0, a1 = int(round(active[0] * fps)), int(round(active[1] * fps))
    else:
        a0, a1 = 0, n - 1
    idx = [i for i in range(max(0, a0), min(n, a1 + 1)) if infos[i]]
    if len(idx) < 3:
        idx = ok
    ctx = {
        "nose": np.mean([infos[i]["nose"] for i in idx], axis=0),
        "hip_y": float(np.mean([infos[i]["hip_y"] for i in idx])),
        "half_sw": float(np.mean([infos[i]["half_sw"] for i in idx])),
    }
    ctx["nose_y"] = float(ctx["nose"][1])

    # tay nào tham gia: cổ tay rời vị trí nghỉ / giơ cao
    side_stats = {}
    for side in ("right", "left"):
        rest = np.mean([b[side]["wrist"] for b in base], axis=0)
        disp = [float(np.linalg.norm(infos[i][side]["wrist"] - rest)) for i in idx]
        hand_idx = [i for i in idx if infos[i][side]["hand"]]
        raised = [infos[i][side]["wrist"][1] > ctx["hip_y"] + 0.12 for i in idx]
        pos = np.array([infos[i][side]["hand"]["palm"] for i in hand_idx]) if hand_idx else np.zeros((0, 3))
        path = float(np.linalg.norm(np.diff(pos * [1, 1, 0.5], axis=0), axis=1).sum()) if len(pos) > 1 else 0.0
        side_stats[side] = {
            "active": (np.mean([d > 0.09 for d in disp]) > 0.3 or np.mean(raised) > 0.45) and len(hand_idx) >= max(2, 0.25 * len(idx)),
            "path": path,
            "hand_idx": hand_idx,
        }
    sides = [s for s in ("right", "left") if side_stats[s]["active"]]
    if not sides:
        # không phát hiện rõ → lấy tay thấy nhiều nhất
        s = max(("right", "left"), key=lambda k: len(side_stats[k]["hand_idx"]))
        if not side_stats[s]["hand_idx"]:
            return None
        sides = [s]

    parts = []
    for side in sides:
        hi_all = side_stats[side]["hand_idx"]
        # chỉ xét pha "làm ký hiệu": tay đã đưa lên gần vị trí xa nhất so với tư thế nghỉ (bỏ pha nâng/hạ tay)
        rest = np.mean([b[side]["wrist"] for b in base], axis=0)
        dd = {i: float(np.linalg.norm(infos[i][side]["wrist"] - rest)) for i in hi_all}
        dmax = max(dd.values()) if dd else 0
        hi = [i for i in hi_all if dd[i] >= 0.6 * dmax] if dmax > 0.12 else hi_all
        if len(hi) < 2:
            hi = hi_all
        # bỏ khoảng trống: lấy đoạn liên tục dài nhất
        runs, cur = [], [hi[0]]
        for i in hi[1:]:
            if i - cur[-1] <= 2:
                cur.append(i)
            else:
                runs.append(cur)
                cur = [i]
        runs.append(cur)
        hi = max(runs, key=len)
        m = len(hi)
        core = hi[int(m * 0.2): max(int(m * 0.2) + 1, int(m * 0.8))] or hi
        start = hi[: max(1, m // 4)]
        end = hi[-max(1, m // 4):]
        H = lambda i: infos[i][side]["hand"]  # noqa: E731
        def shp(ii):
            """Hình dạng chiếm đa số + độ tin cậy (tỉ lệ khung đồng ý)."""
            good = [i for i in ii if abs(H(i)["point"][2]) < 0.8] or ii
            names = [base_shape(shape_name(H(i)["st"], H(i)["touch"], H(i)["touch_mid"], H(i)["spread"]), H(i)["st"]) for i in good]
            best = majority(names)
            return best, names.count(best) / max(1, len(names))
        (s_core, _), (s_start, c0), (s_end, c1) = shp(core), shp(start), shp(end)
        # chỉ báo "đổi hình dạng tay" khi cả đầu và cuối đều rõ ràng
        changed = s_start != s_end and c0 >= 0.6 and c1 >= 0.6 and len(hi) >= 5
        if changed:
            s_core = s_start
        palm0 = unit(np.mean([H(i)["normal"] for i in start], axis=0))
        palm_c = unit(np.mean([H(i)["normal"] for i in core], axis=0))
        point_c = unit(np.mean([H(i)["point"] for i in core], axis=0))
        palm_end = unit(np.mean([H(i)["normal"] for i in end], axis=0))
        pos = np.array([H(i)["palm"] for i in hi])
        # làm mượt vị trí để bớt rung
        if len(pos) >= 3:
            pos = np.vstack([pos[:1], (pos[:-2] + 2 * pos[1:-1] + pos[2:]) / 4, pos[-1:]])
        mv, reps = movement_desc(pos, [H(i)["normal"] for i in hi], None, fps)
        k = len(pos)
        loc_core = where(pos[k // 4: max(k // 4 + 1, 3 * k // 4)].mean(axis=0), ctx, side)
        loc_start = where(pos[: max(1, k // 5)].mean(axis=0), ctx, side)
        loc_end = where(pos[-max(1, len(pos) // 5):].mean(axis=0), ctx, side)
        p = {
            "side": side,
            "shape": s_core,
            "shape_change": [s_start, s_end] if changed else None,
            "palm": dir_name(palm_c, "palm"),
            "palm_end": palm_change(palm_c, palm0, palm_end),
            "point": dir_name(point_c, "point"),
            "location": loc_start if (mv.startswith("đưa tay") and "lặp lại" not in mv and loc_start != loc_end) else loc_core,
            "location_end": None if (loc_end == loc_start or "lặp lại" in mv or not mv.startswith("đưa tay")) else loc_end,
            "movement": mv,
            "path": side_stats[side]["path"],
        }
        parts.append(p)

    # quan hệ hai tay
    relation = None
    if len(parts) == 2:
        both = [i for i in idx if infos[i]["right"]["hand"] and infos[i]["left"]["hand"]]
        if both:
            d = [float(np.linalg.norm((infos[i]["right"]["hand"]["palm"] - infos[i]["left"]["hand"]["palm"]) * [1, 1, 0.5])) for i in both]
            pr, pl = parts
            if min(d) < 0.075:
                relation = "hai bàn tay chạm vào nhau"
            if pr["path"] > 2.2 * pl["path"] and pl["path"] < 0.15:
                pl["movement"] = "giữ yên làm nền"
                relation = relation or "tay trái giữ yên làm nền cho tay phải"
            elif pl["path"] > 2.2 * pr["path"] and pr["path"] < 0.15:
                pr["movement"] = "giữ yên làm nền"
            else:
                vr = np.diff(np.array([infos[i]["right"]["hand"]["palm"] for i in both]), axis=0)
                vl = np.diff(np.array([infos[i]["left"]["hand"]["palm"] for i in both]), axis=0)
                if len(vr) > 2:
                    same = float(np.mean([np.dot(unit(a), unit(b)) for a, b in zip(vr, vl) if np.linalg.norm(a) > 0.01 and np.linalg.norm(b) > 0.01] or [0]))
                    mir = float(np.mean([np.dot(unit(a * [-1, 1, 1]), unit(b)) for a, b in zip(vr, vl) if np.linalg.norm(a) > 0.01 and np.linalg.norm(b) > 0.01] or [0]))
                    if mir > 0.55 and mir > same:
                        relation = relation or "hai tay chuyển động đối xứng"
                    elif same > 0.55:
                        relation = relation or "hai tay chuyển động cùng chiều"
                    elif same < -0.3:
                        relation = relation or "hai tay chuyển động luân phiên"
        # gộp khi hai tay giống nhau
    for p in parts:
        p.pop("path", None)
    return {"hands": hands_label(parts), "parts": parts, "relation": relation, "summary": summarize(parts, relation)}


def hands_label(parts):
    if len(parts) == 2:
        return "Hai tay"
    return "Một tay (tay " + SIDE_VI[parts[0]["side"]] + ")"


def part_sentence(p, cap=True):
    who = "Tay " + SIDE_VI[p["side"]]
    shape = p["shape"]
    if p["shape_change"]:
        a, b = p["shape_change"]
        shape = f"{a}, rồi chuyển thành {b}"
    s = f"{who} {shape}, lòng bàn tay {p['palm']}, đầu ngón tay {p['point']}, đặt {p['location']}"
    mv = p["movement"]
    if mv and mv not in ("giữ yên",):
        s += f", {mv}"
        if p["location_end"] and mv.startswith("đưa tay") and "lặp lại" not in mv and "trở lại" not in mv:
            s += f" tới {p['location_end']}"
    if p.get("palm_end"):
        s += f" (kết thúc lòng bàn tay {p['palm_end']})"
    return s + "."


def summarize(parts, relation):
    out = " ".join(part_sentence(p) for p in parts)
    if relation:
        out += " " + relation[0].upper() + relation[1:] + "."
    return out


def iter_raw(src):
    """Thư mục raw (mỗi video 1 file .json) hoặc file gói qipedc_motion_raw.json(.gz)."""
    if os.path.isdir(src):
        for f in sorted(glob.glob(os.path.join(src, "*.json"))):
            with open(f, encoding="utf-8") as fh:
                yield json.load(fh)
    else:
        import gzip
        opener = gzip.open if src.endswith(".gz") else open
        with opener(src, "rt", encoding="utf-8") as fh:
            data = json.load(fh)
        yield from (data if isinstance(data, list) else [data])


def main(src):
    motions_dir = os.path.join(ROOT, "backend", "data", "motions")
    res = {}
    for raw in iter_raw(src):
        mid = raw["id"]
        mf = os.path.join(motions_dir, mid + ".json")
        if not os.path.exists(mf):
            continue
        active = json.load(open(mf, encoding="utf-8")).get("active")
        try:
            d = describe(raw, active)
        except Exception as e:  # noqa: BLE001
            print("lỗi", mid, e)
            d = None
        if d:
            res[mid] = d
    with open(OUT, "w", encoding="utf-8") as fo:
        json.dump(res, fo, ensure_ascii=False, separators=(",", ":"))
    print(f"Đã mô tả {len(res)} ký hiệu → {OUT}")
    return res


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "raw_motion")
