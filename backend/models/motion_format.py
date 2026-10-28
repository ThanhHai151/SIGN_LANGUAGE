"""
Định dạng chuyển động robot (không phụ thuộc scipy) — dùng lúc chạy API /api/motion.
Phần huấn luyện/chuyển đổi (cần scipy) nằm ở backend/training/retarget_motion.py.
"""

# Thứ tự kênh khi đóng gói 1 khung hình thành mảng phẳng (frontend giải theo keyframes)
FINGER_CHANNELS = [
    ("thumb", ["cmc_x", "cmc_y", "cmc_z", "mcp_x", "mcp_y", "mcp_z", "ip_x"]),
    ("index", ["mcp_x", "pip_x", "dip_x", "spread_z"]),
    ("middle", ["mcp_x", "pip_x", "dip_x", "spread_z"]),
    ("ring", ["mcp_x", "pip_x", "dip_x", "spread_z"]),
    ("pinky", ["mcp_x", "pip_x", "dip_x", "spread_z"]),
]
SIDE_CHANNELS = 3 + 3 + 1 + 3 + sum(len(c) for _, c in FINGER_CHANNELS)  # 31
LAYOUT = (["torso_x", "torso_y", "torso_z"]
          + [f"{s}_{k}" for s in ("right", "left") for k in
             ["arm_x", "arm_y", "arm_z", "fore_x", "fore_y", "fore_z", "twist", "wrist_x", "wrist_y", "wrist_z"]
             + [f"{f}_{c}" for f, cs in FINGER_CHANNELS for c in cs]])


def frames_to_keyframes(motion):
    """Mảng phẳng (đóng gói gọn trên đĩa) → keyframes {t, body, rightArm, ..., leftFingers} cho frontend."""
    idx = {k: i for i, k in enumerate(LAYOUT)}
    fps = float(motion["fps"])
    kfs = []
    for i, row in enumerate(motion["frames"]):
        g = lambda k: row[idx[k]]  # noqa: E731
        kf = {"t": round(i / fps, 3), "body": {"torso": [g("torso_x"), g("torso_y"), g("torso_z")], "head": [0.0, 0.0, 0.0]}}
        for side in ("right", "left"):
            Side = side
            kf[Side + "Arm"] = [g(f"{side}_arm_x"), g(f"{side}_arm_y"), g(f"{side}_arm_z")]
            kf[Side + "Forearm"] = [g(f"{side}_fore_x"), g(f"{side}_fore_y"), g(f"{side}_fore_z")]
            kf[Side + "Twist"] = g(f"{side}_twist")
            kf[Side + "Wrist"] = [g(f"{side}_wrist_x"), g(f"{side}_wrist_y"), g(f"{side}_wrist_z")]
            kf[Side + "Fingers"] = {fname: {c: g(f"{side}_{fname}_{c}") for c in chans} for fname, chans in FINGER_CHANNELS}
        kfs.append(kf)
    return {
        "id": motion["id"],
        "fps": fps,
        "duration_ms": int(round(len(kfs) / fps * 1000)),
        "video_duration_ms": int(round(float(motion.get("dur", 0)) * 1000)),
        "hand_coverage": motion.get("hand_coverage"),
        "active": motion.get("active") or [0.0, len(kfs) / fps],
        "source": "QIPEDC video → MediaPipe Holistic → IK",
        "keyframes": kfs,
    }


