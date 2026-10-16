"""
Phép quay 3D tối giản (chỉ numpy) — thay cho scipy.spatial.transform.Rotation ở những chỗ retargeting cần,
để backend "học" ký hiệu mới được trên máy không cài scipy.

Chỉ hỗ trợ đúng các phần dùng tới: from_euler("XYZ" | "X" | "Y" | "Z", ...) (chữ HOA = quay nội tại như scipy),
from_matrix, from_rotvec, as_matrix, as_euler("XYZ"), as_quat (x, y, z, w).
"""
import math

import numpy as np


def _axis_mat(ax, a):
    c, s = math.cos(a), math.sin(a)
    if ax == "X":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if ax == "Y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    if ax == "Z":
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    raise ValueError(ax)


class Rotation:
    def __init__(self, m):
        self._m = np.asarray(m, dtype=float)

    # ---------------- tạo
    @classmethod
    def from_euler(cls, seq, angles):
        if not seq.isupper():
            raise ValueError("chỉ hỗ trợ quay nội tại (chữ HOA)")
        angles = np.atleast_1d(np.asarray(angles, dtype=float))
        m = np.eye(3)
        for ax, a in zip(seq, angles):
            m = m @ _axis_mat(ax, float(a))
        return cls(m)

    @classmethod
    def from_matrix(cls, m):
        # trực chuẩn hóa (SVD) như scipy
        u, _, vt = np.linalg.svd(np.asarray(m, dtype=float))
        r = u @ vt
        if np.linalg.det(r) < 0:
            u[:, -1] *= -1
            r = u @ vt
        return cls(r)

    @classmethod
    def from_rotvec(cls, v):
        v = np.asarray(v, dtype=float)
        th = float(np.linalg.norm(v))
        if th < 1e-12:
            return cls(np.eye(3))
        k = v / th
        kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        return cls(np.eye(3) + math.sin(th) * kx + (1 - math.cos(th)) * kx @ kx)

    # ---------------- xuất
    def as_matrix(self):
        return self._m.copy()

    def as_euler(self, seq):
        if seq != "XYZ":
            raise ValueError("chỉ hỗ trợ 'XYZ'")
        r = self._m
        b = math.asin(max(-1.0, min(1.0, r[0, 2])))
        if abs(r[0, 2]) < 1 - 1e-9:
            a = math.atan2(-r[1, 2], r[2, 2])
            c = math.atan2(-r[0, 1], r[0, 0])
        else:  # khóa gimbal: dồn toàn bộ vào trục đầu
            a = math.atan2(r[2, 1], r[1, 1])
            c = 0.0
        return np.array([a, b, c])

    def as_quat(self):
        m = self._m
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0:
            s = math.sqrt(tr + 1.0) * 2
            w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
        elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
            s = math.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
            w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
            w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
        else:
            s = math.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
            w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
        q = np.array([x, y, z, w])
        return q / np.linalg.norm(q)
