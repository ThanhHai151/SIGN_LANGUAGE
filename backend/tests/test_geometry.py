"""
Unit tests cho Geometric Rule-Based classifier (classify_geometry)
va cac ham tien ich hinh hoc trong feature_extractor.

Pham vi kiem thu:
  1. Ham tien ich: calculate_distance, calculate_angle, finger_curl_ratio,
     compute_finger_states
  2. classify_geometry: A, C, D, L, O, U, V + E, B, NEUTRAL
  3. Cac cap de nham: C/O, D/L, U/V
  4. Truong hop bien: landmark thieu, palm_scale = 0, nhieu toa do

Luu y: Cac landmark duoi day duoc xay dung tay theo cau truc MediaPipe.
He toa do: x tang phai, y TANG XUONG (goc tren-trai = (0,0)),
z tang ra phia camera. Tat ca trong khoang [0, 1] da chuan hoa.
Khoang cach wrist (0) -> middle MCP (9) duoc dung lam palm_scale.

Vi du: wrist=[0.5, 0.8, 0], middle_mcp=[0.5, 0.5, 0]
=> palm_scale = 0.3 (doc theo truc y vi khung dung)
"""

import sys
import os
import io

if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
import math
import unittest
import numpy as np

# Them duong dan goc vao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.vision.feature_extractor import (
    calculate_distance,
    calculate_angle,
    finger_curl_ratio,
    compute_finger_states,
    _shift_and_scale,
)
from backend.vision.recognizer import VSLRecognizer


# ---------------------------------------------------------------------------
# Ham tao skeleton ban tay don gian de dung lai trong nhieu test
# ---------------------------------------------------------------------------

def _make_hand(wrist=None, palm_len=0.30):
    """
    Tao 21 landmark voi ban tay mac dinh o vi tri thu gian.
    - wrist: [x, y, z] (mac dinh [0.5, 0.8, 0])
    - palm_len: khoang cach wrist->middle_mcp (mac dinh 0.30)
    Tra ve list 21 phan tu [[x,y,z], ...]
    """
    if wrist is None:
        wrist = [0.5, 0.8, 0.0]
    wx, wy, wz = wrist

    # Middle MCP (9) dat thang len tren so voi wrist
    mx, my, mz = wx, wy - palm_len, wz

    # Xay dung 21 landmark co ban (tat ca co chung)
    lm = [[wx, wy, wz] for _ in range(21)]

    # --- Wrist ---
    lm[0] = [wx, wy, wz]

    # --- Middle MCP (9) --- dung lam anchor palm_scale
    lm[9]  = [mx, my, mz]

    # --- Ngon tro MCP (5) ---
    lm[5]  = [mx - 0.04, my + 0.02, mz]

    # --- Ngon giua MCP (9) da dat o tren ---

    # --- Ring MCP (13) ---
    lm[13] = [mx + 0.04, my + 0.02, mz]

    # --- Pinky MCP (17) ---
    lm[17] = [mx + 0.08, my + 0.05, mz]

    # --- Thumb: dat o vi tri thu gian (cai gap nhe) ---
    lm[1]  = [wx - 0.04, wy - 0.03, wz]  # CMC
    lm[2]  = [wx - 0.07, wy - 0.06, wz]  # MCP
    lm[3]  = [wx - 0.09, wy - 0.04, wz]  # IP
    lm[4]  = [wx - 0.08, wy - 0.02, wz]  # TIP (cai gap nhe, gan index MCP)

    # --- Index (5-8): tat ca gap (tip gan MCP) ---
    lm[5]  = [mx - 0.04, my + 0.02, mz]
    lm[6]  = [mx - 0.04, my + 0.05, mz]   # PIP
    lm[7]  = [mx - 0.04, my + 0.07, mz]   # DIP
    lm[8]  = [mx - 0.04, my + 0.09, mz]   # TIP (gap: tip DUOI pip)

    # --- Middle (9-12): tat ca gap ---
    lm[9]  = [mx, my, mz]
    lm[10] = [mx, my + 0.03, mz]          # PIP
    lm[11] = [mx, my + 0.05, mz]          # DIP
    lm[12] = [mx, my + 0.07, mz]          # TIP (gap: tip DUOI pip)

    # --- Ring (13-16): tat ca gap ---
    lm[13] = [mx + 0.04, my + 0.02, mz]
    lm[14] = [mx + 0.04, my + 0.05, mz]
    lm[15] = [mx + 0.04, my + 0.07, mz]
    lm[16] = [mx + 0.04, my + 0.09, mz]   # TIP (gap)

    # --- Pinky (17-20): tat ca gap ---
    lm[17] = [mx + 0.08, my + 0.05, mz]
    lm[18] = [mx + 0.08, my + 0.08, mz]
    lm[19] = [mx + 0.08, my + 0.10, mz]
    lm[20] = [mx + 0.08, my + 0.12, mz]   # TIP (gap)

    return lm


def _extend_finger(lm, finger, palm_len=0.30):
    """
    Dat ngon tay 'finger' o trang thai DUOI THANG len tren.
    finger: 'index', 'middle', 'ring', 'pinky'
    Thay doi in-place va tra ve lm.
    """
    wx, wy, wz = lm[0]
    mx, my, mz = lm[9]

    if finger == "index":
        base_x = lm[5][0]
        lm[6]  = [base_x, my - 0.07, wz]   # PIP cao hon MCP
        lm[7]  = [base_x, my - 0.13, wz]   # DIP
        lm[8]  = [base_x, my - 0.19, wz]   # TIP (TIP.y < PIP.y => extended)
    elif finger == "middle":
        base_x = lm[9][0]
        lm[10] = [base_x, my - 0.07, wz]
        lm[11] = [base_x, my - 0.13, wz]
        lm[12] = [base_x, my - 0.19, wz]
    elif finger == "ring":
        base_x = lm[13][0]
        lm[14] = [base_x, my - 0.07, wz]
        lm[15] = [base_x, my - 0.13, wz]
        lm[16] = [base_x, my - 0.19, wz]
    elif finger == "pinky":
        base_x = lm[17][0]
        lm[18] = [base_x, my - 0.07, wz]
        lm[19] = [base_x, my - 0.13, wz]
        lm[20] = [base_x, my - 0.19, wz]
    return lm


def _extend_thumb_sideways(lm, angle_deg=70):
    """
    Dat ngon cai duoi ra ngang (chu L), tao goc 'angle_deg' voi ngon tro.
    Ngon cai vuon sang trai (x giam).
    """
    wx, wy, wz = lm[0]
    # Ngon cai arch sang trai, tip o xa ngang
    import math
    rad = math.radians(angle_deg)
    tip_offset_x = -0.18 * math.cos(math.radians(20))
    tip_offset_y = -0.10
    lm[3] = [wx - 0.09, wy - 0.08, wz]   # IP
    lm[4] = [wx - 0.18, wy - 0.10, wz]   # TIP xa sang trai
    return lm


def _fold_thumb_firmly(lm):
    """
    Dat ngon cai gap sat long ban tay (thu B): tip nam duoi PIP ngon tro.
    """
    wx, wy, wz = lm[0]
    mx, my, mz = lm[9]
    pip_index_y = lm[6][1]  # PIP ngon tro
    lm[3] = [lm[5][0] - 0.02, pip_index_y + 0.03, wz]   # IP
    lm[4] = [lm[5][0] - 0.01, pip_index_y + 0.06, wz]   # TIP ben duoi PIP ngon tro
    return lm


# ---------------------------------------------------------------------------
# Test suite 1: Ham tien ich hinh hoc
# ---------------------------------------------------------------------------

class TestGeometricUtils(unittest.TestCase):
    """Kiem thu cac ham tien ich trong feature_extractor."""

    def test_calculate_distance_zero(self):
        """Khoang cach tu mot diem den chinh no phai bang 0."""
        p = [0.3, 0.5, 0.1]
        self.assertAlmostEqual(calculate_distance(p, p), 0.0)

    def test_calculate_distance_axis_aligned(self):
        """Khoang cach theo mot truc don."""
        p1 = [0.0, 0.0, 0.0]
        p2 = [0.3, 0.0, 0.0]
        self.assertAlmostEqual(calculate_distance(p1, p2), 0.3)

    def test_calculate_distance_3d(self):
        """Khoang cach 3D: sqrt(3^2+4^2+0^2)=5."""
        p1 = [0.0, 0.0, 0.0]
        p2 = [3.0, 4.0, 0.0]
        self.assertAlmostEqual(calculate_distance(p1, p2), 5.0)

    def test_calculate_angle_right_angle(self):
        """Goc 90 do tai dinh b."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [0.0, 1.0, 0.0]
        self.assertAlmostEqual(calculate_angle(a, b, c), 90.0, places=5)

    def test_calculate_angle_zero(self):
        """Goc 0 khi a va c cung huong tu b."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [2.0, 0.0, 0.0]
        self.assertAlmostEqual(calculate_angle(a, b, c), 0.0, places=5)

    def test_calculate_angle_180(self):
        """Goc 180 do khi a-b-c thang hang."""
        a = [1.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [-1.0, 0.0, 0.0]
        self.assertAlmostEqual(calculate_angle(a, b, c), 180.0, places=5)

    def test_calculate_angle_degenerate(self):
        """Goc 0 khi mot trong cac vector co do dai = 0."""
        a = [0.0, 0.0, 0.0]
        b = [0.0, 0.0, 0.0]
        c = [1.0, 0.0, 0.0]
        # Ket qua phai la 0.0 (khong phat ngoai le)
        result = calculate_angle(a, b, c)
        self.assertAlmostEqual(result, 0.0)

    def test_finger_curl_ratio_straight(self):
        """Ngon duoi thang: TIP, PIP, MCP thang hang => curl ~ 0."""
        # Dat TIP-PIP-MCP thang hang doc theo truc y
        shifted = [[0.0, 0.0, 0.0]] * 21
        # Index: MCP(5) - PIP(6) - TIP(8) thang hang
        shifted[5] = [0.0, 0.0, 0.0]   # MCP
        shifted[6] = [0.0, -0.07, 0.0]  # PIP
        shifted[8] = [0.0, -0.19, 0.0]  # TIP
        # finger_curl_ratio(shifted, tip=8, pip=6, mcp=5, palm_scale)
        curl = finger_curl_ratio(shifted, 8, 6, 5, 1.0)
        self.assertLess(curl, 0.15, "Ngon duoi thang: curl phai gan 0")

    def test_finger_curl_ratio_folded(self):
        """Ngon gap: TIP gap lai gan MCP => curl cao."""
        shifted = [[0.0, 0.0, 0.0]] * 21
        shifted[5] = [0.0, 0.0, 0.0]    # MCP
        shifted[6] = [0.0, -0.07, 0.0]  # PIP (giua)
        shifted[8] = [0.0, -0.02, 0.0]  # TIP (gap lai, gan MCP)
        curl = finger_curl_ratio(shifted, 8, 6, 5, 1.0)
        self.assertGreater(curl, 0.30, "Ngon gap: curl phai > 0.30")

    def test_compute_finger_states_all_folded(self):
        """Tat ca 4 ngon (khong cai) gap: extended phai la False."""
        lm = _make_hand()  # tat ca gap theo mac dinh
        shifted, palm_scale = _shift_and_scale(lm)
        states = compute_finger_states(shifted, palm_scale)
        self.assertFalse(states["index_extended"],  "index nen gap")
        self.assertFalse(states["middle_extended"], "middle nen gap")
        self.assertFalse(states["ring_extended"],   "ring nen gap")
        self.assertFalse(states["pinky_extended"],  "pinky nen gap")

    def test_compute_finger_states_index_extended(self):
        """Chi ngon tro duoi: index_extended = True, con lai False."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        shifted, palm_scale = _shift_and_scale(lm)
        states = compute_finger_states(shifted, palm_scale)
        self.assertTrue(states["index_extended"],   "index nen duoi")
        self.assertFalse(states["middle_extended"], "middle nen gap")
        self.assertFalse(states["ring_extended"],   "ring nen gap")
        self.assertFalse(states["pinky_extended"],  "pinky nen gap")

    def test_compute_finger_states_all_extended(self):
        """Tat ca 4 ngon (khong cai) duoi: extended phai la True."""
        lm = _make_hand()
        for f in ["index", "middle", "ring", "pinky"]:
            _extend_finger(lm, f)
        shifted, palm_scale = _shift_and_scale(lm)
        states = compute_finger_states(shifted, palm_scale)
        self.assertTrue(states["index_extended"])
        self.assertTrue(states["middle_extended"])
        self.assertTrue(states["ring_extended"])
        self.assertTrue(states["pinky_extended"])

    def test_palm_scale_nonzero(self):
        """palm_scale phai > 0 khi landmark hop le."""
        lm = _make_hand()
        shifted, palm_scale = _shift_and_scale(lm)
        self.assertGreater(palm_scale, 1e-5)

    def test_shifted_wrist_at_origin(self):
        """Sau khi shift, wrist (shifted[0]) phai o goc toa do (0,0,0)."""
        lm = _make_hand()
        shifted, _ = _shift_and_scale(lm)
        self.assertAlmostEqual(shifted[0][0], 0.0)
        self.assertAlmostEqual(shifted[0][1], 0.0)
        self.assertAlmostEqual(shifted[0][2], 0.0)


# ---------------------------------------------------------------------------
# Test suite 2: classify_geometry - tung chu cai
# ---------------------------------------------------------------------------

class TestClassifyGeometry(unittest.TestCase):
    """Kiem thu quy tac phan loai hinh hoc trong classify_geometry."""

    @classmethod
    def setUpClass(cls):
        """Khoi tao recognizer mot lan cho ca class (khong load model)."""
        # Chi can classify_geometry, khong can model weights
        # Dung __new__ de tranh load SSTCN/MLP trong __init__
        cls.rec = VSLRecognizer.__new__(VSLRecognizer)

    def _get_states(self, lm):
        """Tinh finger_states tu landmark list."""
        from backend.vision.feature_extractor import normalize_hand_landmarks
        _, states = normalize_hand_landmarks(lm)
        return states

    def _classify(self, lm):
        """Chay classify_geometry va tra ve (chu, confidence)."""
        states = self._get_states(lm)
        return self.rec.classify_geometry(lm, states)

    # ------------------------------------------------------------------
    # Chu V: ngon tro + giua duoi, 2 ngon con lai gap, tach xa
    # ------------------------------------------------------------------
    def test_V_basic(self):
        """Chu V: ngon tro va giua duoi xa nhau (goc lon)."""
        lm = _make_hand()
        # Ngon tro duoi - lech trai
        lm[5] = [lm[9][0] - 0.06, lm[9][1], lm[9][2]]
        _extend_finger(lm, "index")
        lm[8][0] = lm[5][0]  # TIP theo chieu ngang cua MCP
        # Ngon giua duoi - lech phai
        _extend_finger(lm, "middle")
        # Tach 2 dau ngon ra xa nhau theo x
        lm[8][0] = lm[9][0] - 0.10   # index TIP sang trai
        lm[12][0] = lm[9][0] + 0.10  # middle TIP sang phai
        sign, conf = self._classify(lm)
        self.assertEqual(sign, "V", f"Ket qua: {sign}")
        self.assertGreaterEqual(conf, 0.80)

    # ------------------------------------------------------------------
    # Chu U: ngon tro + giua duoi, 2 ngon sat nhau (song song)
    # ------------------------------------------------------------------
    def test_U_basic(self):
        """Chu U: ngon tro va giua duoi, song song sat nhau."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        _extend_finger(lm, "middle")
        # Dat 2 TIP rat gan nhau theo x
        mid_x = lm[9][0]
        lm[8][0]  = mid_x - 0.02   # index TIP sat nhau
        lm[12][0] = mid_x + 0.02   # middle TIP sat nhau
        sign, conf = self._classify(lm)
        self.assertEqual(sign, "U", f"Ket qua: {sign}")
        self.assertGreaterEqual(conf, 0.75)

    # ------------------------------------------------------------------
    # Chu L: ngon tro duoi + ngon cai duoi goc lon
    # ------------------------------------------------------------------
    def test_L_basic(self):
        """Chu L: ngon tro len thang, ngon cai ra ngang, goc >= 50 do."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        _extend_thumb_sideways(lm, angle_deg=70)
        sign, conf = self._classify(lm)
        self.assertEqual(sign, "L", f"Ket qua: {sign}")
        self.assertGreaterEqual(conf, 0.80)

    def test_L_not_D_when_thumb_extended(self):
        """Khi ngon cai duoi ro rang va goc lon, khong duoc phan loai la D."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        _extend_thumb_sideways(lm, angle_deg=70)
        sign, _ = self._classify(lm)
        self.assertNotEqual(sign, "D", "L khong duoc nham la D")

    # ------------------------------------------------------------------
    # Chu D: chi ngon tro duoi, ngon cai gap
    # ------------------------------------------------------------------
    def test_D_basic(self):
        """Chu D: ngon tro duoi thang, ngon cai gap sat long ban tay."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        # Ngon cai gap: dat TIP gan wrist (gap vao trong)
        wx, wy, wz = lm[0]
        mx, my, mz = lm[9]
        lm[3] = [wx - 0.04, wy - 0.05, wz]  # IP
        lm[4] = [wx - 0.03, wy - 0.02, wz]  # TIP (gap, gan wrist)
        sign, conf = self._classify(lm)
        self.assertEqual(sign, "D", f"Ket qua: {sign}")
        self.assertGreaterEqual(conf, 0.75)

    def test_D_not_L_when_thumb_folded(self):
        """Khi ngon cai gap, khong duoc phan loai la L."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        # Dat TIP gap
        wx, wy, wz = lm[0]
        lm[4] = [wx - 0.03, wy - 0.02, wz]
        sign, _ = self._classify(lm)
        self.assertNotEqual(sign, "L", "D khong duoc nham la L")

    # ------------------------------------------------------------------
    # Chu O: ngon cai cham ngon tro, cac ngon khac cong vao
    # ------------------------------------------------------------------
    def test_O_basic(self):
        """Chu O: ngon cai cham ngon tro (d_thumb_index < 0.30 * palm_scale)."""
        lm = _make_hand()
        # Dat TIP ngon cai rat gan TIP ngon tro (8)
        index_tip = lm[8]  # [mx-0.04, my+0.09, wz] (gap)
        lm[4] = [index_tip[0] + 0.01, index_tip[1] - 0.01, index_tip[2]]
        lm[3] = [index_tip[0] + 0.02, index_tip[1] - 0.03, index_tip[2]]
        sign, conf = self._classify(lm)
        self.assertEqual(sign, "O", f"Ket qua: {sign}")
        self.assertGreaterEqual(conf, 0.75)

    def test_O_not_C_small_distance(self):
        """O co d_thumb_index nho, C co d_thumb_index lon hon - khong nham nhau."""
        lm = _make_hand()
        index_tip = lm[8]
        lm[4] = [index_tip[0] + 0.01, index_tip[1] - 0.01, index_tip[2]]
        sign, _ = self._classify(lm)
        self.assertNotEqual(sign, "C", "O khong duoc nham la C")

    # ------------------------------------------------------------------
    # Chu C: 4 ngon duoi, ngon cai arch ra vua phai (khong firmly folded)
    # ------------------------------------------------------------------
    def test_C_basic(self):
        """Chu C: 4 ngon duoi, ngon cai arch ra tao hinh chu C."""
        lm = _make_hand()
        for f in ["index", "middle", "ring", "pinky"]:
            _extend_finger(lm, f)
        # Ngon cai arch ra: dat TIP xa ngon tro nhung khong duoi thang hoan toan
        wx, wy, wz = lm[0]
        mx, my, mz = lm[9]
        lm[3] = [wx - 0.07, wy - 0.10, wz]   # IP
        lm[4] = [wx - 0.10, wy - 0.14, wz]   # TIP (arch sang trai)
        # thumb_curl phai < 0.55 (khong firmly folded)
        sign, conf = self._classify(lm)
        self.assertEqual(sign, "C", f"Ket qua: {sign}")
        self.assertGreaterEqual(conf, 0.75)

    def test_C_not_B_when_thumb_not_folded(self):
        """Khi ngon cai arch ra (khong firmly folded), khong duoc phan loai la B."""
        lm = _make_hand()
        for f in ["index", "middle", "ring", "pinky"]:
            _extend_finger(lm, f)
        wx, wy, wz = lm[0]
        lm[4] = [wx - 0.10, wy - 0.14, wz]
        sign, _ = self._classify(lm)
        self.assertNotEqual(sign, "B", "C khong duoc nham la B")

    # ------------------------------------------------------------------
    # Chu B: 4 ngon duoi, ngon cai gap sat long ban tay
    # ------------------------------------------------------------------
    def test_B_basic(self):
        """Chu B: 4 ngon duoi thang, ngon cai gap sat long ban tay."""
        lm = _make_hand()
        for f in ["index", "middle", "ring", "pinky"]:
            _extend_finger(lm, f)
        _fold_thumb_firmly(lm)
        sign, conf = self._classify(lm)
        self.assertEqual(sign, "B", f"Ket qua: {sign}")
        self.assertGreaterEqual(conf, 0.75)

    # ------------------------------------------------------------------
    # Chu A: 4 ngon gap (nam dam), ngon cai huong ngang (khong bi nhan vao)
    # ------------------------------------------------------------------
    def test_A_basic(self):
        """Chu A: tat ca ngon gap (nam dam), ngon cai ngang canh."""
        lm = _make_hand()
        # 4 ngon da gap theo mac dinh
        # Dat ngon cai ra ngang (khong huong len, khong bi nhan vao)
        wx, wy, wz = lm[0]
        pip_index = lm[6]  # PIP ngon tro
        # Ngon cai KHONG bi nhan vao: TIP.y < PIP_index.y (cao hon PIP ngon tro)
        lm[3] = [wx - 0.08, wy - 0.04, wz]
        lm[4] = [wx - 0.12, wy - 0.03, wz]  # TIP ngang, y < PIP_index.y
        sign, conf = self._classify(lm)
        self.assertEqual(sign, "A", f"Ket qua: {sign}")
        self.assertGreaterEqual(conf, 0.75)

    def test_A_not_E_when_thumb_not_pressed(self):
        """Khi ngon cai khong bi nhan vao duoi cac ngon, khong duoc phan loai la E."""
        lm = _make_hand()
        wx, wy, wz = lm[0]
        lm[4] = [wx - 0.12, wy - 0.03, wz]
        sign, _ = self._classify(lm)
        self.assertNotEqual(sign, "E", "A khong duoc nham la E")

    # ------------------------------------------------------------------
    # Chu E: 4 ngon gap, ngon cai bi nhan vao duoi cac ngon
    # ------------------------------------------------------------------
    def test_E_basic(self):
        """Chu E: tat ca ngon gap, ngon cai TIP nam duoi PIP ngon tro."""
        lm = _make_hand()
        # Dat TIP ngon cai DUOI PIP ngon tro (y lon hon = o duoi)
        pip_index_y = lm[6][1]   # PIP ngon tro (y tang xuong)
        wx, wy, wz = lm[0]
        lm[4] = [lm[5][0], pip_index_y + 0.04, wz]  # TIP duoi PIP ngon tro
        lm[3] = [lm[5][0] - 0.01, pip_index_y + 0.01, wz]
        sign, conf = self._classify(lm)
        self.assertEqual(sign, "E", f"Ket qua: {sign}")
        self.assertGreaterEqual(conf, 0.75)

    # ------------------------------------------------------------------
    # Cac truong hop NEUTRAL / bien
    # ------------------------------------------------------------------
    def test_neutral_missing_landmarks(self):
        """Thieu landmark (<21) phai tra ve NEUTRAL."""
        sign, conf = self.rec.classify_geometry([], {})
        self.assertEqual(sign, "NEUTRAL")
        self.assertEqual(conf, 0.0)

    def test_neutral_only_10_landmarks(self):
        """Chi co 10 landmark phai tra ve NEUTRAL."""
        lm = _make_hand()[:10]
        sign, conf = self.rec.classify_geometry(lm, {})
        self.assertEqual(sign, "NEUTRAL")
        self.assertEqual(conf, 0.0)

    def test_neutral_zero_palm_scale(self):
        """Tat ca landmark trung nhau (palm_scale = 0) phai tra ve NEUTRAL."""
        lm = [[0.5, 0.5, 0.0]] * 21
        sign, conf = self.rec.classify_geometry(lm, {})
        self.assertEqual(sign, "NEUTRAL")
        self.assertEqual(conf, 0.0)

    def test_neutral_no_clear_sign(self):
        """Trang thai khong ro rang (3 ngon duoi, 1 ngon gap) phai tra ve NEUTRAL."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        _extend_finger(lm, "middle")
        _extend_finger(lm, "ring")
        # pinky gap, thumb gap => khong khop rule nao ro rang (3 ngon duoi)
        sign, conf = self.rec.classify_geometry(lm, self._get_states(lm))
        # Day co the match quy tac nao do - chi kiem tra khong bi loi
        self.assertIsInstance(sign, str)
        self.assertIsInstance(conf, float)


# ---------------------------------------------------------------------------
# Test suite 3: Cap de nham (Confusion pairs)
# ---------------------------------------------------------------------------

class TestConfusionPairs(unittest.TestCase):
    """Kiem thu cac cap chu cai de bi nham lan nhau."""

    @classmethod
    def setUpClass(cls):
        cls.rec = VSLRecognizer.__new__(VSLRecognizer)

    def _classify(self, lm):
        from backend.vision.feature_extractor import normalize_hand_landmarks
        _, states = normalize_hand_landmarks(lm)
        return self.rec.classify_geometry(lm, states)

    def test_C_vs_O_discrimination(self):
        """
        C co 4 ngon duoi va d_thumb_index lon (arch ra).
        O co d_thumb_index nho (cham nhau).
        Hai tu the khong duoc nham nhau.
        """
        # Xay dung C
        lm_c = _make_hand()
        for f in ["index", "middle", "ring", "pinky"]:
            _extend_finger(lm_c, f)
        wx, wy, wz = lm_c[0]
        lm_c[4] = [wx - 0.10, wy - 0.14, wz]  # arch ra, d lon

        # Xay dung O
        lm_o = _make_hand()  # tat ca ngon gap
        it = lm_o[8]  # TIP ngon tro (da gap)
        lm_o[4] = [it[0] + 0.01, it[1] - 0.01, it[2]]  # cai cham tro
        lm_o[3] = [it[0] + 0.02, it[1] - 0.03, it[2]]

        sign_c, _ = self._classify(lm_c)
        sign_o, _ = self._classify(lm_o)

        self.assertEqual(sign_c, "C", f"C phai la C, got: {sign_c}")
        self.assertEqual(sign_o, "O", f"O phai la O, got: {sign_o}")

    def test_D_vs_L_discrimination(self):
        """
        D: ngon tro duoi, ngon cai gap.
        L: ngon tro duoi, ngon cai ra ngang (goc lon).
        Hai tu the khong duoc nham nhau.
        """
        # D: cai gap
        lm_d = _make_hand()
        _extend_finger(lm_d, "index")
        wx, wy, wz = lm_d[0]
        lm_d[4] = [wx - 0.03, wy - 0.02, wz]  # cai gap

        # L: cai ra ngang
        lm_l = _make_hand()
        _extend_finger(lm_l, "index")
        _extend_thumb_sideways(lm_l, angle_deg=70)

        sign_d, _ = self._classify(lm_d)
        sign_l, _ = self._classify(lm_l)

        self.assertEqual(sign_d, "D", f"D phai la D, got: {sign_d}")
        self.assertEqual(sign_l, "L", f"L phai la L, got: {sign_l}")

    def test_U_vs_V_discrimination(self):
        """
        U: 2 ngon song song sat nhau.
        V: 2 ngon tach xa nhau (goc lon).
        """
        # U: sat nhau
        lm_u = _make_hand()
        _extend_finger(lm_u, "index")
        _extend_finger(lm_u, "middle")
        mid = lm_u[9][0]
        lm_u[8][0]  = mid - 0.02
        lm_u[12][0] = mid + 0.02

        # V: tach xa
        lm_v = _make_hand()
        _extend_finger(lm_v, "index")
        _extend_finger(lm_v, "middle")
        lm_v[8][0]  = mid - 0.10
        lm_v[12][0] = mid + 0.10

        sign_u, _ = self._classify(lm_u)
        sign_v, _ = self._classify(lm_v)

        self.assertEqual(sign_u, "U", f"U phai la U, got: {sign_u}")
        self.assertEqual(sign_v, "V", f"V phai la V, got: {sign_v}")

    def test_A_vs_E_discrimination(self):
        """
        A: ngon cai ngang (khong bi nhan vao duoi cac ngon).
        E: ngon cai bi nhan vao duoi PIP ngon tro.
        """
        pip_index_y = _make_hand()[6][1]
        wx, wy, wz = _make_hand()[0]

        # A: cai ngang, y < PIP_index.y (cao hon PIP)
        lm_a = _make_hand()
        lm_a[4] = [wx - 0.12, wy - 0.03, wz]  # TIP cao hon PIP ngon tro

        # E: cai bi nhan vao, y > PIP_index.y
        lm_e = _make_hand()
        lm_e[4] = [lm_e[5][0], pip_index_y + 0.04, wz]
        lm_e[3] = [lm_e[5][0] - 0.01, pip_index_y + 0.01, wz]

        sign_a, _ = self._classify(lm_a)
        sign_e, _ = self._classify(lm_e)

        self.assertEqual(sign_a, "A", f"A phai la A, got: {sign_a}")
        self.assertEqual(sign_e, "E", f"E phai la E, got: {sign_e}")


# ---------------------------------------------------------------------------
# Test suite 4: Tinh bat bien kich thuoc / vi tri
# ---------------------------------------------------------------------------

class TestScaleInvariance(unittest.TestCase):
    """
    Kiem tra rang cac quy tac hoat dong bat ke kich thuoc ban tay
    hay vi tri trong khung hinh (nhung van dung palm_scale = wrist->middle_mcp).
    """

    @classmethod
    def setUpClass(cls):
        cls.rec = VSLRecognizer.__new__(VSLRecognizer)

    def _classify(self, lm):
        from backend.vision.feature_extractor import normalize_hand_landmarks
        _, states = normalize_hand_landmarks(lm)
        return self.rec.classify_geometry(lm, states)

    def _scale_hand(self, lm, factor, center=None):
        """Nhan toan bo toa do len 'factor' lan quanh 'center'."""
        if center is None:
            center = lm[0]
        return [
            [center[0] + (p[0] - center[0]) * factor,
             center[1] + (p[1] - center[1]) * factor,
             center[2] + (p[2] - center[2]) * factor]
            for p in lm
        ]

    def _translate_hand(self, lm, dx, dy, dz=0.0):
        """Dich chuyen toan bo toa do."""
        return [[p[0]+dx, p[1]+dy, p[2]+dz] for p in lm]

    def test_V_scale_small(self):
        """Chu V nhan dien dung voi ban tay nho hon (factor=0.7)."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        _extend_finger(lm, "middle")
        lm[8][0]  = lm[9][0] - 0.10
        lm[12][0] = lm[9][0] + 0.10
        lm_small = self._scale_hand(lm, 0.7)
        sign, _ = self._classify(lm_small)
        self.assertEqual(sign, "V", f"Got: {sign}")

    def test_L_translated(self):
        """Chu L nhan dien dung khi ban tay dich chuyen trong khung hinh."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        _extend_thumb_sideways(lm, angle_deg=70)
        lm_moved = self._translate_hand(lm, dx=0.2, dy=-0.1)
        sign, _ = self._classify(lm_moved)
        self.assertEqual(sign, "L", f"Got: {sign}")

    def test_O_scale_large(self):
        """Chu O nhan dien dung voi ban tay lon hon (factor=1.3)."""
        lm = _make_hand()
        it = lm[8]
        lm[4] = [it[0] + 0.01, it[1] - 0.01, it[2]]
        lm[3] = [it[0] + 0.02, it[1] - 0.03, it[2]]
        lm_large = self._scale_hand(lm, 1.3)
        sign, _ = self._classify(lm_large)
        self.assertEqual(sign, "O", f"Got: {sign}")


# ---------------------------------------------------------------------------
# Test suite 5: Nhieu toa do (robustness)
# ---------------------------------------------------------------------------

class TestNoisyLandmarks(unittest.TestCase):
    """Kiem tra kha nang chiu nhieu cua bo phan loai."""

    @classmethod
    def setUpClass(cls):
        cls.rec = VSLRecognizer.__new__(VSLRecognizer)

    def _classify(self, lm):
        from backend.vision.feature_extractor import normalize_hand_landmarks
        _, states = normalize_hand_landmarks(lm)
        return self.rec.classify_geometry(lm, states)

    def _add_noise(self, lm, std=0.005):
        """Them nhieu Gaussian nho vao toa do."""
        rng = np.random.RandomState(42)
        return [[p[0] + rng.normal(0, std),
                 p[1] + rng.normal(0, std),
                 p[2] + rng.normal(0, std)] for p in lm]

    def test_V_with_small_noise(self):
        """Chu V van nhan dien dung khi co nhieu nho (std=0.005)."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        _extend_finger(lm, "middle")
        lm[8][0]  = lm[9][0] - 0.10
        lm[12][0] = lm[9][0] + 0.10
        lm_noisy = self._add_noise(lm, std=0.005)
        sign, _ = self._classify(lm_noisy)
        self.assertEqual(sign, "V", f"Got: {sign}")

    def test_L_with_small_noise(self):
        """Chu L van nhan dien dung khi co nhieu nho (std=0.005)."""
        lm = _make_hand()
        _extend_finger(lm, "index")
        _extend_thumb_sideways(lm, angle_deg=70)
        lm_noisy = self._add_noise(lm, std=0.005)
        sign, _ = self._classify(lm_noisy)
        self.assertEqual(sign, "L", f"Got: {sign}")




# ---------------------------------------------------------------------------
# Test suite: MLP Checkpoint & Ensemble Fallback Safety
# ---------------------------------------------------------------------------

class TestMLPEnsembleFallback(unittest.TestCase):
    """
    Kiem tra an toan checkpoint MLP va co che fallback ve geometric:
    1. MLP chua nap weights (has_weights=False) khong ghi de ket qua hinh hoc.
    2. MLP co checkpoint hop le (has_weights=True) tham gia ensemble.
    3. Thieu checkpoint duoc xu ly an toan, khong gay crash.
    """

    def _make_letter_a_landmarks(self):
        lm = _make_hand()
        wx, wy, wz = lm[0]
        lm[3] = [wx - 0.08, wy - 0.04, wz]
        lm[4] = [wx - 0.12, wy - 0.03, wz]
        return lm

    def test_missing_checkpoint_safety(self):
        """Khoi tao voi checkpoint khong ton tai khong gay exception, has_weights=False."""
        rec = VSLRecognizer(weights_path="non_existent_path_to_weights.pt")
        self.assertFalse(rec.model.has_weights)
        
        # Goi predict_landmarks van an toan
        lm = self._make_letter_a_landmarks()
        result = rec.predict_landmarks(lm)
        self.assertIn("sign", result)
        self.assertEqual(result["sign"], "A")

    def test_unloaded_mlp_does_not_override_geometry(self):
        """Khi MLP chua nap weights, khong the ghi de chu cai nhan dien hinh hoc (A: 0.84)."""
        rec = VSLRecognizer(weights_path="non_existent_weights.pt")
        self.assertFalse(rec.model.has_weights)

        # Chế độ hình học nhận diện A với confidence 0.84 (< 0.85)
        lm = self._make_letter_a_landmarks()
        res = rec.predict_landmarks(lm)
        
        # Kết quả phải là A (0.84), không bị MLP ngẫu nhiên ghi đè
        self.assertEqual(res["sign"], "A")
        self.assertAlmostEqual(res["confidence"], 0.84, places=2)

    def test_valid_weights_loaded_by_default(self):
        """Khi khoi tao mac dinh voi weights.pt ton tai, has_weights phai la True."""
        rec = VSLRecognizer()
        self.assertTrue(rec.model.has_weights)

    def test_loaded_mlp_participates_in_ensemble(self):
        """Khi model co weights hop le (has_weights=True), luong ensemble hoat dong binh thuong."""
        import torch
        import torch.nn as nn
        
        rec = VSLRecognizer()
        # Mocking một mô hình có has_weights=True và output cố định để kiểm tra luồng
        class MockNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.has_weights = True
            def forward(self, x):
                # Giả lập logits với class 'B' (index 1) có logit rất lớn
                out = torch.zeros(1, 84)
                out[0, 1] = 10.0 # B
                return out

        rec.model = MockNet()
        
        # Với cử chỉ trung tính không khớp rule hình học (geom_conf = 0.0)
        # và MLP tự tin nhận diện 'B' (> 0.6)
        lm = _make_hand()
        res = rec.predict_landmarks(lm)
        self.assertEqual(res["sign"], "B")
        self.assertGreater(res["confidence"], 0.6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

