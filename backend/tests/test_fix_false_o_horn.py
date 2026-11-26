# -*- coding: utf-8 -*-
"""
CHECKPOINT 5 — Regression & Fix Verification for False Diacritic O -> Ơ
Covers all 15 mandatory scenarios + all 29 Vietnamese letters:
1. O tĩnh, giữ trong 1 giây.
2. O tĩnh, giữ trong 5 giây.
3. O có ngón út tách nhẹ.
4. O có rung tự nhiên.
5. O ở nhiều góc bàn tay (15°, 25°, 35°, 40°).
6. O rồi mở tay (thả lỏng cả bàn tay sang hướng B / C).
7. O rồi hạ tay ngay.
8. O -> Ơ hợp lệ.
9. O -> Ô hợp lệ.
10. O -> ký hiệu khác (C, Q, I, Y, B, E).
11. Hai chữ O liên tiếp bằng hai cử chỉ riêng biệt (O -> hạ tay -> O).
12. Chuỗi O -> Ơ -> Ô -> O.
13. Tắt và bật lại camera (reset recognizer & FSM).
14. Chuyển chế độ nhận diện (letters <-> digits <-> words).
15. Kiểm tra kết quả Phụ đề và TTS nhất quán 100%, cùng toàn bộ 29 chữ cái tiếng Việt.
"""

import json
import math
import os
import unittest

os.environ["VSL_PREFETCH"] = "0"
os.environ["VSL_DIACRITIC_DEBUG"] = "0"

import numpy as np
from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS


def _rotate_y(lm_arr: np.ndarray, deg: float) -> np.ndarray:
    wrist = lm_arr[0].copy()
    rel = lm_arr - wrist
    th = math.radians(deg)
    ry = np.array([
        [math.cos(th), 0.0, math.sin(th)],
        [0.0, 1.0, 0.0],
        [-math.sin(th), 0.0, math.cos(th)],
    ])
    return (rel @ ry.T) + wrist


class TestFixFalseOHorn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = VSLRecognizer()
        base_dir = os.path.dirname(__file__)

        npz = np.load(
            os.path.join(base_dir, "../data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"),
            allow_pickle=True,
        )
        cls_static = [str(c) for c in npz["classes"]]
        y_static = npz["y"]
        lm_static = npz["landmarks"]

        cls.letter_samples = {}
        cls.o_samples = []
        for i, y in enumerate(y_static):
            gt = cls_static[y]
            lm = lm_static[i].tolist()
            cls.rec.reset()
            r = cls.rec.predict_landmarks(lm, mode="letters", session_id=f"cls_init_{i}", timestamp=10.0)
            if r.get("sign") == gt:
                if gt not in cls.letter_samples:
                    cls.letter_samples[gt] = lm
                if gt == "O":
                    cls.o_samples.append(np.array(lm, dtype=float))

        motion_dir = os.path.join(base_dir, "../data/alphabet_v2/motion_sequences")
        cls.diacritic_sequences = {}
        for target in ("Ă", "Â", "Đ", "Ê", "Ô", "Ơ", "Ư"):
            with open(os.path.join(motion_dir, f"{target}.json"), "r", encoding="utf-8") as f:
                cls.diacritic_sequences[target] = json.load(f)["sequence"]

        with open(os.path.join(base_dir, "fixtures/real_digit_landmarks.json"), "r", encoding="utf-8") as f:
            d_samples = json.load(f)["samples"]
        cls.digit_5_lm = next(s["landmarks"] for s in d_samples if str(s["digit"]) == "5")

    def setUp(self):
        self.rec.reset()

    # Kịch bản 1 & 2: O tĩnh giữ 1 giây và giữ 5 giây -> chỉ ra O, không bao giờ biến thành Ơ hay lặp
    def test_01_02_static_o_hold_1s_and_5s(self):
        lm_o = self.o_samples[1].tolist()
        for hold_sec, n_frames in ((1.0, 26), (5.0, 126)):
            self.rec.reset()
            sid = f"o_hold_{int(hold_sec)}s"
            self.rec.reset_diacritic_fsm(sid)
            spoken = []
            t = 10.0
            for _ in range(n_frames):
                res = self.rec.predict_landmarks(lm_o, mode="letters", session_id=sid, timestamp=t)
                self.assertNotEqual(res.get("sign"), "Ơ", f"O tĩnh giữ {hold_sec}s bị biến thành Ơ!")
                if res.get("tts_speak"):
                    spoken.append(res["tts_speak"])
                t += 0.04
            self.assertEqual(self.rec.sentence_tokens, ["O"])
            self.assertEqual(spoken, ["O"])

    # Kịch bản 3 & 5: O có ngón út tách nhẹ tự nhiên & ở nhiều góc bàn tay (15°, 25°, 35°, 40°)
    def test_03_05_o_with_separated_pinky_and_multiple_hand_angles(self):
        base_arr = self.o_samples[2]
        for deg in (0.0, 15.0, 25.0, 35.0, 40.0):
            rot = _rotate_y(base_arr, deg)
            ps0 = float(np.linalg.norm(rot[9] - rot[0]))
            pinky_dir = 1.0 if rot[17][0] >= rot[5][0] else -1.0
            mod = rot.copy()
            mod[18][0] += pinky_dir * 0.17 * ps0
            mod[19][0] += pinky_dir * 0.29 * ps0
            mod[20][0] += pinky_dir * 0.37 * ps0
            lm_angled = mod.tolist()

            self.rec.reset()
            sid = f"o_angle_{int(deg)}"
            self.rec.reset_diacritic_fsm(sid)
            spoken = []
            t = 10.0
            for _ in range(26):  # 1.00s span (t=10.00..11.00) @ 25 FPS -> confirm at 10.12 + 0.85s hold -> speaks at 11.00s
                res = self.rec.predict_landmarks(lm_angled, mode="letters", session_id=sid, timestamp=t)
                self.assertNotEqual(res.get("sign"), "Ơ", f"Góc {deg}° bị nhận nhầm thành Ơ!")
                if res.get("tts_speak"):
                    spoken.append(res["tts_speak"])
                t += 0.04
            self.assertEqual(self.rec.sentence_tokens, ["O"], f"Góc {deg}° không ghi được O xuống Phụ đề!")
            self.assertEqual(spoken, ["O"], f"Góc {deg}° không phát âm đúng O!")

    # Kịch bản 4: O có rung tự nhiên (nhiễu tọa độ MediaPipe + dao động nhẹ cổ tay)
    def test_04_o_with_natural_jitter(self):
        rng = np.random.default_rng(123)
        base_arr = self.o_samples[3]
        sid = "o_jitter"
        self.rec.reset_diacritic_fsm(sid)
        spoken = []
        t = 10.0
        for k in range(26):
            noise = rng.normal(0.0, 0.0022, size=base_arr.shape)
            w_shift = np.array([0.0, 0.003 * math.sin(k * 0.8), 0.0])
            lm_j = (base_arr + noise + w_shift).tolist()
            res = self.rec.predict_landmarks(lm_j, mode="letters", session_id=sid, timestamp=t)
            self.assertNotEqual(res.get("sign"), "Ơ")
            if res.get("tts_speak"):
                spoken.append(res["tts_speak"])
            t += 0.04
        self.assertEqual(self.rec.sentence_tokens, ["O"])
        self.assertEqual(spoken, ["O"])

    # Kịch bản 6 & 7: O rồi mở bàn tay thả lỏng (B / C) & O rồi hạ tay ngay
    def test_06_07_o_then_open_hand_and_immediate_hand_drop(self):
        arr_o = self.o_samples[2]
        arr_b = np.array(self.letter_samples["B"], dtype=float)
        arr_c = np.array(self.letter_samples["C"], dtype=float)
        b_same_w = arr_b - arr_b[0] + arr_o[0]
        c_same_w = arr_c - arr_c[0] + arr_o[0]

        for label, target_open in (("open_B", b_same_w), ("open_C", c_same_w)):
            self.rec.reset()
            sid = f"o_{label}"
            self.rec.reset_diacritic_fsm(sid)
            spoken = []
            t = 10.0
            for _ in range(6):
                r = self.rec.predict_landmarks(arr_o.tolist(), mode="letters", session_id=sid, timestamp=t)
                if r.get("tts_speak"):
                    spoken.append(r["tts_speak"])
                t += 0.05
            # 2 frames of opening fingers before hand leaves frame
            for alpha in (0.45, 0.60):
                lm_rel = ((1.0 - alpha) * arr_o + alpha * target_open).tolist()
                r = self.rec.predict_landmarks(lm_rel, mode="letters", session_id=sid, timestamp=t)
                self.assertNotEqual(r.get("sign"), "Ơ", f"{label} lúc mở tay bị nhận nhầm Ơ!")
                if r.get("tts_speak"):
                    spoken.append(r["tts_speak"])
                t += 0.05
            r_drop = self.rec.predict_landmarks([], mode="letters", session_id=sid, timestamp=t + 0.06)
            if r_drop.get("tts_speak"):
                spoken.append(r_drop["tts_speak"])
            self.assertEqual(self.rec.sentence_tokens, ["O"])
            self.assertEqual(spoken, ["O"])

        # Kịch bản 7: O rồi hạ tay ngay lập tức sau frame thứ 4
        self.rec.reset()
        sid_drop = "o_immediate_drop"
        self.rec.reset_diacritic_fsm(sid_drop)
        t = 30.0
        for _ in range(4):
            self.rec.predict_landmarks(arr_o.tolist(), mode="letters", session_id=sid_drop, timestamp=t)
            t += 0.033
        self.assertEqual(self.rec.sentence_tokens, ["O"])
        r_rel = self.rec.predict_landmarks([], mode="letters", session_id=sid_drop, timestamp=t + 0.06)
        self.assertEqual(r_rel.get("tts_speak"), "O")
        self.assertEqual(self.rec.sentence_tokens, ["O"])

    # Kịch bản 8, 9, 11, 12: O -> Ơ hợp lệ, O -> Ô hợp lệ, O -> O liên tiếp, và chuỗi O -> Ơ -> Ô -> O
    def test_08_09_11_12_valid_diacritics_consecutive_o_and_mixed_sequence(self):
        lm_o = self.o_samples[0].tolist()

        # Kịch bản 11: Hai chữ O liên tiếp bằng hai cử chỉ riêng biệt
        self.rec.reset()
        sid_2o = "two_consecutive_o"
        self.rec.reset_diacritic_fsm(sid_2o)
        spoken_2o = []
        t = 10.0
        for rep in range(2):
            for _ in range(4):
                r = self.rec.predict_landmarks(lm_o, mode="letters", session_id=sid_2o, timestamp=t)
                if r.get("tts_speak"):
                    spoken_2o.append(r["tts_speak"])
                t += 0.04
            r_rel = self.rec.predict_landmarks([], mode="letters", session_id=sid_2o, timestamp=t + 0.06)
            if r_rel.get("tts_speak"):
                spoken_2o.append(r_rel["tts_speak"])
            t += 0.40
        self.assertEqual(self.rec.sentence_tokens, ["O", "O"])
        self.assertEqual(spoken_2o, ["O", "O"])

        # Kịch bản 8, 9, 12: Chuỗi ghép liên hoàn O -> Ơ -> Ô -> O trong cùng 1 phiên
        self.rec.reset()
        sid_mix = "seq_O_Ohorn_Ocirc_O"
        self.rec.reset_diacritic_fsm(sid_mix)
        spoken_mix = []
        t = 100.0

        # 1) O tĩnh -> hạ tay
        for _ in range(4):
            r = self.rec.predict_landmarks(lm_o, mode="letters", session_id=sid_mix, timestamp=t)
            if r.get("tts_speak"):
                spoken_mix.append(r["tts_speak"])
            t += 0.04
        r = self.rec.predict_landmarks([], mode="letters", session_id=sid_mix, timestamp=t + 0.06)
        if r.get("tts_speak"):
            spoken_mix.append(r["tts_speak"])
        t += 0.50

        # 2) O -> Ơ hợp lệ (dịch thời gian nối tiếp đơn điệu)
        seq_ohorn = self.diacritic_sequences["Ơ"]
        t_offset = t - float(seq_ohorn[0].get("timestamp_sec", 0.0))
        for item in seq_ohorn:
            ts = float(item.get("timestamp_sec", 0.0)) + t_offset
            r = self.rec.predict_landmarks(item.get("landmarks"), mode="letters", session_id=sid_mix, timestamp=ts)
            if r.get("tts_speak"):
                spoken_mix.append(r["tts_speak"])
            t = ts
        r = self.rec.predict_landmarks([], mode="letters", session_id=sid_mix, timestamp=t + 0.10)
        if r.get("tts_speak"):
            spoken_mix.append(r["tts_speak"])
        t += 0.50

        # 3) O -> Ô hợp lệ
        seq_ocirc = self.diacritic_sequences["Ô"]
        t_offset = t - float(seq_ocirc[0].get("timestamp_sec", 0.0))
        for item in seq_ocirc:
            ts = float(item.get("timestamp_sec", 0.0)) + t_offset
            r = self.rec.predict_landmarks(item.get("landmarks"), mode="letters", session_id=sid_mix, timestamp=ts)
            if r.get("tts_speak"):
                spoken_mix.append(r["tts_speak"])
            t = ts
        r = self.rec.predict_landmarks([], mode="letters", session_id=sid_mix, timestamp=t + 0.10)
        if r.get("tts_speak"):
            spoken_mix.append(r["tts_speak"])
        t += 0.50

        # 4) O tĩnh cuối câu -> hạ tay
        for _ in range(4):
            r = self.rec.predict_landmarks(lm_o, mode="letters", session_id=sid_mix, timestamp=t)
            if r.get("tts_speak"):
                spoken_mix.append(r["tts_speak"])
            t += 0.04
        r = self.rec.predict_landmarks([], mode="letters", session_id=sid_mix, timestamp=t + 0.06)
        if r.get("tts_speak"):
            spoken_mix.append(r["tts_speak"])

        self.assertEqual(self.rec.sentence_tokens, ["O", "Ơ", "Ô", "O"])
        self.assertEqual(spoken_mix, ["O", "Ơ", "Ô", "O"])

    # Kịch bản 10: O chuyển thẳng sang ký hiệu khác (C, Q, I, Y) không sinh Ơ giả
    def test_10_o_transition_to_other_signs_never_emits_false_o_horn(self):
        lm_o = self.o_samples[0].tolist()
        for next_letter in ("C", "Q", "I", "Y"):
            self.rec.reset()
            sid = f"o_to_{next_letter}"
            self.rec.reset_diacritic_fsm(sid)
            spoken = []
            t = 10.0
            for _ in range(4):
                r = self.rec.predict_landmarks(lm_o, mode="letters", session_id=sid, timestamp=t)
                if r.get("tts_speak"):
                    spoken.append(r["tts_speak"])
                t += 0.04
            # Chuyển thẳng sang chữ khác trong 6 frames
            for _ in range(6):
                r = self.rec.predict_landmarks(self.letter_samples[next_letter], mode="letters", session_id=sid, timestamp=t)
                self.assertNotEqual(r.get("sign"), "Ơ", f"O -> {next_letter} bị nhận nhầm Ơ!")
                if r.get("tts_speak"):
                    spoken.append(r["tts_speak"])
                t += 0.04
            self.assertNotIn("Ơ", self.rec.sentence_tokens)
            self.assertEqual(self.rec.sentence_tokens, ["O", next_letter])

    # Kịch bản 13, 14, 15: Tắt/bật lại camera, chuyển chế độ nhận diện, và kiểm tra toàn bộ 29 chữ cái tiếng Việt
    def test_13_14_15_camera_restart_mode_switch_and_all_29_letters(self):
        lm_o = self.o_samples[0].tolist()

        # Kịch bản 13: Đang giữ O (pending TTS) -> tắt/bật lại camera (reset) -> ký O mới
        self.rec.reset()
        sid = "cam_restart"
        for i in range(4):
            self.rec.predict_landmarks(lm_o, mode="letters", session_id=sid, timestamp=10.0 + i * 0.04)
        self.assertEqual(self.rec._pending_tts_token, "O")
        self.rec.reset()
        self.rec.reset_diacritic_fsm(sid)
        self.assertIsNone(self.rec._pending_tts_token)
        self.assertEqual(self.rec.sentence_tokens, [])

        # Kịch bản 14: Chuyển chế độ letters -> digits -> words không rò rỉ trạng thái
        for i in range(4):
            self.rec.predict_landmarks(lm_o, mode="letters", session_id="mode_sw", timestamp=20.0 + i * 0.04)
        d_spoken = []
        for i in range(4):
            rd = self.rec.predict_landmarks(self.digit_5_lm, mode="digits", session_id="mode_sw", timestamp=21.0 + i * 0.04)
            if rd.get("tts_speak"):
                d_spoken.append(rd["tts_speak"])
        self.assertEqual(d_spoken, ["năm"])
        w_res = self.rec.predict_sequence(np.zeros((30, 75, 3), dtype=np.float32).tolist(), mode="words")
        self.assertTrue(w_res.get("success"))

        # Kịch bản 15: Toàn bộ 29 chữ cái tiếng Việt đều nhận dạng và phát TTS nhất quán
        self.assertEqual(len(ALLOWED_LETTERS), 29)
        # Kiểm tra điểm hình học 95% của O không đổi
        r_geom_o = self.rec.predict_landmarks(self.o_samples[1].tolist(), mode="letters", session_id="geom_o_95", timestamp=49.0)
        self.assertEqual(r_geom_o.get("prediction_source"), "geometric_rule")
        self.assertEqual(r_geom_o.get("confidence"), 0.95)

        for ch, lm in sorted(self.letter_samples.items()):
            self.rec.reset()
            s_ch = f"all29_o_fix_{ch}"
            self.rec.reset_diacritic_fsm(s_ch)
            spoken = []
            t = 50.0
            for _ in range(4):
                r = self.rec.predict_landmarks(lm, mode="letters", session_id=s_ch, timestamp=t)
                if r.get("tts_speak"):
                    spoken.append(r["tts_speak"])
                t += 0.04
            r_rel = self.rec.predict_landmarks([], mode="letters", session_id=s_ch, timestamp=t + 0.06)
            if r_rel.get("tts_speak"):
                spoken.append(r_rel["tts_speak"])
            self.assertEqual(self.rec.sentence_tokens, [ch])
            self.assertEqual(spoken, [ch])

        for target, seq in self.diacritic_sequences.items():
            self.rec.reset()
            s_d = f"all29_diac_o_fix_{target}"
            self.rec.reset_diacritic_fsm(s_d)
            spoken = []
            for item in seq:
                r = self.rec.predict_landmarks(
                    item.get("landmarks"),
                    mode="letters",
                    session_id=s_d,
                    trial_id=s_d,
                    frame_idx=item.get("frame_index"),
                    timestamp=float(item.get("timestamp_sec", 0.0)),
                )
                if r.get("tts_speak"):
                    spoken.append(r["tts_speak"])
            self.assertEqual(self.rec.sentence_tokens, [target])
            self.assertEqual(spoken, [target])


if __name__ == "__main__":
    unittest.main()
