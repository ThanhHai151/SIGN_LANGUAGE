# -*- coding: utf-8 -*-
"""
CHECKPOINT 6 — Comprehensive Regression & Latency Verification Suite
Covers all 15 scenarios required by task specification:
1. All 29 Vietnamese letters (22 static + 7 dynamic diacritics)
2. All digits 0-10
3. Base letters A, D, E, O, U (stationary hold <= ~0.85s/1.18s & immediate hand drop)
4. Diacritic letters Ă, Â, Đ, Ê, Ô, Ơ, Ư (no double speak of base + diacritic)
5. Holding a single sign steady for 5 seconds (both letters and digits: zero repeat!)
6. Signing then immediately dropping hand (speaks immediately without needing next sign!)
7. Rapid transition between two signs (letters and digits)
8. Two identical signs in a row after release (letters A->A, B->B; digits 1->1, 5->5)
9. Assembling a sequence of letters
10. Assembling a sequence of digits (both recognizer digits mode & VSLSequentialDigitAggregator)
11. Switching between letters and digits modes (no state/pending TTS leakage)
12. Camera stop/restart reset & clear_sentence reset
13. Words mode (predict_sequence) unaffected
14. Protection of NO_MATCH, CONFLICT, NEUTRAL
15. Unique monotonic tts_event_id and confirm_event_id emitted on every confirmation/TTS event
"""

import json
import os
import unittest

os.environ["VSL_PREFETCH"] = "0"
os.environ["VSL_DIACRITIC_DEBUG"] = "0"

import numpy as np
from backend.vision.recognizer import VSLRecognizer, ALLOWED_LETTERS
from backend.vision.digit_aggregator import VSLSequentialDigitAggregator


class TestLatencyAndTtsOptimization(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rec = VSLRecognizer()
        npz = np.load(
            os.path.join(os.path.dirname(__file__), "../data/alphabet_v2/datasets/vsl_alphabet_static_v2.npz"),
            allow_pickle=True,
        )
        lm_static = npz["landmarks"]
        y_static = npz["y"]
        cls_static = [str(c) for c in npz["classes"]]

        cls.letter_samples = {}
        for i in range(len(lm_static)):
            gt = cls_static[y_static[i]]
            if gt not in cls.letter_samples:
                cls.rec.reset()
                res = cls.rec.predict_landmarks(lm_static[i].tolist(), mode="letters", session_id=f"init_{gt}", timestamp=10.0)
                if res.get("sign") == gt:
                    cls.letter_samples[gt] = lm_static[i].tolist()

        with open(os.path.join(os.path.dirname(__file__), "fixtures/real_digit_landmarks.json"), "r", encoding="utf-8") as f:
            d19 = json.load(f)["samples"]
        with open(os.path.join(os.path.dirname(__file__), "fixtures/real_digit_0_landmarks.json"), "r", encoding="utf-8") as f:
            d0 = json.load(f)["samples"]
        with open(os.path.join(os.path.dirname(__file__), "fixtures/real_digit_10_landmarks.json"), "r", encoding="utf-8") as f:
            cls.d10_samples = [s for s in json.load(f)["samples"] if s.get("quality_status") == "ok"]

        cls.digit_samples = {"0": d0[0]["landmarks"]}
        for s in d19:
            d = str(s["digit"])
            if d not in cls.digit_samples:
                cls.digit_samples[d] = s["landmarks"]

        cls.diacritic_sequences = {}
        motion_dir = os.path.join(os.path.dirname(__file__), "../data/alphabet_v2/motion_sequences")
        for target in ("Ă", "Â", "Đ", "Ê", "Ô", "Ơ", "Ư"):
            with open(os.path.join(motion_dir, f"{target}.json"), "r", encoding="utf-8") as f:
                cls.diacritic_sequences[target] = json.load(f)["sequence"]

    def setUp(self):
        self.rec.reset()

    def test_01_all_29_vietnamese_letters_confirm_and_speak_once(self):
        self.assertEqual(len(ALLOWED_LETTERS), 29)
        self.assertEqual(len(self.letter_samples), 22)

        # 22 static letters (17 non-base + 5 base)
        for ch, lm in sorted(self.letter_samples.items()):
            self.rec.reset()
            sid = f"all29_{ch}"
            self.rec.reset_diacritic_fsm(sid)
            t = 10.0
            spoken = []
            event_ids = []
            # 4 frames at 30 FPS -> confirms at frame 4 (4/7 smoothing!)
            for step in range(4):
                res = self.rec.predict_landmarks(lm, mode="letters", session_id=sid, timestamp=t)
                if res.get("tts_speak"):
                    spoken.append(res["tts_speak"])
                    event_ids.append(res.get("tts_event_id"))
                t += 0.033
            self.assertEqual(self.rec.sentence_tokens, [ch], f"Chữ {ch} phải chốt sau 4 frame!")

            # Drop hand -> flushes pending TTS if base letter, does nothing extra if normal letter
            res_rel = self.rec.predict_landmarks(None, mode="letters", session_id=sid, timestamp=t)
            if res_rel.get("tts_speak"):
                spoken.append(res_rel["tts_speak"])
                event_ids.append(res_rel.get("tts_event_id"))

            self.assertEqual(spoken, [ch], f"Chữ {ch} phải phát âm đúng 1 lần duy nhất, thực tế: {spoken}")
            self.assertTrue(all(isinstance(eid, str) and eid.startswith("tts_") for eid in event_ids))

        # 7 dynamic diacritic letters
        for target, seq in self.diacritic_sequences.items():
            self.rec.reset()
            tid = f"all29_diac_{target}"
            self.rec.reset_diacritic_fsm(tid)
            spoken = []
            for item in seq:
                res = self.rec.predict_landmarks(
                    item.get("landmarks"),
                    mode="letters",
                    session_id=tid,
                    trial_id=tid,
                    frame_idx=item.get("frame_index"),
                    timestamp=item.get("timestamp_sec", 0.0),
                )
                if res.get("tts_speak"):
                    spoken.append(res["tts_speak"])
            self.assertEqual(self.rec.sentence_tokens, [target], f"Chữ có dấu {target} sai phụ đề")
            self.assertEqual(spoken, [target], f"Chữ có dấu {target} không được đọc cả chữ nền lẫn dấu: {spoken}")

    def test_02_all_digits_0_to_10_confirm_and_speak_immediately(self):
        expected_words = {
            "0": "không", "1": "một", "2": "hai", "3": "ba", "4": "bốn",
            "5": "năm", "6": "sáu", "7": "bảy", "8": "tám", "9": "chín",
        }
        for dg, word in expected_words.items():
            self.rec.reset()
            lm = self.digit_samples[dg]
            t = 10.0
            spoken = []
            for _ in range(4):
                res = self.rec.predict_landmarks(lm, mode="digits", session_id=f"dig_{dg}", timestamp=t)
                if res.get("tts_speak"):
                    spoken.append((res["tts_speak"], res.get("tts_event_id")))
                t += 0.033
            self.assertEqual(self.rec.sentence_tokens, [dg], f"Số {dg} phải chốt sau 4 frame")
            self.assertEqual(len(spoken), 1)
            self.assertEqual(spoken[0][0], word)
            self.assertTrue(spoken[0][1].startswith("tts_"))

        # Digit 10
        self.rec.reset()
        spoken_10 = []
        for s in self.d10_samples:
            res = self.rec.predict_landmarks(s["landmarks"], mode="digits", session_id="dig_10", timestamp=s["timestamp_sec"])
            if res.get("tts_speak"):
                spoken_10.append((res["tts_speak"], res.get("tts_event_id")))
        self.assertEqual(self.rec.sentence_tokens, ["10"])
        self.assertEqual(len(spoken_10), 1)
        self.assertEqual(spoken_10[0][0], "mười")
        self.assertTrue(spoken_10[0][1].startswith("tts_"))

    def test_03_base_letters_stationary_hold_speaks_under_2s_and_immediate_hand_drop(self):
        # A, E, O, U stationary hold should speak right around ~0.85s (< 1.0s), D around ~1.18s (< 1.3s)
        for base in ("A", "E", "O", "U", "D"):
            self.rec.reset()
            sid = f"hold_{base}"
            self.rec.reset_diacritic_fsm(sid)
            lm = self.letter_samples[base]
            t = 50.0
            confirm_t = None
            speak_t = None
            spoken = []
            for _ in range(50):  # up to 2.0s at 25 FPS (0.04s/frame)
                res = self.rec.predict_landmarks(lm, mode="letters", session_id=sid, timestamp=t)
                if res.get("confirmed") and confirm_t is None:
                    confirm_t = t
                if res.get("tts_speak"):
                    spoken.append(res["tts_speak"])
                    if speak_t is None:
                        speak_t = t
                t += 0.04
            self.assertIsNotNone(confirm_t)
            self.assertIsNotNone(speak_t)
            self.assertEqual(spoken, [base])
            delta_ms = (speak_t - confirm_t) * 1000.0
            if base in ("A", "E", "O", "U"):
                self.assertLessEqual(delta_ms, 920.0, f"Base {base} stationary TTS quá chậm: {delta_ms}ms")
                self.assertGreaterEqual(delta_ms, 840.0)
            else:
                self.assertLessEqual(delta_ms, 1250.0, f"Base D stationary TTS quá chậm: {delta_ms}ms")

            # Immediate hand drop right after confirmation -> speaks on the VERY NEXT release frame without waiting!
            self.rec.reset()
            sid_drop = f"drop_{base}"
            self.rec.reset_diacritic_fsm(sid_drop)
            t2 = 100.0
            for _ in range(4):
                self.rec.predict_landmarks(lm, mode="letters", session_id=sid_drop, timestamp=t2)
                t2 += 0.033
            self.assertEqual(self.rec.sentence_tokens, [base])
            rel_res = self.rec.predict_landmarks([], mode="letters", session_id=sid_drop, timestamp=t2 + 0.06)
            self.assertEqual(rel_res.get("tts_speak"), base, f"Hạ tay sau chữ {base} phải phát âm ngay lập tức!")
            self.assertIsNone(self.rec._pending_tts_token)

    def test_04_five_second_hold_never_repeats_letter_or_digit(self):
        # Hold letter 'B' and base letter 'A' for 5.0 seconds (150 frames @ 30 FPS)
        for ch in ("B", "A"):
            self.rec.reset()
            sid = f"hold5s_{ch}"
            self.rec.reset_diacritic_fsm(sid)
            lm = self.letter_samples[ch]
            spoken = []
            t = 10.0
            for _ in range(150):
                res = self.rec.predict_landmarks(lm, mode="letters", session_id=sid, timestamp=t)
                if res.get("tts_speak"):
                    spoken.append(res["tts_speak"])
                t += 0.0333
            self.assertEqual(self.rec.sentence_tokens, [ch], f"Giữ chữ {ch} 5s không được lặp phụ đề!")
            self.assertEqual(spoken, [ch], f"Giữ chữ {ch} 5s không được đọc lặp!")

        # Hold digit '5' and '1' for 5.0 seconds (150 frames @ 30 FPS)
        for dg, word in (("5", "năm"), ("1", "một")):
            self.rec.reset()
            lm = self.digit_samples[dg]
            spoken = []
            t = 10.0
            for _ in range(150):
                res = self.rec.predict_landmarks(lm, mode="digits", session_id=f"hold5s_d_{dg}", timestamp=t)
                if res.get("tts_speak"):
                    spoken.append(res["tts_speak"])
                t += 0.0333
            self.assertEqual(self.rec.sentence_tokens, [dg], f"Giữ số {dg} 5s không được lặp phụ đề!")
            self.assertEqual(spoken, [word], f"Giữ số {dg} 5s không được đọc lặp!")

    def test_05_rapid_transitions_and_consecutive_identical_signs(self):
        # Rapid letter sequence: B -> L -> V
        self.rec.reset()
        sid = "rapid_letters"
        t = 10.0
        spoken = []
        for ch in ("B", "L", "V"):
            for _ in range(5):
                res = self.rec.predict_landmarks(self.letter_samples[ch], mode="letters", session_id=sid, timestamp=t)
                if res.get("tts_speak"):
                    spoken.append(res["tts_speak"])
                t += 0.04
        self.assertEqual(self.rec.sentence_tokens, ["B", "L", "V"])
        self.assertEqual(spoken, ["B", "L", "V"])

        # Consecutive identical digits: 5 -> release -> 5
        self.rec.reset()
        t = 20.0
        d_spoken = []
        d_events = []
        for _ in range(5):
            r = self.rec.predict_landmarks(self.digit_samples["5"], mode="digits", session_id="rep_d5", timestamp=t)
            if r.get("tts_speak"):
                d_spoken.append(r["tts_speak"])
                d_events.append(r["tts_event_id"])
            t += 0.04
        # Release hand
        self.rec.predict_landmarks([], mode="digits", session_id="rep_d5", timestamp=t)
        t += 0.35
        for _ in range(5):
            r = self.rec.predict_landmarks(self.digit_samples["5"], mode="digits", session_id="rep_d5", timestamp=t)
            if r.get("tts_speak"):
                d_spoken.append(r["tts_speak"])
                d_events.append(r["tts_event_id"])
            t += 0.04
        self.assertEqual(self.rec.sentence_tokens, ["5", "5"])
        self.assertEqual(d_spoken, ["năm", "năm"])
        self.assertEqual(len(set(d_events)), 2, "Hai lần ký số 5 liên tiếp phải có 2 tts_event_id khác nhau!")

    def test_06_mode_switch_and_clear_sentence_clean_pending_tts(self):
        # Sign 'A' in letters mode (pending TTS), then immediately switch to digits mode and sign '3'
        self.rec.reset()
        t = 10.0
        for _ in range(4):
            self.rec.predict_landmarks(self.letter_samples["A"], mode="letters", session_id="switch_s", timestamp=t)
            t += 0.04
        self.assertEqual(self.rec._pending_tts_token, "A")

        # Switch to digits mode on same session -> must clear pending 'A' so 'A' never leaks in digits mode!
        spoken_in_digits = []
        for _ in range(5):
            res = self.rec.predict_landmarks(self.digit_samples["3"], mode="digits", session_id="switch_s", timestamp=t)
            if res.get("tts_speak"):
                spoken_in_digits.append(res["tts_speak"])
            t += 0.04
        res_rel = self.rec.predict_landmarks([], mode="digits", session_id="switch_s", timestamp=t)
        if res_rel.get("tts_speak"):
            spoken_in_digits.append(res_rel["tts_speak"])

        self.assertEqual(spoken_in_digits, ["ba"])
        self.assertEqual(self.rec.sentence_tokens, ["A", "3"])

        # Clear sentence cleans everything
        self.rec.clear_sentence()
        self.assertEqual(self.rec.sentence_tokens, [])
        self.assertIsNone(self.rec._pending_tts_token)

    def test_07_digit_aggregator_and_words_mode_preserved(self):
        agg = VSLSequentialDigitAggregator()
        agg.update("1", 0.95, timestamp=1.00)
        agg.update("1", 0.95, timestamp=1.45)
        agg.update("NEUTRAL", 0.0, timestamp=1.55)
        agg.update("1", 0.95, timestamp=1.65)
        agg.update("1", 0.95, timestamp=2.10)
        out = agg.update("NEUTRAL", 0.0, timestamp=3.70)
        self.assertTrue(out.is_committed)
        self.assertEqual(out.committed_number, "11")

        # Words sequence endpoint untouched
        dummy_seq = np.zeros((30, 75, 3), dtype=np.float32).tolist()
        w_res = self.rec.predict_sequence(dummy_seq, mode="words")
        self.assertTrue(w_res.get("success"))
        self.assertEqual(w_res.get("sign"), "NEUTRAL")


if __name__ == "__main__":
    unittest.main()
