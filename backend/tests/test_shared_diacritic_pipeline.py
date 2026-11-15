"""Unit và integration tests cho Shared Diacritic Pipeline (Checkpoints A-D)."""

import unittest
from unittest.mock import patch

from backend.vision.diacritic_state_machine import (
    DiacriticStateMachine,
    STATE_IDLE,
    STATE_BASE_CANDIDATE,
    STATE_BASE_STABLE,
    STATE_DIACRITIC_CANDIDATE,
    STATE_DIACRITIC_CONFIRMED,
    STATE_COOLDOWN,
)
from backend.vision.recognizer import VSLRecognizer


def make_hand_landmarks(
    wrist_y: float = 0.88,
    palm_scale: float = 0.22,
    extend_index: bool = False,
    extend_middle: bool = False,
    extend_pinky: bool = False,
    extend_thumb: bool = False,
    index_bent_down: bool = False,
    index_curl_scale: float = 1.0,
):
    """Tạo 21 landmarks hợp lệ theo hình học giải phẫu bàn tay (gập chặt vs duỗi thẳng)."""
    pts = [[0.5, wrist_y, 0.0] for _ in range(21)]
    s = palm_scale

    # Thumb: 1..4 (mặc định áp nhẹ, hoặc bung ngang Shaka Y khi extend_thumb=True)
    pts[1] = [0.5 - 0.16 * s, wrist_y - 0.18 * s, 0.0]
    pts[2] = [0.5 - 0.28 * s, wrist_y - 0.40 * s, 0.0]
    if extend_thumb:
        pts[3] = [0.5 - 0.65 * s, wrist_y - 0.55 * s, 0.0]
        pts[4] = [0.5 - 0.98 * s, wrist_y - 0.65 * s, 0.0]
    else:
        pts[3] = [0.5 - 0.30 * s, wrist_y - 0.60 * s, 0.0]
        pts[4] = [0.5 - 0.22 * s, wrist_y - 0.72 * s, 0.0]

    # Index: 5..8
    pts[5] = [0.5 - 0.18 * s, wrist_y - 0.88 * s, 0.0]
    if extend_index:
        pts[6] = [0.5 - 0.19 * s, wrist_y - 1.28 * s, 0.0]
        pts[7] = [0.5 - 0.20 * s, wrist_y - 1.56 * s, 0.0]
        pts[8] = [0.5 - 0.20 * s, wrist_y - 1.82 * s, 0.0]
    elif index_bent_down:
        pts[6] = [0.5 - 0.19 * s, wrist_y - 1.15 * s, 0.0]
        pts[7] = [0.5 - 0.15 * s, wrist_y - 0.85 * s, 0.0]
        pts[8] = [0.5 - 0.10 * s, wrist_y - 0.65 * s, 0.0]
    else:
        c = index_curl_scale
        pts[6] = [0.5 - 0.18 * s, wrist_y - (1.05 + 0.12 * (1.0 - c)) * s, 0.0]
        pts[7] = [0.5 - 0.18 * s, wrist_y - (0.82 + 0.28 * (1.0 - c)) * s, 0.0]
        pts[8] = [0.5 - 0.18 * s, wrist_y - (0.64 + 0.42 * (1.0 - c)) * s, 0.0]

    # Middle: 9..12
    pts[9] = [0.5 + 0.00 * s, wrist_y - 0.94 * s, 0.0]
    if extend_middle:
        pts[10] = [0.5 + 0.00 * s, wrist_y - 1.38 * s, 0.0]
        pts[11] = [0.5 + 0.00 * s, wrist_y - 1.66 * s, 0.0]
        pts[12] = [0.5 + 0.00 * s, wrist_y - 1.92 * s, 0.0]
    else:
        pts[10] = [0.5 + 0.00 * s, wrist_y - 1.08 * s, 0.0]
        pts[11] = [0.5 + 0.00 * s, wrist_y - 0.84 * s, 0.0]
        pts[12] = [0.5 + 0.00 * s, wrist_y - 0.66 * s, 0.0]

    # Ring: 13..16 (cuộn gọn)
    pts[13] = [0.5 + 0.18 * s, wrist_y - 0.88 * s, 0.0]
    pts[14] = [0.5 + 0.18 * s, wrist_y - 1.04 * s, 0.0]
    pts[15] = [0.5 + 0.18 * s, wrist_y - 0.82 * s, 0.0]
    pts[16] = [0.5 + 0.18 * s, wrist_y - 0.66 * s, 0.0]

    # Pinky: 17..20
    pts[17] = [0.5 + 0.34 * s, wrist_y - 0.78 * s, 0.0]
    if extend_pinky:
        pts[18] = [0.5 + 0.35 * s, wrist_y - 1.10 * s, 0.0]
        pts[19] = [0.5 + 0.36 * s, wrist_y - 1.34 * s, 0.0]
        pts[20] = [0.5 + 0.37 * s, wrist_y - 1.56 * s, 0.0]
    else:
        pts[18] = [0.5 + 0.34 * s, wrist_y - 0.94 * s, 0.0]
        pts[19] = [0.5 + 0.34 * s, wrist_y - 0.76 * s, 0.0]
        pts[20] = [0.5 + 0.34 * s, wrist_y - 0.62 * s, 0.0]
    return pts


class TestCheckpointAWholeHandExitAndLifecycle(unittest.TestCase):
    def test_low_in_frame_active_hand_does_not_trigger_exit(self):
        """Bàn tay đặt thấp (wrist_y=0.95) nhưng bàn tay/ngón vẫn dựng signing không bị EXIT Guard."""
        fsm = DiacriticStateMachine(min_base_stable_frames=3, min_base_stable_sec=0.10)
        t = 10.0
        for i in range(4):
            lm = make_hand_landmarks(wrist_y=0.95 + (0.004 if i % 2 == 0 else -0.004))
            res = fsm.update("A", 0.95, lm, timestamp=t)
            self.assertNotIn("EXIT Guard", res.get("reason", ""))
            t += 0.06
        self.assertEqual(fsm.state, STATE_BASE_STABLE)
        self.assertIsNotNone(fsm.baseline)

    def test_whole_hand_plunge_and_sustained_exit_reset_fsm(self):
        """Rơi toàn bàn tay thật sự (đột ngột hoặc kéo dài >= 2 frame) phải kích hoạt EXIT Guard."""
        fsm = DiacriticStateMachine(min_base_stable_frames=3, min_base_stable_sec=0.10)
        t = 20.0
        for _ in range(4):
            fsm.update("A", 0.95, make_hand_landmarks(wrist_y=0.55), timestamp=t)
            t += 0.05
        self.assertEqual(fsm.state, STATE_BASE_STABLE)

        # Rơi kéo dài 2 frame xuống dưới khung hình
        fsm.update("A", 0.95, make_hand_landmarks(wrist_y=0.68), timestamp=t)
        t += 0.10
        res_exit = fsm.update("A", 0.95, make_hand_landmarks(wrist_y=0.84), timestamp=t)
        self.assertEqual(res_exit["state"], STATE_IDLE)
        self.assertIn("EXIT Guard", res_exit["reason"])
        self.assertIsNone(fsm.baseline)

    def test_light_single_frame_downward_jitter_does_not_exit(self):
        """Rung nhẹ 1 frame đi xuống vừa phải rồi bật lại không được kích hoạt EXIT Guard."""
        fsm = DiacriticStateMachine(min_base_stable_frames=3, min_base_stable_sec=0.10)
        t = 30.0
        for _ in range(4):
            fsm.update("A", 0.95, make_hand_landmarks(wrist_y=0.86), timestamp=t)
            t += 0.18
        self.assertEqual(fsm.state, STATE_BASE_STABLE)

        # Jitter 1 frame v_pcy ~ +0.65/s (0.12 trong 0.185s), rồi giữ ổn định
        res_j1 = fsm.update("A", 0.95, make_hand_landmarks(wrist_y=0.98), timestamp=t)
        self.assertNotIn("EXIT Guard", res_j1.get("reason", ""))
        self.assertEqual(fsm.state, STATE_BASE_STABLE)

        t += 0.18
        res_j2 = fsm.update("A", 0.95, make_hand_landmarks(wrist_y=0.91), timestamp=t)
        self.assertNotIn("EXIT Guard", res_j2.get("reason", ""))
        self.assertEqual(fsm.state, STATE_BASE_STABLE)

    def test_lost_landmarks_commits_pending_tts_and_clears_replaceable(self):
        """Mất bàn tay (None/empty) phải reset FSM, commit pending TTS và xóa _replaceable_base_token."""
        rec = VSLRecognizer(smoothing_window=6)
        mock_a = {"letter": "A", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        t = 40.0
        lm_low_a = make_hand_landmarks(wrist_y=0.90)

        with patch.object(rec.alphabet_classifier, "classify_detailed", return_value=mock_a):
            for _ in range(4):
                rec.predict_landmarks(lm_low_a, mode="letters", session_id="s_low", timestamp=t)
                t += 0.10

        self.assertIsNotNone(rec._replaceable_base_token)
        self.assertEqual(rec._pending_tts_token, "A")
        self.assertEqual(rec.sentence_tokens[-1], "A")

        # Giữ thêm vài frame thấp ở wrist_y=0.95 không được làm mất _replaceable_base_token
        with patch.object(rec.alphabet_classifier, "classify_detailed", return_value=mock_a):
            res_low = rec.predict_landmarks(
                make_hand_landmarks(wrist_y=0.95), mode="letters", session_id="s_low", timestamp=t + 0.10
            )
            self.assertIsNone(res_low.get("tts_speak"))
        self.assertIsNotNone(rec._replaceable_base_token)

        # Mất landmarks hoàn toàn -> commit TTS "A", xóa replaceable, reset FSM
        res_lost = rec.predict_landmarks(None, mode="letters", session_id="s_low", timestamp=t + 0.25)
        self.assertEqual(res_lost.get("tts_speak"), "A")
        self.assertIsNone(rec._replaceable_base_token)
        self.assertIsNone(rec._pending_tts_token)
        self.assertEqual(rec.diacritic_fsms["s_low"].state, STATE_IDLE)

    def test_low_in_frame_replaces_base_token_with_diacritic(self):
        """Khi tay đặt thấp trong khung hình, A -> Ă vẫn giữ _replaceable_base_token và thay thế tại chỗ."""
        rec = VSLRecognizer(smoothing_window=6)
        mock_a = {"letter": "A", "confidence": 0.95, "status": "MATCH", "source": "geometric_rule"}
        mock_n = {"letter": "UNKNOWN", "confidence": 0.0, "status": "NO_MATCH", "source": "none"}
        t = 50.0
        session_id = "s_replace_low"

        # 4 frame A nắm gọn ở wrist_y=0.92
        lm_a = make_hand_landmarks(wrist_y=0.92, extend_index=False, extend_middle=False)
        with patch.object(rec.alphabet_classifier, "classify_detailed", return_value=mock_a):
            for _ in range(4):
                rec.predict_landmarks(lm_a, mode="letters", session_id=session_id, timestamp=t)
                t += 0.15

        self.assertEqual(rec.sentence_tokens, ["A"])
        self.assertIsNotNone(rec._replaceable_base_token)

        # Duỗi đơn ngón trỏ (ngón giữa giữ gập) tạo dấu Ă trong khi wrist_y vẫn 0.95 (rất thấp)
        lm_a_breve = make_hand_landmarks(wrist_y=0.95, extend_index=True, extend_middle=False)
        with patch.object(rec.alphabet_classifier, "classify_detailed", return_value=mock_n):
            res_1 = rec.predict_landmarks(lm_a_breve, mode="letters", session_id=session_id, timestamp=t)
            t += 0.15
            res_2 = rec.predict_landmarks(lm_a_breve, mode="letters", session_id=session_id, timestamp=t)

        self.assertEqual(res_2["sign"], "Ă")
        self.assertEqual(res_2["replaced_token"], "A")
        self.assertEqual(rec.sentence_tokens, ["Ă"])
        self.assertEqual(res_2.get("tts_speak"), "Ă")


class TestCheckpointBMultiFrameBaselineAndEntryGuard(unittest.TestCase):
    def test_multiframe_baseline_captures_pre_motion_curl_across_candidates(self):
        """BaselineSnapshot đa frame giữ được độ gập tiền chuyển động dù frame cuối bắt đầu thả lỏng nhẹ."""
        fsm = DiacriticStateMachine(min_base_stable_frames=3, min_base_stable_sec=0.10)
        f1 = make_hand_landmarks(wrist_y=0.75, extend_index=False)
        f2 = make_hand_landmarks(wrist_y=0.75, extend_index=False)
        f3 = make_hand_landmarks(wrist_y=0.75, extend_index=True)  # bắt đầu duỗi mở trỏ ở frame 3
        f1_curl = fsm._extract_frame_features(f1, 1.0).index_curl
        f3_curl = fsm._extract_frame_features(f3, 2.0).index_curl
        fsm.reset()
        self.assertGreater(f1_curl, f3_curl)

        fsm.update("A", 0.95, f1, timestamp=1.00)
        fsm.update("A", 0.95, f2, timestamp=1.08)
        res = fsm.update("A", 0.95, f3, timestamp=1.16)

        self.assertEqual(res["state"], STATE_BASE_STABLE)
        self.assertIsNotNone(fsm.baseline)
        self.assertEqual(fsm.baseline.sample_count, 3)
        self.assertGreater(fsm.baseline.index_curl, f3_curl)

    def test_rising_hand_entry_settles_before_baseline_no_false_circumflex(self):
        """Đưa tay từ dưới lên liên tiếp rồi dừng lại ở chữ A không được chụp baseline sớm gây Â giả."""
        fsm = DiacriticStateMachine(min_base_stable_frames=3, min_base_stable_sec=0.10)
        ys = [0.99, 0.96, 0.92, 0.89, 0.87, 0.867, 0.865]
        ts = [0.00, 0.06, 0.12, 0.18, 0.36, 0.54, 0.72]
        confirmed = []
        for y, t in zip(ys, ts):
            res = fsm.update("A", 0.95, make_hand_landmarks(wrist_y=y), timestamp=t)
            if res["state"] == STATE_DIACRITIC_CONFIRMED:
                confirmed.append(res["final_label"])
        self.assertEqual(confirmed, [])
        self.assertEqual(fsm.state, STATE_BASE_STABLE)

    def test_all_five_base_groups_static_and_shaka_no_false_diacritics(self):
        """Kiểm tra A, E, O, U, D giữ tĩnh và O/U -> Y (Shaka) không kích hoạt sai dấu."""
        for base in ("A", "E", "O", "U", "D"):
            fsm = DiacriticStateMachine(min_base_stable_frames=3, min_base_stable_sec=0.10)
            lm = make_hand_landmarks(wrist_y=0.70, extend_index=(base == "D"))
            t = 5.0
            for _ in range(10):
                res = fsm.update(base, 0.95, lm, timestamp=t)
                self.assertIsNone(res.get("diacritic_detected"), f"Static {base} falsely triggered {res}")
                t += 0.10

        # O -> Y và U -> Y phải không bật Ơ/Ư
        for base in ("O", "U"):
            fsm = DiacriticStateMachine(min_base_stable_frames=3, min_base_stable_sec=0.10)
            t = 10.0
            for _ in range(4):
                fsm.update(base, 0.95, make_hand_landmarks(wrist_y=0.70), timestamp=t)
                t += 0.06
            self.assertEqual(fsm.state, STATE_BASE_STABLE)
            lm_y = make_hand_landmarks(wrist_y=0.70, extend_thumb=True, extend_pinky=True)
            for _ in range(5):
                res = fsm.update("Y", 0.95, lm_y, timestamp=t)
                self.assertNotIn(res.get("diacritic_detected"), ("Ơ", "Ư"))
                t += 0.06

    def test_checkpoint_c_multi_fps_diacritic_timing(self):
        """Shared Diacritic Timing (3 frames / 0.10s base; 1.50s / 35 frames search) hoạt động ở 5..30 FPS."""
        lm_base_a = make_hand_landmarks(wrist_y=0.72)
        lm_breve = make_hand_landmarks(wrist_y=0.72, extend_index=True)
        lm_base_d = make_hand_landmarks(wrist_y=0.72, extend_index=True)
        lm_d_stroke = make_hand_landmarks(wrist_y=0.72, extend_index=False, index_bent_down=True)

        for fps in (5, 10, 15, 24, 30):
            dt = 1.0 / fps
            # A -> Ă: 0.70s giữ A + 0.45s mở ngón trỏ (tối thiểu >= 2 frames ở mọi FPS từ 5..30)
            fsm_a = DiacriticStateMachine(
                min_base_stable_frames=3,
                min_base_stable_sec=0.10,
                max_diacritic_search_sec=1.50,
                max_diacritic_search_frames=35,
            )
            n_base = max(4, int(round(0.70 * fps)))
            n_mot = max(2, int(round(0.45 * fps)))
            t = 10.0
            got_a = []
            for _ in range(n_base):
                res = fsm_a.update("A", 0.95, lm_base_a, timestamp=t)
                if res.get("state") == STATE_DIACRITIC_CONFIRMED:
                    got_a.append(res["final_label"])
                t += dt
            self.assertEqual(fsm_a.state, STATE_BASE_STABLE, f"Failed BASE_STABLE at {fps}fps")
            for _ in range(n_mot):
                res = fsm_a.update("A", 0.95, lm_breve, timestamp=t)
                if res.get("state") == STATE_DIACRITIC_CONFIRMED:
                    got_a.append(res["final_label"])
                t += dt
            self.assertIn("Ă", got_a, f"Missing Ă confirmation at {fps}fps")

            # D -> Đ: 1-frame confirm trên toàn bộ 5..30 FPS
            fsm_d = DiacriticStateMachine(
                min_base_stable_frames=3,
                min_base_stable_sec=0.10,
                max_diacritic_search_sec=1.50,
                max_diacritic_search_frames=35,
            )
            t = 20.0
            for _ in range(n_base):
                fsm_d.update("D", 0.95, lm_base_d, timestamp=t)
                t += dt
            res_d = fsm_d.update("D", 0.95, lm_d_stroke, timestamp=t)
            self.assertEqual(res_d.get("final_label"), "Đ", f"Missing Đ confirmation at {fps}fps")

    def test_checkpoint_c_duplicate_switch_ts_and_slow_drift_reanchor(self):
        """Retry trùng timestamp không đếm đúp switch, và giữ tĩnh trôi chậm nhiều giây không tạo Â giả."""
        fsm = DiacriticStateMachine(
            min_base_stable_frames=3,
            min_base_stable_sec=0.10,
            max_diacritic_search_sec=1.50,
            max_diacritic_search_frames=35,
        )
        lm_a = make_hand_landmarks(wrist_y=0.75)
        t = 1.0
        for _ in range(4):
            fsm.update("A", 0.95, lm_a, timestamp=t)
            t += 0.10
        self.assertEqual(fsm.state, STATE_BASE_STABLE)

        # Retry đúng cùng timestamp không được làm nhảy count lên 2 -> hủy BASE_STABLE
        fsm.update("B", 0.90, make_hand_landmarks(wrist_y=0.75), timestamp=t)
        self.assertEqual(fsm.switch_candidate_count, 1)
        fsm.update("B", 0.90, make_hand_landmarks(wrist_y=0.75), timestamp=t)
        self.assertEqual(fsm.switch_candidate_count, 1)
        self.assertEqual(fsm.current_base, "A")

        # Quay lại A và trôi rất chậm lên trên 0.045 trong suốt 4.5s -> re-anchor, không phát Â giả
        t += 0.10
        for step in range(45):
            y = 0.75 - step * 0.001
            res = fsm.update("A", 0.95, make_hand_landmarks(wrist_y=y), timestamp=t)
            self.assertIsNone(res.get("diacritic_detected"), f"Slow posture drift triggered {res}")
            t += 0.10

    def test_checkpoint_c_tts_deferred_during_diacritic_candidate_and_released(self):
        """Hoãn TTS chữ nền khi đang DIACRITIC_CANDIDATE (tối đa 1.6s), tĩnh ở BASE_STABLE vẫn phát ở 1.2s."""
        rec = VSLRecognizer.__new__(VSLRecognizer)
        rec.prediction_history = []
        rec._pending_tts_token = "A"
        rec._uncommitted_base_time = 10.0
        rec._replaceable_base_token = {"base": "A", "token_index": 0, "timestamp": 10.0}
        rec._neutral_release_count = 0
        rec._seen_neutral_release = False
        rec._last_confirmed_diacritic = None
        rec._exit_trackers = {}

        lm_ok = make_hand_landmarks(wrist_y=0.70)

        # 1.25s nhưng đang DIACRITIC_CANDIDATE -> hoãn TTS để nhường dấu xác nhận
        _, tts1 = rec._update_gesture_release_and_tts(
            final_sign="A",
            p_prim=lm_ok,
            current_clk=11.25,
            release_key="s1",
            recent_signs=["A"],
            fsm_state=STATE_DIACRITIC_CANDIDATE,
            fsm_res_state=STATE_DIACRITIC_CANDIDATE,
        )
        self.assertIsNone(tts1)
        self.assertEqual(rec._pending_tts_token, "A")

        # Vừa chạm DIACRITIC_CONFIRMED -> không xả "A" trước khi replacement block chuyển thành "Ă"
        _, tts2 = rec._update_gesture_release_and_tts(
            final_sign="Ă",
            p_prim=lm_ok,
            current_clk=11.30,
            release_key="s1",
            recent_signs=["Ă"],
            fsm_state=STATE_COOLDOWN,
            fsm_res_state=STATE_DIACRITIC_CONFIRMED,
        )
        self.assertIsNone(tts2)

        # Giữ tĩnh BASE_STABLE quá 1.2s -> nói "A" ngay nhưng vẫn giữ cửa sổ replacement
        _, tts3 = rec._update_gesture_release_and_tts(
            final_sign="A",
            p_prim=lm_ok,
            current_clk=11.35,
            release_key="s1",
            recent_signs=["A"],
            fsm_state=STATE_BASE_STABLE,
            fsm_res_state=STATE_BASE_STABLE,
        )
        self.assertEqual(tts3, "A")
        self.assertIsNone(rec._pending_tts_token)
        self.assertIsNotNone(rec._replaceable_base_token)

        # NO_HAND khóa _replaceable_base_token và flush token chờ nếu còn
        rec._pending_tts_token = "O"
        rec._uncommitted_base_time = 15.0
        _, tts4 = rec._update_gesture_release_and_tts(
            final_sign="NEUTRAL",
            p_prim=None,
            current_clk=15.4,
            release_key="s1",
            recent_signs=[],
            fsm_state=STATE_IDLE,
        )
        self.assertEqual(tts4, "O")
        self.assertIsNone(rec._replaceable_base_token)
        self.assertIsNone(rec._pending_tts_token)

    def test_checkpoint_d_timestamp_micro_skew_and_epoch_contract(self):
        """Micro-skew <= 1ms không reset oan FSM; epoch giây và timestamp=None tương thích ngược."""
        fsm = DiacriticStateMachine(min_base_stable_frames=3, min_base_stable_sec=0.10)
        lm_a = make_hand_landmarks(wrist_y=0.72)
        base_epoch = 1763250000.0

        fsm.update("A", 0.95, lm_a, timestamp=base_epoch)
        fsm.update("A", 0.95, lm_a, timestamp=base_epoch + 0.08)
        fsm.update("A", 0.95, lm_a, timestamp=base_epoch + 0.16)
        self.assertEqual(fsm.state, STATE_BASE_STABLE)

        # Sub-millisecond negative skew (-0.0005s) phải kẹp an toàn và giữ nguyên BASE_STABLE
        res_skew = fsm.update("A", 0.95, lm_a, timestamp=base_epoch + 0.1595)
        self.assertEqual(res_skew["state"], STATE_BASE_STABLE)

        # Nhảy lùi rõ rệt (-0.05s) phải reset an toàn
        res_back = fsm.update("A", 0.95, lm_a, timestamp=base_epoch + 0.10)
        self.assertEqual(res_back["state"], STATE_BASE_CANDIDATE)
        self.assertEqual(fsm.candidate_base_count, 1)

    def test_checkpoint_d_recognizer_timestamp_none_backward_compat(self):
        """VSLRecognizer.predict_landmarks hỗ trợ cả epoch timestamp tường minh lẫn timestamp=None."""
        rec = VSLRecognizer()
        lm_a = make_hand_landmarks(wrist_y=0.72)
        res_none = rec.predict_landmarks(lm_a, mode="letters", session_id="cpd_none", timestamp=None)
        self.assertIn("sign", res_none)
        self.assertIn("tts_speak", res_none)

        res_epoch = rec.predict_landmarks(
            lm_a,
            mode="letters",
            session_id="cpd_epoch",
            trial_id="T1",
            timestamp=1763250100.25,
            backend_receive_ts=1763250100.26,
        )
        self.assertIn("sign", res_epoch)
        self.assertIn("tts_speak", res_epoch)


if __name__ == "__main__":
    unittest.main()
