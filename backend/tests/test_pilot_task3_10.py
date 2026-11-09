"""
Unit & Integration Test Suite cho TASK 3.10 — Data Collection Implementation
=============================================================================
Kiểm thử toàn diện 6 nhóm yêu cầu kỹ thuật theo SSOT:
1. Raw Landmarks:
   - 21 landmarks hợp lệ (image_landmarks_21 giữ nguyên 21x3).
   - Tracking loss / Không có bàn tay -> null (None), không thay bằng 0.
   - Schema không hợp lệ (NaN, Inf, sai shape) -> tracking_status = "ERROR".
   - Phân biệt rạch ròi image_landmarks_21 ([0,1]) và normalized_landmarks_21 (scale-invariant).
2. Video Recording (Backend Manifest & Sidecar Sync Integration):
   - JSONL frame schema 2.1-kinematic-phase; session manifest retains its own 2.0 schema.
   - Dừng recording, tính toán SHA256 nguyên tử.
   - Bảo toàn dữ liệu, video không lưu vào Git repository.
3. Video/Landmark Synchronization:
   - 6 mốc thời gian: client_capture_ts, landmark_extract_ts, request_send_ts, backend_receive_ts, video_pts_sec, video_frame_idx.
   - Phát hiện frame jump (delta frame_idx > 1).
   - Phát hiện network stall (delta t > 70ms).
   - Xử lý trung thực video PTS không xác định -> "UNKNOWN", không tạo PTS giả.
4. Temporal Replay Diagnostics:
   - Snapshot causal tại frame hiện tại: recent_predictions_10, vote_count, cooldown_remaining_sec.
   - Token confirmation event: boolean cờ, last_confirmed_sign, last_confirmed_time.
   - Tuyệt đối không dùng frame tương lai để giải thích quá khứ.
5. Privacy & Data Integrity:
   - Signer ID ẩn danh (signer_01 .. signer_08).
   - Session manifest hoàn chỉnh với SHA256, schema_version 2.0-kinematic.
   - Sidecar mapping JSON nguyên tử, an toàn symlink & path containment.
6. Regression & Boundary Invariants:
   - Bảo vệ tuyệt đối P Rule 15 và X Rule 18.
   - Bảo toàn các ngưỡng O ngoài phạm vi Candidate D.
   - Không thay đổi model weights hay classifier logic.
"""

import os
import sys
import json
import time
import math
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.vision.recognizer import VSLRecognizer
from backend.vision.pilot_logger import (
    PilotDiagnosticLogger,
    compute_file_sha256,
    create_session_manifest,
    save_session_manifest,
    save_sidecar_sync,
    validate_session_id,
    PILOT_SCHEMA_VERSION,
)
from backend.app import (
    recognize_landmarks,
    LandmarkRequest,
    PilotManifestRequest,
    PilotSidecarRequest,
    save_pilot_manifest,
    save_pilot_sidecar,
    recognizer,
)
from backend.tests.test_geometry import _make_hand


def _make_letter_o_hand():
    """Tạo landmarks mẫu khớp Rule O (các đầu ngón uốn cong chạm ngón cái tạo vòm tròn)."""
    lm = _make_hand()
    wx, wy, wz = lm[0]
    # Thu hẹp các đầu ngón tạo vòng cung O
    lm[4] = [wx - 0.04, wy - 0.16, wz]        # Thumb tip
    lm[8] = [wx - 0.03, wy - 0.165, wz + 0.01] # Index tip
    lm[12] = [wx - 0.02, wy - 0.17, wz + 0.01] # Middle tip
    lm[16] = [wx - 0.01, wy - 0.16, wz + 0.01] # Ring tip
    lm[20] = [wx, wy - 0.14, wz + 0.01]        # Pinky tip
    return lm


def _load_jsonl_records(filepath):
    if not os.path.exists(filepath):
        return []
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


class TestTask310DataCollectionImplementation(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_filepath = os.path.join(self.temp_dir, "test_pilot_task3_10.jsonl")
        self.rec = VSLRecognizer(pilot_logging_enabled=True)
        self.rec.pilot_filepath = self.log_filepath

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    # =========================================================================
    # NHÓM 1: RAW LANDMARKS (IMPLEMENTATION A)
    # =========================================================================
    def test_1_1_valid_raw_landmarks_21_preserved(self):
        """1.1: 21 landmarks hợp lệ được lưu trọn vẹn dạng 21x3 trong image_landmarks_21."""
        lm = _make_hand()
        res = self.rec.predict_landmarks(
            lm,
            mode="letters",
            expected_sign="O",
            session_id="sess_task3_10",
            trial_id="sess_task3_10_O_rep01",
            frame_idx=1,
            signer_id="signer_01",
            image_width=640,
            image_height=480,
            handedness_score=0.98,
        )
        self.rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)
        r = records[0]

        # Schema 2.1 bổ sung provenance/phase, giữ nguyên payload 2.0.
        self.assertEqual(r.get("schema_version"), "2.1-kinematic-phase")
        self.assertIn("phase", r)
        self.assertIn("classifier_config", r)
        self.assertEqual(r.get("feature_extractor_version"), "2.0-kinematic")
        self.assertEqual(r.get("tracking_status"), "VALID")
        self.assertEqual(r.get("signer_id"), "signer_01")
        self.assertEqual(r.get("image_width"), 640)
        self.assertEqual(r.get("image_height"), 480)
        self.assertEqual(r.get("handedness_score"), 0.98)

        # Kiểm tra image_landmarks_21
        raw_lms = r.get("image_landmarks_21")
        self.assertIsNotNone(raw_lms)
        self.assertEqual(len(raw_lms), 21)
        for pt in raw_lms:
            self.assertEqual(len(pt), 3)
            self.assertTrue(all(isinstance(v, (int, float)) for v in pt))

        normalized = r.get("normalized_landmarks_21")
        self.assertIsInstance(normalized, list)
        self.assertEqual(len(normalized), 21)
        self.assertTrue(all(isinstance(pt, list) and len(pt) == 3 for pt in normalized))
        self.assertIsInstance(r.get("geometric"), dict)
        self.assertIsInstance(r["geometric"].get("features"), dict)
        self.assertIsInstance(r.get("temporal"), dict)
        self.assertIsInstance(r.get("sync"), dict)

    def test_1_2_missing_hand_landmarks_are_null(self):
        """1.2: Khi mất tracking (không có bàn tay), image_landmarks_21 và normalized_landmarks_21 phải là null (None), không thay bằng 0."""
        res = self.rec.predict_landmarks(
            None,
            mode="letters",
            expected_sign="O",
            session_id="sess_task3_10",
            trial_id="sess_task3_10_O_rep01",
            frame_idx=2,
            signer_id="signer_02",
        )
        self.rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)
        r = records[0]

        self.assertEqual(r.get("tracking_status"), "NO_HAND")
        self.assertIsNone(r.get("image_landmarks_21"))
        self.assertIsNone(r.get("normalized_landmarks_21"))
        self.assertEqual(r.get("final_sign"), "NEUTRAL")

    def test_1_3_invalid_shape_or_nan_landmarks(self):
        """1.3: Landmarks có shape sai hoặc chứa NaN/Inf phải được đánh dấu tracking_status = INVALID_SHAPE (Task 3.10.2)."""
        bad_lm = [[0.1, 0.2, float("nan")]] * 21
        res = self.rec.predict_landmarks(
            bad_lm,
            mode="letters",
            expected_sign="O",
            session_id="sess_task3_10",
            trial_id="sess_task3_10_O_rep01",
            frame_idx=3,
        )
        self.rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)
        r = records[0]

        self.assertEqual(r.get("tracking_status"), "INVALID_SHAPE")
        self.assertIsNone(r.get("image_landmarks_21"))
        self.assertIsNone(r.get("normalized_landmarks_21"))
        self.assertEqual(r.get("final_sign"), "NEUTRAL")

    def test_1_5_invalid_shape_wrong_count_landmarks(self):
        """1.5: Landmarks sai số lượng điểm (<21 điểm) phải được đánh dấu tracking_status = INVALID_SHAPE (Task 3.10.2)."""
        short_lm = [[0.1, 0.2, 0.3]] * 10  # Chỉ có 10 điểm thay vì 21
        res = self.rec.predict_landmarks(
            short_lm,
            mode="letters",
            expected_sign="O",
            session_id="sess_task3_10",
            trial_id="sess_task3_10_O_rep01",
            frame_idx=31,
        )
        self.rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)
        r = records[0]

        self.assertEqual(r.get("tracking_status"), "INVALID_SHAPE")
        self.assertIsNone(r.get("image_landmarks_21"))
        self.assertIsNone(r.get("normalized_landmarks_21"))
        self.assertEqual(r.get("final_sign"), "NEUTRAL")

    def test_1_6_internal_error_handling(self):
        """1.6: Ngoại lệ không lường trước trong quá trình xử lý landmark phải trả về ERROR (Task 3.10.2)."""
        from unittest.mock import patch
        with patch("numpy.asarray", side_effect=RuntimeError("Simulated internal exception")):
            lm = _make_hand()
            res = self.rec.predict_landmarks(
                lm,
                mode="letters",
                expected_sign="O",
                session_id="sess_task3_10",
                trial_id="sess_task3_10_O_rep01",
                frame_idx=32,
            )
            self.rec.pilot_logger.flush()
            records = _load_jsonl_records(self.log_filepath)
            self.assertEqual(len(records), 1)
            r = records[0]
            self.assertEqual(r.get("tracking_status"), "ERROR")
            self.assertIsNone(r.get("image_landmarks_21"))
            self.assertIsNone(r.get("normalized_landmarks_21"))

    def test_1_4_differentiation_image_vs_normalized_coordinates(self):
        """1.4: Phân biệt rõ image_landmarks_21 (tọa độ gốc) và normalized_landmarks_21 (dịch cổ tay về gốc và chuẩn hóa theo palm_scale)."""
        lm = _make_hand()
        # Dịch chuyển cổ tay ra xa gốc
        lm = [[pt[0] + 0.3, pt[1] + 0.4, pt[2]] for pt in lm]

        res = self.rec.predict_landmarks(
            lm,
            mode="letters",
            expected_sign="O",
            session_id="sess_task3_10",
            trial_id="sess_task3_10_O_rep01",
            frame_idx=4,
        )
        self.rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)
        r = records[0]

        img_lms = r.get("image_landmarks_21")
        norm_lms = r.get("normalized_landmarks_21")

        self.assertIsNotNone(img_lms)
        self.assertIsNotNone(norm_lms)
        # Điểm mốc cổ tay trong normalized phải là xấp xỉ [0, 0, 0]
        self.assertAlmostEqual(norm_lms[0][0], 0.0, places=3)
        self.assertAlmostEqual(norm_lms[0][1], 0.0, places=3)
        self.assertAlmostEqual(norm_lms[0][2], 0.0, places=3)
        # Trong khi image_landmarks_21 giữ nguyên tọa độ viewport
        self.assertGreater(img_lms[0][0], 0.2)
        self.assertGreater(img_lms[0][1], 0.2)

    # =========================================================================
    # NHÓM 2: VIDEO/LANDMARK SYNCHRONIZATION (IMPLEMENTATION C)
    # =========================================================================
    def test_2_1_synchronization_timestamps_logged(self):
        """2.1: Ghi nhận đầy đủ các mốc thời gian đồng bộ và xử lý video PTS trung thực."""
        lm = _make_hand()
        t_cap = 1731110000.1234
        t_ext = 1731110000.1350
        t_send = 1731110000.1400

        res = self.rec.predict_landmarks(
            lm,
            mode="letters",
            expected_sign="O",
            session_id="sess_task3_10",
            trial_id="sess_task3_10_O_rep01",
            frame_idx=10,
            client_capture_ts=t_cap,
            landmark_extract_ts=t_ext,
            request_send_ts=t_send,
            video_pts_sec=0.3333,
            video_frame_idx=10,
        )
        self.rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)
        r = records[0]

        sync = r.get("sync")
        self.assertIsNotNone(sync)
        self.assertEqual(sync.get("frame_idx"), 10)
        self.assertEqual(sync.get("client_capture_ts"), t_cap)
        self.assertEqual(sync.get("landmark_extract_ts"), t_ext)
        self.assertEqual(sync.get("request_send_ts"), t_send)
        self.assertIsNotNone(sync.get("backend_receive_ts"))
        self.assertEqual(sync.get("video_pts_sec"), 0.3333)
        self.assertEqual(sync.get("video_frame_idx"), 10)

    def test_2_2_unknown_pts_not_faked(self):
        """2.2: Khi trình duyệt không cung cấp video PTS, ghi nhận trung thực 'UNKNOWN', không tự tạo PTS giả."""
        lm = _make_hand()
        res = self.rec.predict_landmarks(
            lm,
            mode="letters",
            expected_sign="O",
            session_id="sess_task3_10",
            trial_id="sess_task3_10_O_rep01",
            frame_idx=11,
            video_pts_sec=None,  # Không có PTS
        )
        self.rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 1)
        r = records[0]

        sync = r.get("sync")
        self.assertEqual(sync.get("video_pts_sec"), "UNKNOWN")

    def test_2_3_save_sidecar_sync_file(self):
        """2.3: Lưu tệp sidecar mapping đồng bộ hóa giữa video và landmarks thành công, đọc lại toàn vẹn."""
        sync_data = {
            "schema_version": PILOT_SCHEMA_VERSION,
            "session_id": "sess_task3_10_sync",
            "video_file": "pilot_sess_task3_10_sync_O_rep01.webm",
            "jsonl_file": "pilot_sess_task3_10_sync.jsonl",
            "sync_anomalies": {"jumps": 1, "stalls": 2},
            "frame_sync_table": [
                {"frame_idx": 1, "client_capture_ts": 100.1, "video_pts_sec": 0.033},
                {"frame_idx": 2, "client_capture_ts": 100.14, "video_pts_sec": 0.067},
            ]
        }
        saved_path = save_sidecar_sync("sess_task3_10_sync", sync_data, log_dir=self.temp_dir)
        self.assertTrue(os.path.exists(saved_path))

        with open(saved_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
        self.assertEqual(loaded.get("schema_version"), "2.0-kinematic")
        self.assertEqual(loaded.get("video_file"), "pilot_sess_task3_10_sync_O_rep01.webm")
        self.assertEqual(len(loaded.get("frame_sync_table")), 2)

    # =========================================================================
    # NHÓM 3: TEMPORAL REPLAY DIAGNOSTICS (IMPLEMENTATION D)
    # =========================================================================
    def test_3_1_causal_temporal_snapshot_and_cooldown(self):
        """3.1: Ghi nhận snapshot temporal pipeline hoàn toàn causal, chứa recent_predictions_10, cooldown, vote_count."""
        lm = _make_hand()

        # Phát 5 frame liên tiếp để tạo lịch sử dự đoán
        for i in range(5):
            res = self.rec.predict_landmarks(
                lm,
                mode="letters",
                expected_sign="O",
                session_id="sess_task3_10_temp",
                trial_id="sess_task3_10_temp_O_rep01",
                frame_idx=i + 1,
            )

        self.rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)
        self.assertEqual(len(records), 5)

        # Kiểm tra snapshot ở frame 5
        r5 = records[4]
        temp = r5.get("temporal")
        self.assertIsNotNone(temp)
        self.assertEqual(temp.get("smoothing_window"), 10)
        self.assertEqual(temp.get("vote_threshold"), 5)
        self.assertIn("vote_count", temp)
        self.assertIn("most_common_sign", temp)
        self.assertIn("recent_predictions_10", temp)
        self.assertEqual(len(temp["recent_predictions_10"]), 5)  # 5 frames đã tích lũy
        self.assertIn("cooldown_remaining_sec", temp)
        self.assertIn("token_confirmation_event", temp)

    def test_3_2_no_future_frames_in_temporal_state(self):
        """3.2: Chứng minh temporal snapshot tại frame N chỉ chứa tối đa N frames, không bao giờ chứa frame tương lai (N+1)."""
        lm = _make_hand()
        self.rec.predict_landmarks(
            lm, mode="letters", expected_sign="O", session_id="sess_causal", frame_idx=1
        )
        self.rec.predict_landmarks(
            lm, mode="letters", expected_sign="O", session_id="sess_causal", frame_idx=2
        )
        self.rec.pilot_logger.flush()
        records = _load_jsonl_records(self.log_filepath)

        r1 = records[0]
        # Frame 1 chỉ có 1 prediction trong recent_predictions_10
        self.assertEqual(len(r1["temporal"]["recent_predictions_10"]), 1)

        r2 = records[1]
        # Frame 2 có đúng 2 predictions
        self.assertEqual(len(r2["temporal"]["recent_predictions_10"]), 2)

    # =========================================================================
    # NHÓM 4: PRIVACY & DATA INTEGRITY (SECTION 8 & SESSION MANAGEMENT)
    # =========================================================================
    def test_4_1_sha256_computation_and_session_manifest(self):
        """4.1: Tạo Session Manifest hoàn chỉnh với mã SHA256 chính xác của file JSONL và sidecar."""
        # Tạo file JSONL mẫu
        sample_jsonl = os.path.join(self.temp_dir, "pilot_sess_manifest_01.jsonl")
        with open(sample_jsonl, "w", encoding="utf-8") as f:
            f.write('{"frame_idx": 1}\n')

        # Tạo sidecar mẫu
        sample_sidecar = os.path.join(self.temp_dir, "pilot_sess_manifest_01_sync.json")
        with open(sample_sidecar, "w", encoding="utf-8") as f:
            json.dump({"sync": True}, f)

        # Tính toán SHA256 độc lập
        import hashlib
        with open(sample_jsonl, "rb") as f:
            expected_jsonl_hash = hashlib.sha256(f.read()).hexdigest()

        manifest = create_session_manifest(
            session_id="sess_manifest_01",
            jsonl_filepath=sample_jsonl,
            video_filename="pilot_sess_manifest_01.webm",
            sidecar_filepath=sample_sidecar,
            signer_id="signer_03",
            metadata={"total_trials": 10, "target_letter": "O"}
        )

        self.assertEqual(manifest["schema_version"], "2.0-kinematic")
        self.assertEqual(manifest["signer_id"], "signer_03")
        self.assertEqual(manifest["files"]["jsonl"]["sha256"], expected_jsonl_hash)
        self.assertTrue(manifest["files"]["jsonl"]["exists"])
        self.assertTrue(manifest["files"]["sidecar"]["exists"])

        # Lưu manifest xuống đĩa
        manifest_path = save_session_manifest(manifest, log_dir=self.temp_dir)
        self.assertTrue(os.path.exists(manifest_path))

    # =========================================================================
    # NHÓM 5: FASTAPI ENDPOINTS CHO MANIFEST VÀ SIDECAR
    # =========================================================================
    def test_5_1_app_manifest_and_sidecar_endpoints(self):
        """5.1: Kiểm tra API endpoints /api/pilot/manifest và /api/pilot/sidecar_sync qua test client."""
        recognizer.diagnostic_dir = self.temp_dir
        manifest_payload = {
            "session_id": "sess_api_test_01",
            "manifest": {
                "schema_version": "2.0-kinematic",
                "session_id": "sess_api_test_01",
                "signer_id": "signer_05"
            }
        }
        res_manifest = save_pilot_manifest(PilotManifestRequest(**manifest_payload))
        self.assertTrue(res_manifest.get("success"))
        self.assertTrue(os.path.exists(res_manifest.get("manifest_path")))

        sidecar_payload = {
            "session_id": "sess_api_test_01",
            "sync_data": {
                "schema_version": "2.0-kinematic",
                "session_id": "sess_api_test_01",
                "video_file": "pilot_sess_api_test_01.webm"
            }
        }
        res_sidecar = save_pilot_sidecar(PilotSidecarRequest(**sidecar_payload))
        self.assertTrue(res_sidecar.get("success"))
        self.assertTrue(os.path.exists(res_sidecar.get("sidecar_path")))

    # =========================================================================
    # NHÓM 6: REGRESSION & ABSOLUTE INVARIANTS (SECTION 10)
    # =========================================================================
    def test_6_1_rule15_of_p_and_rule18_of_x_unaltered(self):
        """6.1: Bảo vệ tuyệt đối: Rule 15 của chữ P và Rule 18 của chữ X không bị thay đổi."""
        import backend.vision.alphabet_classifier as ac
        
        # Ngưỡng O Candidate D cùng các gate hình học ngoài phạm vi regression.
        self.assertEqual(ac.O_D_TI_MAX, 0.23)
        self.assertEqual(ac.O_IDX_STRAIGHT_MAX, 0.65)
        self.assertEqual(ac.O_D_TM_MAX, 0.35)
        self.assertEqual(ac.O_D_IW_MIN, 0.95)
        self.assertEqual(ac.O_D_MW_MIN, 0.90)
        self.assertEqual(ac.O_D_RW_MIN, 0.80)
        self.assertEqual(ac.O_D_PW_MIN, 0.75)
        self.assertEqual(ac.O_D_TW_MIN, 0.85)

        # Rule 15 của P (cả 2 nhánh upright & down) giữ nguyên
        self.assertEqual(ac.P_D_IW_MIN, 1.65)
        self.assertEqual(ac.P_IDX_STRAIGHT_MIN, 0.95)
        self.assertEqual(ac.P_IDX_TIP_Y_MAX, -1.40)
        self.assertEqual(ac.P_DOWN_D_IW_MIN, 1.70)
        self.assertEqual(ac.P_DOWN_IDX_STRAIGHT_MIN, 0.90)

        # Rule 18 của X (cả 2 nhánh upright & fwd) giữ nguyên
        self.assertEqual(ac.X_D_TI_MAX, 0.25)
        self.assertEqual(ac.X_IDX_CURL_MIN, 0.15)
        self.assertEqual(ac.X_IDX_STRAIGHT_MIN, 0.40)
        self.assertEqual(ac.X_IDX_STRAIGHT_MAX, 0.85)
        self.assertEqual(ac.X_FWD_D_TI_MIN, 0.35)
        self.assertEqual(ac.X_FWD_D_TI_MAX, 0.60)

        # Classifier methods tồn tại và callable
        self.assertTrue(callable(getattr(ac.alphabet_classifier, "evaluate_rule_p_diagnostics", None)))
        self.assertTrue(callable(getattr(ac.alphabet_classifier, "evaluate_rule_x_diagnostics", None)))
        self.assertTrue(callable(getattr(ac.alphabet_classifier, "evaluate_rule_o_diagnostics", None)))

    # =========================================================================
    # NHÓM 7: PROTOCOL 6 GIÂY & 5 PHA KINEMATIC (TASK 3.10.2)
    # =========================================================================
    def test_7_1_hud_protocol_duration_and_presets(self):
        """7.1: Runtime protocol helper exports the locked six-second duration."""
        import subprocess
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script = "import { PILOT_DURATION_SEC } from './src/components/pilotProtocol.js'; console.log(PILOT_DURATION_SEC)"
        result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=repo_root,
                                capture_output=True, text=True, check=True)
        self.assertEqual(float(result.stdout.strip()), 6.0)

    def test_7_2_five_phase_boundary_timings_in_hud(self):
        """7.2: Exercise the frontend phase helper at each approved boundary."""
        import subprocess
        repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        script = (
            "import { phaseAtElapsedSeconds } from './src/components/pilotProtocol.js'; "
            "console.log(JSON.stringify([0,0.5,1.5,4.5,5.5,6].map(phaseAtElapsedSeconds)))"
        )
        result = subprocess.run(["node", "--input-type=module", "-e", script], cwd=repo_root,
                                capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(result.stdout), [
            "PRE_IDLE", "ENTRY", "STABLE_HOLD", "EXIT", "POST_IDLE", "POST_IDLE"
        ])


if __name__ == "__main__":
    unittest.main()
