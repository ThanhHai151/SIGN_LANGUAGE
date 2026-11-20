import numpy as np
from fastapi.testclient import TestClient

from backend.app import app, recognizer
from backend.models import video_cache
from backend.vision import qipedc_practice
from backend.vision.qipedc_practice import reference_path
from backend.vision.skeleton_norm import MIRRORED_JOINTS, reflect_legacy_left_hand


def test_may_gio_practice_checks_the_recorded_landmarks(tmp_path, monkeypatch):
    monkeypatch.setattr(video_cache, "CACHE_DIR", tmp_path)
    reference = reference_path("W02196")
    reference.parent.mkdir()

    motion = np.zeros((30, 75, 3), dtype=np.float32)
    motion[:, 11] = [0.36, 0.45, 0.0]
    motion[:, 12] = [0.64, 0.45, 0.0]
    motion[:, 33:54, 0] = np.linspace(0.44, 0.62, 30)[:, None]
    motion[:, 33:54, 1] = np.linspace(0.47, 0.55, 21)
    np.save(reference, motion, allow_pickle=False)

    base = "/api/qipedc/practice/q-may-gio"
    with TestClient(app) as client:
        assert client.post(f"{base}/prepare").json()["video_id"] == "W02196"
        assert client.post(f"{base}/prepare?video_id=W00489").status_code == 404
        transformed = motion * 1.25 + np.any(motion, axis=-1, keepdims=True) * [0.1, 0.1, 0.0]
        match = client.post(f"{base}/match", json={"sequence": transformed.tolist()}).json()
        assert match["word"] == "mấy giờ?" and match["matched"]

        wrong_sign = motion.copy()
        wrong_sign[:, 33:54, 1] += 0.35
        rejected = client.post(f"{base}/match", json={"sequence": wrong_sign.tolist()}).json()
        assert rejected["matched"] is False
        assert client.post(f"{base}/match", json={"sequence": np.zeros_like(motion).tolist()}).status_code == 422


def test_bilateral_practice_handles_frame_aspect_tracking_gaps_and_opposite_hand(tmp_path, monkeypatch):
    monkeypatch.setattr(video_cache, "CACHE_DIR", tmp_path)
    reference = reference_path("W02196")
    reference.parent.mkdir()
    motion = np.zeros((30, 75, 3), dtype=np.float32)
    motion[:, 11] = [0.36, 0.3, 0.0]
    motion[:, 12] = [0.64, 0.3, 0.0]
    motion[:, 33:54, 0] = np.linspace(0.24, 0.44, 30)[:, None]
    motion[:, 33:54, 1] = np.linspace(0.33, 0.39, 21)
    motion[:, 54:75, 0] = np.linspace(0.28, 0.32, 21)
    motion[:, 54:75, 1] = np.linspace(0.5, 0.57, 21)
    np.save(reference, motion, allow_pickle=False)

    camera = motion.copy()
    camera[..., 1] *= 4 / 3
    camera[5:7, 33:54] = 0
    camera[15:17, 11] = 0

    with TestClient(app) as client:
        path = "/api/qipedc/practice/q-may-gio/match"
        data = {"sequence": camera.tolist(), "aspect_ratio": 4 / 3}
        assert client.post(path, json=data).json()["matched"] is True
        assert client.post(path, json={**data, "aspect_ratio": 0}).status_code == 422

        opposite = camera.copy()
        opposite[:, 33:54] = camera[:, 54:75]
        opposite[:, 54:75] = camera[:, 33:54]
        opposite[..., 0] = np.where(np.any(opposite != 0, axis=-1), 1 - opposite[..., 0], 0)
        data["sequence"] = opposite.tolist()
        assert client.post(path, json=data).json()["matched"] is False
        assert client.post(path, json={**data, "mirror": True}).json()["matched"] is True

        camera[:, 54:75] = 0
        data["sequence"] = camera.tolist()
        assert client.post(path, json=data).json()["reason"] == "missing_both_hands"

        camera[4:24, 33:54] = 0
        data["sequence"] = camera.tolist()
        assert client.post(path, json=data).json()["reason"] == "tracking_gaps"


def test_live_words_can_identify_time_question_without_a_selected_word(tmp_path, monkeypatch):
    monkeypatch.setattr(video_cache, "CACHE_DIR", tmp_path)
    reference = reference_path("W02196")
    reference.parent.mkdir()
    motion = np.zeros((30, 75, 3), dtype=np.float32)
    motion[:, 11] = [0.36, 0.35, 0.0]
    motion[:, 12] = [0.64, 0.35, 0.0]
    motion[:, 33:54, 0] = np.linspace(0.52, 0.60, 21)
    motion[:, 33:54, 1] = 0.65
    motion[:, 54:75, 0] = 0.38
    motion[:, 54:75, 1] = np.r_[np.full(8, 0.65), np.linspace(0.65, 0.31, 12), np.full(10, 0.31)][:, None]
    motion[:, 62, 0] = np.r_[np.full(8, 0.52), np.linspace(0.52, 0.38, 12), np.full(10, 0.38)]
    np.save(reference, motion, allow_pickle=False)

    with TestClient(app) as client:
        path = "/api/recognize_sequence"
        payload = {"sequence": motion.tolist(), "mode": "words", "live_words": True}
        live = client.post(path, json=payload).json()["data"]
        assert live["sign"] == "mấy giờ?" and live["source"] == "qipedc_template"
        assert "diagnostic" not in live

        padded = np.concatenate([np.zeros_like(motion[:5]), motion])
        phase = client.post(path, json={**payload, "sequence": padded.tolist(), "include_diagnostic": True}).json()["data"]["diagnostic"]
        assert phase["status"] == "matched"
        assert 5 <= phase["tap_frame"] < phase["lift_frame"] < len(padded)

        stuck_index = motion.copy()
        stuck_index[8:, 62, 1] = stuck_index[7, 62, 1]
        recovered = client.post(path, json={**payload, "sequence": stuck_index.tolist(), "include_diagnostic": True}).json()["data"]
        assert recovered["entry_id"] == "q-may-gio"
        assert recovered["diagnostic"]["tap_frame"] < recovered["diagnostic"]["lift_frame"]

        held_pose = np.repeat(motion[24:25], 45, axis=0)
        idle = client.post(path, json={**payload, "sequence": held_pose.tolist(), "include_diagnostic": True}).json()["data"]
        assert idle["sign"] == "NEUTRAL" and idle["motion_detected"] is False
        assert not idle["confirmed"]
        assert idle["diagnostic"]["status"] == "no_motion"

        final_phase_only = client.post(path, json={**payload, "sequence": motion[15:].tolist()}).json()["data"]
        assert final_phase_only.get("entry_id") != "q-may-gio"


def test_live_time_sign_can_verify_opening_after_supporting_hand_disappears(tmp_path, monkeypatch):
    monkeypatch.setattr(video_cache, "CACHE_DIR", tmp_path)
    reference = reference_path("W02196")
    reference.parent.mkdir()

    motion = np.zeros((36, 75, 3), dtype=np.float32)
    motion[:, 11] = [0.62, 0.50, 0.0]
    motion[:, 12] = [0.38, 0.50, 0.0]
    motion[:, 14] = [0.36, 0.55, 0.0]
    motion[:, 16] = [0.52, 0.54, 0.0]
    motion[:, 54:75] = [0.53, 0.54, 0.0]
    closed = np.zeros((21, 3), dtype=np.float32)
    closed[9] = [-0.05, 0.03, 0]
    closed[[8, 12, 16, 20]] = [[-0.09, 0.04, 0], [-0.05, 0.04, 0], [-0.04, 0.03, 0], [-0.03, 0.03, 0]]
    opened = closed.copy()
    opened[9] = [0, -0.06, 0]
    opened[[8, 12, 16, 20]] = [[-0.052, -0.115, 0], [0.005, -0.125, 0], [0.045, -0.115, 0], [0.077, -0.09, 0]]

    wrist = np.tile([0.62, 0.49, 0.0], (len(motion), 1))
    wrist[:16, 1] = np.linspace(0.32, 0.49, 16)
    wrist[18:26, 1] = np.linspace(0.49, 0.35, 8)
    wrist[26:, 1] = 0.35
    motion[:, 33:54] = wrist[:, None] + closed
    motion[30:, 33:54] = wrist[30:, None] + opened
    np.save(reference, motion, allow_pickle=False)

    occluded = motion.copy()
    occluded[:16, 54:75] = 0
    occluded[18:, 54:75] = 0
    times = np.arange(len(motion)) * 60
    payload = {"mode": "words", "live_words": True, "include_diagnostic": True}

    def unexpected_fallback(**kwargs):
        raise AssertionError("SSTCN should not guess while the QIPEDC motion is missing landmarks")

    monkeypatch.setattr(recognizer, "predict_sequence", unexpected_fallback)
    with TestClient(app) as client:
        path = "/api/recognize_sequence"
        complete = client.post(path, json={
            **payload, "sequence": occluded.tolist(), "timestamps": times.tolist(),
        }).json()["data"]
        assert complete["entry_id"] == "q-may-gio"
        assert complete["match_method"] == "phases"
        assert complete["confidence"] is None
        assert complete["diagnostic"]["tap_frame"] == 16
        assert complete["diagnostic"]["lift_frame"] == 31

        one_contact = occluded.copy()
        one_contact[17, 54:75] = 0
        for with_pose_wrist in (True, False):
            if not with_pose_wrist:
                one_contact[:, 16] = 0
            result = client.post(path, json={
                **payload, "sequence": one_contact.tolist(), "timestamps": times.tolist(),
            }).json()["data"]
            assert result["diagnostic"]["status"] == ("matched" if with_pose_wrist else "missing_both_hands")

        for timestamps, frames in ((times, occluded[:30]), (times * 2, occluded)):
            interrupted = client.post(path, json={
                **payload, "sequence": frames.tolist(), "timestamps": timestamps[:len(frames)].tolist(),
            }).json()["data"]
            assert interrupted["sign"] == "NEUTRAL"
            assert interrupted["top_k"] == []
            assert interrupted["diagnostic"]["status"] == "missing_both_hands"


def test_live_words_match_dictionary_entries_and_reject_indistinguishable_templates(tmp_path, monkeypatch):
    monkeypatch.setattr(video_cache, "CACHE_DIR", tmp_path)
    reference_path("D0440").parent.mkdir()
    main = np.zeros((24, 75, 3), dtype=np.float32)
    main[:, 9] = [0.47, 0.35, 0]
    main[:, 10] = [0.53, 0.35, 0]
    main[:, 11] = [0.63, 0.51, 0]
    main[:, 12] = [0.37, 0.51, 0]
    main[:, 54:, 0] = np.linspace(0.56, 0.50, 21)
    main[:, 54:, 1] = np.r_[np.linspace(0.70, 0.41, 12), np.linspace(0.41, 0.54, 12)][:, None]
    main[:, 54:, 1] -= np.linspace(0, 0.12, 21)
    np.save(reference_path("D0440"), main)

    optional = main.copy()
    optional[:, 54:, 0] -= np.linspace(0, 0.36, len(optional))[:, None]
    optional[:, 54:, 1] -= 0.29
    np.save(reference_path("W00489"), optional)

    def live_payload(frames, extra=()):
        return {"sequence": frames.tolist(), "live_words": True, "mode": "words", "include_diagnostic": True,
                "qipedc_entries": list(extra)}

    with TestClient(app) as client:
        path = "/api/recognize_sequence"
        main_result = client.post(path, json=live_payload(main)).json()["data"]
        assert main_result["entry_id"] == "q-an-du"
        assert main_result["match_method"] == "trajectory"

        # The active hand can differ from the hand in the dictionary video.
        opposite = optional[:, MIRRORED_JOINTS, :].copy()
        opposite[..., 0] = np.where(np.any(opposite != 0, axis=-1), 1 - opposite[..., 0], 0)
        response = client.post("/api/qipedc/live/prepare?entry_id=q-chao")
        assert response.json()["entries"] == [{"id": "q-chao", "word": "chào", "ready": True, "builtin": False}]
        assert client.post("/api/qipedc/live/prepare?entry_id=invalid").status_code == 404
        assert client.post(path, json=live_payload(opposite)).json()["data"]["sign"] == "NEUTRAL"
        assert client.post(path, json=live_payload(opposite, ("q-chao",))).json()["data"]["entry_id"] == "q-chao"

        # A strong dictionary match can disambiguate the separate wrist/lift detector.
        with monkeypatch.context() as patch:
            patch.setattr(qipedc_practice, "analyze_live_words", lambda *args, **kwargs: {
                "motion_detected": True, "status": "matched", "similarity": 0.75,
            })
            conflict = client.post(path, json=live_payload(opposite, ("q-chao",))).json()["data"]
            assert conflict["entry_id"] == "q-chao"

        np.save(reference_path("D0442"), main)
        ambiguous = client.post(path, json=live_payload(main)).json()["data"]
        assert ambiguous["sign"] == "NEUTRAL"
        assert ambiguous["diagnostic"]["status"] == "ambiguous_word"


def test_legacy_left_hand_reflection_preserves_joints_and_missing_values():
    frames = np.zeros((30, 75, 3), dtype=np.float32)
    frames[:, 11] = [0.65, 0.40, 0.1]
    frames[:, 33] = [0.72, 0.61, -0.2]

    reflected = reflect_legacy_left_hand(frames)
    np.testing.assert_allclose(reflected[0, 12], [0.35, 0.40, 0.1])
    np.testing.assert_allclose(reflected[0, 54], [0.28, 0.61, -0.2], atol=1e-7)
    assert not reflected[:, 33:54].any()
    assert reflect_legacy_left_hand(reflected) is reflected

    frames[10, 54] = [0.45, 0.61, -0.2]
    assert reflect_legacy_left_hand(frames) is frames


def test_live_words_wait_when_a_hand_disappears_during_two_hand_motion(tmp_path, monkeypatch):
    monkeypatch.setattr(video_cache, "CACHE_DIR", tmp_path)
    motion = np.zeros((36, 75, 3), dtype=np.float32)
    motion[:, 11] = [0.63, 0.51, 0]
    motion[:, 12] = [0.37, 0.51, 0]
    motion[:, 15] = [0.58, 0.55, 0]
    motion[:, 16] = [0.42, 0.55, 0]
    motion[:, 33:54, 0] = 0.58 + np.linspace(0, 0.04, 21)
    motion[:, 33:54, 1] = np.linspace(0.56, 0.29, len(motion))[:, None] - np.linspace(0, 0.10, 21)
    motion[:5, 54:, 0] = 0.43 + np.linspace(0, 0.04, 21)
    motion[:5, 54:, 1] = 0.55 - np.linspace(0, 0.10, 21)
    calls = []

    def fallback(**kwargs):
        calls.append(kwargs)
        return {"sign": "Mẹ"}

    monkeypatch.setattr(recognizer, "predict_sequence", fallback)
    path = "/api/recognize_sequence"
    payload = {"mode": "words", "live_words": True, "include_diagnostic": True}
    with TestClient(app) as client:
        blocked = client.post(path, json={**payload, "sequence": motion.tolist()}).json()["data"]
        assert blocked["sign"] == "NEUTRAL"
        assert blocked["diagnostic"]["status"] == "lost_hand_during_interaction"
        assert not calls

        motion[7:11, [15, 16], 1] = 0.85
        released = client.post(path, json={**payload, "sequence": motion.tolist()}).json()["data"]
        assert released["sign"] == "Mẹ"
        assert len(calls) == 1


def test_live_bilateral_words_trim_waist_frames_and_tolerate_reference_occlusion(tmp_path, monkeypatch):
    monkeypatch.setattr(video_cache, "CACHE_DIR", tmp_path)
    reference_path("D0261").parent.mkdir()

    # Build a chest-level two-hand sign (like "năn nỉ"), then pad the reference with waist-level
    # entry/exit frames and occlude one hand during the raised signing phase.
    raised = np.zeros((16, 75, 3), dtype=np.float32)
    raised[:, 9] = [0.47, 0.35, 0]
    raised[:, 10] = [0.53, 0.35, 0]
    raised[:, 11] = [0.63, 0.51, 0]
    raised[:, 12] = [0.37, 0.51, 0]
    y_wave = np.sin(np.linspace(0, 2 * np.pi, len(raised))) * 0.04
    raised[:, 33:54, 0] = 0.52 + np.linspace(0, 0.03, 21)
    raised[:, 33:54, 1] = (0.56 + y_wave)[:, None] - np.linspace(0, 0.10, 21)
    raised[:, 54:, 0] = 0.48 - np.linspace(0, 0.03, 21)
    raised[:, 54:, 1] = (0.56 + y_wave)[:, None] - np.linspace(0, 0.10, 21)

    waist = raised[:6].copy()
    waist[:, 33:54, 1] = 0.90 + np.linspace(0, 0.08, 21)
    waist[:, 54:, 1] = 0.90 + np.linspace(0, 0.08, 21)

    studio_ref = np.concatenate([waist, raised.copy(), waist], axis=0)
    # Simulate self-occlusion of the right hand in the middle of the studio reference.
    studio_ref[8:20, 54:] = 0
    np.save(reference_path("D0261"), studio_ref)

    with TestClient(app) as client:
        res = client.post("/api/recognize_sequence", json={
            "sequence": raised.tolist(),
            "live_words": True,
            "mode": "words",
            "include_diagnostic": True,
        }).json()["data"]
        assert res["entry_id"] == "q-nan-ni"
        assert res["match_method"] == "trajectory"


def test_live_compressed_index_matches_two_stage_and_exposes_metadata(tmp_path, monkeypatch):
    from backend.training.build_qipedc_live_index import pack_index

    monkeypatch.setattr(video_cache, "CACHE_DIR", tmp_path)
    practice_dir = tmp_path / "practice"
    practice_dir.mkdir()

    sign_a = np.zeros((20, 75, 3), dtype=np.float32)
    sign_a[:, 9] = [0.47, 0.35, 0]
    sign_a[:, 10] = [0.53, 0.35, 0]
    sign_a[:, 11] = [0.63, 0.51, 0]
    sign_a[:, 12] = [0.37, 0.51, 0]
    sign_a[:, 54:, 0] = np.linspace(0.56, 0.50, 21)
    sign_a[:, 54:, 1] = np.r_[np.linspace(0.70, 0.41, 10), np.linspace(0.41, 0.54, 10)][:, None] - np.linspace(0, 0.12, 21)
    path_a = practice_dir / "v3_D0440.npy"
    np.save(path_a, sign_a)

    sign_b = sign_a.copy()
    sign_b[:, 54:, 0] -= 0.30
    sign_b[:, 54:, 1] -= 0.25
    path_b = practice_dir / "v3_W00489.npy"
    np.save(path_b, sign_b)

    n_uni, n_bi, skipped = pack_index([("q-an-du", path_a), ("q-chao", path_b)], practice_dir / "live_index.npz")
    assert (n_uni, n_bi, skipped) == (2, 0, 0)
    # Remove individual .npy files to verify matching uses only the compressed index.
    path_a.unlink()
    path_b.unlink()

    with TestClient(app) as client:
        res = client.post("/api/recognize_sequence", json={
            "sequence": sign_b.tolist(),
            "live_words": True,
            "mode": "words",
        }).json()["data"]
        assert res["entry_id"] == "q-chao"
        assert res["match_method"] == "trajectory"

        prep = client.post("/api/qipedc/live/prepare?entry_id=q-chao").json()
        assert prep["indexed_count"] == 2
        assert prep["entries"][0]["ready"] is True and prep["entries"][0]["builtin"] is True
