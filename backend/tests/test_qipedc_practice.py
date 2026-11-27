import io
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.app import app, recognizer
from backend.models import video_cache
from backend.training import build_lexicon
from backend.training.build_qipedc_live_index import sync_qipedc_dictionary
from backend.vision import qipedc_live, qipedc_practice
from backend.vision.qipedc_practice import reference_path
from backend.vision.skeleton_norm import MIRRORED_JOINTS, reflect_legacy_left_hand


def test_dictionary_sync_updates_metadata_for_existing_video_ids(tmp_path, monkeypatch):
    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "qipedc_dictionary.json"
    output = tmp_path / "vsl_lexicon.json"
    source.write_text(json.dumps([["D0001N", "thử", "Động từ", "Giải nghĩa cũ", 0, False]]))
    remote = {"data": [{
        "_id": "D0001N", "word": "thử", "tl": "Động từ ", "description": "Giải nghĩa cũ",
    }]}
    monkeypatch.setattr("urllib.request.urlopen", lambda *a, **kw: io.BytesIO(json.dumps(remote).encode()))
    monkeypatch.setattr(build_lexicon, "SRC_DIR", str(sources))
    monkeypatch.setattr(build_lexicon, "OUT_PATH", str(output))

    def rebuild(*, dictionary_path, output_path):
        output_path.write_bytes(dictionary_path.read_bytes())

    monkeypatch.setattr(build_lexicon, "build", rebuild)
    assert sync_qipedc_dictionary() == (1, False)
    assert not output.exists()
    remote["data"][0]["description"] = "Giải nghĩa mới"
    assert sync_qipedc_dictionary() == (1, True)
    assert json.loads(source.read_text())[0][3] == "Giải nghĩa mới"
    assert output.read_bytes() == source.read_bytes()


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
        padded_res = client.post(path, json={**payload, "sequence": padded.tolist(), "include_diagnostic": True}).json()["data"]
        assert padded_res["entry_id"] == "q-may-gio"
        assert padded_res["diagnostic"]["status"] == "word_matched"

        stuck_index = motion.copy()
        stuck_index[8:, 62, 1] = stuck_index[7, 62, 1]
        recovered = client.post(path, json={**payload, "sequence": stuck_index.tolist(), "include_diagnostic": True}).json()["data"]
        assert recovered["entry_id"] == "q-may-gio"

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
        assert complete["match_method"] in {"trajectory", "hand_shape"}

        no_pose_wrist = occluded.copy()
        no_pose_wrist[:, 16] = 0
        result = client.post(path, json={
            **payload, "sequence": no_pose_wrist.tolist(), "timestamps": times.tolist(),
        }).json()["data"]
        assert result["sign"] == "NEUTRAL"
        assert result["diagnostic"]["status"] != "word_matched"

        interrupted = client.post(path, json={
            **payload, "sequence": occluded[:26].tolist(), "timestamps": times[:26].tolist(),
        }).json()["data"]
        assert interrupted["sign"] == "NEUTRAL"
        assert interrupted["top_k"] == []


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
        return {"sign": "Mẹ", "entry_id": "q-ma-giong-me", "confidence": 0.9999}

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


@pytest.mark.parametrize(("entry_id", "recording_id", "shift_y"), [
    ("q-ky-nang", "D0097", -0.12),
    ("q-xin-loi", "W03990", -0.12),
    ("q-nan-ni", "D0261_close", 0.08),
])
def test_live_bilateral_shape_match_handles_changes_in_signing_height(
    tmp_path, monkeypatch, entry_id, recording_id, shift_y,
):
    from backend.training.build_qipedc_live_index import pack_index

    folder = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference"
    templates = {
        "q-ky-nang": "D0097", "q-xin-loi": "W03990", "q-kha-nang": "D0096",
        "q-nan-ni": "D0261", "q-cau-nguyen": "W00450", "q-vai-lay": "W03770",
    }
    index = tmp_path / "live_index.npz"
    pack_index([(key, folder / f"v3_{video}.npy") for key, video in templates.items()], index)
    monkeypatch.setattr(qipedc_live, "index_path", lambda: index)
    monkeypatch.setenv("VSL_QIPEDC_REFERENCE_DIR", str(folder))

    frames = np.load(folder / f"v3_{recording_id}.npy", allow_pickle=False)
    hands = frames[:, 33:]
    hands[np.any(hands != 0, axis=-1)] += [0.02, shift_y, 0]
    result = qipedc_live.match_live_words(frames, 1.0, [])

    assert result["entry_id"] == entry_id
    assert result["match_method"] in {"trajectory", "hand_shape"}


def test_fast_idle_padded_and_clasped_occluded_signs_match_reliably():
    folder = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference"
    xin_loi = np.load(folder / "v3_W03990.npy", allow_pickle=False)
    ky_nang = np.load(folder / "v3_D0097.npy", allow_pickle=False)

    # 1. Fast 2x sign embedded at the end of a long idle rolling window
    fast_xin_loi = xin_loi[::2]
    fast_padded = np.concatenate([np.repeat(fast_xin_loi[:1], 24, axis=0), fast_xin_loi], axis=0)
    res_fast = qipedc_live.match_live_words(fast_padded, 1.0, [])
    assert res_fast is not None and res_fast["entry_id"] == "q-xin-loi"

    fast_ky_nang = ky_nang[::2]
    fast_kn_padded = np.concatenate([np.repeat(fast_ky_nang[:1], 24, axis=0), fast_ky_nang], axis=0)
    res_kn = qipedc_live.match_live_words(fast_kn_padded, 1.0, [])
    assert res_kn is not None and res_kn["entry_id"] == "q-ky-nang"

    # 2. Clasped-hand self-occlusion: right hand occluded for half of the middle clasp
    occluded = xin_loi.copy()
    occluded[len(occluded) // 3: 2 * len(occluded) // 3, 54:75] = 0.0
    res_occ = qipedc_live.match_live_words(occluded, 1.0, [])
    assert res_occ is not None and res_occ["entry_id"] == "q-xin-loi"

    # 3. Topic-scoped allowed_ids narrows search cleanly without breaking matching
    res_topic = qipedc_live.match_live_words(xin_loi, 1.0, [], allowed_ids={"q-xin-loi", "q-nan-ni", "q-ky-nang"})
    assert res_topic is not None and res_topic["entry_id"] == "q-xin-loi"
    assert qipedc_live.match_live_words(xin_loi, 1.0, [], allowed_ids={"q-ky-nang"}) is None


def test_pose_guided_gap_fill_and_kvsl_2link_ik_augmentation():
    from backend.training.augmentations import augment_hand_occlusion, augment_inter_hand_shift

    # Pose-guided 2-frame hand dropout follows curved pose[15] wrist path
    seq = np.zeros((10, 75, 3), dtype=np.float32)
    seq[:, 11] = [0.6, 0.4, 0.0]
    seq[:, 12] = [0.4, 0.4, 0.0]
    seq[:, 15, 0] = np.linspace(0.5, 0.7, 10)
    seq[:, 15, 1] = 0.5 - 0.1 * np.sin(np.linspace(0, np.pi, 10))
    seq[:, 33:54] = seq[:, 15:16] + 0.02
    seq[4:6, 33:54] = 0.0  # 2-frame fast-motion dropout
    filled = qipedc_practice._fill_short_gaps(seq)
    assert np.all(np.any(filled[4:6, 33:54] != 0, axis=-1))
    np.testing.assert_allclose(filled[4, 33, :2], seq[4, 15, :2] + 0.02, atol=1e-4)

    # KVSL 2-link IK preserves upper-arm / forearm bone lengths when shifting wrists
    ctj = np.zeros((3, 12, 75), dtype=np.float32)
    ctj[:, :, 11] = np.array([[0.6], [0.4], [0.0]], dtype=np.float32)
    ctj[:, :, 13] = np.array([[0.65], [0.55], [0.0]], dtype=np.float32)
    ctj[:, :, 15] = np.array([[0.58], [0.68], [0.0]], dtype=np.float32)
    ctj[:, :, 33:54] = ctj[:, :, 15:16] + 0.01
    ctj[:, :, 54:75] = np.array([[0.42], [0.65], [0.0]], dtype=np.float32)[:, :, None]
    rng = np.random.default_rng(42)
    orig_upper = float(np.linalg.norm(ctj[:2, 0, 13] - ctj[:2, 0, 11]))
    aug = augment_inter_hand_shift(ctj, rng, max_delta=0.05)
    new_upper = float(np.linalg.norm(aug[:2, 0, 13] - aug[:2, 0, 11]))
    assert np.isclose(new_upper, orig_upper, atol=1e-3)
    occ_aug = augment_hand_occlusion(ctj, rng, min_frames=2, max_frames=4)
    assert np.any(occ_aug[:, :, 33:54] == 0) or np.any(occ_aug[:, :, 54:75] == 0)


def test_kha_nang_recovers_when_vertical_edge_on_hand_is_occluded():
    folder = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference"
    kha_nang = np.load(folder / "v3_D0096.npy", allow_pickle=False)

    # Simulate severe edge-on occlusion of the vertical hand (RH in studio D0096)
    # during the perpendicular hold while keeping Holistic pose wrists/indices (pose[16, 20]).
    occluded = kha_nang.copy()
    occluded[15:23, 54:75] = 0.0
    # Shift hands closer/higher like a laptop webcam (where trajectory alone loses D0096)
    mask = np.any(occluded[:, 33:] != 0, axis=-1)
    occluded[:, 33:][mask] += [0.02, -0.10, 0.0]
    occluded[:, [15, 16, 19, 20], 1] -= 0.10

    res = qipedc_live.match_live_words(occluded, 1.0, [])
    assert res is not None and res["entry_id"] == "q-kha-nang"

    # Negative check: 'kỹ năng', 'xin lỗi', 'năn nỉ', 'mấy giờ?' must NOT be misclassified as 'khả năng'.
    for other in ("v3_D0097.npy", "v3_W03990.npy", "v3_D0261.npy", "v3_W02196.npy"):
        other_res = qipedc_live.match_live_words(np.load(folder / other, allow_pickle=False), 1.0, [])
        assert other_res is None or other_res.get("entry_id") != "q-kha-nang"


def test_recognize_sequence_full_diagnostic_trace_explains_motion_ambiguity_and_match(tmp_path, monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference"
    ref_uni = np.load(folder / "v3_D0440.npy", allow_pickle=False)

    with TestClient(app) as client:
        path = "/api/recognize_sequence"

        # 1. Matched QIPEDC word returns full motion_gate, qipedc top_candidates, timings, and explanation
        matched = client.post(path, json={
            "sequence": ref_uni.tolist(),
            "mode": "words",
            "live_words": True,
            "include_diagnostic": True,
        }).json()["data"]
        assert matched["entry_id"] == "q-an-du"
        diag = matched["diagnostic"]
        assert diag["motion_gate"]["motion_detected"] is True
        assert diag["qipedc"]["bucket_hands"] == 1
        assert diag["qipedc"]["top_candidates"][0]["entry_id"] == "q-an-du"
        assert diag["qipedc"]["top_candidates"][0]["word"]
        assert diag["decision"]["explanation"]
        assert diag["timings_ms"]["total_ms"] >= 0
        assert len(diag["pipeline_steps"]) >= 2

        # 2. Stray 2-frame second hand on live webcam is ignored (stays 1-hand and matches q-an-du)
        poisoned = ref_uni.copy()
        poisoned[0:2, 33:54] = poisoned[0:2, 54:75] + [-0.3, 0.25, 0.0]
        clean_1h = client.post(path, json={
            "sequence": poisoned.tolist(),
            "mode": "words",
            "live_words": True,
            "include_diagnostic": True,
        }).json()["data"]
        assert clean_1h["entry_id"] == "q-an-du"
        assert clean_1h["diagnostic"]["qipedc"]["bucket_hands"] == 1
        assert clean_1h["diagnostic"]["qipedc"]["bilateral_forced_by_2frame_rule"] is False

        # 2b. Sustained stationary second hand triggers forced bilateral flag cleanly
        sustained_2h = ref_uni.copy()
        sustained_2h[:, 33:54] = ref_uni[15:16, 54:75] + [-0.35, 0.0, 0.0]
        forced_res = client.post(path, json={
            "sequence": sustained_2h.tolist(),
            "mode": "words",
            "live_words": True,
            "include_diagnostic": True,
        }).json()["data"]
        assert forced_res["sign"] == "NEUTRAL"
        assert forced_res["top_k"] == []
        assert forced_res["diagnostic"]["qipedc"]["bilateral_forced_by_2frame_rule"] is True
        assert forced_res["diagnostic"]["qipedc"]["bucket_hands"] == 2

        # 2c. Ambiguous match explains Top 1 vs Top 2 gap and exposes quick-pick suggestions
        with monkeypatch.context() as patch:
            patch.setattr(video_cache, "CACHE_DIR", tmp_path)
            patch.setattr(qipedc_live, "index_path", lambda: None)
            reference_path("D0440").parent.mkdir(parents=True, exist_ok=True)
            np.save(reference_path("D0440"), ref_uni, allow_pickle=False)
            np.save(reference_path("D0442"), ref_uni, allow_pickle=False)
            ambig = client.post(path, json={
                "sequence": ref_uni.tolist(),
                "mode": "words",
                "live_words": True,
                "include_diagnostic": True,
            }).json()["data"]
            assert ambig["sign"] == "NEUTRAL"
            assert ambig["top_k"] == []
            assert ambig["diagnostic"]["status"] == "ambiguous_word"
            assert ambig["diagnostic"]["qipedc"]["gap_top1_top2"] == 0.0
            assert len(ambig["suggestions"]) == 2
            assert "Top 1 và Top 2" in ambig["diagnostic"]["decision"]["explanation"]

        # 3. Held static pose explains exact extent vs effective motion threshold (0.08 raised or 0.16 resting)
        held = np.repeat(ref_uni[10:11], 24, axis=0)
        no_motion = client.post(path, json={
            "sequence": held.tolist(),
            "mode": "words",
            "live_words": True,
            "include_diagnostic": True,
        }).json()["data"]
        assert no_motion["diagnostic"]["status"] == "no_motion"
        eff_thr = no_motion["diagnostic"]["motion_gate"]["motion_threshold"]
        assert eff_thr in (0.08, 0.16)
        assert no_motion["diagnostic"]["motion_gate"]["max_extent"] < eff_thr
        assert str(eff_thr) in no_motion["diagnostic"]["decision"]["explanation"]

        # 4. When include_diagnostic is False, no diagnostic payload is attached
        plain = client.post(path, json={
            "sequence": ref_uni.tolist(),
            "mode": "words",
            "live_words": True,
            "include_diagnostic": False,
        }).json()["data"]
        assert "diagnostic" not in plain


def test_live_words_vocab_scope_and_time_sign_protected_from_hand_shape_override(tmp_path, monkeypatch):
    folder = Path(__file__).resolve().parents[1] / "data" / "qipedc_reference"
    ref_uni = np.load(folder / "v3_D0440.npy", allow_pickle=False)

    with TestClient(app) as client:
        path = "/api/recognize_sequence"

        # vocab_scope="common" matches common/builtin words cleanly on Tier 1
        common_res = client.post(path, json={
            "sequence": ref_uni.tolist(),
            "mode": "words",
            "live_words": True,
            "vocab_scope": "common",
            "include_diagnostic": True,
        }).json()["data"]
        assert common_res["entry_id"] == "q-an-du"
        assert common_res["diagnostic"]["qipedc"]["tier"] == "priority_common"

        # Rare words are skipped when vocab_scope="common" and matched on Tier 2 when vocab_scope="all"
        rare_clip = np.load(folder / "v3_W03770.npy", allow_pickle=False)
        all_res = client.post(path, json={
            "sequence": rare_clip.tolist(),
            "mode": "words",
            "live_words": True,
            "vocab_scope": "all",
        }).json()["data"]
        assert all_res["entry_id"] == "q-vai-lay"


def test_chung_toi_family_recognized_for_1_to_5_fingers_and_not_confused_with_con_sau():
    from backend.training.build_qipedc_live_index import _existing_npy

    idx = qipedc_live._active_index()
    uni_ids = set(idx["buckets"][1]["ids"])
    family = [
        ("q-chung-toi-dai-tu", "D0134"),
        ("q-chung-toi-2-nguoi-dai-tu", "D0130"),
        ("q-chung-toi-3-nguoi-dai-tu", "D0131"),
        ("q-chung-toi-4-nguoi-dai-tu", "D0132"),
        ("q-chung-toi-5-nguoi-dai-tu", "D0133"),
    ]
    for eid, _ in family:
        assert eid in uni_ids

    with TestClient(app) as client:
        for expected_id, vid in family:
            raw = np.load(_existing_npy(vid), allow_pickle=False).copy()
            # Simulate 4:3 webcam framing + mild position shift without per-sign hardcoding
            webcam_clip = raw.copy()
            webcam_clip[..., 1] *= 4 / 3
            mask = np.any(webcam_clip[:, 33:] != 0, axis=-1)
            webcam_clip[:, 33:][mask] += [0.03, -0.02, 0.0]
            resp = client.post("/api/recognize_sequence", json={
                "sequence": webcam_clip.tolist(),
                "mode": "words",
                "live_words": True,
                "aspect_ratio": 4 / 3,
                "include_diagnostic": True,
            }).json()["data"]
            assert resp["entry_id"] == expected_id, f"Expected {expected_id}, got {resp}"
            assert resp["sign"] != "con sâu"


def test_head_salute_distinguishes_chao_outward_from_tuan_lenh_inward_and_blocks_em_gai_on_chest():
    from backend.training.build_qipedc_live_index import _existing_npy

    chao_raw = np.load(_existing_npy("W00489"), allow_pickle=False).copy()
    chao_cam = chao_raw.copy()
    chao_cam[..., 1] *= 4 / 3

    tl_raw = np.load(_existing_npy("W03683"), allow_pickle=False).copy()
    tl_cam = tl_raw.copy()
    tl_cam[..., 1] *= 4 / 3

    # Chest-only partial arc (should NEVER be misclassified as SSTCN 'Em gái')
    chest_arc = np.zeros((30, 75, 3), dtype=np.float32)
    chest_arc[:, 9] = [0.48, 0.30, 0.0]
    chest_arc[:, 10] = [0.52, 0.30, 0.0]
    chest_arc[:, 11] = [0.64, 0.46, 0.0]
    chest_arc[:, 12] = [0.36, 0.46, 0.0]
    for t in range(30):
        w = np.array([0.42 + 0.08 * (t / 29.0), 0.58, 0.0], dtype=np.float32)
        chest_arc[t, 54:75] = w + np.linspace(0.0, 0.04, 21)[:, None] * np.array([1.0, -0.4, 0.0])

    with TestClient(app) as client:
        out_chao = client.post("/api/recognize_sequence", json={
            "sequence": chao_cam.tolist(),
            "mode": "words", "live_words": True, "aspect_ratio": 4 / 3,
        }).json()["data"]
        assert out_chao["entry_id"] == "q-chao"

        in_tl = client.post("/api/recognize_sequence", json={
            "sequence": tl_cam.tolist(),
            "mode": "words", "live_words": True, "aspect_ratio": 4 / 3,
        }).json()["data"]
        assert in_tl["entry_id"] == "q-tuan-lenh"

        # Even if SSTCN tries to predict 'Em gái' on a chest-only gesture, generic corroboration blocks it.
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(recognizer, "predict_sequence", lambda **kw: {
                "success": True, "sign": "Em gái", "label": "Em gái", "index": 86,
                "entry_id": "q-em-gai", "confidence": 0.95, "top_k": [],
            })
            chest_resp = client.post("/api/recognize_sequence", json={
                "sequence": chest_arc.tolist(),
                "mode": "words", "live_words": True, "aspect_ratio": 4 / 3,
            }).json()["data"]
            assert chest_resp["sign"] != "Em gái"
            assert chest_resp["sign"] == "NEUTRAL"
