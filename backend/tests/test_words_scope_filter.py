"""Tests for Camera `words` recognition scope, 36 GT exclusion, and 3,231 QIPEDC filtering."""

import numpy as np
import pytest
import torch
from fastapi.testclient import TestClient

from backend.app import _live_qipedc_entries, app, recognizer
from backend.models import video_cache
from backend.models.lexicon import load_lexicon
from backend.vision import qipedc_live, qipedc_practice
from backend.vision.word_scope import (
    allowed_live_word_ids,
    allowed_sstcn_word_indices,
    approved_live_word_ids,
    filter_entry_eligible_videos,
    parse_qipedc_video_region,
    validate_extra_qipedc_word,
    validate_word_result_scope,
    verify_reference_video_id,
)


def _sample_sequence(length=30):
    seq = np.zeros((length, 75, 3), dtype=np.float32)
    seq[:, 11] = [0.63, 0.50, 0.0]
    seq[:, 12] = [0.37, 0.50, 0.0]
    seq[:, 13] = [0.68, 0.65, 0.0]
    seq[:, 14] = [0.32, 0.65, 0.0]
    seq[:, 15] = [0.58, 0.54, 0.0]
    seq[:, 16] = [0.42, 0.54, 0.0]
    seq[:, 33:54, 0] = np.linspace(0.56, 0.64, length)[:, None]
    seq[:, 33:54, 1] = 0.50 - np.linspace(0.0, 0.06, 21)
    seq[:, 54:75, 0] = 0.42 + np.linspace(0.0, 0.04, 21)
    seq[:, 54:75, 1] = 0.52 - np.linspace(0.0, 0.06, 21)
    return seq


def test_sstcn_word_scope_does_not_promote_a_word_over_a_letter(monkeypatch):
    def predict_letter(tensor):
        logits = torch.zeros((len(tensor), 102), device=tensor.device)
        logits[:, 0] = 20.0
        logits[:, 52] = 15.0
        return logits

    monkeypatch.setattr(recognizer.sstcn_model, "forward", predict_letter)
    data = recognizer.predict_sequence(_sample_sequence(), mode="words", confirm=False)
    assert data["sign"] == "NEUTRAL"
    assert data["top_k"][0]["entry_id"] == "q-chao"
    assert data["top_k"][0]["confidence"] < 0.01


def test_recorded_word_without_motion_never_reaches_sstcn(monkeypatch):
    def unexpected_prediction(**kwargs):
        pytest.fail("SSTCN cannot confirm a word from a held hand pose")

    monkeypatch.setattr(recognizer, "predict_sequence", unexpected_prediction)
    held = np.repeat(_sample_sequence()[15:16], 30, axis=0)
    with TestClient(app) as client:
        data = client.post("/api/recognize_sequence", json={
            "sequence": held.tolist(), "mode": "words", "include_diagnostic": True,
        }).json()["data"]
    assert data["sign"] == "NEUTRAL"
    assert data["new_token"] is None
    assert data["diagnostic"]["sstcn"] == {"called": False, "skip_reason": "no_motion"}


def test_manual_candidates_include_unconfirmed_words_without_committing(monkeypatch):
    monkeypatch.setattr(
        qipedc_practice, "analyze_live_words", lambda *a, **kw: {"motion_detected": True, "status": "motion_detected"},
    )

    def collect_templates(*args, candidate_scores=None, **kwargs):
        if candidate_scores is not None:
            candidate_scores.extend([
                {"entry_id": "q-muon", "similarity": 0.65},
                {"entry_id": "q-chi", "similarity": 0.6},
                {"entry_id": "q-an-it", "similarity": 0.599},
                {"entry_id": "gt-unverified", "similarity": 0.99},
            ])

    monkeypatch.setattr(qipedc_live, "match_live_words", collect_templates)
    monkeypatch.setattr(qipedc_live, "corroborate_sstcn_entry", lambda *a: False)
    monkeypatch.setattr(recognizer, "predict_sequence", lambda **kw: {
        "success": True, "sign": "Chào", "entry_id": "q-chao", "confidence": 0.93,
        "top_k": [{"entry_id": "q-chao", "term": "Chào", "confidence": 0.93}],
    })
    payload = {"sequence": _sample_sequence().tolist(), "mode": "words", "live_words": True}
    with TestClient(app) as client:
        data = client.post("/api/recognize_sequence", json=payload).json()["data"]
        assert "word_candidates" not in data
        data = client.post("/api/recognize_sequence", json={**payload, "include_word_candidates": True}).json()["data"]

    assert data["sign"] == "NEUTRAL"
    assert data["new_token"] is None
    assert data["top_k"] == []
    assert "diagnostic" not in data
    assert [(item["entry_id"], item["confidence"]) for item in data["word_candidates"]] == [
        ("q-chao", 0.93), ("q-muon", 0.65), ("q-chi", 0.6),
    ]


def test_sstcn_single_hand_does_not_auto_commit_without_template_agreement(monkeypatch):
    monkeypatch.setattr(qipedc_live, "match_live_words", lambda *a, **kw: None)
    monkeypatch.setattr(qipedc_live, "corroborate_sstcn_entry", lambda *a: False)
    monkeypatch.setattr(recognizer, "predict_sequence", lambda **kw: {
        "success": True, "sign": "Chào", "entry_id": "q-chao", "confidence": 0.9995,
        "top_k": [{"entry_id": "q-chao", "term": "Chào", "confidence": 0.9995}],
    })
    sequence = _sample_sequence()
    # A visible hand hanging below the signing space isn't a second signing hand.
    sequence[:, 54:75, 1] += 0.33
    payload = {
        "sequence": sequence.tolist(), "mode": "words", "live_words": True,
        "include_word_candidates": True, "gesture_complete": True,
    }
    with TestClient(app) as client:
        data = client.post("/api/recognize_sequence", json=payload).json()["data"]
    assert data["requires_user_confirmation"] is True
    assert data["word_candidates"][0]["entry_id"] == "q-chao"


def test_single_hand_contour_alone_is_only_a_suggestion(monkeypatch):
    hand_shape = qipedc_live._match_result(0.20, "q-ma-hong", method="hand_shape")
    monkeypatch.setattr(qipedc_live, "_unilateral_core_rank", lambda *args: ([], {}))
    result = qipedc_live._with_unilateral_core(
        hand_shape, _sample_sequence(), 1.0, None, None, None, True, [], None,
    )
    assert result["requires_user_confirmation"] is True
    assert result["entry_id"] == "q-ma-hong"


def test_calibration_profile_matches_only_its_own_explicitly_labelled_sample(tmp_path, monkeypatch):
    monkeypatch.setattr("backend.app._calibration_dir", lambda: str(tmp_path))
    profile = "a" * 32
    other_profile = "b" * 32
    sequence = _sample_sequence(44)
    sequence[:, 54:] = 0
    sequence[:, 33:54, 0] += (0.04 * np.sin(np.linspace(0, np.pi, len(sequence))))[:, None]
    with TestClient(app) as client:
        sample = {"label": "chỉ", "entry_id": "q-chi", "sequence": sequence.tolist(), "calibration_profile": profile}
        for _ in range(2):
            saved = client.post("/api/calibration/save_sample", json=sample)
            assert saved.status_code == 200, saved.json()
            assert saved.json()["sample_count"] == 1
        assert client.post("/api/calibration/save_sample", json={
            **sample, "sequence": np.repeat(sequence[:1], len(sequence), axis=0).tolist(),
        }).status_code == 400
        request = {**sample, "mode": "words", "live_words": True, "gesture_complete": True,
                   "allow_short_sequences": True, "include_word_candidates": True}
        for tested in (profile, other_profile):
            reply = client.post("/api/recognize_sequence", json={**request, "calibration_profile": tested}).json()["data"]
            personal = [item for item in reply.get("word_candidates", []) if item["match_method"] == "personal_template"]
            if tested == profile:
                assert personal[0]["entry_id"] == "q-chi"
                assert reply["requires_user_confirmation"] is True
            else:
                assert not personal
        second = sequence.copy()
        second[:, 33:54, 1] += 0.008
        saved = client.post("/api/calibration/save_sample", json={**sample, "sequence": second.tolist()})
        assert saved.json()["sample_count"] == 2
        reply = client.post("/api/recognize_sequence", json=request).json()["data"]
        assert reply["entry_id"] == "q-chi"
        assert reply["requires_user_confirmation"] is False


def test_workspace_profile_bootstrap_is_loopback_only(tmp_path, monkeypatch):
    from starlette.requests import Request
    from backend.app import get_workspace_calibration_profile
    from fastapi import HTTPException

    monkeypatch.setattr("backend.app._calibration_dir", lambda: str(tmp_path))
    folder = tmp_path / ".live"
    folder.mkdir()
    (folder / ".workspace-profile").write_text("a" * 32, encoding="ascii")
    local_header = (b"x-vsl-workspace-profile", b"1")
    local_host = (b"host", b"localhost:8000")
    local_origin = (b"origin", b"http://localhost:5173")

    def make_request(host, origin, client="127.0.0.1"):
        return Request({"type": "http", "path": "/", "headers": [host, origin, local_header],
                        "client": (client, 55000)})

    response = get_workspace_calibration_profile(make_request(local_host, local_origin))
    assert response.headers["cache-control"] == "no-store"
    for request in (
        make_request(local_host, (b"origin", b"https://example.org")),
        make_request((b"host", b"example.org"), local_origin),
        make_request(local_host, local_origin, "192.0.2.1"),
    ):
        with pytest.raises(HTTPException) as rejected:
            get_workspace_calibration_profile(request)
        assert rejected.value.status_code == 404


@pytest.mark.parametrize("topic", [None, "Bài 3 - Giao tiếp cơ bản và Cảm xúc"])
@pytest.mark.parametrize("confidence", [0.98, 0.9995])
def test_uncorroborated_sstcn_uses_same_gate_with_or_without_topic(monkeypatch, topic, confidence):
    monkeypatch.setattr(
        qipedc_practice, "analyze_live_words", lambda *a, **kw: {"motion_detected": True, "status": "motion_detected"},
    )
    monkeypatch.setattr(qipedc_live, "match_live_words", lambda *a, **kw: None)
    monkeypatch.setattr(qipedc_live, "corroborate_sstcn_entry", lambda *a: False)
    monkeypatch.setattr(recognizer, "predict_sequence", lambda **kw: {
        "success": True, "sign": "Chào", "entry_id": "q-chao", "confidence": confidence,
    })
    with TestClient(app) as client:
        data = client.post("/api/recognize_sequence", json={
            "sequence": _sample_sequence().tolist(), "mode": "words", "live_words": True,
            "topic_id": topic, "include_diagnostic": True,
        }).json()["data"]
    if confidence >= 0.999:
        assert data["sign"] == "Chào"
        assert data["requires_live_consensus"] is True
    else:
        assert data["sign"] == "NEUTRAL"
        assert data["diagnostic"]["sstcn"]["blocked_prediction"]["sign"] == "Chào"


def _make_fake_lexicon():
    entries = [
        {
            "id": "q-valid-nat",
            "word": "từ toàn quốc",
            "keys": ["từ toàn quốc"],
            "source": "QIPEDC",
            "category": "Giao tiếp",
            "video_url": "https://qipedc.moet.gov.vn/videos/W03373.mp4",
            "variants": [{"region": "", "url": "https://qipedc.moet.gov.vn/videos/W03373.mp4"}],
            "region": "",
            "sstcn_index": None,
            "fingerspelling": False,
        },
        {
            "id": "q-valid-south",
            "word": "từ miền nam",
            "keys": ["từ miền nam"],
            "source": "QIPEDC",
            "category": "Giao tiếp",
            "video_url": "https://qipedc.moet.gov.vn/videos/W03374N.mp4",
            "variants": [{"region": "N", "url": "https://qipedc.moet.gov.vn/videos/W03374N.mp4"}],
            "region": "N",
            "sstcn_index": None,
            "fingerspelling": False,
        },
        {
            "id": "q-only-north",
            "word": "từ miền bắc",
            "keys": ["từ miền bắc"],
            "source": "QIPEDC",
            "category": "Giao tiếp",
            "video_url": "https://qipedc.moet.gov.vn/videos/W03375B.mp4",
            "variants": [{"region": "B", "url": "https://qipedc.moet.gov.vn/videos/W03375B.mp4"}],
            "region": "B",
            "sstcn_index": None,
            "fingerspelling": False,
        },
        {
            "id": "q-only-central",
            "word": "từ miền trung",
            "keys": ["từ miền trung"],
            "source": "QIPEDC",
            "category": "Giao tiếp",
            "video_url": "https://qipedc.moet.gov.vn/videos/W03376T.mp4",
            "variants": [{"region": "T", "url": "https://qipedc.moet.gov.vn/videos/W03376T.mp4"}],
            "region": "T",
            "sstcn_index": None,
            "fingerspelling": False,
        },
        {
            "id": "q-mixed-nb",
            "word": "từ hỗn hợp",
            "keys": ["từ hỗn hợp"],
            "source": "QIPEDC",
            "category": "Giao tiếp",
            "video_url": None,
            "variants": [
                {"region": "B", "url": "https://qipedc.moet.gov.vn/videos/W03375B.mp4"},
                {"region": "T", "url": "https://qipedc.moet.gov.vn/videos/W03376T.mp4"},
                {"region": "N", "url": "https://qipedc.moet.gov.vn/videos/W03374N.mp4"},
                {"region": "GT", "url": "https://proj.supabase.co/storage/v1/object/public/v/sign_001.mp4"},
            ],
            "region": "N",
            "sstcn_index": None,
            "fingerspelling": False,
        },
        {
            "id": "gt-toi",
            "word": "tôi",
            "keys": ["tôi"],
            "source": "Giáo trình VSL (102 từ)",
            "category": "Bài 3 - Giao tiếp cơ bản và Cảm xúc",
            "video_url": "https://gajsffqnyujwyllfcmny.supabase.co/storage/v1/object/public/sign_videos/sign_051.mp4",
            "variants": [{"region": "GT", "url": "https://gajsffqnyujwyllfcmny.supabase.co/storage/v1/object/public/sign_videos/sign_051.mp4"}],
            "region": "GT",
            "sstcn_index": 51,
            "fingerspelling": False,
        },
        {
            "id": "q-1",
            "word": "1",
            "keys": ["1"],
            "source": "QIPEDC",
            "category": "Số đếm",
            "video_url": "https://qipedc.moet.gov.vn/videos/D0530.mp4",
            "variants": [{"region": "", "url": "https://qipedc.moet.gov.vn/videos/D0530.mp4"}],
            "region": "",
            "sstcn_index": 29,
            "fingerspelling": False,
        },
        {
            "id": "q-a",
            "word": "a",
            "keys": ["a"],
            "source": "QIPEDC",
            "category": "Bảng chữ cái",
            "video_url": "https://qipedc.moet.gov.vn/videos/D0489.mp4",
            "variants": [{"region": "", "url": "https://qipedc.moet.gov.vn/videos/D0489.mp4"}],
            "region": "",
            "sstcn_index": 0,
            "fingerspelling": True,
        },
    ]
    return {
        "entries": entries,
        "by_id": {e["id"]: e for e in entries},
        "by_sstcn": {e["sstcn_index"]: e for e in entries if e["sstcn_index"] is not None},
        "by_key": {k: e for e in entries for k in e["keys"]},
    }


def test_01_to_04_qipedc_region_parsing_rules():
    # 1. No region suffix -> Nationwide (ACCEPT)
    for sample in (
        "W03373.mp4",
        "W03373",
        "https://qipedc.moet.gov.vn/videos/W03373.mp4",
        "https://qipedc.moet.gov.vn/videos/D0440.mp4?download=1#t=2",
    ):
        parsed = parse_qipedc_video_region(sample)
        assert parsed["eligible"] is True, sample
        assert parsed["region"] == "NATIONWIDE"

    # 2. Suffix N -> South (ACCEPT)
    for sample in ("W03374N.mp4", "W03374N", "https://qipedc.moet.gov.vn/videos/W03314N.mp4"):
        parsed = parse_qipedc_video_region(sample)
        assert parsed["eligible"] is True, sample
        assert parsed["region"] == "N"

    # 3. Suffix B -> North (REJECT), including deceptive query param spoofing
    for sample in (
        "W03375B.mp4",
        "D0001B",
        "https://qipedc.moet.gov.vn/videos/W03375B.mp4?region=N",
    ):
        parsed = parse_qipedc_video_region(sample)
        assert parsed["eligible"] is False, sample
        assert parsed["region"] == "B"
        assert parsed["reason"] == "rejected_region_north"

    # 4. Suffix T -> Central (REJECT), plus unknown/non-QIPEDC/GT/missing -> NEVER nationwide
    for sample, expected_reg in (
        ("W03376T.mp4", "T"),
        ("https://qipedc.moet.gov.vn/videos/D0001T.mp4", "T"),
        ("W03377X.mp4", "UNKNOWN"),
        ("random_clip.mp4", "UNKNOWN"),
        ("https://example.com/videos/W03373.mp4", "UNKNOWN"),
        ("https://gajsffqnyujwyllfcmny.supabase.co/storage/v1/object/public/sign_videos/sign_051.mp4", "GT"),
        (None, "UNKNOWN"),
    ):
        parsed = parse_qipedc_video_region(sample)
        assert parsed["eligible"] is False, sample
        assert parsed["region"] == expected_reg

    # Provenance alias check for D0261_close -> D0261 (Nationwide)
    assert verify_reference_video_id("D0261_close")["eligible"] is True
    assert verify_reference_video_id("W03375B")["eligible"] is False
    assert verify_reference_video_id("unverified_custom")["eligible"] is False


def test_05_and_06_out_of_scope_and_bt_only_entries_rejected():
    fake_lex = _make_fake_lexicon()
    allowed_set = allowed_live_word_ids(
        fake_lex,
        allowed_override={"q-valid-nat", "q-valid-south", "q-only-north", "q-only-central", "q-1", "q-a", "gt-toi"},
    )
    # Only valid word entries with Nationwide/N QIPEDC videos survive intersection
    assert allowed_set == frozenset({"q-valid-nat", "q-valid-south"})

    # Condition B alone rejects B-only / T-only / GT even if named in ALLOWED_WORD_IDS
    for bad_id in ("q-only-north", "q-only-central", "gt-toi", "q-1", "q-a", "q-unknown"):
        res = validate_word_result_scope(bad_id, source="qipedc_template", lex=fake_lex, allowed_ids=allowed_set)
        assert res["valid"] is False, bad_id
        assert res["reject_reason"]

    # Real lexicon entry with B/T-only videos (no_south=True) is rejected
    real_lex = load_lexicon()
    res_choi = validate_word_result_scope("q-choi", source="qipedc_template", lex=real_lex)
    assert res_choi["valid"] is False


def test_07_multi_variant_entry_retains_only_eligible_variants():
    fake_lex = _make_fake_lexicon()
    info = filter_entry_eligible_videos(fake_lex["by_id"]["q-mixed-nb"])
    assert info["eligible"] is True
    assert [v["video_id"] for v in info["eligible_videos"]] == ["W03374N"]
    assert [v["region"] for v in info["eligible_videos"]] == ["N"]
    assert {v["region"] for v in info["rejected_videos"]} == {"B", "T", "GT"}

    # Adversarial case: video_url points to B, variants has N -> primary B is rejected, N is kept
    mixed_primary_b = {
        "id": "q-adv",
        "video_url": "https://qipedc.moet.gov.vn/videos/D0001B.mp4",
        "region": "N",
        "variants": [
            {"region": "B", "url": "https://qipedc.moet.gov.vn/videos/D0001B.mp4"},
            {"region": "N", "url": "https://qipedc.moet.gov.vn/videos/D0001N.mp4"},
        ],
    }
    adv = filter_entry_eligible_videos(mixed_primary_b)
    assert adv["eligible"] is True
    assert adv["selected_video_id"] == "D0001N"
    assert [v["video_id"] for v in adv["eligible_videos"]] == ["D0001N"]


def test_08_extra_words_and_prepare_cannot_bypass_filter(monkeypatch):
    # Direct _live_qipedc_entries blocks B-only, digits, fingerspelling, GT, and missing IDs
    entries = _live_qipedc_entries(extra_ids=("q-chao", "q-1", "q-a", "q-choi", "gt-toi", "does-not-exist"))
    eids = [eid for eid, _ in entries]
    assert "q-chao" in eids
    for forbidden in ("q-1", "q-a", "q-choi", "gt-toi", "does-not-exist"):
        assert forbidden not in eids

    # Approved QIPEDC Live scope covers all extracted Southern/Nationwide word videos (>= 3,231)
    real_lex = load_lexicon()
    approved_scope = approved_live_word_ids(
        real_lex,
        indexed_ids=qipedc_live.indexed_entry_ids(),
        builtin_ids=qipedc_live.BUILTIN_WORDS,
    )
    assert len(approved_scope) >= 3231
    # Unindexed handless QIPEDC entry (outside approved scope) cannot bypass filter
    assert "q-bat-dong" not in approved_scope
    assert validate_extra_qipedc_word("q-only-north", _make_fake_lexicon())["allowed"] is False

    with TestClient(app) as client:
        for disallowed in ("q-1", "q-a", "q-choi", "gt-toi", "q-bat-dong"):
            prep = client.post(f"/api/qipedc/live/prepare?entry_id={disallowed}")
            assert prep.status_code == 200
            item = prep.json()["entries"][0]
            assert item["ready"] is False, disallowed
            assert item["reject_reason"], disallowed

        # Continuous mode recognize_sequence drops disallowed & unindexed extra_words and logs rejection
        monkeypatch.setattr(
            recognizer,
            "predict_sequence",
            lambda **kwargs: {"success": True, "sign": "NEUTRAL", "label": "Neutral", "confidence": 0.0, "confirmed": False, "top_k": []},
        )
        seq = _sample_sequence(30)
        resp = client.post(
            "/api/recognize_sequence",
            json={
                "sequence": seq.tolist(),
                "mode": "words",
                "live_words": True,
                "qipedc_entries": ["q-1", "q-choi", "gt-toi", "q-bat-dong", "q-chao"],
                "include_diagnostic": True,
            },
        )
        assert resp.status_code == 200
        diag = resp.json()["data"]["diagnostic"]
        rejected_ids = {r["entry_id"] for r in diag.get("rejected_extra_words", [])}
        assert {"q-1", "q-choi", "gt-toi", "q-bat-dong"} <= rejected_ids
        assert "q-chao" not in rejected_ids


def test_09_qipedc_live_does_not_search_outside_candidates(tmp_path, monkeypatch):
    from backend.training.build_qipedc_live_index import pack_index

    monkeypatch.setattr(video_cache, "CACHE_DIR", tmp_path)
    pdir = tmp_path / "practice"
    pdir.mkdir()
    motion = _sample_sequence(24)
    p_allowed = pdir / "v3_W00489.npy"
    p_excluded = pdir / "v3_D0440.npy"
    np.save(p_allowed, motion)
    # Excluded sample is an exact copy of query! If searched at all, it would match or tie.
    np.save(p_excluded, motion)

    index_file = pdir / "live_index.npz"
    pack_index([("q-an-du", p_excluded), ("q-chao", p_allowed)], index_file)
    monkeypatch.setattr(qipedc_live, "index_path", lambda: index_file)

    seen_ids = []
    orig_score = qipedc_live._score_templates

    def spy_score(recording, references):
        seen_ids.extend(references.keys())
        return orig_score(recording, references)

    monkeypatch.setattr(qipedc_live, "_score_templates", spy_score)

    # 1. With allowed_ids excluding q-an-du, only q-chao is passed to DTW scoring (plus invalid B video entry ignored)
    res = qipedc_live.match_live_words(
        motion,
        1.0,
        [("q-fake-north", "W03375B")],
        allowed_ids=frozenset({"q-chao", "q-fake-north"}),
    )
    assert set(seen_ids) <= {"q-chao"}
    assert "q-an-du" not in seen_ids
    assert "q-fake-north" not in seen_ids
    assert res is not None and res["entry_id"] == "q-chao"

    # 2. Empty allowed_ids (e.g. non-word topic) triggers 0 scoring and returns None immediately
    seen_ids.clear()
    res_empty = qipedc_live.match_live_words(motion, 1.0, [], allowed_ids=frozenset())
    assert res_empty is None
    assert seen_ids == []

    # 3. Real production index check: if topic is "Bảng chữ cái" (non-word), live_words API skips QIPEDC matching completely
    real_lex = load_lexicon()
    alphabet_allowed = allowed_live_word_ids(
        real_lex,
        indexed_ids=qipedc_live.indexed_entry_ids(),
        builtin_ids=qipedc_live.BUILTIN_WORDS,
        topic_id="Bảng chữ cái",
    )
    assert alphabet_allowed == frozenset()


def test_10_and_11_exclude_36_gt_words_from_sstcn_top_k_continuous_and_record(monkeypatch):
    real_lex = load_lexicon()
    approved_3231 = approved_live_word_ids(
        real_lex,
        indexed_ids=qipedc_live.indexed_entry_ids(),
        builtin_ids=qipedc_live.BUILTIN_WORDS,
    )

    # Default SSTCN scope MUST exclude all 36 gt-* classes and leave exactly 27 QIPEDC-verified classes
    active_default, meta_default = allowed_sstcn_word_indices(recognizer.sstcn_model, real_lex)
    assert len(active_default) == 27
    assert all(39 <= i <= 101 for i in active_default)
    gt_excluded = [x for x in meta_default["excluded"] if x["reason"] == "unverified_gt_curriculum"]
    assert len(gt_excluded) == 36
    assert len(meta_default["pending_gt_indices"]) == 0
    for idx in active_default:
        entry = real_lex["by_sstcn"][idx]
        assert entry["id"] in approved_3231
        assert not entry["id"].startswith("gt-")
        assert filter_entry_eligible_videos(entry)["eligible"] is True

    # Bug regression: topic_id pointing to alphabet ("Bài 1 - Bảng chữ cái") must NOT leak letters 0..28 in words mode
    seq = _sample_sequence(30)
    rec_topic_res = recognizer.predict_sequence(seq.tolist(), mode="words", topic_id="Bài 1 - Bảng chữ cái", threshold=0.01)
    assert rec_topic_res["sign"] == "NEUTRAL"
    assert rec_topic_res["top_k"] == []
    assert rec_topic_res["confirmed"] is False
    assert rec_topic_res["new_token"] is None
    assert rec_topic_res["tts_speak"] is None
    assert rec_topic_res["scope_rejected"] is True
    assert rec_topic_res["reject_reason"] == "topic_no_eligible_words"

    # Topic with < 5 active classes ("Bài 6 - Nghề nghiệp" has only 95: q-co-giao & 101: q-bo-doi)
    # must NEVER pad top_k with gt-* or letter/digit classes
    prof_res = recognizer.predict_sequence(seq.tolist(), mode="words", topic_id="Bài 6 - Nghề nghiệp", threshold=0.01)
    assert len(prof_res["top_k"]) <= 2
    assert {item["index"] for item in prof_res["top_k"]} <= {95, 101}
    assert all(item["entry_id"] in {"q-co-giao", "q-bo-doi"} for item in prof_res["top_k"])

    # Exhaustive check across ALL 36 gt-* classes: forcing peak logit on each GT class index never emits gt-*
    gt_indices = [x["index"] for x in gt_excluded]
    assert len(gt_indices) == 36
    orig_model = recognizer.sstcn_model
    for forced_idx in gt_indices:
        class _ForcedLogitsModel:
            num_classes = orig_model.num_classes
            class_names = orig_model.class_names
            entry_ids = orig_model.entry_ids
            topic_map = orig_model.topic_map
            word_class_indices = orig_model.word_class_indices
            preprocessing_version = orig_model.preprocessing_version

            def __call__(self, tensor_input):
                logits = torch.full((1, self.num_classes), -10.0, dtype=torch.float32)
                logits[0, forced_idx] = 100.0
                return logits

        with monkeypatch.context() as mp:
            mp.setattr(recognizer, "sstcn_model", _ForcedLogitsModel())
            out = recognizer.predict_sequence(
                seq.tolist(),
                mode="words",
                threshold=0.01,
                allowed_word_ids=approved_3231,
            )
            assert not (out.get("entry_id") or "").startswith("gt-"), forced_idx
            assert all(not (item.get("entry_id") or "").startswith("gt-") for item in out.get("top_k", []))

    # Direct API defense-in-depth across BOTH `record` (live_words=False) and `continuous` (live_words=True):
    # (1) /api/sstcn/topics reports 27 classes & 36 excluded, zero gt-* in words/topics
    # (2) If predictor ever leaks gt-toi or q-1, API blocks sign, top_k, subtitle token, and TTS cleanly
    with TestClient(app) as client:
        topics_resp = client.get("/api/sstcn/topics")
        assert topics_resp.status_code == 200
        tjson = topics_resp.json()
        assert tjson["word_classes_count"] == 27
        assert tjson["excluded_gt_count"] == 36
        assert all(not (w.get("entry_id") or "").startswith("gt-") for w in tjson["words"])

        for live_flag in (False, True):
            for leaked_payload, expected_reason in (
                (
                    {
                        "success": True,
                        "sign": "Tôi",
                        "label": "Tôi",
                        "entry_id": "gt-toi",
                        "confidence": 0.99,
                        "confirmed": True,
                        "new_token": "Tôi",
                        "tts_speak": "Tôi",
                        "top_k": [
                            {"index": 51, "term": "Tôi", "entry_id": "gt-toi", "confidence": 0.99},
                            {"index": 67, "term": "Đi", "entry_id": "q-di", "confidence": 0.01},
                        ],
                    },
                    "unverified_gt_curriculum",
                ),
                (
                    {
                        "success": True,
                        "sign": "1",
                        "label": "1",
                        "entry_id": "q-1",
                        "confidence": 0.99,
                        "confirmed": True,
                        "new_token": "1",
                        "tts_speak": "một",
                        "top_k": [{"index": 29, "term": "1", "entry_id": "q-1", "confidence": 0.99}],
                    },
                    "digit_not_word",
                ),
            ):
                with monkeypatch.context() as mp:
                    mp.setattr(qipedc_practice, "analyze_live_words", lambda *a, **kw: {"motion_detected": True, "status": "no_match"})
                    mp.setattr(qipedc_live, "match_live_words", lambda *a, **kw: None)
                    mp.setattr(recognizer, "predict_sequence", lambda **kwargs: dict(leaked_payload))
                    resp = client.post(
                        "/api/recognize_sequence",
                        json={
                            "sequence": seq.tolist(),
                            "mode": "words",
                            "live_words": live_flag,
                            "include_diagnostic": True,
                        },
                    )
                    assert resp.status_code == 200
                    data = resp.json()["data"]
                    assert data["sign"] == "NEUTRAL"
                    assert data["confidence"] == 0.0
                    assert data["confirmed"] is False
                    assert data["new_token"] is None
                    assert data["tts_speak"] is None
                    assert data["top_k"] == []
                    assert data["scope_rejected"] is True
                    assert data["reject_reason"] == expected_reason
                    assert data["diagnostic"]["reject_reason"] == expected_reason
                    assert data["diagnostic"]["rejected_entry_id"] == leaked_payload["entry_id"]


def test_12_other_modes_and_dictionary_browse_unaffected():
    # Dictionary endpoint preserves all 3,327 entries (zero gt-*, 66 camera entries)
    with TestClient(app) as client:
        lex_resp = client.get("/api/lexicon")
        assert lex_resp.status_code == 200
        lex_json = lex_resp.json()
        assert lex_json["total"] == 3327
        gt_entries_in_dict = [e for e in lex_json["entries"] if e["id"].startswith("gt-")]
        assert len(gt_entries_in_dict) == 0
        assert all(not (e.get("video_url") or "").startswith("https://gajsffqnyujwyllfcmny.supabase.co") for e in lex_json["entries"])
        assert all(not any(v.get("region") == "GT" for v in e.get("variants", [])) for e in lex_json["entries"])
        # Numbers 1..10 excluded from live_qipedc_ids in meta
        for digit_id in ("q-1", "q-5", "q-10"):
            assert digit_id not in lex_json["meta"]["live_qipedc_ids"]

        cam_resp = client.get("/api/lexicon", params={"filter": "camera"})
        assert cam_resp.status_code == 200
        assert cam_resp.json()["total"] == 66
        assert all(not e["id"].startswith("gt-") for e in cam_resp.json()["entries"])

        # Static recognition modes remain operational and independent
        v_lm = [[0.5, 0.7, 0.0] for _ in range(21)]
        v_lm[0] = [0.5, 0.8, 0.0]
        v_lm[9] = [0.5, 0.5, 0.0]
        v_lm[8] = [0.42, 0.25, 0.0]; v_lm[6] = [0.45, 0.45, 0.0]
        v_lm[12] = [0.58, 0.25, 0.0]; v_lm[10] = [0.55, 0.45, 0.0]
        v_lm[16] = [0.65, 0.65, 0.0]; v_lm[14] = [0.65, 0.55, 0.0]
        v_lm[20] = [0.70, 0.70, 0.0]; v_lm[18] = [0.70, 0.60, 0.0]
        v_lm[4] = [0.52, 0.65, 0.0]

        for mode_name in ("letters", "digits", "letters_numbers"):
            r = client.post("/api/recognize_landmarks", json={"landmarks": v_lm, "mode": mode_name})
            assert r.status_code == 200
            assert r.json()["success"] is True

        # Sentence mode sequence endpoint unaffected
        seq = _sample_sequence(30)
        rs = client.post("/api/recognize_sequence", json={"sequence": seq.tolist(), "mode": "sentence"})
        assert rs.status_code == 200
        assert "scope_rejected" not in rs.json()["data"]
