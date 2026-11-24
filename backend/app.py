"""
FastAPI Server cho Hệ thống Giao tiếp Hai chiều Ngôn ngữ Ký hiệu Việt Nam (VSL)
Hỗ trợ:
1. API Nhận diện Ký hiệu qua Tọa độ Khung xương & Khung hình Camera (Sign -> Text)
2. API Diễn giải Văn bản/Giọng nói sang Chuỗi Ký hiệu VSL & 3D Avatar Poses (Text -> Sign)
3. API Từ điển VSL Đa phương tiện & Kiểm tra Sức khỏe Hệ thống
"""

import os
import sys
import time
import re
import json
import base64
import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import Response
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Union, Dict, Any, Literal

from backend.models.sign_model import CLASSES
from backend.models.vsl_dictionary import ALL_SIGNS, VSL_ALPHABET, VSL_VOCABULARY
from backend.vision.recognizer import VSLRecognizer
from backend.vision import qipedc_live
from backend.vision.qipedc_live import (
    BUILTIN_WORDS, DEFAULT_WORDS, MAX_EXTRA_WORDS,
    indexed_entry_ids, lost_hand_during_interaction, prepare_live_reference,
)
from backend.vision.qipedc_practice import TIME_SIGN_ENTRY, TIME_SIGN_VIDEO
from backend.vision.word_scope import (
    allowed_live_word_ids,
    approved_live_word_ids,
    validate_extra_qipedc_word,
    validate_word_result_scope,
    verify_reference_video_id,
)
from backend.vision.alphabet_classifier import O_D_TI_MAX, O_IDX_STRAIGHT_MAX
from backend.nlp.text_to_vsl import text_to_vsl_glosses
from backend.models import video_cache
from backend.models.lexicon import load_lexicon, compact_entry, get_entry_by_sstcn, norm_key, sign_description
from fastapi import Query

app = FastAPI(
    title="Vietnamese Sign Language (VSL) Communication API",
    description="Backend Hybrid hỗ trợ dịch hai chiều Ký hiệu tay <-> Văn bản tiếng Việt",
    version="1.0.0"
)

# Cho phép kết nối CORS từ Frontend Web
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Nén gzip phản hồi lớn (bộ từ vựng ~2MB → ~300KB)
app.add_middleware(GZipMiddleware, minimum_size=2000)

# Khởi tạo động cơ nhận diện
recognizer = VSLRecognizer()

# Feature flag & cấu hình nhận diện cử chỉ ngắn (< 30 frame)
FEATURE_ALLOW_SHORT_SEQUENCES = os.environ.get("VSL_ALLOW_SHORT_SEQUENCES", "1").lower() in ("1", "true", "yes")
MIN_GESTURE_FRAMES = 12

# Pydantic Request Models
class LandmarkRequest(BaseModel):
    landmarks: Optional[List[List[float]]] = None
    multi_hands: Optional[List[dict]] = None
    handedness: Optional[str] = "Right"
    mode: Optional[str] = None
    timestamp: Optional[float] = None
    expected_sign: Optional[str] = None
    session_id: Optional[str] = None
    trial_id: Optional[str] = None
    frame_idx: Optional[int] = None
    client_capture_ts: Optional[float] = None
    landmark_extract_ts: Optional[float] = None
    request_send_ts: Optional[float] = None
    signer_id: Optional[str] = None
    image_width: Optional[int] = None
    image_height: Optional[int] = None
    handedness_score: Optional[float] = None
    tracking_confidence: Optional[float] = None
    mediapipe_model_complexity: Optional[int] = None
    processing_resolution: Optional[str] = None
    camera_fps: Optional[float] = None
    video_pts_sec: Optional[Union[float, str]] = None
    video_frame_idx: Optional[Union[int, str]] = None
    phase: Optional[Literal["PRE_IDLE", "ENTRY", "STABLE_HOLD", "EXIT", "POST_IDLE"]] = None

class FrameRequest(BaseModel):
    image_base64: str

class PilotControlRequest(BaseModel):
    enabled: bool
    filepath: Optional[str] = None
    session_id: Optional[str] = None

class PilotFlushRequest(BaseModel):
    session_id: Optional[str] = None
    trial_id: Optional[str] = None

class PilotTrialRequest(BaseModel):
    session_id: str
    trial_id: str

class PilotManifestRequest(BaseModel):
    session_id: str
    manifest: dict

class PilotSidecarRequest(BaseModel):
    session_id: str
    sync_data: dict

class PilotEvaluationRequest(BaseModel):
    session_id: str
    report: Dict[str, Any]
    frames: Optional[List[Dict[str, Any]]] = None

class SequenceRequest(BaseModel):
    sequence: List[List[List[float]]]  # Chuỗi T frames x 75 joints x 3 coords
    threshold: Optional[float] = 0.65
    mode: Optional[str] = None
    topic_id: Optional[str] = None
    timestamps: Optional[List[float]] = None
    debug_meta: Optional[Dict[str, Any]] = None
    live_words: bool = False
    vocab_scope: Literal["all", "common"] = "all"
    qipedc_entries: List[str] = Field(default_factory=list, max_length=MAX_EXTRA_WORDS)
    include_diagnostic: bool = False
    aspect_ratio: float = Field(default=1.0, ge=0.25, le=4.0)
    allow_short_sequences: Optional[bool] = None

class CalibrationSampleRequest(BaseModel):
    label: str
    entry_id: Optional[str] = None
    sequence: List[List[List[float]]]
    timestamps: Optional[List[float]] = None
    signer_id: Optional[str] = "personal"

class QipedcPracticeRequest(BaseModel):
    sequence: List[List[List[float]]] = Field(min_length=4, max_length=180)
    aspect_ratio: float = Field(default=1.0, ge=0.25, le=4.0)
    mirror: bool = False

class TextToSignRequest(BaseModel):
    text: str

@app.get("/api/health")
def health_check():
    sstcn_meta = recognizer.get_sstcn_topics_info()
    return {
        "status": "online",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "sstcn_classes": sstcn_meta["num_classes"],
        "sstcn_word_classes": sstcn_meta["word_classes_count"],
        "sstcn_preprocessing_version": sstcn_meta["preprocessing_version"],
        "sstcn_topics_count": len(sstcn_meta["topics"]),
        "total_classes": len(CLASSES),
        "total_vocabulary": len(VSL_VOCABULARY),
        "total_alphabet": len(VSL_ALPHABET),
        "lexicon_total": load_lexicon()["meta"]["total"],
        "lexicon_common": load_lexicon()["meta"]["common"],
        "classes": CLASSES
    }

@app.get("/api/pilot/config")
def get_pilot_runtime_config():
    """Expose the active classifier configuration for controlled pilot provenance."""
    from backend.vision.recognizer import (
        PILOT_CLASSIFIER_RULE_VERSION,
        PILOT_FEATURE_EXTRACTOR_VERSION,
        PILOT_SCHEMA_VERSION,
    )
    return {
        "success": True,
        "classifier_rule_version": PILOT_CLASSIFIER_RULE_VERSION,
        "schema_version": PILOT_SCHEMA_VERSION,
        "feature_extractor_version": PILOT_FEATURE_EXTRACTOR_VERSION,
        "classifier_config": {
            "O_D_TI_MAX": O_D_TI_MAX,
            "O_IDX_STRAIGHT_MAX": O_IDX_STRAIGHT_MAX,
        },
    }

@app.get("/api/dictionary")
def get_dictionary():
    """Lấy danh mục từ điển ký hiệu VSL đầy đủ kèm mô tả cử chỉ (bảng chữ cái + từ có tư thế robot)"""
    return {
        "alphabet": VSL_ALPHABET,
        "vocabulary": VSL_VOCABULARY,
        "total": len(ALL_SIGNS),
        "lexicon_total": load_lexicon()["meta"]["total"]
    }

@app.get("/api/lexicon")
def get_lexicon(
    q: Optional[str] = Query(None, description="Từ khóa tìm kiếm"),
    filter: Optional[str] = Query("all", description="all | common | camera | robot | alphabet"),
    limit: int = Query(0, ge=0, description="0 = trả về tất cả"),
    offset: int = Query(0, ge=0),
):
    """
    BỘ TỪ VỰNG VSL THỐNG NHẤT: ~3.300 từ QIPEDC (Bộ GD&ĐT, video 3 miền) + 102 từ giáo trình
    (mô hình camera) + từ có tư thế robot 3D. Dùng chung cho Camera → Văn bản và Văn bản → Ký hiệu.
    """
    lex = load_lexicon()
    items = lex["entries"]
    if filter == "common":
        items = [e for e in items if e.get("common") and not e.get("fingerspelling")]
    elif filter == "camera":
        items = sorted([e for e in items if e.get("sstcn_index") is not None], key=lambda e: e["sstcn_index"])
    elif filter == "robot":
        from backend.models.lexicon import motion_id_of, motion_ids, robot_pose_key
        avail = motion_ids()
        items = [e for e in items if robot_pose_key(e) or motion_id_of(e, avail)]
    elif filter == "alphabet":
        items = [e for e in items if e.get("fingerspelling")]
    if q:
        k = norm_key(q)
        items = [e for e in items if any(k in key for key in e.get("keys", [])) or k in norm_key(e.get("definition", ""))]
    total = len(items)
    if limit:
        items = items[offset:offset + limit]
    indexed_ids = indexed_entry_ids()
    live_allowed = allowed_live_word_ids(lex, indexed_ids=indexed_ids, builtin_ids=BUILTIN_WORDS)
    return {
        "success": True,
        "meta": {
            **lex["meta"],
            "live_qipedc_ids": [eid for eid in dict.fromkeys((*BUILTIN_WORDS, *indexed_ids)) if eid in live_allowed],
        },
        "total": total,
        "entries": [compact_entry(e) for e in items],
    }

class PrefetchRequest(BaseModel):
    ids: list[str] = []
    kind: str = "videos"


@app.get("/api/media/{kind}/{filename}")
async def get_media(kind: str, filename: str, request: Request):
    """Video/ảnh mẫu QIPEDC phát từ bộ đệm trên máy (đã faststart, hỗ trợ tua, cache dài hạn)."""
    if kind not in video_cache.KINDS or not filename.endswith(video_cache.KINDS[kind]):
        raise HTTPException(status_code=404, detail="Không hỗ trợ")
    media_id = filename[: -len(video_cache.KINDS[kind])]
    path = await run_in_threadpool(video_cache.ensure_cached, kind, media_id)
    if not path:
        raise HTTPException(status_code=502, detail="Không tải được từ máy chủ QIPEDC")
    status, data, headers = video_cache.read_range(path, request.headers.get("range"))
    headers["Content-Encoding"] = "identity"  # không nén gzip video/ảnh
    media_type = "video/mp4" if kind == "videos" else "image/png"
    return Response(content=data, status_code=status, headers=headers, media_type=media_type)


@app.on_event("startup")
def warm_common_videos():
    """Tải dần ở nền video của các từ THÔNG DỤNG (≈800 video, ~0,7 GB) vào bộ đệm, để Từ điển phát tức thì.
    Máy chủ QIPEDC phản hồi thất thường (0,2–5 giây/video) nên không chờ tới lúc người dùng bấm mới tải.
    Tắt bằng biến môi trường VSL_PREFETCH=0."""
    if os.environ.get("VSL_PREFETCH", "0") == "0" or "pytest" in sys.modules or "unittest" in sys.modules:
        return
    try:
        entries = sorted((e for e in load_lexicon()["entries"] if e.get("common")), key=lambda e: e.get("rank") or 1e9)
        ids = [mediaid_of(e) for e in entries]
        n = video_cache.warm_background("videos", [i for i in ids if i])
        if n:
            print(f"[media] Đang tải nền {n} video từ thông dụng vào bộ đệm (VSL_PREFETCH=0 để tắt)")
    except Exception as exc:  # noqa: BLE001
        print("[media] Không khởi động được tải nền:", exc)


def mediaid_of(entry):
    m = video_cache.VIDEO_URL_RE.search(entry.get("video_url") or "")
    return m.group(1) if m else None


def _qipedc_practice_entry(entry_id, video_id):
    entry = load_lexicon()["by_id"].get(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Mục từ này chưa có video mẫu QIPEDC.")
    default = mediaid_of(entry)
    available = {default, *(mediaid_of({"video_url": variant.get("url")}) for variant in entry.get("variants", []))}
    selected = video_id or default
    if not selected or selected not in available:
        raise HTTPException(status_code=404, detail="Mục từ này chưa có video mẫu QIPEDC phù hợp.")
    return entry, selected


def _live_qipedc_entries(extra_ids=(), allowed_ids=None):
    lex = load_lexicon()
    entries = []
    for entry_id in dict.fromkeys((*DEFAULT_WORDS, *extra_ids)):
        if entry_id == TIME_SIGN_ENTRY:
            continue
        if allowed_ids is not None and entry_id not in allowed_ids:
            continue
        validation = validate_extra_qipedc_word(entry_id, lex)
        if validation["allowed"] and validation["video_id"]:
            entries.append((entry_id, validation["video_id"]))
    return entries


@app.post("/api/qipedc/live/prepare")
def prepare_qipedc_live(entry_id: Optional[str] = None):
    lex = load_lexicon()
    raw_indexed = indexed_entry_ids()
    approved_scope = (
        approved_live_word_ids(lex, indexed_ids=raw_indexed, builtin_ids=BUILTIN_WORDS)
        if raw_indexed
        else None
    )
    eligible_ids = allowed_live_word_ids(
        lex,
        indexed_ids=raw_indexed,
        builtin_ids=BUILTIN_WORDS,
        allowed_override=approved_scope,
    )
    indexed = frozenset(eid for eid in raw_indexed if eid in eligible_ids)
    ids = (entry_id,) if entry_id else BUILTIN_WORDS
    prepared = []
    for item in ids:
        if item and item.startswith("gt-"):
            prepared.append({
                "id": item,
                "word": item,
                "ready": False,
                "builtin": False,
                "reject_reason": "unverified_gt_curriculum",
            })
            continue
        entry = lex["by_id"].get(item)
        if entry is None:
            raise HTTPException(status_code=404, detail="Mục từ này chưa có video mẫu QIPEDC.")
        validation = validate_extra_qipedc_word(item, lex, allowed_override=approved_scope)
        if not validation["allowed"]:
            prepared.append({
                "id": item,
                "word": entry["word"],
                "ready": False,
                "builtin": False,
                "reject_reason": validation["reason"],
            })
            continue
        video_id = validation["video_id"]
        if item in indexed:
            ready = True
        else:
            try:
                ready = prepare_live_reference(video_id)
            except (ImportError, OSError, RuntimeError, ValueError):
                ready = False
        prepared.append({
            "id": item,
            "word": entry["word"],
            "ready": ready,
            "builtin": item in BUILTIN_WORDS or item in indexed,
        })
    return {"entries": prepared, "indexed_count": len(indexed), "max_extra_words": MAX_EXTRA_WORDS}


@app.post("/api/qipedc/practice/{entry_id}/prepare")
def prepare_qipedc_practice(entry_id: str, video_id: Optional[str] = None):
    from backend.vision.qipedc_practice import prepare_reference

    _, video_id = _qipedc_practice_entry(entry_id, video_id)
    try:
        prepare_reference(video_id)
    except ImportError as exc:
        raise HTTPException(status_code=503, detail="Cần cài MediaPipe 0.10.21 để trích xuất video mẫu.") from exc
    except (OSError, RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Chưa trích được mẫu từ video QIPEDC. Hãy thử lại hoặc chọn video khác.") from exc
    return {"success": True, "video_id": video_id}


@app.post("/api/qipedc/practice/{entry_id}/match")
def match_qipedc_practice(entry_id: str, req: QipedcPracticeRequest, video_id: Optional[str] = None):
    from backend.vision.qipedc_practice import compare_recording

    entry, video_id = _qipedc_practice_entry(entry_id, video_id)
    try:
        result = compare_recording(video_id, req.sequence, aspect_ratio=req.aspect_ratio, mirror=req.mirror)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return {"success": True, "entry_id": entry_id, "word": entry["word"], **result}


@app.get("/api/media/diagnose")
def diagnose_media():
    """Kiểm tra tốc độ tải video từ QIPEDC trên máy chạy backend."""
    return video_cache.diagnose()


@app.post("/api/media/prefetch")
def prefetch_media(req: PrefetchRequest):
    """Tải trước ở nền các video sắp xem (ô đang hiện trên màn hình / di chuột tới)."""
    if req.kind not in video_cache.KINDS:
        raise HTTPException(status_code=400, detail="kind không hợp lệ")
    return {"success": True, "queued": video_cache.prefetch(req.kind, req.ids)}


@app.get("/api/motion/{motion_id}")
def get_motion(motion_id: str):
    """
    Chuyển động robot đã HỌC từ video QIPEDC (MediaPipe Holistic → IK → góc khớp robot).
    Trả về keyframes theo thời gian (giây) với đúng tên khớp RobotSigner3D dùng.
    """
    import re as _re
    from backend.models.lexicon import MOTION_DIR
    from backend.models.motion_format import frames_to_keyframes  # không cần scipy
    if not _re.fullmatch(r"[A-Za-z0-9_-]+", motion_id):
        raise HTTPException(status_code=400, detail="Mã không hợp lệ")
    path = os.path.join(MOTION_DIR, motion_id + ".json")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Chưa có chuyển động cho ký hiệu này")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {"success": True, "motion": frames_to_keyframes(data)}

class LearnRequest(BaseModel):
    id: str
    fps: float = 8.0
    dur: float = 0.0
    frames: list = []


@app.post("/api/motion/learn")
def learn_motion(req: LearnRequest):
    """Nhận điểm khớp MediaPipe (trình duyệt trích từ video QIPEDC) của 1 từ chưa có chuyển động
    → chuyển sang góc khớp robot, lưu lại và trả về để robot diễn ngay."""
    from backend.models.motion_learn import learn
    return learn(req.model_dump() if hasattr(req, "model_dump") else req.dict())


@app.post("/api/motion/import")
async def import_raw_motions(request: Request):
    """Nhập HÀNG LOẠT điểm khớp đã trích từ video (công cụ cho nhà phát triển).

    Nhận form (application/x-www-form-urlencoded) với trường data = base64(gzip(JSON danh sách
    {id, fps, dur, frames})) — gửi được bằng một form POST thông thường từ trang QIPEDC (nơi kịch bản
    extract_motion_browser.js lưu kết quả), không cần quyền gọi API khác nguồn. Mỗi mục được kiểm tra
    và xử lý y như /api/motion/learn (lưu điểm khớp gốc, chuyển sang robot, viết diễn tả)."""
    import gzip
    from urllib.parse import parse_qs
    from backend.models.motion_learn import learn
    body = await request.body()
    try:
        data = parse_qs(body.decode("ascii")).get("data", [""])[0]
        raws = json.loads(gzip.decompress(base64.b64decode(data)).decode("utf-8"))
        if not isinstance(raws, list):
            raise ValueError
    except Exception:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="Dữ liệu không hợp lệ")

    def run():
        ok, bad = [], []
        for raw in raws[:400]:
            r = learn(raw if isinstance(raw, dict) else {})
            (ok if r.get("success") else bad).append(str((raw or {}).get("id"))[:40] if r.get("success") else [str((raw or {}).get("id"))[:40], r.get("reason")])
        return {"success": True, "imported": len(ok), "failed": bad, "ids": ok}
    return await run_in_threadpool(run)


@app.get("/api/lexicon/{entry_id}")
def get_lexicon_entry(entry_id: str):
    e = load_lexicon()["by_id"].get(entry_id)
    if not e:
        raise HTTPException(status_code=404, detail="Không tìm thấy mục từ")
    ce = compact_entry(e)
    ce["sign_description"] = sign_description(ce.get("motion_id"))
    return {"success": True, "entry": ce}

VALID_CONVERSION_MODES = {"letters", "digits", "words", "sentence", "letters_numbers"}

@app.post("/api/recognize_landmarks")
def recognize_landmarks(req: LandmarkRequest, request: Request):
    """
    Nhận diện cử chỉ từ danh sách 21 điểm mốc bàn tay hoặc 2 bàn tay do MediaPipe Web trích xuất
    Đảm bảo độ trễ siêu thấp (<10ms) và tính chính xác cao.
    Hỗ trợ lưu đầy đủ 21 landmarks thô, đồng bộ video RGB và temporal diagnostics (Task 3.10).
    """
    backend_receive_ts = time.time()
    try:
        expected_sign = req.expected_sign or request.headers.get("x-expected-sign")
        session_id = req.session_id or request.headers.get("x-session-id")
        trial_id = req.trial_id or request.headers.get("x-trial-id")
        raw_frame_idx = req.frame_idx if req.frame_idx is not None else request.headers.get("x-frame-idx")
        frame_idx = None
        if raw_frame_idx is not None:
            try:
                frame_idx = int(raw_frame_idx)
            except (ValueError, TypeError):
                frame_idx = None

        mode = req.mode or request.headers.get("x-conversion-mode")
        if mode is not None and mode not in VALID_CONVERSION_MODES:
            raise HTTPException(
                status_code=400,
                detail=f"Chế độ nhận diện không hợp lệ: '{mode}'. Hỗ trợ: 'letters', 'digits', 'words', 'sentence', 'letters_numbers'."
            )

        client_capture_ts = req.client_capture_ts
        if client_capture_ts is None and request.headers.get("x-client-capture-ts"):
            try:
                client_capture_ts = float(request.headers.get("x-client-capture-ts"))
            except (ValueError, TypeError):
                client_capture_ts = None

        landmark_extract_ts = req.landmark_extract_ts
        if landmark_extract_ts is None and request.headers.get("x-landmark-extract-ts"):
            try:
                landmark_extract_ts = float(request.headers.get("x-landmark-extract-ts"))
            except (ValueError, TypeError):
                landmark_extract_ts = None

        request_send_ts = req.request_send_ts
        if request_send_ts is None and request.headers.get("x-request-send-ts"):
            try:
                request_send_ts = float(request.headers.get("x-request-send-ts"))
            except (ValueError, TypeError):
                request_send_ts = None

        signer_id = req.signer_id or request.headers.get("x-signer-id")
        video_pts_sec = req.video_pts_sec if req.video_pts_sec is not None else request.headers.get("x-video-pts-sec")
        video_frame_idx = req.video_frame_idx if req.video_frame_idx is not None else request.headers.get("x-video-frame-idx")

        result = recognizer.predict_landmarks(
            landmarks_list=req.landmarks or [],
            multi_hands=req.multi_hands,
            expected_sign=expected_sign,
            session_id=session_id,
            mode=mode,
            timestamp=req.timestamp,
            trial_id=trial_id,
            frame_idx=frame_idx,
            client_capture_ts=client_capture_ts,
            landmark_extract_ts=landmark_extract_ts,
            request_send_ts=request_send_ts,
            backend_receive_ts=backend_receive_ts,
            video_pts_sec=video_pts_sec,
            video_frame_idx=video_frame_idx,
            signer_id=signer_id,
            image_width=req.image_width,
            image_height=req.image_height,
            handedness_score=req.handedness_score,
            tracking_confidence=req.tracking_confidence,
            mediapipe_model_complexity=req.mediapipe_model_complexity,
            processing_resolution=req.processing_resolution,
            camera_fps=req.camera_fps,
            phase=req.phase,
        )
        return {"success": True, "data": result}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pilot/control")
def control_pilot_logging(req: PilotControlRequest):
    """Bật/tắt chế độ ghi nhật ký chẩn đoán pilot từ UI (Task 7F.5B / 7F.5D Fix B-01)."""
    if req.filepath is not None:
        raise HTTPException(
            status_code=400,
            detail="Chỉ định filepath qua API bị từ chối vì lý do an toàn bảo mật. Vui lòng chỉ truyền session_id."
        )

    if req.enabled:
        raw_sess = req.session_id
        if raw_sess is not None:
            from backend.vision.pilot_logger import validate_session_id
            try:
                clean_sess = validate_session_id(raw_sess)
            except ValueError as ve:
                raise HTTPException(status_code=400, detail=str(ve))
        else:
            clean_sess = f"session_{time.strftime('%Y%m%d_%H%M%S')}"

        target_dir = getattr(
            recognizer,
            "diagnostic_dir",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diagnostics", "pilot")
        )

        try:
            # Kiểm tra xung đột session trước khi tạo file mới
            if recognizer.pilot_logging_enabled and recognizer.active_pilot_session_id:
                if clean_sess != recognizer.active_pilot_session_id:
                    raise HTTPException(
                        status_code=409,
                        detail=f"Phiên '{recognizer.active_pilot_session_id}' đang hoạt động. "
                               f"Vui lòng tắt phiên hiện tại trước khi bật phiên mới '{clean_sess}'."
                    )
                logger = recognizer.pilot_logger
                return {
                    "success": True,
                    "enabled": True,
                    "session_id": recognizer.active_pilot_session_id,
                    "filepath": recognizer.pilot_filepath,
                    "stats": logger.get_stats() if logger else None
                }

            from backend.vision.pilot_logger import create_safe_pilot_file
            target_file = create_safe_pilot_file(target_dir, clean_sess)
            logger = recognizer.enable_pilot_logging(session_id=clean_sess, filepath=target_file)
            return {
                "success": True,
                "enabled": True,
                "session_id": clean_sess,
                "filepath": target_file,
                "stats": logger.get_stats()
            }
        except HTTPException:
            raise
        except ValueError as ve:
            raise HTTPException(status_code=400, detail=str(ve))
        except RuntimeError as re_err:
            raise HTTPException(status_code=409, detail=str(re_err))
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    else:
        active_session_id = getattr(recognizer, "active_pilot_session_id", None)
        if (
            recognizer.pilot_logging_enabled
            and req.session_id is not None
            and req.session_id != active_session_id
        ):
            raise HTTPException(
                status_code=409,
                detail=f"Không thể đóng session '{req.session_id}': backend đang ghi session '{active_session_id}'.",
            )
        recognizer.disable_pilot_logging()
        return {"success": True, "enabled": False, "session_id": active_session_id}


@app.post("/api/pilot/flush")
def flush_pilot_log(req: Optional[PilotFlushRequest] = None):
    """Return the logger barrier result and counters; HTTP 200 alone is not a flush acknowledgement."""
    logger = getattr(recognizer, "pilot_logger", None)
    session_id = getattr(recognizer, "active_pilot_session_id", None)
    filepath = getattr(recognizer, "pilot_filepath", None)
    expected_session_id = req.session_id if req else None
    stats_before = None
    stats = None
    stats_error = None
    if logger is not None:
        try:
            stats_before = logger.get_stats()
        except Exception as exc:
            stats_error = str(exc)
    flush_attempted = False
    flush_success = False
    error_code = None
    error = None

    if logger is None:
        error_code = "logger_unavailable"
        error = "No pilot logger is available to verify a flush."
    elif not isinstance(session_id, str) or not session_id or not isinstance(filepath, str) or not filepath:
        error_code = "logger_unavailable"
        error = "Logger session or output filepath is unavailable."
    elif expected_session_id is not None and expected_session_id != session_id:
        error_code = "session_mismatch"
        error = f"Requested session '{expected_session_id}' does not match active session '{session_id}'."
    elif not getattr(recognizer, "pilot_logging_enabled", False):
        error_code = "logging_disabled"
        error = "Pilot logging is disabled; no flush was attempted."
    elif getattr(logger, "state", None) != getattr(logger, "STATE_OPEN", "OPEN") or not getattr(logger, "is_running", False):
        error_code = "logger_unavailable"
        error = "Pilot logger is not open and running."
    else:
        flush_attempted = True
        try:
            flush_success = bool(logger.flush())
        except Exception as exc:
            error_code = "flush_failed"
            error = str(exc)

    if logger is not None:
        try:
            stats = logger.get_stats()
        except Exception as exc:
            stats_error = stats_error or str(exc)
    if flush_attempted and not flush_success and error_code is None:
        before_dropped = (stats_before or {}).get("total_dropped", 0)
        after_dropped = (stats or {}).get("total_dropped", 0)
        error_code = "io_error" if after_dropped > before_dropped else "flush_failed"
        error = "Logger reported an I/O/drop error during the flush barrier." if error_code == "io_error" else "Logger flush barrier did not complete successfully."

    required_stats = (
        "queued_count", "logged_count", "pending_in_queue",
        "dropped_queue_full", "dropped_io_error", "total_dropped",
    )
    stats_complete = stats is not None and all(stats.get(key) is not None for key in required_stats)
    if error_code is None and not stats_complete:
        error_code = "stats_unavailable"
        error = stats_error or "Logger did not provide all required accounting counters."
    if error_code is None and stats["total_dropped"] > 0:
        error_code = "io_error"
        error = "Logger counters report one or more dropped records."
    if error_code is None and (stats["pending_in_queue"] != 0 or stats["queued_count"] != stats["logged_count"]):
        error_code = "record_accounting_mismatch"
        error = "Queue is not empty or queued_count does not equal logged_count."

    success = bool(flush_attempted and flush_success and error_code is None)
    counter_fields = {
        key: stats.get(key) if stats is not None else None
        for key in required_stats
    }
    return {
        "success": success,
        "flush_success": flush_success,
        "flush_attempted": flush_attempted,
        "flush_semantics": "FIFO worker barrier processed without a logger-reported drop before the barrier; this does not guarantee fsync or physical-media durability.",
        "error_code": error_code,
        "error": error,
        "session_id": session_id,
        "requested_session_id": expected_session_id,
        "filepath": filepath,
        "counter_scope": "session_cumulative",
        "stats": stats,
        **counter_fields,
    }


@app.get("/api/pilot/stats")
def get_pilot_stats():
    """Lấy thống kê ghi nhật ký pilot (Task 7F.5A)."""
    stats = recognizer.pilot_logger.get_stats() if hasattr(recognizer, "pilot_logger") and recognizer.pilot_logger else None
    return {
        "success": True,
        "stats": stats,
        "session_id": getattr(recognizer, "active_pilot_session_id", None),
        "filepath": getattr(recognizer, "pilot_filepath", None),
        "has_static_head": bool(getattr(recognizer, "has_static_head", False)),
        "pilot_logging_enabled": bool(getattr(recognizer, "pilot_logging_enabled", False))
    }


def _pilot_trial_context(session_id: str, trial_id: str):
    from backend.vision.pilot_logger import validate_session_id, validate_trial_id
    try:
        clean_session = validate_session_id(session_id)
        clean_trial = validate_trial_id(trial_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error_code": "invalid_trial_id", "error": str(exc)})

    active_session = getattr(recognizer, "active_pilot_session_id", None)
    logger = getattr(recognizer, "pilot_logger", None)
    if active_session != clean_session:
        raise HTTPException(status_code=409, detail={
            "error_code": "session_mismatch", "session_id": active_session,
            "requested_session_id": clean_session,
        })
    if (
        logger is None or not getattr(recognizer, "pilot_logging_enabled", False) or
        getattr(logger, "state", None) != getattr(logger, "STATE_OPEN", "OPEN") or
        not getattr(logger, "is_running", False)
    ):
        raise HTTPException(status_code=503, detail={"error_code": "logger_unavailable"})
    return logger, clean_session, clean_trial


@app.post("/api/pilot/trial/open")
def open_pilot_trial(req: PilotTrialRequest):
    """Reserve a unique trial accounting scope before its first recognition frame."""
    logger, session_id, trial_id = _pilot_trial_context(req.session_id, req.trial_id)
    try:
        created = logger.register_trial(session_id, trial_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error_code": "invalid_trial_id", "error": str(exc)})
    if not created:
        raise HTTPException(status_code=409, detail={"error_code": "duplicate_trial_id"})
    return {
        "success": True, "session_id": session_id, "trial_id": trial_id,
        "counter_scope": "session_trial", "trial_exists": True,
        "stats": logger.get_trial_stats(session_id, trial_id),
    }


@app.get("/api/pilot/trial/accounting")
def get_pilot_trial_accounting(session_id: str = Query(...), trial_id: str = Query(...)):
    """Read backend-owned counters for one trial, independent of frontend responses."""
    logger, clean_session, clean_trial = _pilot_trial_context(session_id, trial_id)
    try:
        stats = logger.get_trial_stats(clean_session, clean_trial)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error_code": "invalid_trial_id", "error": str(exc)})
    if stats is None:
        raise HTTPException(status_code=404, detail={
            "error_code": "trial_not_found", "session_id": clean_session,
            "trial_id": clean_trial, "trial_exists": False,
        })
    verified = bool(
        stats["sealed"] and stats["flush_barrier_passed"] and stats["record_count"] > 0 and
        stats["pending_count"] == 0 and stats["queued_count"] == stats["logged_count"] and
        stats["total_dropped"] == 0 and stats["late_record_count"] == 0
    )
    return {
        "success": True, "session_id": clean_session, "trial_id": clean_trial,
        "counter_scope": "session_trial", "trial_exists": True,
        "trial_verified": verified,
        "verification_status": "VERIFIED" if verified else "UNKNOWN_OR_INVALID",
        "flush_barrier_passed": stats["flush_barrier_passed"],
        "error_code": None if verified else (
            "record_dropped" if stats["total_dropped"] else
            "queue_pending" if stats["pending_count"] else
            "late_record_after_seal" if stats["late_record_count"] else
            "flush_not_verified" if not stats["flush_barrier_passed"] else
            "trial_has_no_records" if stats["record_count"] == 0 else
            "trial_accounting_mismatch"
        ),
        "stats": stats,
    }


@app.post("/api/pilot/trial/flush")
def flush_pilot_trial(req: PilotTrialRequest):
    """Seal a trial scope, drain its pre-barrier records, and return scoped accounting."""
    logger, session_id, trial_id = _pilot_trial_context(req.session_id, req.trial_id)
    try:
        sealed = logger.seal_trial(session_id, trial_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail={"error_code": "invalid_trial_id", "error": str(exc)})
    if sealed is None:
        raise HTTPException(status_code=404, detail={
            "error_code": "trial_not_found", "session_id": session_id,
            "trial_id": trial_id, "trial_exists": False,
        })
    try:
        barrier_passed = bool(logger.flush())
    except Exception:
        barrier_passed = False
    logger.mark_trial_flush(session_id, trial_id, barrier_passed)
    stats = logger.get_trial_stats(session_id, trial_id)
    verified = bool(
        barrier_passed and stats and stats["record_count"] > 0 and
        stats["pending_count"] == 0 and stats["queued_count"] == stats["logged_count"] and
        stats["total_dropped"] == 0 and stats["late_record_count"] == 0
    )
    error_code = None if verified else (
        "flush_failed" if not barrier_passed else
        "record_dropped" if stats["total_dropped"] else
        "queue_pending" if stats["pending_count"] else
        "late_record_after_seal" if stats["late_record_count"] else
        "trial_has_no_records" if stats["record_count"] == 0 else
        "trial_accounting_mismatch"
    )
    return {
        "success": verified, "flush_success": barrier_passed,
        "flush_semantics": "FIFO worker barrier processed; does not guarantee fsync or physical-media durability.",
        "session_id": session_id, "trial_id": trial_id,
        "filepath": getattr(recognizer, "pilot_filepath", None),
        "counter_scope": "session_trial", "trial_exists": True,
        "trial_verified": verified,
        "verification_status": "VERIFIED" if verified else "UNKNOWN_OR_INVALID",
        "error_code": error_code, "stats": stats,
    }


@app.post("/api/pilot/manifest")
def save_pilot_manifest(req: PilotManifestRequest):
    """Lưu session manifest kèm SHA256 cho phiên thử nghiệm pilot (Task 3.10 Section 8)."""
    try:
        from backend.vision.pilot_logger import save_session_manifest, validate_session_id
        validate_session_id(req.session_id)
        target_dir = getattr(
            recognizer,
            "diagnostic_dir",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diagnostics", "pilot")
        )
        saved_path = save_session_manifest(req.manifest, log_dir=target_dir)
        return {"success": True, "manifest_path": saved_path}
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pilot/sidecar_sync")
def save_pilot_sidecar(req: PilotSidecarRequest):
    """Lưu sidecar synchronization file ánh xạ video và JSONL (Task 3.10 Implementation C)."""
    try:
        from backend.vision.pilot_logger import save_sidecar_sync, validate_session_id
        validate_session_id(req.session_id)
        target_dir = getattr(
            recognizer,
            "diagnostic_dir",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diagnostics", "pilot")
        )
        saved_path = save_sidecar_sync(req.session_id, req.sync_data, log_dir=target_dir)
        return {"success": True, "sidecar_path": saved_path}
    except FileExistsError:
        raise HTTPException(status_code=409, detail="Diagnostic sidecar for this trial already exists; refusing to overwrite it.")
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pilot/save_evaluation")
def save_pilot_evaluation(req: PilotEvaluationRequest):
    """
    Lưu kết quả chẩn đoán Controlled Pilot 29 chữ cái vào backend/data/diagnostics/
    Bao gồm file JSON tổng kết và file JSONL chi tiết từng frame (nếu có).
    """
    try:
        diag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diagnostics")
        os.makedirs(diag_dir, exist_ok=True)

        clean_sess = re.sub(r'[^a-zA-Z0-9_-]', '_', req.session_id)
        if not clean_sess:
            clean_sess = f"session_{int(time.time())}"

        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        json_filename = f"pilot_eval_29_{clean_sess}_{timestamp_str}.json"
        json_path = os.path.join(diag_dir, json_filename)

        req.report["saved_at"] = time.strftime("%Y-%m-%d %H:%M:%S")
        req.report["server_timestamp"] = time.time()
        req.report["filename"] = json_filename

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(req.report, f, ensure_ascii=False, indent=2)

        jsonl_filename = None
        jsonl_path = None
        if req.frames:
            jsonl_filename = f"pilot_eval_29_{clean_sess}_{timestamp_str}_frames.jsonl"
            jsonl_path = os.path.join(diag_dir, jsonl_filename)
            with open(jsonl_path, "w", encoding="utf-8") as f:
                for frame in req.frames:
                    f.write(json.dumps(frame, ensure_ascii=False) + "\n")

        return {
            "success": True,
            "filename": json_filename,
            "filepath": json_path,
            "frames_filename": jsonl_filename,
            "frames_filepath": jsonl_path,
            "total_letters_evaluated": len(req.report.get("letters", {})),
            "summary": req.report.get("summary", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi khi lưu file JSON chẩn đoán: {str(e)}")


@app.get("/api/pilot/evaluations")
def list_pilot_evaluations():
    """Liệt kê danh sách các file báo cáo chẩn đoán trong backend/data/diagnostics/"""
    try:
        diag_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "diagnostics")
        if not os.path.exists(diag_dir):
            return {"success": True, "evaluations": []}

        evaluations = []
        for fname in os.listdir(diag_dir):
            if fname.startswith("pilot_eval_29_") and fname.endswith(".json"):
                fpath = os.path.join(diag_dir, fname)
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        data = json.load(f)
                    evaluations.append({
                        "filename": fname,
                        "session_id": data.get("session_id", fname),
                        "saved_at": data.get("saved_at", ""),
                        "mode": data.get("config", {}).get("mode", "single_letter"),
                        "total_letters": len(data.get("letters", {})),
                        "summary": data.get("summary", {})
                    })
                except Exception:
                    continue

        evaluations.sort(key=lambda x: x.get("saved_at", ""), reverse=True)
        return {"success": True, "evaluations": evaluations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _annotate_qipedc_candidates(qipedc_trace, by_id):
    if not isinstance(qipedc_trace, dict):
        return
    for key in ("top_candidates",):
        for item in qipedc_trace.get(key) or ():
            entry = by_id.get(item.get("entry_id"))
            item["word"] = entry["word"] if entry else item.get("entry_id")
            if entry:
                item["common"] = bool(entry.get("common"))
    shape = qipedc_trace.get("bilateral_shape")
    if isinstance(shape, dict):
        for item in shape.get("top_candidates") or ():
            entry = by_id.get(item.get("entry_id"))
            item["word"] = entry["word"] if entry else item.get("entry_id")
            if entry:
                item["common"] = bool(entry.get("common"))


def _build_sequence_explanation(
    *,
    final_sign,
    final_label,
    source,
    confidence,
    status,
    motion_trace,
    qipedc_trace,
    sstcn_result,
    raw_len,
    threshold,
):
    if status == "insufficient_landmarks":
        return f"Thiếu khung hình hợp lệ có đủ 2 vai và bàn tay ({raw_len} frame gửi lên)."
    if status == "no_motion" and motion_trace:
        return (
            f"Biên độ chuyển động tay quá nhỏ: trái={motion_trace.get('left_extent', 0)}, "
            f"phải={motion_trace.get('right_extent', 0)} (ngưỡng yêu cầu >= {motion_trace.get('motion_threshold', 0.16)})."
        )
    if status == "lost_hand_during_interaction":
        return "Phát hiện 2 tay đưa gần nhau nhưng sau đó mất dấu 1 tay, nên tạm chặn dự đoán."
    if status in ("missing_both_hands", "tracking_gaps"):
        return f"Dữ liệu bám tay bị đứt đoạn ({status}), chưa đủ để xác nhận."
    if status in ("matched", "word_matched"):
        if qipedc_trace and qipedc_trace.get("top_candidates"):
            top1 = qipedc_trace["top_candidates"][0]
            top2 = qipedc_trace["top_candidates"][1] if len(qipedc_trace["top_candidates"]) > 1 else None
            gap = qipedc_trace.get("gap_top1_top2")
            method = qipedc_trace.get("verdict") or "qipedc"
            tail = (
                f", hơn đối thủ #{2} “{top2['word']}” ({top2['distance']}) một khoảng {gap}"
                if top2 and gap is not None else ""
            )
            return (
                f"Khớp mẫu QIPEDC “{final_label}” ({method}, nhóm {qipedc_trace.get('bucket_hands')} tay/"
                f"{qipedc_trace.get('bucket_total_words')} từ, dist={top1['distance']} <= {qipedc_trace.get('max_distance')}{tail})."
            )
        return f"Khớp mẫu QIPEDC “{final_label}” (độ tương đồng {confidence})."
    if status == "ambiguous_word" and qipedc_trace:
        cands = qipedc_trace.get("top_candidates") or []
        summary = ", ".join(f"“{c.get('word')}” ({c.get('distance')})" for c in cands[:3])
        extra = " [CẢNH BÁO: tay thứ 2 lọt >=2 frame nên bị ép sang nhóm 2 tay]" if qipedc_trace.get("bilateral_forced_by_2frame_rule") else ""
        return (
            f"Bị từ chối do khoảng cách Top 1 và Top 2 quá sát (gap={qipedc_trace.get('gap_top1_top2')} < "
            f"{qipedc_trace.get('min_separation')}, nhóm {qipedc_trace.get('bucket_hands')} tay/{qipedc_trace.get('bucket_total_words')} từ): "
            f"{summary}.{extra}"
        )
    if raw_len < 30 and sstcn_result is None:
        q_reason = ""
        if qipedc_trace:
            if qipedc_trace.get("variants_reject_reason"):
                q_reason = f" QIPEDC bỏ qua vì {qipedc_trace['variants_reject_reason']}."
            elif qipedc_trace.get("top_candidates"):
                top1 = qipedc_trace["top_candidates"][0]
                q_reason = (
                    f" QIPEDC gần nhất là “{top1.get('word')}” (dist={top1.get('distance')} > ngưỡng {qipedc_trace.get('max_distance')})."
                )
        return f"Chưa đủ 30 frame ({raw_len}/30) để chạy SSTCN và chưa khớp mẫu QIPEDC.{q_reason}"
    if sstcn_result is not None:
        if sstcn_result.get("fallback_blocked"):
            return f"SSTCN bị chặn: {sstcn_result.get('blocked_reason')} — {sstcn_result.get('label')}."
        top_k = sstcn_result.get("top_k") or []
        top_str = ", ".join(f"“{t.get('term')}” ({round(t.get('confidence', 0) * 100)}%)" for t in top_k[:3])
        q_note = ""
        if qipedc_trace:
            if qipedc_trace.get("variants_reject_reason"):
                q_note = f"QIPEDC không chạy ({qipedc_trace['variants_reject_reason']}). "
            elif qipedc_trace.get("top_candidates"):
                top1 = qipedc_trace["top_candidates"][0]
                q_note = f"QIPEDC ngoài ngưỡng (gần nhất: “{top1.get('word')}” dist={top1.get('distance')}). "
        if final_sign != "NEUTRAL":
            return f"{q_note}SSTCN chọn “{final_label}” ({round(float(confidence or 0) * 100)}% >= {round(float(threshold or 0.65) * 100)}%). Top 3: {top_str}."
        return f"{q_note}SSTCN dưới ngưỡng {round(float(threshold or 0.65) * 100)}% (cao nhất: {top_str or 'không có'})."
    return final_label or "Không xác định"


@app.post("/api/recognize_sequence")
def recognize_sequence(req: SequenceRequest, request: Request):
    """
    Nhận diện chuỗi chuyển động ký hiệu VSL qua mô hình SSTCN_Attention (T frames x 75 joints).
    Hỗ trợ tự động chuẩn hóa body-centric v2, lọc chủ đề topic_id và adaptive timestamp resampling.
    """
    try:
        t_req_start = time.perf_counter()
        mode = req.mode or request.headers.get("x-conversion-mode")
        live_words = req.live_words and mode == "words"
        sequence, timestamps = req.sequence, req.timestamps
        raw_len = len(sequence)
        diag_enabled = bool(req.include_diagnostic)
        motion_trace = {} if diag_enabled else None
        qipedc_trace = {} if diag_enabled else None
        timings_ms = {}
        pipeline_steps = []
        diagnostic = {}
        lex = load_lexicon() if mode == "words" else None
        raw_indexed = indexed_entry_ids() if mode == "words" else ()
        approved_scope = (
            approved_live_word_ids(lex, indexed_ids=raw_indexed, builtin_ids=BUILTIN_WORDS)
            if mode == "words" and raw_indexed
            else None
        )

        allow_short = req.allow_short_sequences if req.allow_short_sequences is not None else FEATURE_ALLOW_SHORT_SEQUENCES
        min_required_len = MIN_GESTURE_FRAMES if allow_short else 30

        if not live_words and len(sequence) < min_required_len:
            return {"success": True, "data": {
                "success": True, "sign": "NEUTRAL", "label": "Đang chờ động tác từ vựng…",
                "confidence": 0.0, "top_k": [], "confirmed": False, "new_token": None,
                "motion_detected": False,
                **diagnostic,
            }}

        if live_words:
            from backend.vision.qipedc_practice import analyze_live_words

            valid_extra_ids = []
            rejected_extra_words = []
            for extra_id in req.qipedc_entries or ():
                check = validate_extra_qipedc_word(extra_id, lex, allowed_override=approved_scope)
                if check["allowed"]:
                    valid_extra_ids.append(extra_id)
                else:
                    rejected_extra_words.append({"entry_id": extra_id, "reason": check["reason"]})

            allowed_live_ids = allowed_live_word_ids(
                lex,
                indexed_ids=raw_indexed,
                builtin_ids=BUILTIN_WORDS,
                extra_ids=valid_extra_ids,
                topic_id=req.topic_id,
                allowed_override=approved_scope,
            )
            enable_time_sign = (
                TIME_SIGN_ENTRY in allowed_live_ids
                and req.topic_id in (None, "", "Bài 2 - Số đếm và Thời gian")
                and verify_reference_video_id(TIME_SIGN_VIDEO)["eligible"]
            )

            window = sequence[-120:]
            t0 = time.perf_counter()
            analyze_kwargs = {
                "enable_time_sign": enable_time_sign,
                "timestamps": timestamps[-120:] if timestamps is not None and len(timestamps) == len(sequence) else None,
            }
            if diag_enabled:
                analyze_kwargs["trace"] = motion_trace
            details = analyze_live_words(window, req.aspect_ratio, **analyze_kwargs)
            timings_ms["analyze_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            if diag_enabled:
                pipeline_steps.append({
                    "step": "1_motion_and_time_gate",
                    "ms": timings_ms["analyze_ms"],
                    "motion_detected": details.get("motion_detected"),
                    "status": details.get("status"),
                    "max_extent": motion_trace.get("max_extent"),
                    "threshold": motion_trace.get("motion_threshold"),
                })
            if details["motion_detected"]:
                if allowed_live_ids:
                    entries = _live_qipedc_entries(valid_extra_ids, allowed_ids=allowed_live_ids)
                    if details["status"] == "matched" and TIME_SIGN_ENTRY in allowed_live_ids:
                        entries.append((TIME_SIGN_ENTRY, TIME_SIGN_VIDEO))
                    t1 = time.perf_counter()
                    match_kwargs = {"allowed_ids": allowed_live_ids, "vocab_scope": req.vocab_scope}
                    if diag_enabled:
                        match_kwargs["trace"] = qipedc_trace
                    generic = qipedc_live.match_live_words(window, req.aspect_ratio, entries, **match_kwargs)
                    timings_ms["qipedc_match_ms"] = round((time.perf_counter() - t1) * 1000, 1)
                    if diag_enabled:
                        _annotate_qipedc_candidates(qipedc_trace, lex["by_id"])
                        pipeline_steps.append({
                            "step": "2_qipedc_dtw_match",
                            "ms": timings_ms["qipedc_match_ms"],
                            "tier": qipedc_trace.get("tier"),
                            "verdict": qipedc_trace.get("verdict"),
                            "bucket_hands": qipedc_trace.get("bucket_hands"),
                            "bucket_total_words": qipedc_trace.get("bucket_total_words"),
                            "forced_bilateral": qipedc_trace.get("bilateral_forced_by_2frame_rule"),
                            "gap_top1_top2": qipedc_trace.get("gap_top1_top2"),
                            "top1": (qipedc_trace.get("top_candidates") or [None])[0],
                        })
                    # Preserve a confirmed two-phase "mấy giờ?" match against weak hand_shape overrides
                    # (only allow a direct trajectory match to disambiguate it).
                    can_override_time = (
                        generic is not None
                        and generic.get("status") == "word_matched"
                        and generic.get("match_method") == "trajectory"
                    )
                    if generic and generic.get("entry_id") != TIME_SIGN_ENTRY and (
                        details["status"] != "matched" or can_override_time
                    ):
                        details = generic
                elif details["status"] != "matched":
                    details = {"motion_detected": True, "status": "no_eligible_candidates"}
            if rejected_extra_words:
                details = {**details, "rejected_extra_words": rejected_extra_words}
            if details["motion_detected"] and details["status"] not in ("matched", "word_matched", "ambiguous_word", "no_eligible_candidates"):
                t2 = time.perf_counter()
                lost = lost_hand_during_interaction(window, req.aspect_ratio)
                timings_ms["lost_hand_ms"] = round((time.perf_counter() - t2) * 1000, 1)
                if lost:
                    details = {"motion_detected": True, "status": "lost_hand_during_interaction"}
                if diag_enabled:
                    pipeline_steps.append({
                        "step": "3_lost_hand_check",
                        "ms": timings_ms["lost_hand_ms"],
                        "lost_hand_during_interaction": bool(lost),
                    })
            active = details["motion_detected"]
            if details["status"] in ("matched", "word_matched"):
                matched_entry_id = details.get("entry_id", TIME_SIGN_ENTRY)
                scope_check = validate_word_result_scope(
                    matched_entry_id,
                    source="qipedc_template",
                    lex=lex,
                    allowed_ids=allowed_live_ids,
                    include_pending_gt_sstcn=False,
                )
                if not scope_check["valid"]:
                    details = {
                        **details,
                        "status": "scope_rejected",
                        "rejected_entry_id": matched_entry_id,
                        "reject_reason": scope_check["reject_reason"],
                    }
                    if diag_enabled:
                        timings_ms["total_ms"] = round((time.perf_counter() - t_req_start) * 1000, 1)
                        diagnostic = {"diagnostic": {
                            **details,
                            "raw_frames": raw_len,
                            "topic_id": req.topic_id,
                            "motion_gate": motion_trace,
                            "qipedc": qipedc_trace,
                            "timings_ms": timings_ms,
                            "pipeline_steps": pipeline_steps,
                        }}
                    return {"success": True, "data": {
                        "success": True,
                        "sign": "NEUTRAL",
                        "label": "Từ nằm ngoài phạm vi nhận diện cho phép",
                        "confidence": 0.0,
                        "top_k": [],
                        "confirmed": False,
                        "new_token": None,
                        "tts_speak": None,
                        "motion_detected": active,
                        "scope_rejected": True,
                        "reject_reason": scope_check["reject_reason"],
                        "word_scope_valid": False,
                        **diagnostic,
                    }}
                entry = lex["by_id"][matched_entry_id]
                if diag_enabled:
                    timings_ms["total_ms"] = round((time.perf_counter() - t_req_start) * 1000, 1)
                    explanation = _build_sequence_explanation(
                        final_sign=entry["word"], final_label=entry["word"], source="qipedc_template",
                        confidence=details.get("similarity"), status=details["status"],
                        motion_trace=motion_trace, qipedc_trace=qipedc_trace, sstcn_result=None,
                        raw_len=raw_len, threshold=req.threshold,
                    )
                    diagnostic = {"diagnostic": {
                        **details,
                        "raw_frames": raw_len,
                        "topic_id": req.topic_id,
                        "motion_gate": motion_trace,
                        "qipedc": qipedc_trace,
                        "sstcn": {"called": False, "skip_reason": "qipedc_matched"},
                        "timings_ms": timings_ms,
                        "pipeline_steps": pipeline_steps,
                        "decision": {
                            "final_sign": entry["word"],
                            "final_label": entry["word"],
                            "source": "qipedc_template",
                            "confidence": details.get("similarity"),
                            "reason_code": details["status"],
                            "explanation": explanation,
                        },
                    }}
                return {"success": True, "data": {
                    "success": True, "sign": entry["word"], "label": entry["word"],
                    "entry_id": entry["id"], "category": "QIPEDC", "source": "qipedc_template",
                    "confidence": details["similarity"],
                    "video_url": scope_check["selected_video_url"] or entry["video_url"],
                    "region": scope_check["region"],
                    "word_scope_valid": True,
                    "scope_rejected": False,
                    "top_k": [],
                    "match_method": details.get("match_method"),
                    "confirmed": False, "new_token": None,
                    **diagnostic,
                }}
            missing_tracking = details["status"] in ("missing_both_hands", "tracking_gaps", "lost_hand_during_interaction")
            ambiguous = details["status"] == "ambiguous_word"
            # In continuous live_words mode, only allow <30-frame SSTCN fallback when explicitly requested
            # (e.g. end-of-gesture flush) so mid-gesture partial windows don't trigger premature SSTCN guesses.
            live_sstcn_min_len = MIN_GESTURE_FRAMES if req.allow_short_sequences is True else 30
            if ambiguous or missing_tracking or not active or len(sequence) < live_sstcn_min_len:
                label = "Động tác gần nhiều ký hiệu. Hãy thử lại." if ambiguous else (
                    "Chưa đủ dữ liệu bàn tay để xác nhận" if missing_tracking else "Đang chờ động tác từ vựng…"
                )
                if details["status"] == "lost_hand_during_interaction":
                    label = "Mất dấu một tay khi hai tay ở gần nhau. Hãy giữ cả hai cổ tay trong khung."
                if diag_enabled:
                    timings_ms["total_ms"] = round((time.perf_counter() - t_req_start) * 1000, 1)
                    skip_reason = (
                        "ambiguous_word" if ambiguous else
                        details["status"] if missing_tracking else
                        "no_motion" if not active else
                        f"sequence_len_lt_{min_required_len} ({raw_len})"
                    )
                    explanation = _build_sequence_explanation(
                        final_sign="NEUTRAL", final_label=label, source="qipedc_gate",
                        confidence=0.0, status=details["status"],
                        motion_trace=motion_trace, qipedc_trace=qipedc_trace, sstcn_result=None,
                        raw_len=raw_len, threshold=req.threshold,
                    )
                    diagnostic = {"diagnostic": {
                        **details,
                        "raw_frames": raw_len,
                        "topic_id": req.topic_id,
                        "motion_gate": motion_trace,
                        "qipedc": qipedc_trace,
                        "sstcn": {"called": False, "skip_reason": skip_reason},
                        "timings_ms": timings_ms,
                        "pipeline_steps": pipeline_steps,
                        "decision": {
                            "final_sign": "NEUTRAL",
                            "final_label": label,
                            "source": "qipedc_gate",
                            "confidence": 0.0,
                            "reason_code": skip_reason,
                            "explanation": explanation,
                        },
                    }}
                suggestions = []
                if ambiguous and details.get("suggestions"):
                    by_id = lex["by_id"]
                    for item in details["suggestions"]:
                        e = by_id.get(item.get("entry_id"))
                        if e:
                            suggestions.append({
                                "entry_id": e["id"],
                                "word": e["word"],
                                "confidence": item.get("similarity", 0.0),
                                "distance": item.get("distance"),
                                "video_url": e.get("video_url", ""),
                            })
                return {"success": True, "data": {
                    "success": True, "sign": "NEUTRAL", "label": label,
                    "confidence": 0.0, "top_k": [], "suggestions": suggestions,
                    "confirmed": False, "new_token": None,
                    "motion_detected": active,
                    **diagnostic,
                }}
            if len(sequence) >= 30:
                timestamps = timestamps[-30:] if timestamps is not None and len(timestamps) == len(sequence) else None
                sequence = sequence[-30:]
        else:
            if len(sequence) >= 30:
                timestamps = timestamps[-30:] if timestamps is not None and len(timestamps) == len(sequence) else None
                sequence = sequence[-30:]
        word_scope_kwargs = (
            {"allowed_word_ids": approved_scope, "include_pending_gt_sstcn": False}
            if mode == "words"
            else {}
        )
        t3 = time.perf_counter()
        result = recognizer.predict_sequence(
            sequence_data=sequence,
            threshold=req.threshold,
            mode=mode,
            debug_meta=req.debug_meta,
            topic_id=req.topic_id,
            timestamps=timestamps,
            confirm=not live_words,
            allow_short_sequences=allow_short,
            **word_scope_kwargs,
        )
        timings_ms["sstcn_ms"] = round((time.perf_counter() - t3) * 1000, 1)
        scope_rejected_diag = None
        if mode == "words" and isinstance(result, dict) and result.get("success", True):
            raw_top_k = result.get("top_k") or []
            filtered_top_k = []
            for item in raw_top_k:
                if not isinstance(item, dict):
                    continue
                item_idx = item.get("index")
                item_entry = lex["by_sstcn"].get(item_idx) if item_idx is not None else None
                item_entry_id = item.get("entry_id") or (item_entry["id"] if item_entry else None)
                item_check = validate_word_result_scope(
                    item_entry_id,
                    source="sstcn",
                    lex=lex,
                    allowed_ids=approved_scope,
                    include_pending_gt_sstcn=False,
                    sign_label=item.get("term"),
                )
                if item_check["valid"]:
                    filtered_top_k.append({**item, "entry_id": item_check["entry_id"]})
            result["top_k"] = filtered_top_k

            if result.get("scope_rejected"):
                result["top_k"] = []
                scope_rejected_diag = {"status": "scope_rejected", "reject_reason": result.get("reject_reason")}
            elif result.get("sign") not in (None, "", "NEUTRAL"):
                scope_check = validate_word_result_scope(
                    result.get("entry_id"),
                    source="sstcn",
                    lex=lex,
                    allowed_ids=approved_scope,
                    include_pending_gt_sstcn=False,
                    sign_label=result.get("sign"),
                )
                if not scope_check["valid"]:
                    scope_rejected_diag = {
                        "status": "scope_rejected",
                        "rejected_sign": result.get("sign"),
                        "rejected_entry_id": scope_check.get("entry_id") or result.get("entry_id"),
                        "reject_reason": scope_check["reject_reason"],
                    }
                    result = {
                        **result,
                        "sign": "NEUTRAL",
                        "label": "Từ nằm ngoài phạm vi nhận diện cho phép",
                        "confidence": 0.0,
                        "confirmed": False,
                        "new_token": None,
                        "replaced_token": None,
                        "tts_speak": None,
                        "top_k": [],
                        "entry_id": None,
                        "video_url": "",
                        "region": scope_check.get("region"),
                        "scope_rejected": True,
                        "reject_reason": scope_check["reject_reason"],
                        "word_scope_valid": False,
                    }
                else:
                    checked_entry_id = scope_check.get("entry_id") or result.get("entry_id")
                    if live_words and not req.topic_id and not qipedc_live.corroborate_sstcn_entry(
                        window, req.aspect_ratio, checked_entry_id
                    ):
                        result = {
                            **result,
                            "sign": "NEUTRAL",
                            "label": "Đang chờ động tác từ vựng…",
                            "confidence": 0.0,
                            "confirmed": False,
                            "new_token": None,
                            "replaced_token": None,
                            "tts_speak": None,
                            "top_k": [],
                            "entry_id": None,
                            "video_url": "",
                            "fallback_blocked": True,
                            "blocked_reason": "sstcn_uncorroborated_by_qipedc",
                            "word_scope_valid": True,
                            "scope_rejected": False,
                        }
                    else:
                        result["word_scope_valid"] = True
                        result["scope_rejected"] = False
        if diag_enabled:
            timings_ms["total_ms"] = round((time.perf_counter() - t_req_start) * 1000, 1)
            pipeline_steps.append({
                "step": "4_sstcn_fallback" if live_words else "1_sstcn_predict",
                "ms": timings_ms["sstcn_ms"],
                "sign": result.get("sign"),
                "confidence": result.get("confidence"),
                "threshold": req.threshold,
                "blocked_reason": result.get("blocked_reason") or result.get("reject_reason"),
                "top1": (result.get("top_k") or [None])[0],
            })
            base_details = {**(details if live_words else {}), **(scope_rejected_diag or {})}
            explanation = _build_sequence_explanation(
                final_sign=result.get("sign"), final_label=result.get("label"), source="sstcn",
                confidence=result.get("confidence"), status=base_details.get("status"),
                motion_trace=motion_trace, qipedc_trace=qipedc_trace, sstcn_result=result,
                raw_len=raw_len, threshold=req.threshold,
            )
            diagnostic = {"diagnostic": {
                **base_details,
                "raw_frames": raw_len,
                "topic_id": req.topic_id,
                "motion_gate": motion_trace,
                "qipedc": qipedc_trace,
                "sstcn": {
                    "called": True,
                    "sign": result.get("sign"),
                    "confidence": result.get("confidence"),
                    "threshold": req.threshold,
                    "blocked_reason": result.get("blocked_reason") or result.get("reject_reason"),
                    "inference_time_ms": result.get("inference_time_ms"),
                    "top_k": result.get("top_k") or [],
                },
                "timings_ms": timings_ms,
                "pipeline_steps": pipeline_steps,
                "decision": {
                    "final_sign": result.get("sign"),
                    "final_label": result.get("label"),
                    "source": "sstcn",
                    "confidence": result.get("confidence"),
                    "reason_code": result.get("reject_reason") or result.get("blocked_reason") or (
                        "sstcn_matched" if result.get("sign") != "NEUTRAL" else "sstcn_below_threshold"
                    ),
                    "explanation": explanation,
                },
            }}
        result.update(diagnostic)
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/sstcn/topics")
def get_sstcn_topics():
    """Trả về danh mục chủ đề và từ vựng động của checkpoint SSTCN đang nạp."""
    try:
        info = recognizer.get_sstcn_topics_info()
        return {"success": True, **info}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def _calibration_dir() -> str:
    base = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "personal_calibration")
    os.makedirs(base, exist_ok=True)
    return base


def _safe_slug(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_\-]+", "_", norm_key(text).replace(" ", "_")).strip("_")
    return cleaned or "sign"


@app.post("/api/calibration/save_sample")
def save_calibration_sample(req: CalibrationSampleRequest):
    """Lưu mẫu 75 khớp Holistic do người dùng tự quay để fine-tune cá nhân (Personal Calibration)."""
    from backend.vision.skeleton_norm import resample_sequence_by_timestamps, resample_sequence_linear

    try:
        label = (req.label or "").strip()
        if not label:
            raise HTTPException(status_code=400, detail="label không được để trống")
        arr = np.asarray(req.sequence, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (75, 3) or len(arr) < 4:
            raise HTTPException(status_code=400, detail=f"Kích thước chuỗi không hợp lệ: {arr.shape}")
        if not np.all(np.isfinite(arr)):
            arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        if not np.any(arr):
            raise HTTPException(status_code=400, detail="Chuỗi landmark toàn 0")

        if req.timestamps and len(req.timestamps) == len(arr):
            resampled = resample_sequence_by_timestamps(arr, req.timestamps, target_len=30)
        else:
            resampled = resample_sequence_linear(arr, target_len=30)

        cal_root = _calibration_dir()
        slug = _safe_slug(label)
        label_dir = os.path.join(cal_root, slug)
        os.makedirs(label_dir, exist_ok=True)

        ts_ms = int(time.time() * 1000)
        npy_name = f"sample_{ts_ms}.npy"
        meta_name = f"sample_{ts_ms}.json"
        np.save(os.path.join(label_dir, npy_name), resampled.astype(np.float32))
        meta = {
            "label": label,
            "entry_id": req.entry_id,
            "signer_id": req.signer_id or "personal",
            "raw_frames": int(len(arr)),
            "saved_shape": [30, 75, 3],
            "created_at_ms": ts_ms,
            "file": npy_name,
        }
        with open(os.path.join(label_dir, meta_name), "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        sample_count = len([f for f in os.listdir(label_dir) if f.endswith(".npy")])
        return {
            "success": True,
            "label": label,
            "slug": slug,
            "sample_count": sample_count,
            "file": npy_name,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/calibration/summary")
def get_calibration_summary():
    """Thống kê số lượng mẫu Personal Calibration đã ghi theo từng nhãn."""
    cal_root = _calibration_dir()
    items = []
    total = 0
    for slug in sorted(os.listdir(cal_root)):
        sub = os.path.join(cal_root, slug)
        if not os.path.isdir(sub):
            continue
        npys = [f for f in os.listdir(sub) if f.endswith(".npy")]
        if not npys:
            continue
        label = slug
        metas = sorted([f for f in os.listdir(sub) if f.endswith(".json")])
        entry_id = None
        if metas:
            try:
                with open(os.path.join(sub, metas[-1]), "r", encoding="utf-8") as f:
                    m = json.load(f)
                    label = m.get("label") or slug
                    entry_id = m.get("entry_id")
            except Exception:
                pass
        total += len(npys)
        items.append({"slug": slug, "label": label, "entry_id": entry_id, "count": len(npys)})
    return {"success": True, "total_samples": total, "labels": items}

@app.get("/api/vsl_102_classes")
def get_vsl_102_classes():
    """
    Trả về danh mục 102 lớp từ vựng VSL chuẩn hóa kèm mô tả và link video giáo trình.
    """
    try:
        from backend.models.vsl_102_dictionary import SIGN_METADATA_102
        signs = []
        for s in SIGN_METADATA_102:
            item = dict(s)
            entry = get_entry_by_sstcn(s["index"])
            # Ảnh Supabase cũ dùng link có token đã hết hạn → thay bằng ảnh QIPEDC của mục từ tương ứng
            item["image_url"] = (entry or {}).get("image_url") or (entry or {}).get("thumb_url") or ""
            if entry:
                item["entry_id"] = entry["id"]
                item["word"] = entry["word"]
                item["variants"] = entry.get("variants", [])
                item["definition"] = entry.get("definition", "")
            signs.append(item)
        return {
            "success": True,
            "total": len(signs),
            "signs": signs,
            "classes": [s["term"] for s in signs]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/recognize_frame")
def recognize_frame(req: FrameRequest):
    """
    Nhận diện trực tiếp từ ảnh chụp webcam Base64 bằng OpenCV
    """
    try:
        # Giải mã ảnh base64
        img_data = req.image_base64
        if "," in img_data:
            img_data = img_data.split(",")[1]
        decoded = base64.b64decode(img_data)
        np_arr = np.frombuffer(decoded, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

        if img is None:
            raise HTTPException(status_code=400, detail="Không thể giải mã ảnh")

        h, w, _ = img.shape
        # Phân tích sơ bộ bằng OpenCV: phát hiện màu da / bounding box bàn tay
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        lower_skin = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin = np.array([20, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower_skin, upper_skin)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        hand_detected = len(contours) > 0 and any(cv2.contourArea(c) > 2000 for c in contours)

        return {
            "success": True,
            "hand_detected": hand_detected,
            "resolution": f"{w}x{h}",
            "message": "Ảnh đã được xử lý bởi OpenCV"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/letters")
def fingerspell_alphabet():
    """Toàn bộ bảng chữ cái ngón tay (chữ cái, không có dấu thanh) — giao diện tải sẵn một lần để chọn chữ là robot diễn ngay."""
    from backend.nlp.text_to_vsl import letter_clips, _letter_item, remove_accents
    from backend.models.lexicon import motion_ids
    lex, avail = load_lexicon(), motion_ids()
    items = {k: _letter_item(k, remove_accents(k).upper(), lex, avail, k) for k in letter_clips()}
    return {"success": True, "letters": items}


@app.get("/api/letter/{name}")
def fingerspell_letter(name: str):
    """Một chữ cái của bảng chữ cái ngón tay tiếng Việt (A, Ă, Â…; không có dấu thanh)
    — robot diễn đoạn giữ hình chữ cái học từ video QIPEDC."""
    from backend.nlp.text_to_vsl import letter_clips, _letter_item, remove_accents
    from backend.models.lexicon import motion_ids
    key = norm_key(name)
    if key not in letter_clips():
        raise HTTPException(status_code=404, detail="Không có chữ cái này")
    item = _letter_item(key, remove_accents(key).upper(), load_lexicon(), motion_ids(), key)
    return {"success": True, "sequence": [item]}


@app.post("/api/text_to_sign")
def convert_text_to_sign(req: TextToSignRequest):
    """
    Chuyển đổi văn bản / giọng nói tiếng Việt thành chuỗi ký hiệu VSL và tư thế 3D Avatar
    """
    try:
        sequence = text_to_vsl_glosses(req.text)
        total_duration = sum(item["duration_ms"] for item in sequence)
        return {
            "success": True,
            "input_text": req.text,
            "sequence": sequence,
            "total_tokens": len(sequence),
            "total_duration_ms": total_duration
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear_sentence")
def clear_sentence():
    """Xóa trắng phụ đề đã nhận diện"""
    result = recognizer.clear_sentence()
    return {"success": True, "data": result}

@app.get("/api/dataset_info")
def get_dataset_info():
    """Lấy thông tin tập dữ liệu huấn luyện VSL và các nguồn tải về"""
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    meta_path = os.path.join(data_dir, "dataset_metadata.json")
    if os.path.exists(meta_path):
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {
        "status": "not_generated",
        "message": "Chưa khởi tạo dữ liệu. Gọi POST /api/train_model để tải và tạo dữ liệu."
    }

@app.post("/api/train_model")
def trigger_training():
    """Kích hoạt tải dữ liệu và huấn luyện mô hình PyTorch VSLGestureNet"""
    try:
        from backend.training.download_dataset import build_augmented_dataset
        from backend.training.train_model import train
        build_augmented_dataset(samples_per_class=100)
        train()
        return {
            "success": True,
            "message": "Đã tạo dữ liệu và huấn luyện mô hình PyTorch thành công!",
            "weights_path": "backend/models/weights.pt"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================================
# API WORD DEBUG (chỉ dùng để test Word Mode, không ảnh hưởng nhận diện)
# ==========================================================
class WordDebugSessionRequest(BaseModel):
    action: str  # "start" | "stop"
    target_class: Optional[int] = None
    target_word: Optional[str] = None
    trial: Optional[int] = 1
    session_prefix: Optional[str] = "A01_1_T01"


class WordDebugEventRequest(BaseModel):
    event_type: str
    frame_count: Optional[int] = None
    frames_before_reset: Optional[int] = None
    reason: Optional[str] = None
    consecutive_missing_frames: Optional[int] = None
    tracker_engine: Optional[str] = None
    active_instance_engine: Optional[str] = None
    requested_engine: Optional[str] = None
    landmarks_source: Optional[str] = None
    debounce_threshold: Optional[int] = None
    pose_zero_frames: Optional[int] = None
    left_hand_zero_frames: Optional[int] = None
    right_hand_zero_frames: Optional[int] = None
    detail: Optional[str] = None


@app.get("/api/debug/word/targets")
def get_word_debug_targets():
    """63 từ Word Mode (class 39..101) lấy từ SIGN_DICT_102, cùng nguồn với recognizer."""
    from backend.vision.word_debugger import get_word_targets
    targets = get_word_targets()
    return {"success": True, "total": len(targets), "targets": targets}


@app.post("/api/debug/word/session")
def manage_word_debug_session(req: WordDebugSessionRequest):
    from backend.vision.word_debugger import word_debugger
    if req.action == "start":
        result = word_debugger.start_session(
            target_class=req.target_class,
            target_word=req.target_word,
            trial=req.trial if req.trial is not None else 1,
            checkpoint_info=recognizer.get_sstcn_checkpoint_info(),
            session_prefix=req.session_prefix or "A01_1_T01"
        )
        if not result.get("success"):
            raise HTTPException(status_code=400, detail=result.get("message", "Không thể bắt đầu phiên"))
        return result
    if req.action == "stop":
        return word_debugger.stop_session()
    raise HTTPException(status_code=400, detail="action phải là 'start' hoặc 'stop'")


@app.post("/api/debug/word/event")
def log_word_debug_event(req: WordDebugEventRequest):
    from backend.vision.word_debugger import word_debugger
    data = req.model_dump(exclude_none=True) if hasattr(req, "model_dump") else req.dict(exclude_none=True)
    event_type = data.pop("event_type")
    return {"success": word_debugger.log_frontend_event(event_type, data)}


@app.get("/api/debug/word/status")
def get_word_debug_status():
    from backend.vision.word_debugger import word_debugger
    return word_debugger.get_status()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
