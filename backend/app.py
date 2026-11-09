"""
FastAPI Server cho Hệ thống Giao tiếp Hai chiều Ngôn ngữ Ký hiệu Việt Nam (VSL)
Hỗ trợ:
1. API Nhận diện Ký hiệu qua Tọa độ Khung xương & Khung hình Camera (Sign -> Text)
2. API Diễn giải Văn bản/Giọng nói sang Chuỗi Ký hiệu VSL & 3D Avatar Poses (Text -> Sign)
3. API Từ điển VSL Đa phương tiện & Kiểm tra Sức khỏe Hệ thống
"""

from asyncio import timeouts
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
from pydantic import BaseModel
from typing import List, Optional, Union, Dict, Any, Literal

from backend.models.sign_model import CLASSES
from backend.models.vsl_dictionary import ALL_SIGNS, VSL_ALPHABET, VSL_VOCABULARY
from backend.vision.recognizer import VSLRecognizer
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
    sequence: List[List[List[float]]]  # Chuỗi 30 frames x 75 joints x 3 coords
    threshold: Optional[float] = 0.65

class TextToSignRequest(BaseModel):
    text: str

@app.get("/api/health")
def health_check():
    return {
        "status": "online",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "sstcn_classes": 102,
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
        from backend.models.lexicon import motion_id_of, motion_ids
        avail = motion_ids()
        items = [e for e in items if e.get("pose_key") or motion_id_of(e, avail)]
    elif filter == "alphabet":
        items = [e for e in items if e.get("fingerspelling")]
    if q:
        k = norm_key(q)
        items = [e for e in items if any(k in key for key in e.get("keys", [])) or k in norm_key(e.get("definition", ""))]
    total = len(items)
    if limit:
        items = items[offset:offset + limit]
    return {
        "success": True,
        "meta": lex["meta"],
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


@app.post("/api/recognize_sequence")
def recognize_sequence(req: SequenceRequest):
    """
    Nhận diện chuỗi chuyển động ký hiệu VSL qua mô hình SSTCN_Attention (30 frames x 75 joints).
    Độ chính xác cao vượt trội (93.87%) trên 102 lớp từ vựng VSL.
    """
    try:
        result = recognizer.predict_sequence(
            sequence_data=req.sequence,
            threshold=req.threshold
        )
        return {"success": True, "data": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
