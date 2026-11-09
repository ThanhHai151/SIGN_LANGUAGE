"""
FastAPI router for Controlled Pilot V3.
Provides clean REST API endpoints for session lifecycle, trial state machine,
monotonic per-frame recording, trial accounting, session recovery, and export.
"""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from backend.vision.controlled_pilot_v3.constants import SessionMode
from backend.vision.controlled_pilot_v3.session_manager import PilotSessionManager

router = APIRouter(prefix="/api/pilot/v3", tags=["Controlled Pilot V3"])

# Shared global session manager instance (injected via init_pilot_v3_router)
_session_manager: Optional[PilotSessionManager] = None


def set_session_manager(mgr: PilotSessionManager) -> None:
    global _session_manager
    _session_manager = mgr


def get_session_manager() -> PilotSessionManager:
    if _session_manager is None:
        raise HTTPException(status_code=500, detail="Controlled Pilot V3 Session Manager is not initialized.")
    return _session_manager


class CreateSessionRequest(BaseModel):
    mode: str = Field(default="FULL_29", description="FULL_29 or SINGLE_LETTER")
    single_letter: Optional[str] = Field(default=None, description="Expected letter if SINGLE_LETTER mode")
    session_id: Optional[str] = Field(default=None, description="Optional custom session ID")


class RecoverSessionRequest(BaseModel):
    session_id: str


class FrameRequest(BaseModel):
    landmarks: Optional[List[List[float]]] = None
    handedness: str = "Right"
    tracking_confidence: Optional[float] = None
    handedness_score: Optional[float] = None
    client_capture_ts: Optional[float] = None


class FinishTrialRequest(BaseModel):
    accept: bool = True


@router.post("/session/create")
def create_session(req: CreateSessionRequest):
    mgr = get_session_manager()
    try:
        mode = SessionMode(req.mode.upper())
        status = mgr.create_session(
            mode=mode,
            single_letter=req.single_letter,
            session_id=req.session_id,
        )
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/session/start")
def start_session():
    mgr = get_session_manager()
    try:
        status = mgr.start_session()
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/session/pause")
def pause_session():
    mgr = get_session_manager()
    try:
        status = mgr.pause_session()
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/session/resume")
def resume_session():
    mgr = get_session_manager()
    try:
        status = mgr.resume_session()
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/session/stop")
def stop_session(reason: Optional[str] = "User requested stop"):
    mgr = get_session_manager()
    try:
        status = mgr.stop_session(reason=reason)
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/session/status")
def get_session_status():
    mgr = get_session_manager()
    return {"success": True, "data": mgr.get_status()}


@router.post("/session/recover")
def recover_session(req: RecoverSessionRequest):
    mgr = get_session_manager()
    try:
        status = mgr.recover_session(req.session_id)
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/trial/start")
def start_trial():
    mgr = get_session_manager()
    try:
        status = mgr.start_trial()
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/trial/countdown")
def start_countdown():
    mgr = get_session_manager()
    try:
        status = mgr.start_trial_countdown()
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/trial/record_start")
def start_record():
    mgr = get_session_manager()
    try:
        status = mgr.start_trial_recording()
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/frame")
def record_frame(req: FrameRequest):
    mgr = get_session_manager()
    try:
        res = mgr.record_frame(
            landmarks=req.landmarks,
            handedness=req.handedness,
            tracking_confidence=req.tracking_confidence,
            handedness_score=req.handedness_score,
            client_capture_ts=req.client_capture_ts,
        )
        return {"success": True, "data": res}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/trial/finish")
def finish_trial(req: Optional[FinishTrialRequest] = None):
    mgr = get_session_manager()
    accept = req.accept if req else True
    try:
        summary = mgr.finish_trial(accept=accept)
        return {"success": True, "data": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/trial/retry")
def retry_trial(reason: Optional[str] = "User requested retry"):
    mgr = get_session_manager()
    try:
        status = mgr.retry_trial(reason=reason)
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/trial/skip")
def skip_trial(reason: Optional[str] = "User skipped"):
    mgr = get_session_manager()
    try:
        status = mgr.skip_trial(reason=reason)
        return {"success": True, "data": status}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/summary")
def get_session_summary():
    mgr = get_session_manager()
    try:
        summary = mgr.generate_session_summary()
        return {"success": True, "data": summary}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/export")
def export_session():
    mgr = get_session_manager()
    try:
        summary = mgr.generate_session_summary()
        export_path = f"{mgr.storage.get_session_dir(mgr.session_id)}/export.json"
        import json, os
        if os.path.exists(export_path):
            with open(export_path, "r", encoding="utf-8") as f:
                return json.load(f)
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
