"""
FastAPI Server cho Hệ thống Giao tiếp Hai chiều Ngôn ngữ Ký hiệu Việt Nam (VSL)
Hỗ trợ:
1. API Nhận diện Ký hiệu qua Tọa độ Khung xương & Khung hình Camera (Sign -> Text)
2. API Diễn giải Văn bản/Giọng nói sang Chuỗi Ký hiệu VSL & 3D Avatar Poses (Text -> Sign)
3. API Từ điển VSL Đa phương tiện & Kiểm tra Sức khỏe Hệ thống
"""

import os
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
from typing import List, Optional

from backend.models.sign_model import CLASSES
from backend.models.vsl_dictionary import ALL_SIGNS, VSL_ALPHABET, VSL_VOCABULARY
from backend.vision.recognizer import VSLRecognizer
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

class FrameRequest(BaseModel):
    image_base64: str

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
    if os.environ.get("VSL_PREFETCH", "1") == "0":
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

@app.post("/api/recognize_landmarks")
def recognize_landmarks(req: LandmarkRequest):
    """
    Nhận diện cử chỉ từ danh sách 21 điểm mốc bàn tay hoặc 2 bàn tay do MediaPipe Web trích xuất
    Đảm bảo độ trễ siêu thấp (<10ms) và tính chính xác cao.
    """
    try:
        result = recognizer.predict_landmarks(
            landmarks_list=req.landmarks or [],
            multi_hands=req.multi_hands
        )
        return {"success": True, "data": result}
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
