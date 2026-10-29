"""
Module trích xuất Landmark bàn tay từ Video nguồn QIPEDC/VSL.
============================================================
Công cụ phục vụ Giai đoạn 2C: Trích xuất 21 điểm mốc chuẩn hóa MediaPipe Hands
từ video thật để phục vụ tạo bộ fixture kiểm thử (Ground Truth) cho hệ thống VSL Connect AI.

Đặc điểm kỹ thuật:
- Chuẩn tọa độ: Normalized coordinates [x, y, z] (x: [0, 1], y: [0, 1], z tương đối cổ tay).
- Khớp với đầu vào của backend.vision.recognizer.classify_geometry và feature_extractor.
- Hỗ trợ chọn frame theo index hoặc timestamp (giây).
- Bảo toàn siêu dữ liệu nguồn (mã video, frame, timestamp, độ tin cậy, trạng thái chất lượng).
- An toàn: Không tự ý ghi đè file khi chưa có cờ overwrite=True.
"""

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional, Tuple, Union

# Cấu hình logging
logger = logging.getLogger("extract_video_landmarks")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Kiểm tra thư viện phụ thuộc
try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False

try:
    import mediapipe as mp
    HAS_MEDIAPIPE = True
except ImportError:
    HAS_MEDIAPIPE = False


def validate_landmark_format(landmarks: Any) -> Tuple[bool, str]:
    """
    Xác minh danh sách 21 điểm mốc bàn tay có đúng quy cách của MediaPipe Hands hay không.
    Trả về: (is_valid: bool, error_message: str)
    """
    if landmarks is None:
        return False, "Landmark là None"
    if not isinstance(landmarks, (list, tuple)):
        return False, f"Định dạng không hợp lệ: mong muốn list/tuple, nhận {type(landmarks).__name__}"
    if len(landmarks) != 21:
        return False, f"Số điểm không đúng: mong muốn 21 điểm, nhận {len(landmarks)} điểm"
    
    for idx, pt in enumerate(landmarks):
        if not isinstance(pt, (list, tuple)) or len(pt) < 3:
            return False, f"Điểm mốc {idx} không đủ 3 tọa độ (x, y, z): {pt}"
        try:
            x, y, z = float(pt[0]), float(pt[1]), float(pt[2])
        except (ValueError, TypeError):
            return False, f"Điểm mốc {idx} chứa tọa độ không phải số: {pt}"
        
        # Kiểm tra tính hợp lệ cơ bản của tọa độ chuẩn hóa
        if not (-0.5 <= x <= 1.5) or not (-0.5 <= y <= 1.5):
            return False, f"Điểm mốc {idx} có tọa độ x/y bất thường nằm ngoài phạm vi quan sát: ({x}, {y})"

    return True, "Hợp lệ"


def extract_landmarks_from_frame(
    frame,
    hands_detector=None,
    target_handedness: str = "Right"
) -> Tuple[Optional[List[List[float]]], str, float]:
    """
    Trích xuất 21 điểm mốc chuẩn hóa từ 1 khung hình OpenCV (BGR).
    
    Tham số:
        frame: Ảnh mảng numpy (BGR)
        hands_detector: Đối tượng mp.solutions.hands.Hands (nếu đã khởi tạo)
        target_handedness: "Right" hoặc "Left" (theo góc nhìn của người ký hiệu)
        
    Trả về:
        (landmarks_21, status_str, confidence_score)
    """
    if not HAS_MEDIAPIPE:
        return None, "dependency_missing_mediapipe", 0.0
    
    if frame is None or frame.size == 0:
        return None, "empty_frame", 0.0

    # Chuyển đổi BGR sang RGB cho MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    close_detector_after = False
    if hands_detector is None:
        hands_detector = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )
        close_detector_after = True

    try:
        results = hands_detector.process(rgb_frame)
        if not results.multi_hand_landmarks:
            return None, "no_hand_detected", 0.0

        chosen_landmarks = None
        best_score = 0.0

        # Nếu có thông tin handedness từ MediaPipe
        # Lưu ý: MediaPipe Handedness mặc định phản ánh góc nhìn camera (gương).
        # Khi người đối diện dùng tay phải, trên ảnh camera nó ở bên trái màn hình.
        # Nếu có multi_handedness, duyệt qua để chọn đúng tay hoặc lấy bàn tay có confidence cao nhất.
        if results.multi_handedness and len(results.multi_handedness) == len(results.multi_hand_landmarks):
            for hand_landmarks, handedness_info in zip(results.multi_hand_landmarks, results.multi_handedness):
                label = handedness_info.classification[0].label
                score = float(handedness_info.classification[0].score)
                # MediaPipe gán label 'Left'/'Right' theo góc nhìn tự thân
                if label.lower() == target_handedness.lower() or target_handedness == "Any":
                    chosen_landmarks = hand_landmarks
                    best_score = score
                    break
        
        # Nếu chưa chọn được theo handedness, lấy bàn tay đầu tiên được phát hiện
        if chosen_landmarks is None and len(results.multi_hand_landmarks) > 0:
            chosen_landmarks = results.multi_hand_landmarks[0]
            best_score = 0.8  # Default estimate

        # Chuyển đổi sang định dạng list 21 điểm [[x, y, z], ...]
        lm_list = []
        for lm in chosen_landmarks.landmark:
            lm_list.append([round(float(lm.x), 6), round(float(lm.y), 6), round(float(lm.z), 6)])

        is_valid, err_msg = validate_landmark_format(lm_list)
        if not is_valid:
            return None, f"invalid_format: {err_msg}", 0.0

        return lm_list, "ok", best_score

    finally:
        if close_detector_after:
            hands_detector.close()


def extract_landmarks_from_video(
    video_path: str,
    target_frames: Optional[List[int]] = None,
    target_timestamps: Optional[List[float]] = None,
    target_handedness: str = "Right"
) -> List[Dict[str, Any]]:
    """
    Đọc video và trích xuất landmarks tại các frame hoặc timestamp chỉ định.
    
    Trả về danh sách bản ghi:
    [
        {
            "video_id": "D0530",
            "source_file": "backend/data/media_cache/videos/D0530.mp4",
            "frame_idx": 71,
            "timestamp_sec": 2.37,
            "handedness": "Right",
            "coordinate_type": "normalized_mediapipe_hands",
            "quality_status": "ok",
            "confidence": 0.95,
            "landmarks": [[x, y, z], ... 21 points ...]
        }, ...
    ]
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Không tìm thấy file video: {video_path}")
    
    if not HAS_OPENCV:
        raise RuntimeError("OpenCV (cv2) chưa được cài đặt trong môi trường hiện tại.")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Không thể mở video bằng OpenCV: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.97
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_id = os.path.splitext(os.path.basename(video_path))[0]

    # Xác định danh sách frame cần trích xuất
    frames_to_process = set()
    if target_frames:
        for f in target_frames:
            if 0 <= f < total_frames:
                frames_to_process.add(int(f))
            else:
                logger.warning(f"Frame {f} nằm ngoài phạm vi [0, {total_frames-1}] của {video_id}")
    
    if target_timestamps:
        for t in target_timestamps:
            f = int(round(t * fps))
            if 0 <= f < total_frames:
                frames_to_process.add(f)
            else:
                logger.warning(f"Timestamp {t}s (frame {f}) nằm ngoài phạm vi của {video_id}")

    # Nếu không chỉ định frame, mặc định chọn frame ở giữa video
    if not frames_to_process:
        frames_to_process.add(total_frames // 2)

    sorted_frames = sorted(list(frames_to_process))
    results = []

    hands_detector = None
    if HAS_MEDIAPIPE:
        hands_detector = mp.solutions.hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5
        )

    try:
        for fidx in sorted_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
            ret, frame = cap.read()
            ts = round(fidx / fps, 4)

            if not ret or frame is None:
                record = {
                    "video_id": video_id,
                    "source_file": video_path.replace("\\", "/"),
                    "frame_idx": fidx,
                    "timestamp_sec": ts,
                    "handedness": target_handedness,
                    "coordinate_type": "normalized_mediapipe_hands",
                    "quality_status": "frame_read_failed",
                    "confidence": 0.0,
                    "landmarks": None
                }
                results.append(record)
                continue

            if not HAS_MEDIAPIPE:
                record = {
                    "video_id": video_id,
                    "source_file": video_path.replace("\\", "/"),
                    "frame_idx": fidx,
                    "timestamp_sec": ts,
                    "handedness": target_handedness,
                    "coordinate_type": "normalized_mediapipe_hands",
                    "quality_status": "dependency_missing_mediapipe",
                    "confidence": 0.0,
                    "landmarks": None
                }
                results.append(record)
                continue

            lms, status, conf = extract_landmarks_from_frame(frame, hands_detector, target_handedness)
            record = {
                "video_id": video_id,
                "source_file": video_path.replace("\\", "/"),
                "frame_idx": fidx,
                "timestamp_sec": ts,
                "handedness": target_handedness,
                "coordinate_type": "normalized_mediapipe_hands",
                "quality_status": status,
                "confidence": round(conf, 4),
                "landmarks": lms
            }
            results.append(record)

    finally:
        cap.release()
        if hands_detector:
            hands_detector.close()

    return results


def save_landmark_fixture(
    fixture_data: Dict[str, Any],
    output_path: str,
    overwrite: bool = False
) -> None:
    """
    Lưu dữ liệu mốc kiểm thử vào file JSON.
    Bảo vệ chống ghi đè khi chưa được cho phép.
    """
    if os.path.exists(output_path) and not overwrite:
        raise FileExistsError(f"Tệp fixture đã tồn tại và overwrite=False: {output_path}")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(fixture_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Đã lưu fixture mốc landmark thành công: {output_path}")

