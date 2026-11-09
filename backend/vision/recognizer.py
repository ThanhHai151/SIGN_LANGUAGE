"""
Động cơ Nhận diện Cử chỉ VSL Độ chính xác cao (High-Precision Hybrid VSL Recognizer Engine)
Kết hợp 2 chế độ:
1. Mô hình Deep Learning Spatio-Temporal: SSTCN_Attention (30 frames x 75 joints) nhận diện 102 từ vựng VSL.
2. Bộ kiểm tra ràng buộc hình học ngón tay (Geometric Fingertip & Hand Angle Constraints) cho bảng chữ cái tức thì (<5ms).
Hỗ trợ chống rung (temporal smoothing), chống lặp (cooldown debounce) và ghép câu tự động.
"""

import os
import re
import json
import time
import threading
import collections
import math
from typing import Optional, Any, Dict, List
import numpy as np
import torch

from backend.models.sstcn_model import load_sstcn_model
from backend.models.vsl_102_dictionary import SIGN_DICT_102, SIGN_METADATA_102, get_sign_info
from backend.models.sign_model import get_trained_model, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from backend.models.vsl_dictionary import ALL_SIGNS
from backend.models.lexicon import get_entry_by_sstcn
from backend.vision.feature_extractor import normalize_hand_landmarks, calculate_distance, calculate_angle
from backend.vision.alphabet_classifier import (
    alphabet_classifier,
    O_D_TI_MAX,
    O_IDX_STRAIGHT_MAX,
)

from backend.models.alphabet_static_head import (
    load_static_alphabet_head,
    DEFAULT_HEAD_WEIGHTS_PATH,
)
from backend.vision.diacritic_state_machine import (
    DiacriticStateMachine,
    BASE_GROUPS,
    STATE_IDLE,
    STATE_BASE_CANDIDATE,
    STATE_BASE_STABLE,
    STATE_DIACRITIC_CANDIDATE,
    STATE_DIACRITIC_CONFIRMED,
    STATE_COOLDOWN,
)

PILOT_CLASSIFIER_RULE_VERSION = "Candidate-D"
PILOT_SCHEMA_VERSION = "2.1-kinematic-phase"
PILOT_FEATURE_EXTRACTOR_VERSION = "2.0-kinematic"

# Bảng 29 chữ cái tiếng Việt VSL chuẩn hóa (loại bỏ F, J, W, Z)
VIETNAMESE_ALPHABET_29 = [
    "A", "Ă", "Â", "B", "C", "D", "Đ", "E", "Ê", "G", "H", "I", "K", "L",
    "M", "N", "O", "Ô", "Ơ", "P", "Q", "R", "S", "T", "U", "Ư", "V", "X", "Y"
]

# Danh sách chuẩn duy nhất 29 chữ cái tiếng Việt cho chế độ nhận diện chữ cái
ALLOWED_LETTERS = set(VIETNAMESE_ALPHABET_29)

# Metadata mô tả bổ sung cho các chữ cái có dấu động
DIACRITIC_SIGNS_META = {
    "Ă": {"label": "Ă", "type": "alphabet", "description": "Chữ cái Ă (dáng chữ A mở xòe ngón V)"},
    "Â": {"label": "Â", "type": "alphabet", "description": "Chữ cái Â (dáng chữ A vẽ dấu nón nhọn)"},
    "Ê": {"label": "Ê", "type": "alphabet", "description": "Chữ cái Ê (dáng chữ E vẽ dấu nón nhọn)"},
    "Ô": {"label": "Ô", "type": "alphabet", "description": "Chữ cái Ô (dáng chữ O vẽ dấu nón nhọn)"},
    "Ơ": {"label": "Ơ", "type": "alphabet", "description": "Chữ cái Ơ (dáng chữ O vẩy móc râu ngón út)"},
    "Ư": {"label": "Ư", "type": "alphabet", "description": "Chữ cái Ư (dáng chữ U bật râu ngón út)"},
    "Đ": {"label": "Đ", "type": "alphabet", "description": "Chữ cái Đ (dáng chữ D gập vạch ngang ngón trỏ)"},
}
ALLOWED_DIGITS = {str(i) for i in range(1, 10)}
DIGITS_MODE_SIGNS = {str(i) for i in range(0, 10)}
ALLOWED_LETTERS_AND_DIGITS = ALLOWED_LETTERS | ALLOWED_DIGITS

# 10 chữ cái tĩnh bổ sung qua mạng nơ-ron VSLStaticAlphabetHead (Task 6B/6C)
ALLOWED_SUPPLEMENTAL_LETTERS = {"G", "H", "K", "M", "N", "P", "Q", "R", "U", "X"}

# 3 chữ cái nền tĩnh hỗ trợ fallback từ VSLStaticAlphabetHead khi geometric rules NO_MATCH (Task 7E.5B)
ALLOWED_FALLBACK_BASE_LETTERS = {"A", "D", "E"}
DEFAULT_FALLBACK_BASE_THRESHOLD = 0.70

# Ngưỡng tin cậy thực nghiệm (experimental thresholds) cho từng chữ cái bổ sung
# Q được gán ngưỡng cao (0.85) để phòng ngừa dương tính giả do nhầm lẫn hình chiếu 2D với X
DEFAULT_SUPPLEMENTAL_THRESHOLDS = {
    "G": 0.70,
    "H": 0.70,
    "K": 0.75,
    "M": 0.70,
    "N": 0.75,
    "P": 0.70,
    "Q": 0.85,
    "R": 0.75,
    "U": 0.70,
    "X": 0.70,
}

# Bảng ánh xạ phát âm TTS chuẩn cho từng chữ số (chỉ đọc tên số, không có tiền tố "số")
DIGIT_TTS_MAP = {
    "0": "không",
    "1": "một",
    "2": "hai",
    "3": "ba",
    "4": "bốn",
    "5": "năm",
    "6": "sáu",
    "7": "bảy",
    "8": "tám",
    "9": "chín",
    "10": "mười",
}

def _sanitize_static_head_error(err) -> str:
    """Làm sạch thông báo lỗi nạp Static Head, loại bỏ đường dẫn tuyệt đối nhạy cảm."""
    if err is None:
        return None
    raw_msg = " ".join(str(err).split())
    sanitized = re.sub(r"[A-Za-z]:[\\/][^\s'\":]+", "<path>", raw_msg)
    sanitized = re.sub(r"(?<=\s)/[^\s'\":]+/[^\s'\":]+", "<path>", sanitized)
    err_type = type(err).__name__ if not isinstance(err, str) else "Error"
    full_str = f"{err_type}: {sanitized}" if not sanitized.startswith(err_type) else sanitized
    return full_str[:320]


class VSLRecognizer:
    def __init__(
        self,
        confidence_threshold=0.70,
        smoothing_window=10,
        cooldown_sec=1.2,
        weights_path=None,
        static_head_weights_path=None,
        supplemental_thresholds=None,
        fallback_base_threshold=None,
        pilot_logging_enabled=None,
    ):
        self.confidence_threshold = confidence_threshold
        self.smoothing_window = smoothing_window
        self.cooldown_sec = cooldown_sec
        
        # 1. Mô hình SSTCN_Attention 102 lớp (Chuỗi động 30 frames)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.sstcn_model = load_sstcn_model(device=self.device)
        
        # 2. Mô hình nơ-ron MLP tĩnh cho cử chỉ đơn lẻ
        self.model = get_trained_model(weights_path=weights_path)
        self.alphabet_classifier = alphabet_classifier
        from backend.vision.digit_10_detector import VSL10GestureDetector
        self.digit_10_detector = VSL10GestureDetector()

        # 3. Mô-đun bổ sung 10 chữ cái tĩnh VSL (VSLStaticAlphabetHead - Task 6B/6C)
        self.supplemental_thresholds = dict(DEFAULT_SUPPLEMENTAL_THRESHOLDS)
        if supplemental_thresholds and isinstance(supplemental_thresholds, dict):
            self.supplemental_thresholds.update(supplemental_thresholds)

        # 4. Ngưỡng fallback chữ cái nền tĩnh A, D, E (Task 7E.5B)
        self.fallback_base_threshold = float(fallback_base_threshold) if fallback_base_threshold is not None else DEFAULT_FALLBACK_BASE_THRESHOLD

        self.static_head_weights_path = static_head_weights_path or DEFAULT_HEAD_WEIGHTS_PATH
        self.static_alphabet_head = None
        self.static_head_meta = {}
        self.has_static_head = False
        self.pytorch_version = str(getattr(torch, "__version__", "unknown"))
        self.static_head_load_status = "uninitialized"
        self.static_head_load_error = None

        try:
            if os.path.exists(self.static_head_weights_path):
                self.static_alphabet_head, self.static_head_meta = load_static_alphabet_head(
                    weights_path=self.static_head_weights_path,
                    device=str(self.device) if torch.cuda.is_available() else "cpu",
                )
                self.has_static_head = True
                self.static_head_load_status = "loaded"
                self.static_head_load_error = None
            else:
                self.has_static_head = False
                self.static_head_load_status = "missing_weights_file"
                self.static_head_load_error = "Static head weights file not found"
        except Exception as e:
            print(f"[VSLRecognizer] Warning: Failed to load static alphabet head: {e}")
            self.static_alphabet_head = None
            self.has_static_head = False
            self.static_head_load_status = "failed"
            self.static_head_load_error = _sanitize_static_head_error(e)
        
        # 3. Bộ đệm thời gian chống rung cho chế độ tĩnh
        self.prediction_history = collections.deque(maxlen=smoothing_window)
        self._pilot_temporal_lock = threading.RLock()
        self._pilot_temporal_states = {}
        
        # 4. Quản lý câu và từ
        self.last_confirmed_sign = None
        self.last_confirmed_time = 0
        self.sentence_tokens = []
        
        # 5. Quản lý FSM phân giải dấu cho từng camera session (Task 7C/7D)
        self.diacritic_fsms = {}
        self.debug_diacritic = os.environ.get("VSL_DIACRITIC_DEBUG", "0") == "1"
        
        # 5. Diagnostic Logging State (Legacy rate-limited ~2.2 FPS)
        self.diagnostic_enabled = os.environ.get("VSL_DIAGNOSTIC", "0") == "1"
        self.diagnostic_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "diagnostics")
        self._last_diag_log_time = 0.0
        self._diag_lock = threading.Lock()
        self._diag_frame_counter = 0

        # Controlled Pilot V3 manages diagnostics outside this recognizer via adapter.
        self.pilot_logging_enabled = False
        self.pilot_logger = None
        self.pilot_filepath = None
        self.active_pilot_session_id = None
        self._pilot_lock = threading.Lock()
        self._pilot_frame_counter = 0

    def enable_pilot_logging(self, session_id=None, filepath=None):
        return None

    def disable_pilot_logging(self):
        pass

    def _next_pilot_frame_idx(self) -> int:
        with self._diag_lock:
            self._pilot_frame_counter += 1
            return self._pilot_frame_counter

    def _build_pilot_geometric_payload(
        self,
        status,
        letter,
        reason=None,
        matched_candidates=None,
        features=None,
    ) -> dict:
        """
        Chuẩn hóa thông tin chẩn đoán Geometric Rule cho Pilot JSONL (Task 5):
        - Ghi nhận reason, matched_candidates, features và h_rule_diagnostics nếu có sẵn.
        - Tuyệt đối không tạo giá trị giả khi trường gốc không tồn tại hoặc là None.
        """
        clean_candidates = None
        if isinstance(matched_candidates, (list, tuple)):
            clean_candidates = []
            for item in matched_candidates:
                if isinstance(item, (list, tuple)) and len(item) >= 2:
                    c_let = str(item[0]) if item[0] is not None else None
                    try:
                        c_conf = round(float(item[1]), 4) if item[1] is not None else None
                    except (ValueError, TypeError):
                        c_conf = None
                    clean_candidates.append([c_let, c_conf])
                else:
                    clean_candidates.append(item)

        clean_features = None
        h_rule_diag = None
        k_rule_diag = None
        i_rule_diag = None
        m_rule_diag = None
        n_rule_diag = None
        o_rule_diag = None
        t_rule_diag = None
        u_rule_diag = None
        s_rule_diag = None
        p_rule_diag = None
        q_rule_diag = None
        r_rule_diag = None
        x_rule_diag = None
        if isinstance(features, dict):
            clean_features = {}
            for k, v in features.items():
                if v is None:
                    clean_features[k] = None
                elif isinstance(v, (bool, np.bool_)):
                    clean_features[k] = bool(v)
                elif isinstance(v, (int, float, np.number)):
                    try:
                        fv = float(v)
                        clean_features[k] = round(fv, 4) if np.isfinite(fv) else None
                    except (ValueError, TypeError):
                        clean_features[k] = None
                else:
                    clean_features[k] = v

            h_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_h_diagnostics"):
                try:
                    h_rule_diag = self.alphabet_classifier.evaluate_rule_h_diagnostics(clean_features)
                except Exception:
                    h_rule_diag = None

            k_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_k_diagnostics"):
                try:
                    k_rule_diag = self.alphabet_classifier.evaluate_rule_k_diagnostics(clean_features)
                except Exception:
                    k_rule_diag = None

            i_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_i_diagnostics"):
                try:
                    i_rule_diag = self.alphabet_classifier.evaluate_rule_i_diagnostics(clean_features)
                except Exception:
                    i_rule_diag = None

            m_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_m_diagnostics"):
                try:
                    m_rule_diag = self.alphabet_classifier.evaluate_rule_m_diagnostics(clean_features)
                except Exception:
                    m_rule_diag = None

            n_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_n_diagnostics"):
                try:
                    n_rule_diag = self.alphabet_classifier.evaluate_rule_n_diagnostics(clean_features)
                except Exception:
                    n_rule_diag = None

            o_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_o_diagnostics"):
                try:
                    o_rule_diag = self.alphabet_classifier.evaluate_rule_o_diagnostics(clean_features)
                except Exception:
                    o_rule_diag = None

            t_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_t_diagnostics"):
                try:
                    t_rule_diag = self.alphabet_classifier.evaluate_rule_t_diagnostics(clean_features)
                except Exception:
                    t_rule_diag = None

            u_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_u_diagnostics"):
                try:
                    u_rule_diag = self.alphabet_classifier.evaluate_rule_u_diagnostics(clean_features)
                except Exception:
                    u_rule_diag = None

            s_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_s_diagnostics"):
                try:
                    s_rule_diag = self.alphabet_classifier.evaluate_rule_s_diagnostics(clean_features)
                except Exception:
                    s_rule_diag = None

            p_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_p_diagnostics"):
                try:
                    p_rule_diag = self.alphabet_classifier.evaluate_rule_p_diagnostics(clean_features)
                except Exception:
                    p_rule_diag = None

            q_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_q_diagnostics"):
                try:
                    q_rule_diag = self.alphabet_classifier.evaluate_rule_q_diagnostics(clean_features)
                except Exception:
                    q_rule_diag = None

            r_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_r_diagnostics"):
                try:
                    r_rule_diag = self.alphabet_classifier.evaluate_rule_r_diagnostics(clean_features)
                except Exception:
                    r_rule_diag = None

            x_rule_diag = None
            if hasattr(self.alphabet_classifier, "evaluate_rule_x_diagnostics"):
                try:
                    x_rule_diag = self.alphabet_classifier.evaluate_rule_x_diagnostics(clean_features)
                except Exception:
                    x_rule_diag = None

        return {
            "status": status,
            "letter": letter,
            "reason": reason,
            "matched_candidates": clean_candidates,
            "features": clean_features,
            "h_rule_diagnostics": h_rule_diag,
            "k_rule_diagnostics": k_rule_diag,
            "i_rule_diagnostics": i_rule_diag,
            "m_rule_diagnostics": m_rule_diag,
            "n_rule_diagnostics": n_rule_diag,
            "o_rule_diagnostics": o_rule_diag,
            "t_rule_diagnostics": t_rule_diag,
            "u_rule_diagnostics": u_rule_diag,
            "s_rule_diagnostics": s_rule_diag,
            "p_rule_diagnostics": p_rule_diag,
            "q_rule_diagnostics": q_rule_diag,
            "r_rule_diagnostics": r_rule_diag,
            "x_rule_diagnostics": x_rule_diag,
        }

    def _build_pilot_static_head_payload(self, head_info=None) -> dict:
        """
        Chuẩn hóa thông tin chẩn đoán Static Head cho Pilot JSONL (Task 5):
        - Bổ sung has_static_head, load_status, pytorch_version, load_error.
        """
        base = head_info if isinstance(head_info, dict) else {}
        has_head = bool(getattr(self, "has_static_head", False))
        load_status = getattr(self, "static_head_load_status", "loaded" if has_head else "failed")
        if not has_head and load_status == "loaded":
            load_status = "disabled"
        return {
            "raw_predicted_letter": base.get("raw_predicted_letter"),
            "confidence": base.get("confidence"),
            "top_k": base.get("top_k"),
            "has_static_head": has_head,
            "load_status": load_status,
            "pytorch_version": getattr(self, "pytorch_version", str(getattr(torch, "__version__", "unknown"))),
            "load_error": getattr(self, "static_head_load_error", None),
        }

    def _build_pilot_frame_record(
        self,
        actual_frame_idx: int,
        clean_expected_sign: Optional[str],
        final_sign: str,
        final_conf: float,
        prediction_source: str,
        confirmed: bool,
        tracking_status: str,
        p_prim: Optional[list],
        session_id: Optional[str] = None,
        trial_id: Optional[str] = None,
        signer_id: Optional[str] = None,
        geom_payload: Optional[dict] = None,
        head_payload: Optional[dict] = None,
        gating_status: str = "none",
        threshold_applied: Optional[float] = None,
        reject_reason: Optional[str] = None,
        current_time: Optional[float] = None,
        client_capture_ts: Optional[float] = None,
        landmark_extract_ts: Optional[float] = None,
        request_send_ts: Optional[float] = None,
        backend_receive_ts: Optional[float] = None,
        video_pts_sec: Optional[Any] = None,
        video_frame_idx: Optional[Any] = None,
        image_width: Optional[int] = None,
        image_height: Optional[int] = None,
        handedness: Optional[str] = "Right",
        handedness_score: Optional[float] = None,
        tracking_confidence: Optional[float] = None,
        temporal_snapshot: Optional[dict] = None,
        phase: Optional[str] = None,
    ) -> dict:
        """
        Xây dựng bản ghi JSONL hoàn chỉnh theo Schema Version 2.0-kinematic (Task 3.10).
        Bảo toàn 100% các trường legacy và bổ sung đầy đủ:
        - raw_landmarks_21 (image_landmarks_21): viewport [0, 1] x [0, 1] kèm z tương đối
        - normalized_landmarks_21: scale-invariant wrist-shifted & palm-normalized
        - sync metadata: 6 mốc thời gian, frame indices, phát hiện frame drop / stall
        - temporal diagnostics snapshot: causal snapshot (10 recent predictions, vote count, cooldown)
        """
        image_landmarks_21 = None
        normalized_landmarks_21 = None

        if p_prim and len(p_prim) >= 21:
            try:
                # 1. Image coordinates (viewport [0, 1] x [0, 1] with relative depth z)
                image_landmarks_21 = [
                    [round(float(pt[0]), 6), round(float(pt[1]), 6), round(float(pt[2]), 6)]
                    for pt in p_prim[:21]
                ]
                # 2. Normalized coordinates (wrist-relative shifted and palm_scale normalized)
                w = p_prim[0]
                m = p_prim[9]
                dx = float(w[0]) - float(m[0])
                dy = float(w[1]) - float(m[1])
                dz = float(w[2]) - float(m[2])
                d_palm = math.sqrt(dx * dx + dy * dy + dz * dz)
                if d_palm > 1e-6:
                    normalized_landmarks_21 = [
                        [
                            round((float(pt[0]) - float(w[0])) / d_palm, 4),
                            round((float(pt[1]) - float(w[1])) / d_palm, 4),
                            round((float(pt[2]) - float(w[2])) / d_palm, 4),
                        ]
                        for pt in p_prim[:21]
                    ]
            except Exception:
                image_landmarks_21 = None
                normalized_landmarks_21 = None

        sync_meta = {
            "frame_idx": actual_frame_idx,
            "client_capture_ts": client_capture_ts,
            "landmark_extract_ts": landmark_extract_ts,
            "request_send_ts": request_send_ts,
            "backend_receive_ts": round(float(backend_receive_ts), 4) if backend_receive_ts is not None else None,
            "video_pts_sec": video_pts_sec if (video_pts_sec is not None and video_pts_sec != "") else "UNKNOWN",
            "video_frame_idx": video_frame_idx if (video_frame_idx is not None and video_frame_idx != "") else "UNKNOWN",
        }

        if temporal_snapshot is None:
            recent_signs = [s for s, c in self.prediction_history if c >= self.confidence_threshold]
            most_common_sign = "NEUTRAL"
            vote_count = 0
            if recent_signs:
                mc, cnt = collections.Counter(recent_signs).most_common(1)[0]
                most_common_sign = mc
                vote_count = cnt

            curr_clk = current_time if current_time is not None else time.time()
            last_clk = self.last_confirmed_time
            cd_rem = round(max(0.0, float(self.cooldown_sec - (curr_clk - last_clk))), 4) if last_clk else 0.0

            temporal_snapshot = {
                "smoothing_window": self.smoothing_window,
                "vote_threshold": self.smoothing_window // 2,
                "vote_count": vote_count,
                "most_common_sign": most_common_sign,
                "recent_predictions_10": [
                    {"sign": s, "confidence": round(float(c), 4)}
                    for s, c in list(self.prediction_history)
                ],
                "last_confirmed_sign": self.last_confirmed_sign,
                "last_confirmed_time": round(float(last_clk), 4) if last_clk else None,
                "cooldown_sec": round(float(self.cooldown_sec), 2),
                "cooldown_remaining_sec": cd_rem,
                "token_confirmation_event": bool(confirmed),
                "new_token_emitted": None,
                "model_has_weights": bool(getattr(getattr(self, "model", None), "has_weights", False)),
                "has_static_head": bool(getattr(self, "has_static_head", False)),
            }

        curr_time_float = float(current_time) if current_time is not None else time.time()

        return {
            "schema_version": PILOT_SCHEMA_VERSION,
            "timestamp": round(curr_time_float, 4),
            "session_id": str(session_id) if session_id else None,
            "trial_id": str(trial_id) if trial_id else None,
            "signer_id": str(signer_id) if signer_id else "anonymous",
            "frame_idx": actual_frame_idx,
            "expected_sign": clean_expected_sign,
            "tracking_status": tracking_status,
            "image_width": image_width,
            "image_height": image_height,
            "handedness": handedness,
            "handedness_score": round(float(handedness_score), 4) if handedness_score is not None else None,
            "tracking_confidence": round(float(tracking_confidence), 4) if tracking_confidence is not None else None,
            "feature_extractor_version": PILOT_FEATURE_EXTRACTOR_VERSION,
            "classifier_rule_version": PILOT_CLASSIFIER_RULE_VERSION,
            "classifier_config": {
                "O_D_TI_MAX": O_D_TI_MAX,
                "O_IDX_STRAIGHT_MAX": O_IDX_STRAIGHT_MAX,
            },
            "phase": phase,
            "image_landmarks_21": image_landmarks_21,
            "normalized_landmarks_21": normalized_landmarks_21,
            "final_sign": final_sign,
            "final_confidence": round(float(final_conf), 4),
            "prediction_source": prediction_source,
            "confirmed": bool(confirmed),
            "has_static_head": bool(getattr(self, "has_static_head", False)),
            "geometric": geom_payload if geom_payload is not None else self._build_pilot_geometric_payload("NO_MATCH", None),
            "static_head": head_payload if head_payload is not None else self._build_pilot_static_head_payload(),
            "gating_status": gating_status,
            "threshold_applied": round(float(threshold_applied), 4) if threshold_applied is not None else None,
            "reject_reason": reject_reason,
            "sync": sync_meta,
            "temporal": temporal_snapshot,
        }

    def predict_sequence(self, sequence_data, threshold=None):
        """
        Nhận diện chuỗi chuyển động 30 frames qua SSTCN_Attention.
        sequence_data: List hoặc np.ndarray kích thước (30, 75, 3) hoặc flattened
        75 joints = 33 Pose + 21 Left Hand + 21 Right Hand
        """
        if threshold is None:
            threshold = self.confidence_threshold

        try:
            arr = np.array(sequence_data, dtype=np.float32)
            
            # Xử lý các định dạng input linh hoạt
            if arr.shape == (30, 75, 3):
                pass
            elif arr.shape == (30, 225):
                arr = arr.reshape(30, 75, 3)
            elif arr.size == 30 * 75 * 3:
                arr = arr.reshape(30, 75, 3)
            elif len(arr) == 30 and isinstance(sequence_data[0], (list, np.ndarray)):
                arr = arr.reshape(30, 75, 3)
            else:
                return {
                    "success": False,
                    "error": f"Kích thước chuỗi không hợp lệ: {arr.shape}. Yêu cầu: (30, 75, 3)"
                }

            # Chuyển đổi sang định dạng đầu vào của SSTCN: (Batch=1, Channels=3, Time=30, Joints=75)
            # arr: (30, 75, 3) -> transpose(2, 0, 1) -> (3, 30, 75)
            tensor_seq = arr.transpose(2, 0, 1)
            tensor_seq = tensor_seq[np.newaxis, ...]  # (1, 3, 30, 75)
            tensor_input = torch.tensor(tensor_seq, dtype=torch.float32).to(self.device)

            with torch.no_grad():
                logits = self.sstcn_model(tensor_input)
                probs = torch.softmax(logits, dim=1).squeeze(0)
                
                # Lấy Top 1 và Top 5
                top_k_probs, top_k_indices = torch.topk(probs, k=min(5, len(probs)))
                top_k_probs = top_k_probs.cpu().numpy()
                top_k_indices = top_k_indices.cpu().numpy()

                max_prob = float(top_k_probs[0])
                pred_idx = int(top_k_indices[0])

            confirmed = False
            new_token_added = None
            current_time = time.time()

            # Kiểm tra ngưỡng tin cậy
            if max_prob >= threshold and pred_idx < 102:
                sign_info = dict(get_sign_info(pred_idx))
                # Nối với bộ từ vựng thống nhất (cùng mục từ mà chiều Văn bản → Ký hiệu dùng)
                entry = get_entry_by_sstcn(pred_idx)
                if entry:
                    sign_info["entry_id"] = entry["id"]
                    sign_info["variants"] = entry.get("variants", [])
                    sign_info["image_url"] = entry.get("image_url") or entry.get("thumb_url") or ""
                else:
                    sign_info["image_url"] = ""
                pred_term = sign_info["term"]

                # Kiểm tra debounce cooldown
                if pred_term != self.last_confirmed_sign or (current_time - self.last_confirmed_time > self.cooldown_sec):
                    self.last_confirmed_sign = pred_term
                    self.last_confirmed_time = current_time
                    confirmed = True
                    self.sentence_tokens.append(pred_term)
                    new_token_added = pred_term
            else:
                pred_idx = 102
                sign_info = {
                    "index": 102,
                    "term": "NEUTRAL",
                    "label": "Đang theo dõi cử chỉ...",
                    "description": "Chờ cử chỉ rõ ràng hoặc độ tin cậy chưa đạt ngưỡng",
                    "category": "Trạng thái",
                    "video_url": "",
                    "image_url": ""
                }

            top_k_list = [
                {
                    "index": int(idx),
                    "term": SIGN_DICT_102.get(int(idx), "UNKNOWN"),
                    "confidence": round(float(p), 3)
                }
                for p, idx in zip(top_k_probs, top_k_indices)
            ]

            return {
                "success": True,
                "sign": sign_info["term"],
                "label": sign_info.get("label", sign_info["term"]),
                "index": pred_idx,
                "confidence": round(max_prob, 2),
                "category": sign_info.get("category", ""),
                "description": sign_info.get("description", ""),
                "video_url": sign_info.get("video_url", ""),
                "image_url": sign_info.get("image_url", ""),
                "entry_id": sign_info.get("entry_id"),
                "variants": sign_info.get("variants", []),
                "confirmed": confirmed,
                "new_token": new_token_added,
                "sentence": " ".join(self.sentence_tokens),
                "top_k": top_k_list
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }

    def classify_geometry(self, landmarks, finger_states):
        """
        Kiem tra rang buoc hinh hoc ngon tay cho bang chu cai tuc thi (<5ms).

        Landmark MediaPipe (21 diem, toa do goc chua shift):
          0=Wrist, 1-4=Thumb(CMC,MCP,IP,TIP), 5-8=Index(MCP,PIP,DIP,TIP),
          9-12=Middle(MCP,PIP,DIP,TIP), 13-16=Ring(MCP,PIP,DIP,TIP), 17-20=Pinky(MCP,PIP,DIP,TIP)

        He toa do MediaPipe: x tang phai, y tang xuong, z tang ra gan camera.
        Toa do SHIFTED = tuong doi so voi wrist (landmark 0).
        Tat ca khoang cach duoc chuan hoa theo palm_scale (wrist->Middle MCP, landmark 9).

        Tra ve: (ten_chu, confidence) hoac ("NEUTRAL", 0.0) neu khong du can cu.

        Thu tu uu tien (dac biet -> chung chung):
          V/U -> Y/I -> YEU_THUONG -> L -> D -> O -> C -> B -> A/E -> 5-ngon-mo
        """
        # --- Kiem tra dau vao ---
        if not landmarks or len(landmarks) < 21:
            return "NEUTRAL", 0.0

        p = landmarks  # toa do goc (chua shift)

        # --- Chuan bi toa do shifted va palm_scale ---
        wrist = p[0]
        s = [[p[i][0] - wrist[0], p[i][1] - wrist[1], p[i][2] - wrist[2]]
             for i in range(21)]

        palm_scale = calculate_distance(s[0], s[9])
        if palm_scale < 1e-5:
            return "NEUTRAL", 0.0

        # --- Lay trang thai ngon tay tu finger_states ---
        thumb  = finger_states.get("thumb_extended", False)
        index  = finger_states.get("index_extended", False)
        middle = finger_states.get("middle_extended", False)
        ring   = finger_states.get("ring_extended", False)
        pinky  = finger_states.get("pinky_extended", False)

        # Curl ratio: 0.0=duoi thang, 1.0=gap hoan toan
        thumb_curl  = finger_states.get("thumb_curl",  0.5)
        index_curl  = finger_states.get("index_curl",  0.5)
        middle_curl = finger_states.get("middle_curl", 0.5)
        ring_curl   = finger_states.get("ring_curl",   0.5)
        pinky_curl  = finger_states.get("pinky_curl",  0.5)

        # --- Khoang cach chuan hoa theo palm_scale (toa do shifted) ---
        d_thumb_index  = calculate_distance(s[4], s[8])  / palm_scale
        d_index_middle = calculate_distance(s[8], s[12]) / palm_scale
        d_thumb_pinky  = calculate_distance(s[4], s[20]) / palm_scale
        d_thumb_middle = calculate_distance(s[4], s[12]) / palm_scale

        # --- Nguong trang thai ngon ---
        # Curl > 0.30 => ngon "gap" (folded), dung cho ngon cai vi extended phuc tap hon
        CURL_FOLDED = 0.30
        # Curl > 0.55 => ngon cai "gap hoan toan" (sat long ban tay)
        THUMB_FIRMLY_FOLDED = 0.55

        index_folded  = not index
        middle_folded = not middle
        ring_folded   = not ring
        pinky_folded  = not pinky
        thumb_folded  = thumb_curl > CURL_FOLDED
        thumb_firmly_folded = thumb_curl > THUMB_FIRMLY_FOLDED

        # ================================================================
        # RULE 0: Chu H, Chu K, Chu I, Chu Y, Chu T, Chu U, Chu S, Chu P, Chu Q, Chu O, Chu R, Chu X (Uu tien kiem tra truoc V/U/Y tu AlphabetClassifier)
        # ================================================================
        if hasattr(self, "alphabet_classifier") and self.alphabet_classifier is not None:
            c_let, c_conf = self.alphabet_classifier.classify(p)
            if c_let in ("H", "K", "I", "Y", "T", "U", "S", "P", "Q", "O", "R", "X"):
                return c_let, c_conf

        # ================================================================
        # RULE 1: Chu V va U
        # Dieu kien: Chi ngon tro (8) VA ngon giua (12) duoi thang,
        #            ngon ap ut (16) va ut (20) gap chat.
        # Phan biet V va U bang to hop goc mo va khoang cach:
        #   - V: Hai ngon xoe hinh chu V ro ret:
        #        + Ket hop ca khoang cach va goc mo: (d_index_middle > 0.28 va angle_vu > 12 do)
        #        + Hoac dac trung mo rat rong do goc camera: (angle_vu > 18 do hoac d_index_middle > 0.45)
        #   - U: Hai ngon song song khap sat nhau, goc mo va khoang cach deu hep.
        # ================================================================
        if index and middle and not ring and not pinky:
            if ring_folded and pinky_folded:
                angle_vu = calculate_angle(s[8], s[0], s[12])
                is_v_shape = (d_index_middle > 0.28 and angle_vu > 12) or angle_vu > 18 or d_index_middle > 0.45
                if is_v_shape:
                    return "V", 0.88
                else:
                    return "U", 0.86

        # ================================================================
        # RULE 2: Chu Y va I
        # Dieu kien: chi ngon ut (20) duoi, 3 ngon giua (8,12,16) gap
        # Phan biet Y/I: Y co them ngon cai duoi (thumb extended)
        # ================================================================
        if pinky and not index and not middle and not ring:
            if index_folded and middle_folded and ring_folded:
                if thumb:
                    return "Y", 0.90
                return "I", 0.90

        # ================================================================
        # RULE 3: Chu YEU_THUONG (ILY)
        # Dieu kien: ngon cai (4) + tro (8) + ut (20) cung duoi,
        #            ngon giua (12) va ap ut (16) gap
        # ================================================================
        if thumb and index and not middle and not ring and pinky:
            if middle_folded and ring_folded:
                return "YEU_THUONG", 0.92

        # ================================================================
        # RULE 4: Chu L va D
        # Dieu kien chung: Chi ngon tro (8) duoi, 3 ngon con (12,16,20) gap.
        # Phan biet L va D bang to hop dac trung hinh hoc:
        #   - L: Ngon cai duoi thang (thumb and not thumb_folded), vuon ngang tao goc lon
        #        (angle_ti >= 50 do VA do dang ngang thumb_lateral >= 0.35).
        #   - D: Ngon cai khong vuon ngang (ngon cai gap hoac thu sat vao long ban tay / ngon giua),
        #        dau ngon tro duoi huong len tren.
        # ================================================================
        if index and not middle and not ring and not pinky:
            if middle_folded and ring_folded and pinky_folded:
                angle_ti = calculate_angle(s[4], s[0], s[8])
                thumb_lateral = abs(s[4][0] - s[0][0]) / palm_scale

                # L: Ngon cai vuon ngang ro rang tao chu L vuong van
                if thumb and not thumb_firmly_folded and angle_ti >= 50 and thumb_lateral >= 0.35:
                    return "L", 0.88

                # D: Chi ngon tro duoi thang, ngon cai khong vuon ngang chu L
                is_index_pointing_up = s[8][1] < s[5][1]
                thumb_not_L = (not thumb) or thumb_firmly_folded or angle_ti < 50 or thumb_lateral < 0.35
                if is_index_pointing_up and thumb_not_L:
                    return "D", 0.84

        # ================================================================
        # RULE 5: Chu O
        # Dieu kien: Ngon cai va ngon tro tao vong tron nho (d_thumb_index < 0.30),
        #            ngon cai tiep xuc huong ve dau ngon tro (goc tip-contact >= 80 do),
        #            cac ngon khong duoi thang toan bo de tao vong tron O.
        # ================================================================
        if d_thumb_index < 0.30:
            angle_tip_contact = calculate_angle(s[3], s[4], s[8])
            if angle_tip_contact >= 80 and not (index and middle and ring and pinky):
                return "O", 0.86

        # ================================================================
        # RULE 6: Chu C va B
        # C: 4 ngon mo/khum cong, ngon cai arch ra ngoai tao vom C
        #    (0.30 <= d_thumb_index < 1.25, goc mo ngon cai so voi tro >= 15 do)
        # B: 4 ngon duoi thang, ngon cai gap sat long ban tay
        #    (goc mo ngon cai < 15 do hoac d_thumb_index < 0.30)
        # ================================================================
        if index and middle and ring and pinky and not thumb:
            angle_ti_cb = calculate_angle(s[4], s[0], s[8])
            # Truong hop C: ngon cai arch ra ngoai
            if not thumb_firmly_folded and 0.30 <= d_thumb_index < 1.25 and angle_ti_cb >= 15:
                if d_thumb_middle < 1.40:
                    return "C", 0.83

            # Truong hop B: ngon cai gap sat long ban tay
            if thumb_firmly_folded or angle_ti_cb < 15 or d_thumb_index < 0.30:
                if d_index_middle < 0.65:
                    return "B", 0.86

        # ================================================================
        # RULE 7: Chu A va E
        # Dieu kien chung: Ca 4 ngon (tro/giua/ap-ut/ut) deu gap thanh nam dam.
        # Phan biet A/E:
        #   E: Ngon cai gap/tuck vao trong long ban tay duoi cac ngon (d_thumb_index < 0.35 hoac thumb_curl > 0.40)
        #   A: Ngon cai nam doc/canh ngoai nam dam, khong bi ep vao trong (thumb_curl <= 0.40 va d_thumb_index >= 0.35)
        # ================================================================
        if not index and not middle and not ring and not pinky:
            if index_folded and middle_folded and ring_folded and pinky_folded:
                if thumb_firmly_folded or thumb_curl > 0.40 or d_thumb_index < 0.35:
                    return "E", 0.84
                if thumb_curl <= 0.40 and d_thumb_index >= 0.35:
                    return "A", 0.84

        # ================================================================
        # RULE 8: Chu B (fallback)
        # ================================================================
        if index and middle and ring and pinky and not thumb:
            if thumb_firmly_folded and d_index_middle < 0.65:
                return "B", 0.85

        # ================================================================
        # RULE 9: Ca 5 ngon mo (XIN_CHAO / CAM_ON / XIN_VUI_LONG)
        # Dung toa do wrist goc (p[0][1]) de xac dinh vi tri tay trong khung hinh.
        # ================================================================
        if index and middle and ring and pinky and thumb:
            wrist_y_norm = p[0][1]  # toa do wrist goc [0,1]
            if wrist_y_norm < 0.40:
                return "XIN_CHAO", 0.88
            elif wrist_y_norm > 0.65:
                return "XIN_VUI_LONG", 0.85
            return "CAM_ON", 0.86

        return "NEUTRAL", 0.0

    def classify_dual_hands(self, dual_meta, p_prim, p_sec):
        """
        Nhận diện các cử chỉ phối hợp hai bàn tay (Two-Handed Coordination)
        """
        if not dual_meta.get("is_dual_hand", False):
            return None, 0.0

        p_states = dual_meta["primary_states"]
        s_states = dual_meta["secondary_states"]
        wrist_dist = dual_meta["wrist_distance"]
        height_delta = dual_meta["height_delta"]
        symm = dual_meta["symmetry_score"]

        prim_fist = not p_states.get("index_extended") and not p_states.get("middle_extended")
        sec_fist = not s_states.get("index_extended") and not s_states.get("middle_extended")
        prim_flat = p_states.get("index_extended") and p_states.get("middle_extended") and p_states.get("ring_extended")
        sec_flat = s_states.get("index_extended") and s_states.get("middle_extended") and s_states.get("ring_extended")

        # 1. Giúp đỡ (GIUP_DO): 1 tay nắm đấm đặt trên bàn tay kia mở phẳng
        if (prim_fist and sec_flat) or (sec_fist and prim_flat):
            if wrist_dist < 0.45:
                return "GIUP_DO", 0.96

        # 2. Yêu thương: Hai tay bắt chéo hoặc 2 tay cùng làm ILY
        prim_ily = p_states.get("thumb_extended") and p_states.get("index_extended") and p_states.get("pinky_extended")
        sec_ily = s_states.get("thumb_extended") and s_states.get("index_extended") and s_states.get("pinky_extended")
        if (prim_ily and sec_ily) or (wrist_dist < 0.28 and symm >= 0.6):
            return "YEU_THUONG", 0.95

        # 3. Gia đình (GIA_DINH): Hai tay tạo vòng tròn đối xứng trước ngực
        if wrist_dist < 0.38 and symm >= 0.7:
            return "GIA_DINH", 0.93

        # 4. Cảm ơn hai tay: Cả 2 tay mở, 1 tay cao hơn tay kia
        if prim_flat and sec_flat and abs(height_delta) > 0.08:
            return "CAM_ON", 0.94

        # 5. Xin chào hai tay: 1 tay giơ cao
        if (p_prim[0][1] < 0.42 or p_sec[0][1] < 0.42) and (prim_flat or sec_flat):
            return "XIN_CHAO", 0.95

        return None, 0.0

    def predict_landmarks(self, *args, **kwargs):
        """Serialize recognizer state and isolate explicitly tagged pilot trials."""
        # Pilot requests always carry both identifiers. Their temporal votes and
        # confirmation cooldown belong to that trial, even if an older request
        # arrives after a newer trial has already started.
        session_id = kwargs.get("session_id", args[4] if len(args) > 4 else None)
        mode = kwargs.get("mode", args[5] if len(args) > 5 else None)
        trial_id = kwargs.get("trial_id", args[7] if len(args) > 7 else None)
        pilot_key = None
        if session_id is not None and trial_id is not None and mode == "letters":
            pilot_key = (str(session_id), str(trial_id), mode)

        # The recognizer contains shared mutable temporal state. Serialize calls
        # so switching to a trial-local state cannot leak into another request.
        with self._pilot_temporal_lock:
            if pilot_key is None:
                return self._predict_landmarks_impl(*args, **kwargs)

            previous_history = self.prediction_history
            previous_confirmed_sign = self.last_confirmed_sign
            previous_confirmed_time = self.last_confirmed_time
            state = self._pilot_temporal_states.get(pilot_key)
            if state is None:
                state = {
                    "prediction_history": collections.deque(maxlen=self.smoothing_window),
                    "last_confirmed_sign": None,
                    "last_confirmed_time": 0,
                }
                self._pilot_temporal_states[pilot_key] = state

            self.prediction_history = state["prediction_history"]
            self.last_confirmed_sign = state["last_confirmed_sign"]
            self.last_confirmed_time = state["last_confirmed_time"]
            try:
                return self._predict_landmarks_impl(*args, **kwargs)
            finally:
                state["prediction_history"] = self.prediction_history
                state["last_confirmed_sign"] = self.last_confirmed_sign
                state["last_confirmed_time"] = self.last_confirmed_time
                self.prediction_history = previous_history
                self.last_confirmed_sign = previous_confirmed_sign
                self.last_confirmed_time = previous_confirmed_time

    def _predict_landmarks_impl(
        self,
        landmarks_list,
        landmarks_secondary=None,
        multi_hands=None,
        expected_sign=None,
        session_id=None,
        mode=None,
        timestamp=None,
        trial_id=None,
        frame_idx=None,
        client_capture_ts=None,
        landmark_extract_ts=None,
        request_send_ts=None,
        backend_receive_ts=None,
        video_pts_sec=None,
        video_frame_idx=None,
        signer_id=None,
        image_width=None,
        image_height=None,
        handedness_score=None,
        tracking_confidence=None,
        phase=None,
    ):
        """
        Nhận diện cử chỉ đơn frame hỗ trợ cả 1 tay và 2 tay.
        Khi mode == 'letters_numbers', backend chi tra ve 27 chu cai VSL hoac NEUTRAL.
        Hỗ trợ lưu đầy đủ 21 landmarks thô, đồng bộ video RGB và temporal diagnostics (Task 3.10).
        """
        if backend_receive_ts is None:
            backend_receive_ts = time.time()
        from backend.vision.feature_extractor import extract_dual_hand_features

        # Chuẩn hóa nhãn ground truth: chỉ dùng cho logging chẩn đoán, tuyệt đối KHÔNG vào bộ phân loại
        clean_expected_sign = (
            str(expected_sign).strip().upper()
            if expected_sign and str(expected_sign).strip().upper() in ALLOWED_LETTERS
            else None
        )
        if frame_idx is not None:
            try:
                actual_frame_idx = int(frame_idx)
            except (ValueError, TypeError):
                actual_frame_idx = self._next_pilot_frame_idx()
        else:
            actual_frame_idx = self._next_pilot_frame_idx()

        # Phân tách và nhận biết 4 chế độ độc lập + alias tương thích ngược:
        # 1. 'letters': Chế độ Chữ cái chuyên biệt (27 chữ cái VSL A-Z, Đ).
        # 2. 'digits': Chế độ Chữ số chuyên biệt (1-9 qua VSLDigitClassifier).
        # 3. 'letters_numbers': Chế độ lai tương thích ngược (legacy hybrid alias).
        # 4. 'words', 'sentence' hoặc None: Chế độ từ vựng / ghép câu / mặc định cũ.
        is_letters_mode = (mode == "letters")
        is_digits_mode  = (mode == "digits")
        is_legacy_mode  = (mode == "letters_numbers")
        is_letters_filter = (is_letters_mode or is_legacy_mode)

        # Quản lý bộ đệm smoothing khi chuyển chế độ:
        # Nếu phiên làm việc đổi mode, xóa sạch lịch sử dự đoán của phiên trước đó
        # để tránh kết quả của chế độ cũ gây nhiễu cho chế độ mới.
        session_key = str(session_id) if session_id else "_default_"
        if not hasattr(self, "_session_last_modes"):
            self._session_last_modes = {}
        current_time = time.time() if timestamp is None else float(timestamp)

        if self._session_last_modes.get(session_key) != mode:
            self.prediction_history.clear()
            self.last_confirmed_sign = None
            self._session_last_modes[session_key] = mode
            if hasattr(self, "digit_10_detector"):
                self.digit_10_detector.reset()
            if session_key in self.diacritic_fsms:
                self.diacritic_fsms[session_key].reset()

        p_prim = None
        p_sec = None
        handedness = "Right"

        if multi_hands and len(multi_hands) > 0:
            p_prim = multi_hands[0].get("landmarks", [])
            handedness = multi_hands[0].get("handedness", "Right")
            if len(multi_hands) > 1:
                p_sec = multi_hands[1].get("landmarks", [])
        elif landmarks_list is not None:
            p_prim = landmarks_list
            if landmarks_secondary:
                p_sec = landmarks_secondary

        if not p_prim or len(p_prim) < 21:
            self.prediction_history.append(("NEUTRAL", 0.0))
            if is_digits_mode and hasattr(self, "digit_10_detector"):
                self.digit_10_detector.reset()
            if session_key in self.diacritic_fsms:
                self.diacritic_fsms[session_key].reset()

            if self.pilot_logging_enabled and self.pilot_logger is not None:
                tracking_status = "NO_HAND" if (not p_prim or len(p_prim) == 0) else "INVALID_SHAPE"
                record = self._build_pilot_frame_record(
                    actual_frame_idx=actual_frame_idx,
                    clean_expected_sign=clean_expected_sign,
                    final_sign="NEUTRAL",
                    final_conf=0.0,
                    prediction_source="none",
                    confirmed=False,
                    tracking_status=tracking_status,
                    p_prim=None,
                    session_id=session_id,
                    trial_id=trial_id,
                    signer_id=signer_id,
                    geom_payload=self._build_pilot_geometric_payload(
                        "INVALID_INPUT",
                        None,
                        reason="No valid hand landmarks detected",
                    ),
                    head_payload=self._build_pilot_static_head_payload(),
                    gating_status="none",
                    threshold_applied=None,
                    reject_reason="No valid hand landmarks detected",
                    current_time=current_time,
                    client_capture_ts=client_capture_ts,
                    landmark_extract_ts=landmark_extract_ts,
                    request_send_ts=request_send_ts,
                    backend_receive_ts=backend_receive_ts,
                    video_pts_sec=video_pts_sec,
                    video_frame_idx=video_frame_idx,
                    image_width=image_width,
                    image_height=image_height,
                    handedness=handedness,
                    handedness_score=handedness_score,
                    tracking_confidence=tracking_confidence,
                    phase=phase,
                )
                self.pilot_logger.log_frame(record, filepath=getattr(self, "pilot_filepath", None))

            return {
                "sign": "NEUTRAL",
                "label": "Chờ cử chỉ...",
                "type": "unknown",
                "confidence": 0.0,
                "confirmed": False,
                "new_token": None,
                "sentence": " ".join(self.sentence_tokens)
            }

        # Kiểm tra tính hợp lệ của tọa độ landmark (tránh NaN, Inf hoặc hình dạng không đúng 21x3)
        try:
            arr_check = np.asarray(p_prim, dtype=np.float32)
            if arr_check.shape != (21, 3) or not np.isfinite(arr_check).all():
                if self.pilot_logging_enabled and self.pilot_logger is not None:
                    record = self._build_pilot_frame_record(
                        actual_frame_idx=actual_frame_idx,
                        clean_expected_sign=clean_expected_sign,
                        final_sign="NEUTRAL",
                        final_conf=0.0,
                        prediction_source="none",
                        confirmed=False,
                        tracking_status="INVALID_SHAPE",
                        p_prim=None,
                        session_id=session_id,
                        trial_id=trial_id,
                        signer_id=signer_id,
                        geom_payload=self._build_pilot_geometric_payload(
                            "INVALID_INPUT",
                            None,
                            reason="Non-finite or invalid shape landmarks",
                        ),
                        head_payload=self._build_pilot_static_head_payload(),
                        gating_status="none",
                        threshold_applied=None,
                        reject_reason="Non-finite or invalid shape landmarks",
                        current_time=current_time,
                        client_capture_ts=client_capture_ts,
                        landmark_extract_ts=landmark_extract_ts,
                        request_send_ts=request_send_ts,
                        backend_receive_ts=backend_receive_ts,
                        video_pts_sec=video_pts_sec,
                        video_frame_idx=video_frame_idx,
                        image_width=image_width,
                        image_height=image_height,
                        handedness=handedness,
                        handedness_score=handedness_score,
                        tracking_confidence=tracking_confidence,
                        phase=phase,
                    )
                    self.pilot_logger.log_frame(record, filepath=getattr(self, "pilot_filepath", None))
                return {
                    "sign": "NEUTRAL",
                    "label": "Chờ cử chỉ...",
                    "type": "unknown",
                    "confidence": 0.0,
                    "confirmed": False,
                    "new_token": None,
                    "sentence": " ".join(self.sentence_tokens)
                }
        except Exception as e:
            if self.pilot_logging_enabled and self.pilot_logger is not None:
                record = self._build_pilot_frame_record(
                    actual_frame_idx=actual_frame_idx,
                    clean_expected_sign=clean_expected_sign,
                    final_sign="NEUTRAL",
                    final_conf=0.0,
                    prediction_source="none",
                    confirmed=False,
                    tracking_status="ERROR",
                    p_prim=None,
                    session_id=session_id,
                    trial_id=trial_id,
                    signer_id=signer_id,
                    geom_payload=self._build_pilot_geometric_payload(
                        "INVALID_INPUT",
                        None,
                        reason=f"Landmark validation exception: {e}",
                    ),
                    head_payload=self._build_pilot_static_head_payload(),
                    gating_status="none",
                    threshold_applied=None,
                    reject_reason=f"Landmark validation exception: {e}",
                    current_time=current_time,
                    client_capture_ts=client_capture_ts,
                    landmark_extract_ts=landmark_extract_ts,
                    request_send_ts=request_send_ts,
                    backend_receive_ts=backend_receive_ts,
                    video_pts_sec=video_pts_sec,
                    video_frame_idx=video_frame_idx,
                    image_width=image_width,
                    image_height=image_height,
                    handedness=handedness,
                    handedness_score=handedness_score,
                    tracking_confidence=tracking_confidence,
                    phase=phase,
                )
                self.pilot_logger.log_frame(record, filepath=getattr(self, "pilot_filepath", None))
            return {
                "sign": "NEUTRAL",
                "label": "Chờ cử chỉ...",
                "type": "unknown",
                "confidence": 0.0,
                "confirmed": False,
                "new_token": None,
                "sentence": " ".join(self.sentence_tokens)
            }

        # -------------------------------------------------------------
        # XỬ LÝ CHUYÊN BIỆT CHO CHẾ ĐỘ CHỮ SỐ (digits mode)
        # Độc lập từng ký hiệu, không ghép chuỗi số liên tiếp
        # -------------------------------------------------------------
        if is_digits_mode:
            # 1. ƯU TIÊN 1: Nhận diện cử chỉ động riêng cho Số 10 VSL
            det10_res = self.digit_10_detector.update(p_prim, timestamp=current_time)
            if det10_res["detected"]:
                self.prediction_history.clear()
                label_token = "10"
                tts_text = DIGIT_TTS_MAP["10"]
                
                # Phụ đề trong chế độ chữ số bỏ hoàn toàn từ "Số", chỉ hiển thị số thuần túy ("10")
                if self.sentence_tokens and (self.sentence_tokens[-1] in ("1", "Số 1")):
                    self.sentence_tokens[-1] = label_token
                elif not (self.sentence_tokens and self.sentence_tokens[-1] == label_token):
                    self.sentence_tokens.append(label_token)
                    
                self.last_confirmed_sign = "10"
                self.last_confirmed_time = current_time
                
                self._log_diagnostic(
                    p_prim=p_prim,
                    handedness=handedness,
                    finger_states=None,
                    geom_sign="10",
                    geom_conf=det10_res["confidence"],
                    has_mlp=False,
                    nn_sign="NEUTRAL",
                    nn_conf=0.0,
                    final_sign="10",
                    final_conf=det10_res["confidence"],
                    confirmed=True,
                    expected_sign=expected_sign,
                    session_id=session_id,
                    mode=mode,
                    classifier_details={"detector_10": det10_res}
                )
                
                return {
                    "sign": "10",
                    "label": "Số 10",
                    "tts_text": tts_text,
                    "type": "digit",
                    "confidence": round(float(det10_res["confidence"]), 2),
                    "confirmed": True,
                    "new_token": label_token,
                    "sentence": " ".join(self.sentence_tokens),
                    "description": "Chữ số 10",
                    "detector_10": det10_res
                }

            # Kiểm tra thời gian cooldown sau khi vừa chốt Số 10
            is_in_10_cooldown = (current_time - self.digit_10_detector.last_detected_time < self.digit_10_detector.config.cooldown_sec)

            # 2. Phân loại hình học đơn frame cho chữ số 0-9 độc lập
            from backend.vision.digit_classifier import digit_classifier
            digit_res = digit_classifier.classify_detailed(p_prim)
            if (not is_in_10_cooldown) and digit_res["status"] == "MATCH" and digit_res["digit"] in DIGITS_MODE_SIGNS:
                raw_digit = digit_res["digit"]
                raw_conf = digit_res["confidence"]
            else:
                raw_digit = "NEUTRAL"
                raw_conf = 0.0

            # Chặn tuyệt đối: chế độ digits không bao giờ nhận diện chữ cái hoặc từ vựng
            if raw_digit not in DIGITS_MODE_SIGNS:
                raw_digit = "NEUTRAL"
                raw_conf = 0.0

            self.prediction_history.append((raw_digit, raw_conf))

            # Smoothing cho chữ số đơn tức thời
            recent_signs = [s for s, c in self.prediction_history if c >= self.confidence_threshold]
            confirmed = False
            new_token_added = None
            tts_text = None

            # Hoãn chốt số 1 nếu detector đang quan sát chuyển động xoay/nghiêng của Số 10
            is_observing_10 = det10_res.get("is_observing", False)

            if len(recent_signs) >= self.smoothing_window // 2:
                smoothed_digit, count = collections.Counter(recent_signs).most_common(1)[0]
                smoothed_conf = raw_conf if count >= (self.smoothing_window // 2) else 0.0
                if count >= (self.smoothing_window // 2) and smoothed_digit in DIGITS_MODE_SIGNS:
                    should_defer_one = (smoothed_digit == "1" and is_observing_10)
                    if not should_defer_one:
                        if smoothed_digit != self.last_confirmed_sign or (current_time - self.last_confirmed_time > self.cooldown_sec):
                            self.last_confirmed_sign = smoothed_digit
                            self.last_confirmed_time = current_time
                            confirmed = True
                            label_token = smoothed_digit
                            self.sentence_tokens.append(label_token)
                            new_token_added = label_token
                            tts_text = DIGIT_TTS_MAP.get(smoothed_digit, smoothed_digit)
            else:
                smoothed_digit = raw_digit
                smoothed_conf = raw_conf

            display_sign = raw_digit if raw_digit in DIGITS_MODE_SIGNS else "NEUTRAL"
            is_valid_digit_sign = (display_sign != "NEUTRAL" and display_sign in DIGITS_MODE_SIGNS)
            display_label = f"Số {display_sign}" if is_valid_digit_sign else "Đang theo dõi cử chỉ..."

            if confirmed and not tts_text and is_valid_digit_sign:
                tts_text = DIGIT_TTS_MAP.get(display_sign, display_sign)

            self._log_diagnostic(
                p_prim=p_prim,
                handedness=handedness,
                finger_states=None,
                geom_sign=digit_res.get("digit", "NEUTRAL"),
                geom_conf=digit_res.get("confidence", 0.0),
                has_mlp=False,
                nn_sign="NEUTRAL",
                nn_conf=0.0,
                final_sign=display_sign,
                final_conf=raw_conf,
                confirmed=confirmed,
                expected_sign=expected_sign,
                session_id=session_id,
                mode=mode,
                classifier_details=digit_res
            )

            return {
                "sign": display_sign,
                "label": display_label,
                "tts_text": tts_text,
                "type": "digit" if is_valid_digit_sign else "unknown",
                "confidence": round(float(raw_conf), 2),
                "confirmed": confirmed,
                "new_token": new_token_added,
                "sentence": " ".join(self.sentence_tokens),
                "description": f"Chữ số {display_sign}" if is_valid_digit_sign else ""
            }

        # -------------------------------------------------------------
        # XỬ LÝ CHUYÊN BIỆT CHO CHẾ ĐỘ CHỮ CÁI (letters mode)
        # -------------------------------------------------------------
        if is_letters_mode:
            alpha_res = self.alphabet_classifier.classify_detailed(p_prim)
            status = alpha_res.get("status")
            geom_status = status
            geom_letter = alpha_res.get("letter") if status == "MATCH" else None
            geom_reason = alpha_res.get("reason")
            geom_matched_candidates = alpha_res.get("matched_candidates")
            geom_features = alpha_res.get("features")

            head_info = {
                "raw_predicted_letter": None,
                "confidence": None,
                "top_k": None,
            }
            gating_status = "none"
            threshold_applied = None
            reject_reason = None

            # Bước 1: Ưu tiên tuyệt đối 12 luật hình học hiện có
            if status == "MATCH" and alpha_res.get("letter") in ALLOWED_LETTERS:
                final_sign = alpha_res["letter"]
                final_conf = alpha_res["confidence"]
                recognition_source = "geometric_rule"
                alpha_res["source"] = "geometric_rule"
                gating_status = "passed_geometric"
            elif status in ("NO_MATCH", "UNKNOWN") and self.has_static_head and p_prim and len(p_prim) >= 21:
                # Bước 2: Gọi static alphabet head cho fallback A/D/E (Task 7E.5B) và 10 chữ cái bổ sung (Task 6C) khi luật hình học không khớp
                try:
                    feat_72, _ = normalize_hand_landmarks(p_prim)
                    head_res = self.static_alphabet_head.predict_feature(feat_72[:72], confidence_threshold=0.0)
                    head_letter = head_res.get("raw_predicted_letter", "UNKNOWN")
                    head_conf = float(head_res.get("confidence", 0.0))
                    head_top_k = head_res.get("top_k", [])
                    head_info = {
                        "raw_predicted_letter": head_letter,
                        "confidence": round(head_conf, 4),
                        "top_k": head_top_k,
                    }

                    req_threshold = self.supplemental_thresholds.get(head_letter, 0.70)

                    # Hỗ trợ bất biến đối xứng trục X qua cổ tay cho chữ G và H (tay trái / camera mirror):
                    geom_feat = alpha_res.get("features") if isinstance(alpha_res, dict) else None
                    is_primary_accepted = (
                        (head_letter in ALLOWED_FALLBACK_BASE_LETTERS and head_conf >= self.fallback_base_threshold)
                        or (head_letter in ALLOWED_SUPPLEMENTAL_LETTERS and head_conf >= req_threshold)
                    )
                    if not is_primary_accepted and isinstance(geom_feat, dict):
                        idx_dx = float(geom_feat.get("idx_dx", 0.0))
                        d_iw = float(geom_feat.get("d_IW", 0.0))
                        d_mw = float(geom_feat.get("d_MW", 0.0))
                        d_rw = float(geom_feat.get("d_RW", 99.0))
                        d_pw = float(geom_feat.get("d_PW", 99.0))
                        i_curl = float(geom_feat.get("index_curl", 1.0))
                        m_curl = float(geom_feat.get("middle_curl", 0.0))

                        is_g_mir_cand = (
                            idx_dx < -0.20 and d_iw >= 1.50 and (d_iw - d_mw) >= 0.55
                            and d_rw <= 1.25 and d_pw <= 1.25 and i_curl <= 0.15
                        )
                        is_h_mir_cand = (
                            d_iw >= 1.25 and d_rw >= 1.35 and d_pw >= 1.35
                            and d_mw <= 1.05 and m_curl >= 0.40
                        )
                        if is_g_mir_cand or is_h_mir_cand:
                            w_x = float(p_prim[0][0])
                            p_mir = [[2.0 * w_x - float(pt[0]), float(pt[1]), float(pt[2])] for pt in p_prim]
                            feat_mir, _ = normalize_hand_landmarks(p_mir)
                            mir_res = self.static_alphabet_head.predict_feature(feat_mir[:72], confidence_threshold=0.0)
                            mir_letter = mir_res.get("raw_predicted_letter", "UNKNOWN")
                            mir_conf = float(mir_res.get("confidence", 0.0))
                            target_mir = "G" if (is_g_mir_cand and mir_letter == "G") else ("H" if (is_h_mir_cand and mir_letter == "H") else None)
                            if target_mir is not None:
                                t_thresh = self.supplemental_thresholds.get(target_mir, 0.70)
                                if mir_conf >= t_thresh:
                                    head_letter = target_mir
                                    head_conf = mir_conf
                                    head_top_k = mir_res.get("top_k", [])
                                    req_threshold = t_thresh
                                    head_info = {
                                        "raw_predicted_letter": head_letter,
                                        "confidence": round(head_conf, 4),
                                        "top_k": head_top_k,
                                    }

                    # 2a. Gated Static Head Fallback cho 3 chữ cái nền A, D, E (Task 7E.5B)
                    if head_letter in ALLOWED_FALLBACK_BASE_LETTERS and head_conf >= self.fallback_base_threshold:
                        final_sign = head_letter
                        final_conf = head_conf
                        recognition_source = "static_head_fallback"
                        gating_status = "passed_fallback"
                        threshold_applied = self.fallback_base_threshold
                        alpha_res = {
                            "status": "MATCH",
                            "letter": head_letter,
                            "confidence": head_conf,
                            "source": "static_head_fallback",
                            "head_top_k": head_top_k,
                            "reject_reason": None,
                        }
                    # 2b. 10 chữ cái bổ sung chưa có geometric rules (Task 6C)
                    elif head_letter in ALLOWED_SUPPLEMENTAL_LETTERS and head_conf >= req_threshold:
                        final_sign = head_letter
                        final_conf = head_conf
                        recognition_source = "static_head"
                        gating_status = "passed_supplemental"
                        threshold_applied = req_threshold
                        alpha_res = {
                            "status": "MATCH",
                            "letter": head_letter,
                            "confidence": head_conf,
                            "source": "static_head",
                            "head_top_k": head_top_k,
                            "reject_reason": None,
                        }
                    else:
                        final_sign = "NEUTRAL"
                        final_conf = 0.0
                        recognition_source = "none"
                        gating_status = "rejected"
                        thresh_used = self.fallback_base_threshold if head_letter in ALLOWED_FALLBACK_BASE_LETTERS else req_threshold
                        threshold_applied = thresh_used
                        reject_reason = (
                            f"Letter '{head_letter}' not in supplemental/fallback set"
                            if head_letter not in ALLOWED_SUPPLEMENTAL_LETTERS and head_letter not in ALLOWED_FALLBACK_BASE_LETTERS
                            else f"Confidence {head_conf:.2f} < threshold {thresh_used:.2f}"
                        )
                        alpha_res = {
                            "status": "NO_MATCH",
                            "letter": "UNKNOWN",
                            "confidence": 0.0,
                            "source": "static_head_rejected",
                            "head_letter": head_letter,
                            "head_confidence": head_conf,
                            "reject_reason": reject_reason,
                            "head_top_k": head_top_k,
                        }
                except Exception as e:
                    final_sign = "NEUTRAL"
                    final_conf = 0.0
                    recognition_source = "none"
                    gating_status = "error"
                    threshold_applied = None
                    reject_reason = str(e)
                    alpha_res = {
                        "status": "NO_MATCH",
                        "letter": "UNKNOWN",
                        "confidence": 0.0,
                        "source": "static_head_error",
                        "reject_reason": str(e),
                    }
            else:
                # Xử lý an toàn cho CONFLICT, INVALID_INPUT, hoặc khi không có static head
                final_sign = "NEUTRAL"
                final_conf = 0.0
                recognition_source = "none"
                gating_status = "none"
                threshold_applied = None
                reject_reason = geom_reason or f"Geometric status: {geom_status}"
                alpha_res["source"] = "none"

            # -------------------------------------------------------------
            # Chế độ letters: Nhận diện bảng 29 chữ cái tiếng Việt chuẩn VSL
            # Chưa kích hoạt lại DiacriticStateMachine trong nhiệm vụ này
            # -------------------------------------------------------------
            if session_key in self.diacritic_fsms:
                self.diacritic_fsms[session_key].reset()

            # Chặn tuyệt đối: chế độ letters chỉ cho phép 29 chữ cái tiếng Việt trong ALLOWED_LETTERS (không chứa F, J, W, Z)
            if final_sign not in ALLOWED_LETTERS:
                final_sign = "NEUTRAL"
                final_conf = 0.0

            # Khi nhận diện rõ chữ H, loại bỏ các frame V cũ trong bộ đệm smoothing (do quá trình đưa/xoay 2 ngón tay)
            if final_sign == "H" and final_conf >= self.confidence_threshold:
                if any(s == "V" for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s != "V"],
                        maxlen=self.smoothing_window,
                    )

            # Khi nhận diện rõ chữ K, loại bỏ các frame V hoặc D cũ trong bộ đệm smoothing (do quá trình tạo dáng)
            if final_sign == "K" and final_conf >= self.confidence_threshold:
                if any(s in ("V", "D") for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s not in ("V", "D")],
                        maxlen=self.smoothing_window,
                    )

            # Khi nhận diện rõ chữ I, loại bỏ các frame Y hoặc B cũ trong bộ đệm smoothing (do quá trình khép ngón)
            if final_sign == "I" and final_conf >= self.confidence_threshold:
                if any(s in ("Y", "B") for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s not in ("Y", "B")],
                        maxlen=self.smoothing_window,
                    )

            # Khi nhận diện rõ chữ U, loại bỏ các frame V cũ trong bộ đệm smoothing (do quá trình khép 2 ngón)
            if final_sign == "U" and final_conf >= self.confidence_threshold:
                if any(s == "V" for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s != "V"],
                        maxlen=self.smoothing_window,
                    )

            # Khi nhận diện rõ chữ T, loại bỏ các frame B cũ trong bộ đệm smoothing
            if final_sign == "T" and final_conf >= self.confidence_threshold:
                if any(s in ("B", "O") for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s not in ("B", "O")],
                        maxlen=self.smoothing_window,
                    )

            # Khi nhận diện rõ chữ S, loại bỏ các frame A, E, T cũ trong bộ đệm smoothing
            if final_sign == "S" and final_conf >= self.confidence_threshold:
                if any(s in ("A", "E", "T") for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s not in ("A", "E", "T")],
                        maxlen=self.smoothing_window,
                    )

            # Khi nhận diện rõ chữ P, loại bỏ các frame K hoặc D cũ trong bộ đệm smoothing
            if final_sign == "P" and final_conf >= self.confidence_threshold:
                if any(s in ("K", "D") for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s not in ("K", "D")],
                        maxlen=self.smoothing_window,
                    )

            # Khi nhận diện rõ chữ Q, loại bỏ các frame G cũ trong bộ đệm smoothing
            if final_sign == "Q" and final_conf >= self.confidence_threshold:
                if any(s == "G" for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s != "G"],
                        maxlen=self.smoothing_window,
                    )

            # Khi nhận diện rõ chữ O, loại bỏ các frame A hoặc B cũ trong bộ đệm smoothing
            if final_sign == "O" and final_conf >= self.confidence_threshold:
                if any(s in ("A", "B") for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s not in ("A", "B")],
                        maxlen=self.smoothing_window,
                    )

            # Khi nhận diện rõ chữ R, loại bỏ các frame U hoặc V cũ trong bộ đệm smoothing
            if final_sign == "R" and final_conf >= self.confidence_threshold:
                if any(s in ("U", "V") for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s not in ("U", "V")],
                        maxlen=self.smoothing_window,
                    )

            # Khi nhận diện rõ chữ X, loại bỏ các frame A, S, O cũ trong bộ đệm smoothing
            if final_sign == "X" and final_conf >= self.confidence_threshold:
                if any(s in ("A", "S", "O") for s, _ in self.prediction_history):
                    self.prediction_history = collections.deque(
                        [(s, c) for s, c in self.prediction_history if s not in ("A", "S", "O")],
                        maxlen=self.smoothing_window,
                    )

            self.prediction_history.append((final_sign, final_conf))

            # Smoothing cho chữ cái
            recent_signs = [s for s, c in self.prediction_history if c >= self.confidence_threshold]
            confirmed = False
            new_token_added = None
            most_common = None
            count = 0

            if len(recent_signs) >= self.smoothing_window // 2:
                most_common, count = collections.Counter(recent_signs).most_common(1)[0]
                if count >= (self.smoothing_window // 2) and most_common in ALLOWED_LETTERS:
                    current_time_clock = time.time() if timestamp is None else float(timestamp)
                    if most_common != self.last_confirmed_sign or (current_time_clock - self.last_confirmed_time > self.cooldown_sec):
                        self.last_confirmed_sign = most_common
                        self.last_confirmed_time = current_time_clock
                        confirmed = True
                        sign_info = ALL_SIGNS.get(most_common) or DIACRITIC_SIGNS_META.get(most_common, {})
                        label_text = sign_info.get("label", most_common)
                        self.sentence_tokens.append(label_text)
                        new_token_added = label_text

            sign_meta = ALL_SIGNS.get(final_sign) or DIACRITIC_SIGNS_META.get(final_sign, {})
            default_display = final_sign if final_sign != "NEUTRAL" else "Đang theo dõi cử chỉ..."
            display_label = sign_meta.get("label", default_display)
            sign_type = "alphabet" if final_sign in ALLOWED_LETTERS else "unknown"

            if self.pilot_logging_enabled and self.pilot_logger is not None:
                vote_count_val = count if (len(recent_signs) >= self.smoothing_window // 2 and most_common in ALLOWED_LETTERS) else 0
                most_common_val = most_common if recent_signs else "NEUTRAL"
                curr_clk = current_time_clock if 'current_time_clock' in locals() else (time.time() if timestamp is None else float(timestamp))
                last_clk = self.last_confirmed_time
                cd_rem_val = round(max(0.0, float(self.cooldown_sec - (curr_clk - last_clk))), 4) if last_clk else 0.0

                temporal_snapshot = {
                    "smoothing_window": self.smoothing_window,
                    "vote_threshold": self.smoothing_window // 2,
                    "vote_count": vote_count_val,
                    "most_common_sign": most_common_val,
                    "recent_predictions_10": [
                        {"sign": s, "confidence": round(float(c), 4)}
                        for s, c in list(self.prediction_history)
                    ],
                    "last_confirmed_sign": self.last_confirmed_sign,
                    "last_confirmed_time": round(float(last_clk), 4) if last_clk else None,
                    "cooldown_sec": round(float(self.cooldown_sec), 2),
                    "cooldown_remaining_sec": cd_rem_val,
                    "token_confirmation_event": bool(confirmed),
                    "new_token_emitted": new_token_added,
                    "model_has_weights": bool(getattr(getattr(self, "model", None), "has_weights", False)),
                    "has_static_head": bool(getattr(self, "has_static_head", False)),
                }
                record = self._build_pilot_frame_record(
                    actual_frame_idx=actual_frame_idx,
                    clean_expected_sign=clean_expected_sign,
                    final_sign=final_sign,
                    final_conf=final_conf,
                    prediction_source=recognition_source,
                    confirmed=confirmed,
                    tracking_status="VALID",
                    p_prim=p_prim,
                    session_id=session_id,
                    trial_id=trial_id,
                    signer_id=signer_id,
                    geom_payload=self._build_pilot_geometric_payload(
                        geom_status,
                        geom_letter,
                        reason=geom_reason,
                        matched_candidates=geom_matched_candidates,
                        features=geom_features,
                    ),
                    head_payload=self._build_pilot_static_head_payload(head_info),
                    gating_status=gating_status,
                    threshold_applied=threshold_applied,
                    reject_reason=reject_reason,
                    current_time=current_time,
                    client_capture_ts=client_capture_ts,
                    landmark_extract_ts=landmark_extract_ts,
                    request_send_ts=request_send_ts,
                    backend_receive_ts=backend_receive_ts,
                    video_pts_sec=video_pts_sec,
                    video_frame_idx=video_frame_idx,
                    image_width=image_width,
                    image_height=image_height,
                    handedness=handedness,
                    handedness_score=handedness_score,
                    tracking_confidence=tracking_confidence,
                    phase=phase,
                    temporal_snapshot=temporal_snapshot,
                )
                self.pilot_logger.log_frame(record, filepath=getattr(self, "pilot_filepath", None))
            else:
                self._log_diagnostic(
                    p_prim=p_prim,
                    handedness=handedness,
                    finger_states=None,
                    geom_sign=alpha_res.get("letter", "NEUTRAL"),
                    geom_conf=alpha_res.get("confidence", 0.0),
                    has_mlp=False,
                    nn_sign="NEUTRAL",
                    nn_conf=0.0,
                    final_sign=final_sign,
                    final_conf=final_conf,
                    confirmed=confirmed,
                    expected_sign=clean_expected_sign,
                    session_id=session_id,
                    mode=mode,
                    classifier_details=alpha_res
                )

            return {
                "sign": final_sign,
                "label": display_label,
                "type": sign_type,
                "confidence": round(float(final_conf), 2),
                "confirmed": confirmed,
                "new_token": new_token_added,
                "sentence": " ".join(self.sentence_tokens),
                "description": sign_meta.get("description", ""),
                "source": recognition_source,
                "prediction_source": recognition_source,
                "classifier_details": alpha_res
            }

        # -------------------------------------------------------------
        # XỬ LÝ CHO CÁC CHẾ ĐỘ: letters, letters_numbers (legacy), words, sentence, None
        # -------------------------------------------------------------
        feature_vector, dual_meta = extract_dual_hand_features(p_prim, p_sec)
        finger_states = dual_meta["primary_states"]

        # Kiểm tra hai tay trước
        dual_sign, dual_conf = (None, 0.0)
        is_blocked_vocab = False
        if dual_meta["is_dual_hand"] and p_sec:
            dual_sign, dual_conf = self.classify_dual_hands(dual_meta, p_prim, p_sec)
            if is_letters_filter and dual_sign not in ALLOWED_LETTERS:
                if dual_conf >= 0.85:
                    is_blocked_vocab = True
                dual_sign, dual_conf = (None, 0.0)

        # Kiểm tra hình học đơn tay (cho chữ cái)
        geom_sign, geom_conf = self.classify_geometry(p_prim, finger_states)
        if is_letters_filter and geom_sign not in ALLOWED_LETTERS:
            if geom_conf >= 0.85:
                is_blocked_vocab = True
            geom_sign, geom_conf = ("NEUTRAL", 0.0)

        # Đánh giá bộ phân loại chữ số (CHỈ CHẠY TRONG CHẾ ĐỘ LAI letters_numbers ĐỂ TƯƠNG THÍCH NGƯỢC)
        # Trong chế độ 'letters', tuyệt đối KHÔNG gọi digit_classifier
        digit_sign, digit_conf = ("NEUTRAL", 0.0)
        if is_legacy_mode:
            from backend.vision.digit_classifier import digit_classifier
            digit_res = digit_classifier.classify_detailed(p_prim)
            if digit_res["status"] == "MATCH" and digit_res["digit"] in ALLOWED_DIGITS:
                digit_sign = digit_res["digit"]
                digit_conf = digit_res["confidence"]

        # Phân xử ưu tiên giữa chữ cái và chữ số trong letters_numbers (legacy):
        selected_geom_sign = geom_sign
        selected_geom_conf = geom_conf

        if is_legacy_mode and digit_sign in ALLOWED_DIGITS:
            if digit_sign in {"3", "4", "5", "7", "8", "9"}:
                selected_geom_sign = digit_sign
                selected_geom_conf = digit_conf
            elif digit_sign in {"1", "2", "6"}:
                if expected_sign == digit_sign or geom_sign == "NEUTRAL":
                    selected_geom_sign = digit_sign
                    selected_geom_conf = digit_conf

        # Chạy suy luận qua mạng nơ-ron MLP nếu đã nạp weights hợp lệ
        has_mlp = getattr(self.model, "has_weights", False) and not is_blocked_vocab
        nn_sign, nn_conf = "NEUTRAL", 0.0
        if has_mlp:
            with torch.no_grad():
                tensor_in = torch.from_numpy(feature_vector[:72]).unsqueeze(0)
                logits = self.model(tensor_in).clone()
                if is_letters_filter:
                    # Logit masking: loai bo hoan toan 57 lop tu vung (index >= 27)
                    logits[:, 27:] = -float('inf')
                probs = torch.softmax(logits, dim=-1).squeeze(0).numpy()
                top_idx = int(np.argmax(probs))
                nn_sign = IDX_TO_CLASS[top_idx] if top_idx < len(IDX_TO_CLASS) else "NEUTRAL"
                nn_conf = float(probs[top_idx]) if top_idx < len(probs) else 0.0
                if is_letters_filter and nn_sign not in ALLOWED_LETTERS:
                    nn_sign, nn_conf = "NEUTRAL", 0.0

        # Ensemble logic
        if dual_sign and dual_conf >= 0.85:
            final_sign = dual_sign
            final_conf = dual_conf
        elif selected_geom_conf >= 0.85:
            final_sign = selected_geom_sign
            final_conf = selected_geom_conf
        elif has_mlp and nn_conf > 0.6:
            final_sign = nn_sign
            final_conf = max(nn_conf, selected_geom_conf)
        else:
            final_sign = selected_geom_sign
            final_conf = selected_geom_conf if not has_mlp else max(nn_conf, selected_geom_conf)

        # Chặn tầng cuối cùng tùy theo từng chế độ
        if is_letters_mode:
            # Chế độ letters: CHỈ cho phép 27 chữ cái VSL
            if final_sign not in ALLOWED_LETTERS:
                final_sign = "NEUTRAL"
                final_conf = 0.0
        elif is_legacy_mode:
            # Chế độ letters_numbers: cho phép chữ cái và chữ số
            if final_sign not in ALLOWED_LETTERS_AND_DIGITS:
                final_sign = "NEUTRAL"
                final_conf = 0.0

        if final_sign == "H" and final_conf >= self.confidence_threshold:
            if any(s == "V" for s, _ in self.prediction_history):
                self.prediction_history = collections.deque(
                    [(s, c) for s, c in self.prediction_history if s != "V"],
                    maxlen=self.smoothing_window,
                )

        if final_sign == "S" and final_conf >= self.confidence_threshold:
            if any(s in ("A", "E", "T") for s, _ in self.prediction_history):
                self.prediction_history = collections.deque(
                    [(s, c) for s, c in self.prediction_history if s not in ("A", "E", "T")],
                    maxlen=self.smoothing_window,
                )

        if final_sign == "P" and final_conf >= self.confidence_threshold:
            if any(s in ("K", "D") for s, _ in self.prediction_history):
                self.prediction_history = collections.deque(
                    [(s, c) for s, c in self.prediction_history if s not in ("K", "D")],
                    maxlen=self.smoothing_window,
                )

        if final_sign == "Q" and final_conf >= self.confidence_threshold:
            if any(s == "G" for s, _ in self.prediction_history):
                self.prediction_history = collections.deque(
                    [(s, c) for s, c in self.prediction_history if s != "G"],
                    maxlen=self.smoothing_window,
                )

        if final_sign == "R" and final_conf >= self.confidence_threshold:
            if any(s in ("U", "V") for s, _ in self.prediction_history):
                self.prediction_history = collections.deque(
                    [(s, c) for s, c in self.prediction_history if s not in ("U", "V")],
                    maxlen=self.smoothing_window,
                )

        if final_sign == "X" and final_conf >= self.confidence_threshold:
            if any(s in ("A", "S", "O") for s, _ in self.prediction_history):
                self.prediction_history = collections.deque(
                    [(s, c) for s, c in self.prediction_history if s not in ("A", "S", "O")],
                    maxlen=self.smoothing_window,
                )

        self.prediction_history.append((final_sign, final_conf))

        # Đếm tần suất trong cửa sổ làm mượt
        recent_signs = [s for s, c in self.prediction_history if c >= self.confidence_threshold]
        confirmed = False
        new_token_added = None
        most_common = None
        count = 0

        if len(recent_signs) >= self.smoothing_window // 2:
            most_common, count = collections.Counter(recent_signs).most_common(1)[0]
            if count >= (self.smoothing_window // 2) and most_common != "NEUTRAL":
                valid_smoothing = True
                if is_letters_mode:
                    valid_smoothing = (most_common in ALLOWED_LETTERS)
                elif is_legacy_mode:
                    valid_smoothing = (most_common in ALLOWED_LETTERS_AND_DIGITS)

                if valid_smoothing:
                    current_time = time.time()
                    if most_common != self.last_confirmed_sign or (current_time - self.last_confirmed_time > self.cooldown_sec):
                        self.last_confirmed_sign = most_common
                        self.last_confirmed_time = current_time
                        confirmed = True
                        
                        sign_info = ALL_SIGNS.get(most_common, {})
                        default_lbl = f"Số {most_common}" if most_common in ALLOWED_DIGITS else most_common
                        label_text = sign_info.get("label", default_lbl)
                        self.sentence_tokens.append(label_text)
                        new_token_added = label_text

        sign_meta = ALL_SIGNS.get(final_sign, {})
        default_display = f"Số {final_sign}" if final_sign in ALLOWED_DIGITS else (final_sign if final_sign != "NEUTRAL" else "Đang theo dõi cử chỉ...")
        display_label = sign_meta.get("label", default_display)
        sign_type = "digit" if final_sign in ALLOWED_DIGITS else sign_meta.get("type", "unknown")

        # Diagnostic logging (chỉ kích hoạt khi VSL_PILOT_LOGGING=1 hoặc rec.pilot_logging_enabled=True, hoặc legacy VSL_DIAGNOSTIC=1)
        if self.pilot_logging_enabled and self.pilot_logger is not None:
            vote_count_val = count if (len(recent_signs) >= self.smoothing_window // 2) else 0
            most_common_val = most_common if recent_signs else "NEUTRAL"
            curr_clk = current_time if current_time is not None else time.time()
            last_clk = self.last_confirmed_time
            cd_rem_val = round(max(0.0, float(self.cooldown_sec - (curr_clk - last_clk))), 4) if last_clk else 0.0

            temporal_snapshot = {
                "smoothing_window": self.smoothing_window,
                "vote_threshold": self.smoothing_window // 2,
                "vote_count": vote_count_val,
                "most_common_sign": most_common_val,
                "recent_predictions_10": [
                    {"sign": s, "confidence": round(float(c), 4)}
                    for s, c in list(self.prediction_history)
                ],
                "last_confirmed_sign": self.last_confirmed_sign,
                "last_confirmed_time": round(float(last_clk), 4) if last_clk else None,
                "cooldown_sec": round(float(self.cooldown_sec), 2),
                "cooldown_remaining_sec": cd_rem_val,
                "token_confirmation_event": bool(confirmed),
                "new_token_emitted": new_token_added,
                "model_has_weights": bool(getattr(getattr(self, "model", None), "has_weights", False)),
                "has_static_head": bool(getattr(self, "has_static_head", False)),
            }
            record = self._build_pilot_frame_record(
                actual_frame_idx=actual_frame_idx,
                clean_expected_sign=clean_expected_sign,
                final_sign=final_sign,
                final_conf=final_conf,
                prediction_source="geometric_rule" if geom_sign != "NEUTRAL" else "none",
                confirmed=confirmed,
                tracking_status="VALID",
                p_prim=p_prim,
                session_id=session_id,
                trial_id=trial_id,
                signer_id=signer_id,
                geom_payload=self._build_pilot_geometric_payload(
                    "MATCH" if geom_sign != "NEUTRAL" else "NO_MATCH",
                    geom_sign if geom_sign != "NEUTRAL" else None,
                ),
                head_payload=self._build_pilot_static_head_payload(),
                gating_status="none",
                threshold_applied=None,
                reject_reason=None,
                current_time=current_time,
                client_capture_ts=client_capture_ts,
                landmark_extract_ts=landmark_extract_ts,
                request_send_ts=request_send_ts,
                backend_receive_ts=backend_receive_ts,
                video_pts_sec=video_pts_sec,
                video_frame_idx=video_frame_idx,
                image_width=image_width,
                image_height=image_height,
                handedness=handedness,
                handedness_score=handedness_score,
                tracking_confidence=tracking_confidence,
                phase=phase,
                temporal_snapshot=temporal_snapshot,
            )
            self.pilot_logger.log_frame(record, filepath=getattr(self, "pilot_filepath", None))
        else:
            self._log_diagnostic(
                p_prim=p_prim,
                handedness=handedness,
                finger_states=finger_states,
                geom_sign=geom_sign,
                geom_conf=geom_conf,
                has_mlp=has_mlp,
                nn_sign=nn_sign,
                nn_conf=nn_conf,
                final_sign=final_sign,
                final_conf=final_conf,
                confirmed=confirmed,
                expected_sign=clean_expected_sign,
                session_id=session_id,
                mode=mode
            )

        return {
            "sign": final_sign,
            "label": display_label,
            "type": sign_type,
            "confidence": round(float(final_conf), 2),
            "confirmed": confirmed,
            "new_token": new_token_added,
            "sentence": " ".join(self.sentence_tokens),
            "description": sign_meta.get("description", f"Chữ số {final_sign}" if final_sign in ALLOWED_DIGITS else "")
        }


    def _log_diagnostic(self, p_prim, handedness, finger_states, geom_sign, geom_conf,
                        has_mlp, nn_sign, nn_conf, final_sign, final_conf, confirmed,
                        expected_sign=None, session_id=None, mode=None,
                        classifier_details=None):
        """Ghi nhat ky chan doan JSONL khi duoc bat, gioi han ~2 mau/giay, an toan tuyet doi voi I/O."""
        is_enabled = os.environ.get("VSL_DIAGNOSTIC", "0") == "1" or getattr(self, "diagnostic_enabled", False)
        if not is_enabled:
            return

        now = time.time()
        with self._diag_lock:
            if now - self._last_diag_log_time < 0.45:
                return
            self._last_diag_log_time = now
            self._diag_frame_counter += 1
            frame_idx = self._diag_frame_counter

        try:
            target_dir = getattr(self, "diagnostic_dir", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "diagnostics"))
            os.makedirs(target_dir, exist_ok=True)
            log_date = time.strftime("%Y%m%d")
            filepath = getattr(self, "diagnostic_filepath", None) or os.path.join(target_dir, f"diagnostic_{log_date}.jsonl")

            from backend.vision.feature_extractor import _shift_and_scale
            _, palm_scale = _shift_and_scale(p_prim)

            history_snapshot = [s for s, c in self.prediction_history]

            safe_finger_states = {}
            if isinstance(finger_states, dict):
                safe_finger_states = {
                    k: (bool(v) if "extended" in k else round(float(v), 4))
                    for k, v in finger_states.items()
                }
            elif classifier_details and isinstance(classifier_details.get("features"), dict):
                feat = classifier_details["features"]
                safe_finger_states = {
                    "thumb_extended": bool(feat.get("thumb_ext", False)),
                    "index_extended": bool(feat.get("index_ext", False)),
                    "middle_extended": bool(feat.get("middle_ext", False)),
                    "ring_extended": bool(feat.get("ring_ext", False)),
                    "pinky_extended": bool(feat.get("pinky_ext", False)),
                    "thumb_curl": 0.0,
                    "index_curl": round(float(feat.get("index_curl", 0.0)), 4),
                    "middle_curl": 0.0,
                    "ring_curl": 0.0,
                    "pinky_curl": 0.0
                }
            else:
                safe_finger_states = {
                    "thumb_extended": False,
                    "index_extended": False,
                    "middle_extended": False,
                    "ring_extended": False,
                    "pinky_extended": False,
                    "thumb_curl": 0.0,
                    "index_curl": 0.0,
                    "middle_curl": 0.0,
                    "ring_curl": 0.0,
                    "pinky_curl": 0.0
                }

            valid_count = len([s for s, c in self.prediction_history if c >= self.confidence_threshold])

            record = {
                "timestamp": round(now, 3),
                "session_id": str(session_id) if session_id else "default",
                "frame_idx": frame_idx,
                "mode": str(mode) if mode else "default",
                "expected_sign": str(expected_sign) if expected_sign else None,
                "handedness": handedness or "Right",
                "raw_landmarks": [[round(float(coord), 6) for coord in pt] for pt in p_prim],
                "palm_scale": round(float(palm_scale), 6),
                "finger_states": safe_finger_states,
                "geometry": {
                    "sign": geom_sign,
                    "confidence": round(float(geom_conf), 4)
                },
                "mlp": {
                    "has_weights": bool(has_mlp),
                    "sign": nn_sign,
                    "confidence": round(float(nn_conf), 4)
                },
                "ensemble": {
                    "sign": final_sign,
                    "confidence": round(float(final_conf), 4)
                },
                "smoothing": {
                    "confirmed": bool(confirmed),
                    "recent_history": history_snapshot,
                    "valid_frames_count": valid_count
                }
            }
            if classifier_details and isinstance(classifier_details, dict):
                record["classifier_details"] = {
                    "status": classifier_details.get("status"),
                    "digit": classifier_details.get("digit"),
                    "confidence": round(float(classifier_details.get("confidence", 0.0)), 4),
                    "reason": classifier_details.get("reason"),
                    "matched_digits": classifier_details.get("matched_digits", []),
                    "features": classifier_details.get("features")
                }
            with self._diag_lock:
                with open(filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            pass

    def clear_sentence(self):
        """Xóa trắng câu hiện tại"""
        self.sentence_tokens = []
        self.last_confirmed_sign = None
        if hasattr(self, "digit_10_detector"):
            self.digit_10_detector.reset()
        return {"sentence": ""}

    def reset(self):
        """Reset toàn bộ trạng thái bộ nhận diện"""
        self.prediction_history.clear()
        self.last_confirmed_sign = None
        self.last_confirmed_time = 0
        self.sentence_tokens = []
        if hasattr(self, "digit_10_detector"):
            self.digit_10_detector.reset()
        self.reset_diacritic_fsm()

    def reset_diacritic_fsm(self, session_id=None):
        """Reset máy trạng thái dấu cho một session cụ thể hoặc toàn bộ (Task 7D)."""
        if session_id is None:
            for fsm in self.diacritic_fsms.values():
                fsm.reset()
        else:
            session_key = str(session_id)
            if session_key in self.diacritic_fsms:
                self.diacritic_fsms[session_key].reset()
