"""
Động cơ Nhận diện Cử chỉ VSL Độ chính xác cao (High-Precision Hybrid VSL Recognizer Engine)
Kết hợp 2 chế độ:
1. Mô hình Deep Learning Spatio-Temporal: SSTCN_Attention (30 frames x 75 joints) nhận diện 102 từ vựng VSL.
2. Bộ kiểm tra ràng buộc hình học ngón tay (Geometric Fingertip & Hand Angle Constraints) cho bảng chữ cái tức thì (<5ms).
Hỗ trợ chống rung (temporal smoothing), chống lặp (cooldown debounce) và ghép câu tự động.
"""

import os
import json
import time
import threading
import collections
import numpy as np
import torch

from backend.models.sstcn_model import load_sstcn_model
from backend.models.vsl_102_dictionary import SIGN_DICT_102, SIGN_METADATA_102, get_sign_info
from backend.models.sign_model import get_trained_model, CLASSES, CLASS_TO_IDX, IDX_TO_CLASS
from backend.models.vsl_dictionary import ALL_SIGNS
from backend.models.lexicon import get_entry_by_sstcn
from backend.vision.feature_extractor import normalize_hand_landmarks, calculate_distance, calculate_angle
from backend.vision.alphabet_classifier import alphabet_classifier

# Danh sach 27 chu cai VSL duoc phep trong che do letters_numbers
ALLOWED_LETTERS = set(CLASSES[:27])
ALLOWED_DIGITS = {str(i) for i in range(1, 10)}
DIGITS_MODE_SIGNS = {str(i) for i in range(0, 10)}
ALLOWED_LETTERS_AND_DIGITS = ALLOWED_LETTERS | ALLOWED_DIGITS

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

class VSLRecognizer:
    def __init__(self, confidence_threshold=0.70, smoothing_window=10, cooldown_sec=1.2, weights_path=None):
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
        
        # 3. Bộ đệm thời gian chống rung cho chế độ tĩnh
        self.prediction_history = collections.deque(maxlen=smoothing_window)
        
        # 4. Quản lý câu và từ
        self.last_confirmed_sign = None
        self.last_confirmed_time = 0
        self.sentence_tokens = []
        
        # 5. Diagnostic Logging State
        self.diagnostic_enabled = os.environ.get("VSL_DIAGNOSTIC", "0") == "1"
        self.diagnostic_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "diagnostics")
        self._last_diag_log_time = 0.0
        self._diag_lock = threading.Lock()
        self._diag_frame_counter = 0

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

    def predict_landmarks(self, landmarks_list, landmarks_secondary=None, multi_hands=None, expected_sign=None, session_id=None, mode=None, timestamp=None):
        """
        Nhận diện cử chỉ đơn frame hỗ trợ cả 1 tay và 2 tay.
        Khi mode == 'letters_numbers', backend chi tra ve 27 chu cai VSL hoac NEUTRAL.
        """
        from backend.vision.feature_extractor import extract_dual_hand_features

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

        p_prim = None
        p_sec = None
        handedness = "Right"

        if multi_hands and len(multi_hands) > 0:
            p_prim = multi_hands[0].get("landmarks", [])
            handedness = multi_hands[0].get("handedness", "Right")
            if len(multi_hands) > 1:
                p_sec = multi_hands[1].get("landmarks", [])
        elif landmarks_list and len(landmarks_list) >= 21:
            p_prim = landmarks_list
            if landmarks_secondary and len(landmarks_secondary) >= 21:
                p_sec = landmarks_secondary

        if not p_prim or len(p_prim) < 21:
            self.prediction_history.append(("NEUTRAL", 0.0))
            if is_digits_mode and hasattr(self, "digit_10_detector"):
                self.digit_10_detector.reset()
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
                return {
                    "sign": "NEUTRAL",
                    "label": "Chờ cử chỉ...",
                    "type": "unknown",
                    "confidence": 0.0,
                    "confirmed": False,
                    "new_token": None,
                    "sentence": " ".join(self.sentence_tokens)
                }
        except Exception:
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

            # Chỉ chấp nhận khi MATCH và chữ cái thuộc tập hợp cho phép
            if status == "MATCH" and alpha_res.get("letter") in ALLOWED_LETTERS:
                final_sign = alpha_res["letter"]
                final_conf = alpha_res["confidence"]
            else:
                # Xử lý an toàn cho NO_MATCH, CONFLICT, INVALID_INPUT:
                # Bảo lưu NEUTRAL, không để mạng nơ-ron MLP tự ý ép nhãn thiếu căn cứ
                final_sign = "NEUTRAL"
                final_conf = 0.0

            # Chặn tuyệt đối: chế độ letters không bao giờ trả về chữ số hoặc từ vựng
            if final_sign not in ALLOWED_LETTERS:
                final_sign = "NEUTRAL"
                final_conf = 0.0

            self.prediction_history.append((final_sign, final_conf))

            # Smoothing cho chữ cái
            recent_signs = [s for s, c in self.prediction_history if c >= self.confidence_threshold]
            confirmed = False
            new_token_added = None

            if len(recent_signs) >= self.smoothing_window // 2:
                most_common, count = collections.Counter(recent_signs).most_common(1)[0]
                if count >= (self.smoothing_window // 2) and most_common in ALLOWED_LETTERS:
                    current_time = time.time()
                    if most_common != self.last_confirmed_sign or (current_time - self.last_confirmed_time > self.cooldown_sec):
                        self.last_confirmed_sign = most_common
                        self.last_confirmed_time = current_time
                        confirmed = True
                        sign_info = ALL_SIGNS.get(most_common, {})
                        label_text = sign_info.get("label", most_common)
                        self.sentence_tokens.append(label_text)
                        new_token_added = label_text

            sign_meta = ALL_SIGNS.get(final_sign, {})
            default_display = final_sign if final_sign != "NEUTRAL" else "Đang theo dõi cử chỉ..."
            display_label = sign_meta.get("label", default_display)
            sign_type = "alphabet" if final_sign in ALLOWED_LETTERS else "unknown"

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
                expected_sign=expected_sign,
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
                "description": sign_meta.get("description", "")
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

        self.prediction_history.append((final_sign, final_conf))

        # Đếm tần suất trong cửa sổ làm mượt
        recent_signs = [s for s, c in self.prediction_history if c >= self.confidence_threshold]
        confirmed = False
        new_token_added = None

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

        # Diagnostic logging (chỉ kích hoạt khi VSL_DIAGNOSTIC=1 hoặc rec.diagnostic_enabled=True)
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
            expected_sign=expected_sign,
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
