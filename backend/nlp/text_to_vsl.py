"""
Bộ chuyển đổi Ngôn ngữ Tự nhiên Tiếng Việt sang Ngôn ngữ Ký hiệu (Vietnamese Text-to-VSL NLP Engine)
Quy trình nâng cao:
1. Chuẩn hóa câu & phân đoạn đoạn văn dài thành các mệnh đề ngữ nghĩa.
2. Lược bỏ hư từ tiếng Việt không dùng trong VSL (theo chuẩn ngữ pháp VSL: Time - Topic - Comment).
3. Bản đồ ánh xạ ngữ nghĩa đa tầng (Multi-layer Semantic Synonym Mapping & Keyword Matching).
4. Tối ưu thời lượng nhịp chuyển động liên tục (Continuous Gesture Timing & Co-articulation).
5. Chỉ kích hoạt Fingerspelling cho từ viết tắt hoặc tên riêng thực sự cần thiết.
"""

import re
import unicodedata
from backend.models.vsl_dictionary import VSL_VOCABULARY, VSL_ALPHABET

# Các hư từ trong ngữ pháp tiếng Việt nói thường được lược bỏ hoặc tinh gọn trong VSL
STOPWORDS = {
    "là", "thì", "mà", "những", "các", "của", "đang", "đã", "sẽ", "một", "rất",
    "vì", "chúng", "vào", "trong", "cho", "việc", "hoặc", "nhưng", "cũng", "để",
    "mình", "điều", "đó", "khỏi", "có", "thường", "bởi", "tại", "do", "với", "được"
}

# Bản đồ ánh xạ từ đồng nghĩa và cụm từ phức tạp trong đời sống sang VSL Gloss
SEMANTIC_SYNONYM_MAP = {
    "những ngày cuối tuần": "CUOI_TUAN",
    "ngày cuối tuần": "CUOI_TUAN",
    "cuối tuần": "CUOI_TUAN",
    "lịch trình cụ thể": "LICH_TRINH",
    "lịch trình": "LICH_TRINH",
    "kế hoạch": "LICH_TRINH",
    "thời gian biểu": "LICH_TRINH",
    "phụ thuộc vào": "PHU_THUOC",
    "phụ thuộc": "PHU_THUOC",
    "tùy thuộc": "PHU_THUOC",
    "mức độ bận rộn": "BAN_RON",
    "bận rộn": "BAN_RON",
    "bận": "BAN_RON",
    "trong tuần": "TRONG_TUAN",
    "dành một ít thời gian": "THOI_GIAN",
    "dành thời gian": "THOI_GIAN",
    "thời gian": "THOI_GIAN",
    "một ít": "IT",
    "chút ít": "IT",
    "ít": "IT",
    "học tập": "HOC_BAI",
    "học bài": "HOC_BAI",
    "học": "HOC_BAI",
    "dự án cá nhân": "DU_AN",
    "dự án": "DU_AN",
    "cá nhân": "CA_NHAN",
    "đảm bảo": "DAM_BAO",
    "chắc chắn": "DAM_BAO",
    "đủ thời gian": "THOI_GIAN",
    "có đủ": "DU",
    "đủ": "DU",
    "thư giãn": "THU_GIAN",
    "nghỉ ngơi": "THU_GIAN",
    "nếu có thể": "CO_THE",
    "nếu": "NEU",
    "có thể": "CO_THE",
    "thích": "THICH",
    "yêu thích": "THICH",
    "gặp gỡ": "GAP_GO",
    "gặp bạn": "GAP_GO",
    "gặp": "GAP_GO",
    "bạn bè": "BAN_BE",
    "bạn thân": "BAN_BE",
    "bạn": "BAN",
    "khám phá": "KHAM_PHA",
    "tìm hiểu": "KHAM_PHA",
    "trải nghiệm": "KHAM_PHA",
    "địa điểm mới": "DIA_DIEM",
    "địa điểm": "DIA_DIEM",
    "nơi chốn": "DIA_DIEM",
    "mới": "MOI",
    "mới lạ": "MOI",
    "cơ hội": "CO_HOI",
    "thói quen thường ngày": "THOI_QUEN",
    "thói quen": "THOI_QUEN",
    "thường ngày": "HANG_NGAY",
    "hàng ngày": "HANG_NGAY",
    "nấu ăn": "NAU_AN",
    "nấu nướng": "NAU_AN",
    "lau nhà": "LAU_NHA",
    "dọn dẹp": "LAU_NHA",
    "đi ngủ": "DI_NGU",
    "ngủ": "DI_NGU",
    "rửa tay": "RUA_TAY",
    "đọc sách": "DOC_SACH",
    "viết bài": "VIET_BAI",
    "làm việc": "LAM_VIEC",
    "khỏe mạnh": "KHOE_MANH",
    "mệt mỏi": "MET_MOI",
    "tôi": "TOI",
    "của tôi": "TOI",
    "không có": "KHONG",
    "không": "KHONG"
}

def remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt để dùng cho đánh vần chữ cái ngón tay cơ bản"""
    s = input_str.replace('Đ', 'D').replace('đ', 'd')
    nfkd_form = unicodedata.normalize('NFKD', s)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])

def text_to_vsl_glosses(text: str):
    """
    Chuyển đổi câu văn hoặc đoạn văn dài tiếng Việt thành chuỗi ký hiệu VSL liên tục,
    tối ưu cho ngữ pháp VSL và loại bỏ tình trạng đánh vần từng chữ cái vô nghĩa.
    """
    if not text or not text.strip():
        return []

    cleaned = text.lower().strip()
    cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
    cleaned = re.sub(r'[,;.?!—\-_()"]+', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    result_sequence = []
    
    # 1. Chuẩn bị danh sách tra cứu ưu tiên từ dài đến ngắn
    vocab_entries = []
    for phrase, gloss_key in SEMANTIC_SYNONYM_MAP.items():
        if gloss_key in VSL_VOCABULARY:
            vocab_entries.append((phrase, gloss_key, VSL_VOCABULARY[gloss_key]))
            
    for gloss_key, item in VSL_VOCABULARY.items():
        for kw in item.get("keywords", []):
            vocab_entries.append((kw, gloss_key, item))

    vocab_entries.sort(key=lambda x: len(x[0].split()), reverse=True)

    words = cleaned.split()
    i = 0
    total_words = len(words)

    while i < total_words:
        matched = False

        for length in range(min(5, total_words - i), 0, -1):
            candidate = " ".join(words[i:i + length])
            
            for phrase, gloss_key, item in vocab_entries:
                if candidate == phrase:
                    if not result_sequence or result_sequence[-1]["gloss"] != gloss_key:
                        result_sequence.append({
                            "token": candidate,
                            "gloss": gloss_key,
                            "label": item["label"],
                            "type": "vocabulary",
                            "duration_ms": 1400,
                            "description": item["description"],
                            "motion": item.get("motion", "static"),
                            "avatar_pose": item.get("avatar_pose", {}),
                            "fingers_state": item.get("fingers_state", {})
                        })
                    i += length
                    matched = True
                    break
            if matched:
                break

        if not matched:
            word = words[i]
            if word in STOPWORDS:
                i += 1
                continue

            single_matched = False
            for phrase, gloss_key, item in vocab_entries:
                if word == phrase:
                    if not result_sequence or result_sequence[-1]["gloss"] != gloss_key:
                        result_sequence.append({
                            "token": word,
                            "gloss": gloss_key,
                            "label": item["label"],
                            "type": "vocabulary",
                            "duration_ms": 1400,
                            "description": item["description"],
                            "motion": item.get("motion", "static"),
                            "avatar_pose": item.get("avatar_pose", {}),
                            "fingers_state": item.get("fingers_state", {})
                        })
                    single_matched = True
                    break

            if not single_matched:
                unaccented_word = remove_accents(word).upper()
                if len(unaccented_word) <= 4 and all(c in VSL_ALPHABET for c in unaccented_word):
                    for char in unaccented_word:
                        alpha_item = VSL_ALPHABET[char]
                        result_sequence.append({
                            "token": char,
                            "gloss": f"ALPHA_{char}",
                            "label": f"Chữ {char}",
                            "type": "fingerspelling",
                            "duration_ms": 800,
                            "description": alpha_item["description"],
                            "motion": "static",
                            "fingers_state": alpha_item.get("fingers_state", {}),
                            "avatar_pose": {
                                "right_arm": [35, 20, 0],
                                "right_wrist": [15, 0, 0]
                            }
                        })
            i += 1

    return result_sequence
