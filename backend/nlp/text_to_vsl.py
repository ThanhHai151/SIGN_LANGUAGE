"""
Bộ chuyển đổi Ngôn ngữ Tự nhiên Tiếng Việt sang Ngôn ngữ Ký hiệu (Vietnamese Text-to-VSL NLP Engine)
Tra cứu trên BỘ TỪ VỰNG THỐNG NHẤT (backend/data/vsl_lexicon.json: ~3.300 từ QIPEDC + 102 từ giáo trình
+ từ có tư thế robot), cùng nguồn với mô hình camera — hai chiều dùng chung một bộ từ.
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
from backend.models.lexicon import load_lexicon, gloss_of, display_label, norm_key, motion_id_of, motion_ids, motion_clip, sign_description

# Các hư từ trong ngữ pháp tiếng Việt nói thường được lược bỏ hoặc tinh gọn trong VSL
STOPWORDS = {
    "là", "thì", "mà", "những", "các", "của", "đang", "đã", "sẽ", "một", "rất",
    "vì", "chúng", "vào", "trong", "cho", "việc", "hoặc", "nhưng", "cũng", "để",
    "mình", "điều", "đó", "khỏi", "có", "thường", "bởi", "tại", "do", "với", "được"
}

# Hư từ / trợ từ bổ sung: không có ký hiệu riêng trong VSL, bỏ qua thay vì đánh vần
STOPWORDS |= {
    "và", "rồi", "với", "này", "ấy", "nhé", "nha", "ạ", "à", "ơi", "nhỉ", "chứ", "vậy",
    "lại", "nữa", "đều", "sự", "chiếc", "quá", "lắm", "thật", "luôn", "hãy", "từ", "khi",
    "nên", "nhưng", "thế", "đây", "kia", "hết", "cùng", "bị", "cái",
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

# Từ đồng nghĩa / cách viết khác → từ có trong bộ từ vựng
SYNONYM_WORDS = {
    "cám ơn": "cảm ơn", "xin chào": "chào",
    "má": "mẹ", "mẹ": "mẹ", "ba": "bố", "cha": "bố",
    "làm ơn": "xin vui lòng", "vui lòng": "xin vui lòng",
    "học": "học tập", "học bài": "học tập", "nghỉ ngơi": "thư giãn", "bạn thân": "bạn bè",
    "thường ngày": "hàng ngày", "nấu ăn": "nấu nướng", "dọn dẹp": "lau nhà",
}


def remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt để dùng cho đánh vần chữ cái ngón tay cơ bản"""
    s = input_str.replace('Đ', 'D').replace('đ', 'd')
    nfkd_form = unicodedata.normalize('NFKD', s)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


def _resolve(phrase: str, lex: dict):
    """Tìm mục từ cho một cụm từ: khớp trực tiếp → đồng nghĩa → bản đồ ngữ nghĩa cũ (theo tư thế robot)."""
    entry = lex["by_key"].get(phrase)
    if entry:
        return entry
    syn = SYNONYM_WORDS.get(phrase)
    if syn and syn in lex["by_key"]:
        return lex["by_key"][syn]
    pose_key = SEMANTIC_SYNONYM_MAP.get(phrase)
    if pose_key:
        return lex["by_pose"].get(pose_key)
    return None


_QIPEDC_VIDEO_RE = re.compile(r"qipedc\.moet\.gov\.vn/videos/([A-Za-z0-9_-]+)\.mp4")


def _video_id(entry: dict):
    m = _QIPEDC_VIDEO_RE.search(entry.get("video_url") or "")
    return m.group(1) if m else None


def _vocab_item(token: str, entry: dict, available_motions: set = None) -> dict:
    gloss = gloss_of(entry)
    motion_id = motion_id_of(entry, available_motions)
    legacy = VSL_VOCABULARY.get(entry.get("pose_key") or "", {})
    return {
        "token": token,
        "gloss": gloss,
        "label": display_label(entry, token),
        "word": entry["word"],
        "entry_id": entry["id"],
        "type": "vocabulary",
        "duration_ms": 1400,
        "description": entry.get("how_to") or entry.get("definition") or legacy.get("description", ""),
        "definition": entry.get("definition", ""),
        "how_to": entry.get("how_to", ""),
        "pos": entry.get("pos", ""),
        "source": entry.get("source", ""),
        # Chỉ 1 video minh họa: đúng video QIPEDC mà robot đã học chuyển động (không chia vùng miền).
        # Video giáo trình lưu trên Supabase hay bị chặn/treo → không dùng.
        "video_url": entry.get("video_url") if "qipedc.moet.gov.vn" in (entry.get("video_url") or "") else None,
        "thumb_url": entry.get("thumb_url"),
        "variants": [],
        "pose_key": entry.get("pose_key"),
        # Robot diễn được nếu có chuyển động học từ video (motion_id) hoặc tư thế vẽ tay (pose_key)
        "motion_id": motion_id,
        # Đoạn có động tác trong video mẫu (bỏ phần đứng yên đầu/cuối) — video & robot chỉ phát đoạn này
        "clip": list(motion_clip(motion_id)) if motion_id else None,
        "has_pose": bool(entry.get("pose_key") or motion_id),
        # Mã video QIPEDC: từ chưa có chuyển động → trình duyệt học từ video này rồi robot diễn
        "video_id": _video_id(entry),
        # Chữ cái để robot đánh vần tạm trong lúc chưa có / đang học chuyển động
        "spell": [c for c in remove_accents(entry["word"]).upper() if c in VSL_ALPHABET][:12],
        # Diễn tả cụ thể cách làm ký hiệu (phân tích từ video người thật)
        "sign_description": sign_description(motion_id),
        "sstcn_index": entry.get("sstcn_index"),
        "motion_type": legacy.get("motion", "static"),
        "avatar_pose": legacy.get("avatar_pose", {}),
        "fingers_state": legacy.get("fingers_state", {}),
    }


def _spell_items(word: str, lex: dict) -> list:
    items = []
    for char in remove_accents(word).upper():
        if char not in VSL_ALPHABET:
            continue
        alpha_item = VSL_ALPHABET[char]
        entry = lex["by_key"].get(char.lower())
        items.append({
            "token": char,
            "gloss": f"ALPHA_{char}",
            "label": f"Chữ {char}",
            "word": char,
            "entry_id": entry["id"] if entry else None,
            "type": "fingerspelling",
            "duration_ms": 800,
            "description": alpha_item["description"],
            "video_url": None,  # đánh vần: robot diễn tư thế chữ cái, không cần video
            "variants": [],
            "pose_key": char,
            "has_pose": True,
            "motion_type": "static",
            "fingers_state": alpha_item.get("fingers_state", {}),
            "avatar_pose": {"right_arm": [35, 20, 0], "right_wrist": [15, 0, 0]},
        })
    return items


def text_to_vsl_glosses(text: str):
    """
    Chuyển câu văn / đoạn văn tiếng Việt thành chuỗi ký hiệu VSL liên tục:
    1. Ghép cụm từ dài nhất có trong bộ từ vựng (tối đa 6 âm tiết).
    2. Bỏ hư từ không dùng trong VSL (là, thì, mà, của...).
    3. Từ không có trong từ điển (tên riêng, từ mới) → đánh vần chữ cái ngón tay.
    """
    if not text or not text.strip():
        return []

    lex = load_lexicon()
    available = motion_ids()
    cleaned = norm_key(text)
    cleaned = re.sub(r'[\r\n\t]+', ' ', cleaned)
    cleaned = re.sub(r'[—_()"\[\]]+', ' ', cleaned)
    cleaned = re.sub(r'\s-\s|\s-|-\s', ' ', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()

    words = cleaned.split()
    result = []
    i = 0
    n = len(words)
    max_len = lex["max_phrase_len"]

    def push(item):
        if not result or result[-1]["gloss"] != item["gloss"] or item["type"] == "fingerspelling":
            result.append(item)

    while i < n:
        matched = False
        # Cụm nhiều âm tiết trước (kể cả cụm chứa hư từ như "có thể", "của tôi")
        for length in range(min(max_len, n - i), 1, -1):
            phrase = " ".join(words[i:i + length])
            entry = _resolve(phrase, lex)
            if entry:
                push(_vocab_item(phrase, entry, available))
                i += length
                matched = True
                break
        if matched:
            continue

        word = words[i]
        if word in STOPWORDS:
            i += 1
            continue
        entry = _resolve(word, lex)
        if entry:
            push(_vocab_item(word, entry, available))
        elif re.fullmatch(r"\d+", word):
            for d in word:
                e = lex["by_key"].get(d)
                if e:
                    push(_vocab_item(d, e, available))
        else:
            result.extend(_spell_items(word, lex))
        i += 1

    return result
