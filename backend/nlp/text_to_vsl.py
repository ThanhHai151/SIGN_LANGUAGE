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

import json
import os
import re
import unicodedata
from backend.models.vsl_dictionary import VSL_VOCABULARY, VSL_ALPHABET
from backend.models.lexicon import blocked_motion_ids, robot_pose_key, region_label, load_lexicon, gloss_of, display_label, norm_key, motion_id_of, motion_ids, motion_clip, sign_description

# Các hư từ trong ngữ pháp tiếng Việt nói thường được lược bỏ hoặc tinh gọn trong VSL
STOPWORDS = {
    "là", "thì", "mà", "những", "các", "của", "đang", "đã", "sẽ", "rất",
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

# NGUYÊN TẮC CHÍNH XÁC: một từ chỉ được diễn bằng ký hiệu khi bộ từ vựng có đúng mục từ đó
# (trùng từ gốc của mục QIPEDC). Không dùng ký hiệu của từ "gần nghĩa": thà đánh vần còn hơn diễn sai.
# Bảng dưới đây chỉ gồm các cách viết / cách gọi KHÁC CỦA CÙNG MỘT KHÁI NIỆM, chỉ dùng khi từ gõ vào
# không có mục riêng. Mỗi dòng cần người biết ngôn ngữ ký hiệu rà lại trước khi bổ sung thêm.
SAME_SIGN_WORDS = {
    # cách viết khác
    "ký hiệu": "kí hiệu", "ngôn ngữ ký hiệu": "ngôn ngữ kí hiệu", "cám ơn": "cảm ơn",
    # cách gọi khác của cùng một người / vật
    "ba mẹ": "bố mẹ", "ba má": "bố mẹ",
    "trường": "nhà trường", "trường học": "nhà trường", "học sinh": "học trò",
    "giáo viên": "thầy cô", "thầy": "thầy giáo",
    "vở": "quyển vở", "tranh": "bức tranh", "thư viện": "phòng thư viện",
    "tối": "buổi tối", "trời": "bầu trời", "nghề": "ngành nghề", "tết": "tết âm",
    "anh trai": "anh ruột", "nhà": "nhà cửa",
    # "ốm" đứng riêng là "gầy" theo cách nói miền Nam (mục QIPEDC "gầy (miền Nam: ốm)"); "bị ốm/bệnh" là đau ốm
    "bị ốm": "ốm đau", "bị bệnh": "ốm đau", "bệnh": "ốm đau",
    # dạng rút gọn / dạng láy của cùng một từ
    "xin chào": "chào", "học": "học tập", "làm": "làm việc", "gặp": "gặp gỡ", "giúp": "giúp đỡ",
    "nấu": "nấu nướng", "nấu ăn": "nấu nướng", "đủ": "đầy đủ",
    "chậm": "chậm chạp", "vui": "vui mừng", "ngon": "ngon miệng", "sống": "sinh sống",
    "yêu thương": "thương yêu", "đáng yêu": "dễ thương", "nghỉ ngơi": "thư giãn",
    "thường ngày": "hàng ngày", "gì": "là gì",
    # số đếm viết bằng chữ
    "một": "1", "hai": "2", "ba": "3", "bốn": "4", "sáu": "6", "bảy": "7", "tám": "8", "chín": "9", "mười": "10",
}


def remove_accents(input_str: str) -> str:
    """Loại bỏ dấu tiếng Việt để dùng cho đánh vần chữ cái ngón tay cơ bản"""
    s = input_str.replace('Đ', 'D').replace('đ', 'd')
    nfkd_form = unicodedata.normalize('NFKD', s)
    return "".join([c for c in nfkd_form if not unicodedata.combining(c)])


# Từ thường trùng tên riêng trong QIPEDC: "anh" (đại từ / anh trai) ≠ "Anh (nước Anh)".
# Gõ đứng một mình thì không dùng ký hiệu tên nước; "nước anh" vẫn khớp cụm đầy đủ.
AMBIGUOUS_PROPER_KEYS = {"anh"}


def _usable(entry: dict, available: set = None) -> bool:
    """Robot diễn được: có chuyển động đã học, tư thế vẽ sẵn được phép dùng, hoặc có video QIPEDC
    (miền Nam / dùng chung) để trình duyệt học ngay khi cần."""
    vid = _video_id(entry)
    return bool(motion_id_of(entry, available) or robot_pose_key(entry) or (vid and vid not in blocked_motion_ids()))


def headword_keys(word: str) -> list:
    """Các cách gõ ứng ĐÚNG với một từ gốc của từ điển (khóa đầu tiên là từ gốc bỏ phần trong ngoặc).

    Phần trong ngoặc của nhãn QIPEDC chỉ được coi là cách gõ khác khi chắc chắn là cùng một từ:
      "má (giống: mẹ)" → má, mẹ;  "đẹp đẽ (như đẹp)" → đẹp đẽ, đẹp;  "gầy (miền Nam: ốm)" → gầy, ốm;
      "võng (cái võng)" → võng, cái võng;  "về 5 (về nhà)" → về 5, về nhà;  "1 000 000 (một triệu)".
    Chú thích nghĩa như "cao (nhà)", "bình sữa (em bé)", "cộng (Toán học)" KHÔNG phải là từ khóa
    (gõ "nhà" không được ra ký hiệu "cao")."""
    word = unicodedata.normalize("NFC", word or "")
    raw_base = re.sub(r"\([^)]*\)", " ", word)
    base = norm_key(raw_base)
    keys = [base] if base else []
    if re.search(r"[,/;]", raw_base):           # "anh hai, anh cả" / "chiên/rán": mỗi vế là một cách gọi
        keys += [norm_key(part) for part in re.split(r"[,/;]", raw_base)]
    stem = re.sub(r"\s*\d+$", "", base)           # "về 5" → "về"
    numeric = bool(re.fullmatch(r"[\d\s.,]+", base or "x"))
    for inner in re.findall(r"\(([^)]*)\)", word):
        inner = inner.strip()
        m = re.match(r"(?i)(?:giống\s*:|như\s+|miền nam\s*:)\s*(.+)", inner)
        if m:
            keys.append(norm_key(m.group(1)))
            continue
        k = norm_key(inner)
        if not k or "," in inner or len(k.split()) > 5:
            continue
        if numeric or (stem and re.search(r"(^| )" + re.escape(stem) + r"( |$)", k)):
            keys.append(k)
    return list(dict.fromkeys(k for k in keys if k))


_exact_cache = {}


def exact_index(lex: dict) -> dict:
    """Chỉ mục từ gõ vào → mục từ có ĐÚNG từ gốc đó (bỏ các khóa "gần nghĩa" của bộ từ vựng cũ).
    Nhiều mục trùng khóa: ưu tiên mục robot diễn được (có video miền Nam / dùng chung), mục mà khóa là
    chính từ gốc, mục không kèm chú thích, rồi tới mục thông dụng hơn."""
    cache_key = id(lex["entries"])
    if cache_key not in _exact_cache:
        best = {}
        for e in lex["entries"]:
            hk = headword_keys(e["word"])
            for i, k in enumerate(hk):
                pri = (1 if _video_id(e) else 0, 1 if k == hk[0] else 0, -e["word"].count("("), -e.get("rank", 0))
                if k not in best or pri > best[k][0]:
                    best[k] = (pri, e)
        _exact_cache.clear()
        _exact_cache[cache_key] = {k: v[1] for k, v in best.items()}
    return _exact_cache[cache_key]


def _resolve(phrase: str, lex: dict, available: set = None, multi: bool = False):
    """Tìm mục từ cho một cụm từ: trùng đúng từ gốc → cách gọi khác của cùng khái niệm (SAME_SIGN_WORDS).

    Ưu tiên mục robot diễn được. Với cụm nhiều âm tiết (multi=True), bỏ qua mục robot chưa diễn được
    để các từ ngắn hơn có ký hiệu được dùng thay vì đánh vần cả cụm."""
    idx = exact_index(lex)
    candidates = []
    entry = idx.get(phrase)
    if entry and entry.get("proper_noun") and phrase in AMBIGUOUS_PROPER_KEYS:
        entry = None
    if entry:
        candidates.append(entry)
    syn = SAME_SIGN_WORDS.get(phrase)
    if syn and idx.get(norm_key(syn)):
        candidates.append(idx[norm_key(syn)])
    for c in candidates:
        if _usable(c, available):
            return c
    # Có mục từ nhưng robot chưa diễn được (chưa có video miền Nam / dùng chung): vẫn trả về để giao diện
    # hiện đúng từ và báo "chưa có ký hiệu miền Nam"; robot đánh vần từ đó.
    return candidates[0] if candidates else None


_QIPEDC_VIDEO_RE = re.compile(r"qipedc\.moet\.gov\.vn/videos/([A-Za-z0-9_-]+)\.mp4")


def _video_id(entry: dict):
    m = _QIPEDC_VIDEO_RE.search(entry.get("video_url") or "")
    return m.group(1) if m else None


def _vocab_item(token: str, entry: dict, available_motions: set = None) -> dict:
    gloss = gloss_of(entry)
    motion_id = motion_id_of(entry, available_motions)
    pose_key = robot_pose_key(entry)
    legacy = VSL_VOCABULARY.get(pose_key or "", {})
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
        # Chỉ 1 video minh họa: video QIPEDC miền Nam (hoặc dùng chung cả nước) mà robot học chuyển động.
        # Video giáo trình lưu trên Supabase hay bị chặn/treo → không dùng.
        "video_url": entry.get("video_url") if "qipedc.moet.gov.vn" in (entry.get("video_url") or "") else None,
        "thumb_url": entry.get("thumb_url"),
        "variants": [],
        "pose_key": pose_key,
        "region": entry.get("region"),
        "region_label": region_label(entry),
        "no_south": bool(entry.get("no_south")),
        # Robot diễn được nếu có chuyển động học từ video miền Nam (motion_id)
        "motion_id": motion_id,
        # Đoạn có động tác trong video mẫu (bỏ phần đứng yên đầu/cuối) — video & robot chỉ phát đoạn này
        "clip": list(motion_clip(motion_id)) if motion_id else None,
        "has_pose": bool(pose_key or motion_id),
        # Mã video QIPEDC: từ chưa có chuyển động → trình duyệt học từ video này rồi robot diễn
        # (chuyển động không đạt kiểm định chất lượng → không học lại, robot đánh vần)
        "video_id": None if _video_id(entry) in blocked_motion_ids() else _video_id(entry),
        # Chữ cái để robot đánh vần tạm trong lúc chưa có / đang học chuyển động
        "spell": [c for c in remove_accents(entry["word"]).upper() if c in VSL_ALPHABET][:12],
        # Diễn tả cụ thể cách làm ký hiệu (phân tích từ video người thật)
        "sign_description": sign_description(motion_id),
        "sstcn_index": entry.get("sstcn_index"),
        "motion_type": legacy.get("motion", "static"),
        "avatar_pose": legacy.get("avatar_pose", {}),
        "fingers_state": legacy.get("fingers_state", {}),
    }


_compounds = None


def vietnamese_compounds() -> set:
    """Danh sách từ ghép tiếng Việt (2–4 tiếng) dùng để nhận ra ranh giới từ.
    Nguồn: backend/data/sources/vi_compounds.txt (xem README cùng thư mục). Thiếu tệp → tập rỗng."""
    global _compounds
    if _compounds is None:
        path = os.path.join(os.path.dirname(__file__), "..", "data", "sources", "vi_compounds.txt")
        try:
            with open(path, encoding="utf-8") as f:
                _compounds = {norm_key(line) for line in f if line.strip()}
        except OSError:
            _compounds = set()
    return _compounds


def _compound_len(words: list, i: int) -> int:
    """Số tiếng của từ ghép dài nhất bắt đầu tại words[i] (0 nếu không phải từ ghép đã biết)."""
    comps = vietnamese_compounds()
    if not comps:
        return 0
    for length in range(min(4, len(words) - i), 1, -1):
        if " ".join(words[i:i + length]) in comps:
            return length
    return 0


LETTER_CLIPS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "letter_clips.json")
_letter_cache = {"mtime": None, "clips": {}}

# Dấu thanh (ký tự kết hợp Unicode) → tên dấu trong bảng chữ cái ngón tay
TONE_MARKS = {"\u0301": "sắc", "\u0300": "huyền", "\u0309": "hỏi", "\u0303": "ngã", "\u0323": "nặng"}


def letter_clips() -> dict:
    """Chữ cái / dấu thanh → {motion_id, clip}: đoạn giữ hình chữ cái trong video bảng chữ cái QIPEDC
    (backend/training/build_letter_clips.py)."""
    try:
        mtime = os.path.getmtime(LETTER_CLIPS_PATH)
    except OSError:
        return {}
    if _letter_cache["mtime"] != mtime:
        try:
            with open(LETTER_CLIPS_PATH, encoding="utf-8") as f:
                _letter_cache["clips"] = json.load(f)
        except (OSError, ValueError):
            _letter_cache["clips"] = {}
        _letter_cache["mtime"] = mtime
    return _letter_cache["clips"]


def split_syllable(syl: str):
    """"Trường" → (["t", "r", "ư", "ơ", "n", "g"], "huyền"): chữ cái tiếng Việt (giữ ă â ê ô ơ ư đ) + dấu thanh."""
    tone = None
    out = []
    for ch in unicodedata.normalize("NFD", syl.lower()):
        if ch in TONE_MARKS:
            tone = TONE_MARKS[ch]
            continue
        out.append(ch)
    letters = [c for c in unicodedata.normalize("NFC", "".join(out)) if c.isalpha() or c.isdigit()]
    return letters, tone


def _letter_item(name: str, ascii_key: str, lex: dict, available: set, group: str, is_tone: bool = False) -> dict:
    """Một bước đánh vần: chữ cái (hoặc dấu thanh) diễn bằng chuyển động học từ video bảng chữ cái QIPEDC;
    chưa có chuyển động → tư thế vẽ tay của chữ cái không dấu tương ứng."""
    clip = letter_clips().get(name)
    mid = clip["motion_id"] if clip and clip["motion_id"] in available else None
    label = f"Dấu {name}" if is_tone else f"Chữ {name.upper()}"
    alpha = VSL_ALPHABET.get(ascii_key, {})
    dur = int(round((clip["clip"][1] - clip["clip"][0]) * 1000)) + 250 if mid else 800
    return {
        "token": label if is_tone else name.upper(),
        "gloss": f"ALPHA_{name.upper()}",
        "label": label,
        "word": name,
        "entry_id": None,
        "type": "fingerspelling",
        "duration_ms": dur,
        "description": f"Bảng chữ cái ngón tay QIPEDC: {label}" if mid else alpha.get("description", label),
        "video_url": None,
        "variants": [],
        # Chuyển động học từ video chữ cái QIPEDC, chỉ phát đoạn giữ hình chữ cái
        "motion_id": mid,
        "clip": list(clip["clip"]) if mid else None,
        "pose_key": None if (mid or is_tone) else (ascii_key if ascii_key in VSL_ALPHABET else None),
        "has_pose": bool(mid or ascii_key in VSL_ALPHABET),
        "spelled_word": group,   # từ đang được đánh vần (giao diện hiển thị cả từ)
        "motion_type": "dynamic" if mid else "static",
        "fingers_state": alpha.get("fingers_state", {}),
        "avatar_pose": {},
    }


def _spell_items(word: str, lex: dict, group: str = None, available: set = None) -> list:
    """Đánh vần một từ theo bảng chữ cái ngón tay tiếng Việt: từng chữ cái (kể cả Ă Â Ê Ô Ơ Ư Đ),
    dấu thanh làm sau các chữ cái của mỗi tiếng."""
    available = motion_ids() if available is None else available
    group = group or word
    items = []
    for syl in word.split():
        letters, tone = split_syllable(syl)
        for c in letters:
            ascii_key = remove_accents(c).upper()
            if c in letter_clips() or ascii_key in VSL_ALPHABET:
                items.append(_letter_item(c, ascii_key, lex, available, group))
        if tone and tone in letter_clips():
            items.append(_letter_item(tone, "", lex, available, group, is_tone=True))
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
            entry = _resolve(phrase, lex, available, multi=True)
            if entry and not _usable(entry, available) and not _video_id(entry):
                entry = None   # cụm chưa diễn được → thử các từ ngắn hơn / đánh vần
            if entry:
                push(_vocab_item(phrase, entry, available))
                i += length
                matched = True
                break
        if matched:
            continue

        # Từ ghép tiếng Việt không có trong bộ ký hiệu ("sinh nhật", "công nhân", "kết hôn"...):
        # KHÔNG tách ra từng tiếng để diễn (ký hiệu "nhật" là nước Nhật, "nhân" là phép nhân, "hôn" là hôn)
        # mà đánh vần cả từ. Chỉ tách khi MỌI tiếng đều là từ có ký hiệu riêng ("về quê" → về + quê).
        comp_len = _compound_len(words, i)
        if comp_len:
            parts = words[i:i + comp_len]
            if not all(p in STOPWORDS or ((e := _resolve(p, lex, available)) and _usable(e, available)) for p in parts):
                result.extend(_spell_items(" ".join(parts), lex, available=available))
                i += comp_len
                continue

        word = words[i]
        if word in STOPWORDS:
            i += 1
            continue
        entry = _resolve(word, lex, available)
        if entry and not _usable(entry, available) and not entry.get("proper_noun"):
            # Có mục từ nhưng không có video miền Nam / dùng chung nào để robot học (vd. "tôi", "cảm ơn"
            # chỉ có trong giáo trình) → đánh vần đúng chữ thay vì diễn một ký hiệu chưa kiểm chứng
            result.extend(_spell_items(word, lex, available=available))
        elif entry:
            push(_vocab_item(word, entry, available))
        elif re.fullmatch(r"\d+", word):
            for d in word:
                e = exact_index(lex).get(d)
                if e:
                    push(_vocab_item(d, e, available))
        else:
            result.extend(_spell_items(word, lex, available=available))
        i += 1

    return result
