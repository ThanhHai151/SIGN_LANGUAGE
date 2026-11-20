"""Multi-source VSL Label Harmonizer & Topic Classifier.

Maps raw labels from heterogeneous sources (SSTCN-102, Hugging Face VOYA_VSL,
ViSignLanguage-Video, Kaggle/QIPEDC, Zenodo VSL400, Personal Calibration) onto
the unified vsl_lexicon.json (`entry_id`), while preventing Northern/Southern
regional sign collisions.
"""

import re
import unicodedata
from typing import Dict, Any, Optional, Tuple

from backend.models.lexicon import load_lexicon, norm_key, STANDARD_REGION
from backend.models.vsl_102_dictionary import SIGN_METADATA_102

_REGION_PAREN_RE = re.compile(r"\s*\((bắc|nam|trung|bac|nam|trung)\)\s*$", re.IGNORECASE)
_REGION_SUFFIX_RE = re.compile(r"(?:_|\s+)(B|T|N)$")
_VIDEO_REGION_RE = re.compile(r"^[DW]\d+(B|T|N)$", re.IGNORECASE)

SYNONYMS = {
    "xin chao": "chào",
    "cam on": "cảm ơn",
    "cám ơn": "cảm ơn",
    "xin loi": "xin lỗi",
    "tam biet": "tạm biệt",
    "chuc mung": "chúc mừng",
    "bac si": "bác sĩ",
    "bác sỹ": "bác sĩ",
    "gia dinh": "gia đình",
    "ban than": "bạn thân",
    "benh vien": "bệnh viện",
    "truong hoc": "trường học",
}

# 10+ chủ đề chuẩn hóa (học theo cấu trúc 10 chủ đề của KVSL + nhóm từ điển VSL Connect AI)
TOPIC_KEYWORDS = {
    "Chào hỏi & Giao tiếp": ["chào", "cảm ơn", "xin lỗi", "tạm biệt", "chúc mừng", "tên", "hẹn gặp", "giao tiếp", "phiên dịch", "hiểu", "biết"],
    "Câu hỏi": ["là gì", "bao giờ", "bao nhiêu", "ở đâu", "tại sao", "lí do", "lý do", "không?", "mấy", "ai", "khi nào", "hỏi"],
    "Gia đình & Con người": ["gia đình", "bố", "mẹ", "cha", "ông", "bà", "anh", "chị", "em", "con", "bạn", "tôi", "chúng tôi", "người", "trẻ em", "học sinh", "giáo viên", "bác sĩ", "thợ"],
    "Thời gian": ["thời gian", "giờ", "ngày", "tháng", "năm", "hôm nay", "hôm qua", "ngày mai", "sáng", "trưa", "chiều", "tối", "đêm", "tuần"],
    "Hành động": ["ăn", "uống", "đi", "ngủ", "học", "làm", "đọc", "viết", "xem", "nghe", "nói", "mua", "bán", "chạy", "đứng", "ngồi", "mở", "đóng", "tặng", "giúp"],
    "Cảm xúc & Tính cách": ["vui", "buồn", "giận", "sợ", "yêu", "thích", "ghét", "hồi hộp", "phân vân", "mệt", "đói", "khỏe", "đẹp", "xấu", "ngon", "tốt", "cáu"],
    "Y tế & Sức khỏe": ["bệnh", "đau", "thuốc", "bác sĩ", "bệnh viện", "cấp cứu", "chóng mặt", "khỏi", "cách ly", "sốt", "ho"],
    "Địa điểm & Giao thông": ["nhà", "trường", "siêu thị", "cửa hàng", "bến xe", "công viên", "hà nội", "địa chỉ", "kẹt xe", "lạc", "đường", "xe", "thành phố"],
    "Đồ vật & Ẩm thực": ["tiền", "sách", "bút", "bàn", "ghế", "điện thoại", "mì", "cơm", "nước", "trà", "phở", "bánh", "món"],
    "Thời tiết & Thiên nhiên": ["mưa", "nắng", "gió", "bão", "nóng", "lạnh", "trời", "mây", "hoa", "cây", "biển", "núi"],
}


def parse_region_and_clean_label(raw_label: str, video_id: Optional[str] = None) -> Tuple[str, Optional[str]]:
    """Tách thông tin vùng miền (B/T/N) khỏi nhãn thô mà không làm mất nghĩa gốc."""
    text = unicodedata.normalize("NFC", str(raw_label or "")).strip()
    region = None

    m_paren = _REGION_PAREN_RE.search(text)
    if m_paren:
        r_word = norm_key(m_paren.group(1))
        region = {"bac": "B", "bắc": "B", "nam": "N", "trung": "T"}.get(r_word)
        text = _REGION_PAREN_RE.sub("", text).strip()
    else:
        m_suf = _REGION_SUFFIX_RE.search(text)
        if m_suf and len(text) > 3:
            region = m_suf.group(1).upper()
            text = _REGION_SUFFIX_RE.sub("", text).strip()

    if region is None and video_id:
        m_vid = _VIDEO_REGION_RE.match(str(video_id).strip())
        if m_vid:
            region = m_vid.group(1).upper()

    return text, region


def infer_topic(label: str, lexicon_entry: Optional[dict] = None) -> str:
    """Phân nhóm chủ đề nhất quán cho Topic-Scoped Masking (học từ KVSL)."""
    if lexicon_entry:
        sstcn_idx = lexicon_entry.get("sstcn_index")
        if sstcn_idx is not None:
            for item in SIGN_METADATA_102:
                if int(item["index"]) == int(sstcn_idx) and item.get("category"):
                    return item["category"]
        if lexicon_entry.get("category") and lexicon_entry["category"] not in ("", "Từ vựng VSL", "Từ vựng"):
            return lexicon_entry["category"]

    low = norm_key(label)
    for topic, kws in TOPIC_KEYWORDS.items():
        if any(kw in low for kw in kws):
            return topic
    return "Từ vựng thông dụng"


class LabelHarmonizer:
    """Ánh xạ nhãn đa nguồn về từ điển chuẩn vsl_lexicon.json và tách biến thể vùng miền."""

    def __init__(self, split_regions: bool = True, preferred_region: str = STANDARD_REGION):
        self.lex = load_lexicon()
        self.split_regions = split_regions
        self.preferred_region = preferred_region
        # Chỉ mục bổ sung không dấu cho những nguồn như LNT / tên thư mục không dấu
        self.no_accent_index: Dict[str, dict] = {}
        from backend.models.lexicon import strip_accents
        for entry in self.lex["entries"]:
            for k in entry.get("keys", []):
                na = strip_accents(k)
                self.no_accent_index.setdefault(na, entry)

    def resolve(
        self,
        raw_label: str,
        video_id: Optional[str] = None,
        explicit_region: Optional[str] = None,
    ) -> Dict[str, Any]:
        clean, parsed_region = parse_region_and_clean_label(raw_label, video_id=video_id)
        region = (explicit_region or parsed_region or "").upper() or None

        key = norm_key(clean)
        if key in SYNONYMS:
            clean = SYNONYMS[key]
            key = norm_key(clean)

        entry = self.lex["by_key"].get(key)
        if entry is None:
            from backend.models.lexicon import strip_accents
            entry = self.no_accent_index.get(strip_accents(key))

        base_label = entry["word"] if entry else clean
        entry_id = entry["id"] if entry else None

        # Chống xung đột Bắc - Trung - Nam: nếu nguồn ghi rõ khác vùng chuẩn và split_regions=True
        if self.split_regions and region and region != self.preferred_region:
            region_suffix = {"B": "Bắc", "T": "Trung", "N": "Nam"}.get(region, region)
            canonical_label = f"{base_label} ({region_suffix})"
            class_key = f"{entry_id or norm_key(base_label)}::{region}"
        else:
            canonical_label = base_label
            class_key = f"{entry_id or norm_key(base_label)}"

        topic = infer_topic(base_label, lexicon_entry=entry)
        return {
            "raw_label": raw_label,
            "clean_label": clean,
            "canonical_label": canonical_label,
            "class_key": class_key,
            "entry_id": entry_id,
            "region": region or (entry.get("region") if entry else None),
            "topic": topic,
            "matched_lexicon": entry is not None,
        }
