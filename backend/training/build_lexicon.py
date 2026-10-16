"""
Dựng BỘ TỪ VỰNG VSL THỐNG NHẤT (backend/data/vsl_lexicon.json)
================================================================

Một nguồn dữ liệu duy nhất cho cả 2 chiều của ứng dụng:
  - Camera → Văn bản: 102 lớp của mô hình SSTCN_Attention được gắn vào từ tương ứng.
  - Văn bản → Ký hiệu: tra cứu toàn bộ từ điển, phát video mẫu + robot 3D (nếu có tư thế).

Nguồn dữ liệu (backend/data/sources/):
  1. qipedc_dictionary.json — Danh mục Ngôn ngữ ký hiệu tiếng Việt QIPEDC
     (Bộ GD&ĐT, dự án Nâng cao chất lượng giáo dục học sinh khiếm thính cấp tiểu học),
     https://qipedc.moet.gov.vn/dictionary — 4.362 video ~ 3.300 từ, thu thập từ người
     Điếc ở 3 vùng miền Bắc / Trung / Nam. Mỗi dòng: [mã video, từ, từ loại, giải nghĩa, type, có ảnh minh họa].
  2. backend/models/vsl_102_dictionary.py — 102 ký hiệu giáo trình (lớp đầu ra của SSTCN).
  3. backend/models/vsl_dictionary.py + src/utils/handPoses.js — các từ robot 3D đã có tư thế.
  4. vi_syllable_freq.tsv — tần suất âm tiết tiếng Việt trong hội thoại (OpenSubtitles 2018,
     danh sách FrequencyWords của Hermit Dave, CC BY-SA 4.0) dùng để xếp hạng mức độ thông dụng.

Chạy lại:  python -m backend.training.build_lexicon
"""

import json
import os
import re
import sys
import unicodedata
from collections import defaultdict

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)

from backend.models.vsl_102_dictionary import SIGN_METADATA_102  # noqa: E402
from backend.models.vsl_dictionary import VSL_VOCABULARY  # noqa: E402

DATA_DIR = os.path.join(ROOT, "backend", "data")
SRC_DIR = os.path.join(DATA_DIR, "sources")
OUT_PATH = os.path.join(DATA_DIR, "vsl_lexicon.json")
HAND_POSES_JS = os.path.join(ROOT, "src", "utils", "handPoses.js")

QIPEDC_BASE = "https://qipedc.moet.gov.vn"
COMMON_TOP_N = 700          # số từ QIPEDC thông dụng nhất được đánh dấu "common"
REGION_ORDER = ["B", "T", "N"]  # ưu tiên miền Bắc làm mặc định
REGION_NAMES = {"B": "Miền Bắc", "T": "Miền Trung", "N": "Miền Nam"}

# Từ tiếng Việt → khóa tư thế robot có sẵn trong handPoses.js nhưng không có trong VSL_VOCABULARY
EXTRA_POSE_WORDS = {
    "ăn": "AN", "ăn cơm": "AN", "ngủ": "NGU", "đi ngủ": "NGU", "uống": "UONG",
    "uống nước": "UONG", "đi": "DI", "không": "KHONG",
}

# Cặp (tư thế robot, từ khóa) KHÔNG được ghép vì khác nghĩa (từ khóa cũ chỉ là gần nghĩa)
POSE_LINK_BLOCK = {("BENH_VIEN", "bác sĩ"), ("BENH_VIEN", "y tá"), ("NEU", "giả sử"), ("CO_THE", "khả năng")}

# Cách đọc số trong giáo trình 102 → khóa tra cứu thêm
NUMBER_WORDS = {
    1: "một", 2: "hai", 3: "ba", 4: "bốn", 5: "năm",
    6: "sáu", 7: "bảy", 8: "tám", 9: "chín", 10: "mười",
}


# ---------------------------------------------------------------- tiện ích
def nfc(s):
    return unicodedata.normalize("NFC", s or "")


def norm_key(s):
    """Khóa so khớp: chữ thường, NFC, bỏ dấu câu thừa, gộp khoảng trắng."""
    s = nfc(s).lower().strip()
    s = s.replace("？", "?")
    s = re.sub(r"[?!.,;:\"“”]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def strip_accents(s):
    s = s.replace("đ", "d").replace("Đ", "D")
    return "".join(c for c in unicodedata.normalize("NFD", s) if not unicodedata.combining(c))


def slug(s):
    return re.sub(r"[^a-z0-9]+", "-", strip_accents(norm_key(s))).strip("-") or "x"


def keys_from_word(word):
    """Sinh các khóa tra cứu từ nhãn QIPEDC:
    'võng (cái võng)' → ['võng', 'cái võng'];  'má (giống: mẹ)' → ['má', 'mẹ'];
    'Miến Điện (nước Mi-an-ma)' → ['miến điện', 'nước mi-an-ma'];  'tên là gì?' → ['tên là gì']."""
    word = nfc(word)
    keys = []
    base = re.sub(r"\([^)]*\)", " ", word)
    if norm_key(base):
        keys.append(norm_key(base))
    for inner in re.findall(r"\(([^)]*)\)", word):
        inner = inner.strip()
        m = re.match(r"(?i)giống\s*:\s*(.+)", inner)
        if m:
            keys.append(norm_key(m.group(1)))
            continue
        # bỏ chú thích ngữ pháp / chuyên môn / đánh số, giữ cụm từ thật
        if re.fullmatch(r"[\d\s,./-]*", inner):
            continue
        if re.search(r"(?i)(đại từ|danh từ|động từ|tính từ|toán học|hành động|\d+ người)", inner):
            continue
        k = norm_key(inner)
        if k and len(k.split()) <= 5:
            keys.append(k)
    out = []
    for k in keys:
        if k and k not in out:
            out.append(k)
    return out


def split_region(video_id):
    """D0001B → ('D0001', 'B');  W00004N → ('W00004', 'N');  D0002 → ('D0002', None)."""
    m = re.match(r"^([DW]\d+)([BNT])?$", video_id)
    if not m:
        return video_id, None
    return m.group(1), m.group(2)


def load_syllable_freq():
    path = os.path.join(SRC_DIR, "vi_syllable_freq.tsv")
    freq = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) == 2:
                    freq[parts[0]] = max(freq.get(parts[0], 0), int(parts[1]))
    return freq


def load_pose_keys():
    if not os.path.exists(HAND_POSES_JS):
        return set()
    with open(HAND_POSES_JS, encoding="utf-8") as f:
        return set(re.findall(r"^  ([A-Z_]+): \{", f.read(), re.M))


def commonness(word, freq, proper):
    """Điểm thông dụng ước lượng từ tần suất âm tiết trong hội thoại.
    Từ ghép lấy âm tiết hiếm nhất và giảm dần theo số âm tiết; danh từ riêng bị hạ điểm."""
    syl = [s for s in norm_key(re.sub(r"\([^)]*\)", " ", word)).split() if s]
    if not syl:
        return 0.0
    counts = [freq.get(s, 0) for s in syl]
    score = min(counts) * (0.3 ** (len(syl) - 1))
    if proper:
        score *= 0.02
    inner = " ".join(re.findall(r"\(([^)]*)\)", word))
    if inner and not re.search(r"(?i)giống\s*:", inner):
        score *= 0.5  # nghĩa chuyên biệt (Toán học, chỉ người, số nhiều...) ít gặp hơn nghĩa thường
    return float(score)


# ---------------------------------------------------------------- dựng dữ liệu
def build():
    with open(os.path.join(SRC_DIR, "qipedc_dictionary.json"), encoding="utf-8") as f:
        rows = json.load(f)
    freq = load_syllable_freq()
    pose_keys = load_pose_keys()

    # 1) Gom các video QIPEDC theo (từ, giải nghĩa)
    groups = {}
    order = []
    for vid, word, pos, definition, _type, has_img in rows:
        word = nfc(word).strip()
        k = (word, nfc(definition).strip())
        if k not in groups:
            groups[k] = {"word": word, "pos": nfc(pos).strip(), "definition": nfc(definition).strip(),
                         "ids": [], "image": None}
            order.append(k)
        g = groups[k]
        g["ids"].append(vid)
        if has_img and not g["image"]:
            g["image"] = f"{QIPEDC_BASE}/Anh/{vid[:5]}.png"

    entries = []
    used_ids = set()
    for k in order:
        g = groups[k]
        videos = {}
        others = []
        for vid in g["ids"]:
            base, region = split_region(vid)
            url = f"{QIPEDC_BASE}/videos/{vid}.mp4"
            if region and region not in videos:
                videos[region] = url
            else:
                others.append(url)
        variants = [{"region": r, "label": REGION_NAMES[r], "url": videos[r]} for r in REGION_ORDER if r in videos]
        for i, url in enumerate(others):
            variants.append({"region": "", "label": "Cách thể hiện" + (f" {i + 1}" if len(others) > 1 else ""), "url": url})
        primary_id = g["ids"][0]
        for r in REGION_ORDER:
            if r in videos:
                primary_id = next(v for v in g["ids"] if v.endswith(r))
                break
        if others and not videos:
            primary_id = g["ids"][0]

        eid = "q-" + slug(g["word"])
        n = 2
        while eid in used_ids:
            eid = f"q-{slug(g['word'])}-{n}"
            n += 1
        used_ids.add(eid)

        word = g["word"]
        proper = bool(re.match(r"^[A-ZĐÂĂÊÔƠƯÁÀẢÃẠÉÈẺẼẸÍÌỈĨỊÓÒỎÕỌÚÙỦŨỤÝỲỶỸỴ]", word)) and not word.isupper()
        entries.append({
            "id": eid,
            "word": word,
            "keys": keys_from_word(word),
            "pos": g["pos"],
            "definition": g["definition"],
            "how_to": "",
            "source": "QIPEDC",
            "category": g["pos"] or "Khác",
            "video_url": f"{QIPEDC_BASE}/videos/{primary_id}.mp4",
            "thumb_url": f"{QIPEDC_BASE}/thumbs/{primary_id}.png",
            "image_url": g["image"],
            "variants": variants,
            "sstcn_index": None,
            "pose_key": None,
            "proper_noun": proper,
            "score": commonness(word, freq, proper),
            "common": False,
        })

    by_key = defaultdict(list)

    def reindex():
        by_key.clear()
        for e in entries:
            for key in e["keys"]:
                by_key[key].append(e)

    def best(key):
        cands = by_key.get(key) or []
        return max(cands, key=lambda e: e["score"]) if cands else None

    def best_link(key):
        """Ghép từ giáo trình / robot vào mục QIPEDC: chỉ nhận mục có từ gốc trùng khớp
        (ưu tiên mục không có chú thích trong ngoặc) hoặc mục ghi rõ 'giống: <từ>'.
        Mục có chú thích nghĩa khác như '(động vật)', '(Toán học)' bị loại để tránh ghép sai nghĩa."""
        good = []
        for e in by_key.get(key) or []:
            w = e["word"]
            base = norm_key(re.sub(r"\([^)]*\)", " ", w))
            inner = " ".join(re.findall(r"\(([^)]*)\)", w))
            if re.search(r"(?i)giống\s*:", inner) and key in e["keys"]:
                good.append((1, e))
            elif base == key and not inner:
                good.append((2, e))
            elif base == key and re.search(r"(?i)^(cái|con|quả|người|chỉ người|đt|dt)\b", inner.strip()):
                good.append((0, e))
        if not good:
            return None
        return max(good, key=lambda t: (t[0], t[1]["score"]))[1]

    reindex()

    # 2) Gắn 102 lớp giáo trình (camera SSTCN) — bổ sung từ còn thiếu trong QIPEDC
    for s in SIGN_METADATA_102:
        term = nfc(s["term"])
        keys = [norm_key(term)]
        m = re.match(r"^Số (\d+)$", term)
        if m:
            num = int(m.group(1))
            keys = [str(num), norm_key(term)] + ([NUMBER_WORDS[num]] if num in NUMBER_WORDS else [])
        if len(term) == 1 or term in ("Ă", "Â", "Ê", "Ô", "Ơ", "Ư", "Đ"):
            keys = [norm_key(term)]
        target = None
        for k in keys:
            target = best_link(k)
            if target and target["sstcn_index"] is None:
                break
            target = None
        curriculum_variant = {"region": "GT", "label": "Giáo trình 102", "url": s["video_url"]}
        if target:
            target["sstcn_index"] = s["index"]
            target["how_to"] = s["description"]
            target["category"] = s["category"]
            target["variants"].append(curriculum_variant)
            for k in keys:
                if k not in target["keys"]:
                    target["keys"].append(k)
        else:
            entries.append({
                "id": "gt-" + slug(term),
                "word": term if term.isupper() or term.startswith("Số") else term[0].lower() + term[1:],
                "keys": keys,
                "pos": "",
                "definition": "",
                "how_to": s["description"],
                "source": "Giáo trình VSL (102 từ)",
                "category": s["category"],
                "video_url": s["video_url"],
                "thumb_url": None,
                "image_url": None,
                "variants": [curriculum_variant],
                "sstcn_index": s["index"],
                "pose_key": None,
                "proper_noun": False,
                "score": commonness(term, freq, False),
                "common": False,
            })
    reindex()

    # 3) Gắn tư thế robot 3D (từ vựng cũ của ứng dụng) — bổ sung từ còn thiếu
    pose_words = {}
    for key, item in VSL_VOCABULARY.items():
        if key not in pose_keys:
            continue
        words = [item["label"]] + [w for w in item.get("keywords", []) if re.search(r"[^\x00-\x7f]", w) or " " in w]
        pose_words[key] = (item, [norm_key(w) for w in words])
    for w, key in EXTRA_POSE_WORDS.items():
        if key in pose_keys:
            pose_words.setdefault(key, ({"label": w.capitalize(), "description": ""}, []))[1].append(norm_key(w))

    for key, (item, words) in pose_words.items():
        target = None
        for w in words:
            if (key, w) in POSE_LINK_BLOCK:
                continue
            target = best_link(w)
            if target:
                break
        if target:
            if not target["pose_key"]:
                target["pose_key"] = key
            if not target["how_to"] and item.get("description"):
                target["how_to"] = item["description"]
            for w in words:
                if w not in target["keys"]:
                    target["keys"].append(w)
        else:
            label = nfc(item["label"])
            entries.append({
                "id": "app-" + slug(label),
                "word": label[0].lower() + label[1:] if len(label) > 1 else label,
                "keys": list(dict.fromkeys(words)),
                "pos": "",
                "definition": "",
                "how_to": item.get("description", ""),
                "source": "Robot 3D (ứng dụng)",
                "category": "Giao tiếp",
                "video_url": None,
                "thumb_url": None,
                "image_url": None,
                "variants": [],
                "sstcn_index": None,
                "pose_key": key,
                "proper_noun": False,
                "score": commonness(label, freq, False),
                "common": False,
            })
    reindex()

    # 4) Phân loại chữ cái / số (dùng để đánh vần, không tính vào danh sách thông dụng)
    for e in entries:
        w = e["word"].strip()
        if len(w) == 1 and w.isalpha():
            e["category"] = "Bảng chữ cái"
            e["fingerspelling"] = True
        elif re.fullmatch(r"(Số )?\d+", w):
            e["category"] = "Số đếm" if e["source"] == "QIPEDC" else e["category"]
        e.setdefault("fingerspelling", False)

    # 5) Xếp hạng thông dụng
    ranked = sorted(entries, key=lambda e: -e["score"])
    for rank, e in enumerate(ranked, 1):
        e["rank"] = rank
    qipedc_ranked = [e for e in ranked if e["source"] == "QIPEDC" and not e["proper_noun"] and not e["fingerspelling"]]
    common_ids = {e["id"] for e in qipedc_ranked[:COMMON_TOP_N]}
    for e in entries:
        e["common"] = e["id"] in common_ids or e["sstcn_index"] is not None or e["pose_key"] is not None
        e["score"] = round(e["score"], 2)

    entries.sort(key=lambda e: e["rank"])
    meta = {
        "name": "Bộ từ vựng Ngôn ngữ Ký hiệu Việt Nam thống nhất",
        "total": len(entries),
        "total_videos": sum(len(e["variants"]) for e in entries),
        "common": sum(1 for e in entries if e["common"]),
        "sstcn_linked": sum(1 for e in entries if e["sstcn_index"] is not None),
        "robot_pose_linked": sum(1 for e in entries if e["pose_key"]),
        "sources": {
            "QIPEDC": {
                "url": "https://qipedc.moet.gov.vn/dictionary",
                "description": "Danh mục Ngôn ngữ ký hiệu tiếng Việt (~4.000 ký hiệu) – Dự án QIPEDC, Bộ Giáo dục và Đào tạo; video người Điếc 3 miền Bắc/Trung/Nam.",
                "entries": sum(1 for e in entries if e["source"] == "QIPEDC"),
            },
            "Giáo trình VSL (102 từ)": {
                "description": "102 ký hiệu giáo trình (6 bài) – lớp đầu ra của mô hình camera SSTCN_Attention.",
            },
            "Tần suất": {
                "url": "https://github.com/hermitdave/FrequencyWords",
                "description": "Tần suất âm tiết tiếng Việt trong phụ đề hội thoại (OpenSubtitles 2018, CC BY-SA 4.0) dùng xếp hạng mức thông dụng.",
            },
        },
    }
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"meta": meta, "entries": entries}, f, ensure_ascii=False, separators=(",", ":"))
    return meta


if __name__ == "__main__":
    m = build()
    print(json.dumps(m, ensure_ascii=False, indent=2))
