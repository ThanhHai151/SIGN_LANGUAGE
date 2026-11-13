import json
import os
import sys

sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

with open("scratch/sstcn_102_dump.json", encoding="utf-8") as f:
    dump = json.load(f)

proposed_30 = [
    "Xin chào", "Cảm ơn", "Xin lỗi", "Vui", "Buồn", "Yêu", "Ăn", "Uống", "Ngủ", "Đi",
    "Về", "Học", "Làm việc", "Chơi", "Đọc", "Viết", "Bố", "Mẹ", "Anh", "Chị",
    "Em", "Ông", "Bà", "Gia đình", "Giáo viên", "Học sinh", "Bác sĩ", "Công an", "Công nhân", "Nghề nghiệp"
]

core_words = [d for d in dump if d["index"] >= 39]
print(f"Total core words: {len(core_words)}")

def norm(s):
    if not s:
        return ""
    import unicodedata
    s = unicodedata.normalize("NFC", s.strip().lower())
    return s

# Print all 63 core words
print("\n--- ALL 63 CORE VSL WORDS (Index 39 - 101) ---")
for w in core_words:
    print(f"Idx {w['index']:2d}: term=\"{w['term']}\", label=\"{w['label']}\", q_word=\"{w['q_word']}\", cat=\"{w['category']}\", q_id=\"{w['q_entry_id']}\"")

# Check each proposed word
evaluations = []
for idx, p in enumerate(proposed_30, 1):
    p_norm = norm(p)
    
    # 1. Exact match on term, label, or q_word in core words (39-101)
    exact_match = None
    alias_match = None
    
    for w in core_words:
        term_norm = norm(w["term"])
        label_norm = norm(w["label"])
        q_word_norm = norm(w["q_word"]) if w["q_word"] else ""
        
        # Check exact match
        if p_norm == term_norm or p_norm == label_norm or p_norm == q_word_norm:
            exact_match = w
            break
            
        # Check alias match (e.g. "Xin chào" vs "Chào", "Giáo viên" vs "Cô giáo", "Anh" vs "Anh trai", "Chị" vs "Chị gái", "Em" vs "Em trai"/"Em gái"/"Em út")
        # Common aliases:
        # "xin chào" <-> "chào"
        if p_norm == "xin chào" and term_norm == "chào":
            alias_match = (w, "Trong code là 'Chào' (index 52)")
        elif p_norm == "giáo viên" and term_norm == "cô giáo":
            alias_match = (w, "Trong code là 'Cô giáo' (index 95)")
        elif p_norm == "anh" and term_norm == "anh trai":
            alias_match = (w, "Trong code là 'Anh trai' (index 91)")
        elif p_norm == "chị" and term_norm == "chị gái":
            alias_match = (w, "Trong code là 'Chị gái' (index 92)")
        elif p_norm == "em" and term_norm in ("em trai", "em gái", "em út"):
            alias_match = (w, f"Trong code phân tách thành '{w['term']}' (index {w['index']})")
        elif p_norm == "học" and "học" in term_norm:
            alias_match = (w, f"Trong code có '{w['term']}' (index {w['index']})")
    
    if exact_match:
        evaluations.append({
            "stt": idx,
            "proposed": p,
            "status": "YES",
            "sstcn_index": exact_match["index"],
            "code_label": exact_match["term"],
            "display_name": exact_match["label"],
            "q_entry_id": exact_match["q_entry_id"],
            "q_word": exact_match["q_word"],
            "q_video_url": exact_match["q_video_url"],
            "variants_count": exact_match["variants_count"],
            "variants": exact_match["variants"],
            "note": "Khớp chính xác nhãn trong code."
        })
    elif alias_match:
        w, note = alias_match
        evaluations.append({
            "stt": idx,
            "proposed": p,
            "status": "ALIAS",
            "sstcn_index": w["index"],
            "code_label": w["term"],
            "display_name": w["label"],
            "q_entry_id": w["q_entry_id"],
            "q_word": w["q_word"],
            "q_video_url": w["q_video_url"],
            "variants_count": w["variants_count"],
            "variants": w["variants"],
            "note": note
        })
    else:
        evaluations.append({
            "stt": idx,
            "proposed": p,
            "status": "NO",
            "sstcn_index": None,
            "code_label": "NOT FOUND",
            "display_name": "NOT FOUND",
            "q_entry_id": None,
            "q_word": None,
            "q_video_url": None,
            "variants_count": 0,
            "variants": [],
            "note": "Không tồn tại trong 63 output từ vựng của SSTCN_Attention."
        })

with open("scratch/proposed_evaluations.json", "w", encoding="utf-8") as f:
    json.dump(evaluations, f, ensure_ascii=False, indent=2)

print("\n--- EVALUATION SUMMARY ---")
yes_cnt = sum(1 for e in evaluations if e["status"] == "YES")
alias_cnt = sum(1 for e in evaluations if e["status"] == "ALIAS")
no_cnt = sum(1 for e in evaluations if e["status"] == "NO")
print(f"YES: {yes_cnt}, ALIAS: {alias_cnt}, NO: {no_cnt}")
for e in evaluations:
    print(f"[{e['stt']:2d}] {e['proposed']:15s} -> {e['status']:5s} | Idx: {str(e['sstcn_index']):4s} | Label: {e['code_label']:12s} | Note: {e['note']}")
