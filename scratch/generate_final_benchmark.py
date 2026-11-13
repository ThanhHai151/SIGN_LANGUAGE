import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/sstcn_102_dump.json", encoding="utf-8") as f:
    dump = json.load(f)

with open("backend/data/vsl_lexicon.json", encoding="utf-8") as f:
    lex = json.load(f)

by_sstcn = {e["sstcn_index"]: e for e in lex["entries"] if e.get("sstcn_index") is not None}

# 30 chosen words
selected_indices = [
    52,  # S01: Chào
    53,  # S02: Xin lỗi
    55,  # S03: Vui
    56,  # S04: Buồn
    51,  # S05: Tôi
    60,  # S06: Bạn
    58,  # S07: Khỏe
    44,  # S08: Hôm nay
    75,  # S09: Ăn
    76,  # S10: Uống
    73,  # S11: Ngủ
    67,  # S12: Đi
    68,  # S13: Đứng
    69,  # S14: Ngồi
    71,  # S15: Chạy
    77,  # S16: Tắm
    80,  # S17: Gia đình
    83,  # S18: Bố
    84,  # S19: Mẹ
    81,  # S20: Ông
    82,  # S21: Bà
    91,  # S22: Anh trai
    92,  # S23: Chị gái
    86,  # S24: Em gái
    94,  # S25: Yêu
    95,  # S26: Cô giáo
    96,  # S27: Công nhân
    98,  # S28: Bác sĩ
    100, # S29: Công an
    101  # S30: Bộ đội
]

benchmark_30 = []
for idx, sstcn_idx in enumerate(selected_indices, 1):
    meta = dump[sstcn_idx]
    entry = by_sstcn.get(sstcn_idx, {})
    
    bid = f"S{idx:02d}"
    term = meta["term"]
    label = meta["label"]
    cat = meta["category"]
    desc = meta["description"]
    
    qid = entry.get("id") or meta.get("q_entry_id")
    qword = entry.get("word") or meta.get("q_word")
    vid = entry.get("video_url") or meta.get("video_url_meta")
    variants = entry.get("variants", [])
    v_count = len(variants)
    
    benchmark_30.append({
        "bid": bid,
        "sstcn_index": sstcn_idx,
        "code_label": term,
        "display_name": label,
        "category": cat,
        "q_entry_id": qid,
        "q_word": qword,
        "ref_video_url": vid,
        "variants_count": v_count,
        "variants": variants,
        "description": desc
    })

with open("scratch/benchmark_30_final.json", "w", encoding="utf-8") as f:
    json.dump(benchmark_30, f, ensure_ascii=False, indent=2)

print(f"Compiled {len(benchmark_30)} benchmark words.")
for b in benchmark_30:
    print(f"{b['bid']} | Idx: {b['sstcn_index']:3d} | `{b['code_label']:12s}` | {b['display_name']:12s} | {b['category']:30s} | {b['q_entry_id']:15s} | Var: {b['variants_count']}")

