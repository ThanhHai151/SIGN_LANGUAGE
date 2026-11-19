"""
Dựng danh sách từ ghép tiếng Việt (backend/data/sources/vi_compounds.txt) từ một danh sách từ tiếng Việt.

    python -m backend.training.build_compounds <đường_dẫn_Viet74K.txt>

Chỉ giữ từ 2–4 tiếng, viết thường, chỉ gồm chữ cái tiếng Việt. Danh sách này giúp bộ chuyển văn bản → ký hiệu
nhận ra ranh giới từ: từ ghép không có ký hiệu ("sinh nhật") được đánh vần cả từ thay vì diễn từng tiếng
với nghĩa khác ("nhật" = nước Nhật).
"""
import os
import re
import sys
import unicodedata

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "sources", "vi_compounds.txt")


def build(src: str) -> int:
    out = set()
    with open(src, encoding="utf-8") as f:
        for line in f:
            w = unicodedata.normalize("NFC", line.strip()).lower()
            if not w or re.search(r"[^a-zà-ỹđ ]", w):
                continue
            parts = w.split()
            if 2 <= len(parts) <= 4:
                out.add(" ".join(parts))
    with open(OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(sorted(out)) + "\n")
    return len(out)


if __name__ == "__main__":
    print("Số từ ghép:", build(sys.argv[1]))
