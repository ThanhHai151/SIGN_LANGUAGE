"""
Tìm kiếm chính xác tất cả mục liên quan đến M và N trong qipedc_dictionary.json
"""
import os
import json

WORKSPACE = "d:/SIGN_LANGUAGE"
qipedc_path = os.path.join(WORKSPACE, "backend/data/sources/qipedc_dictionary.json")

with open(qipedc_path, "r", encoding="utf-8") as fp:
    q_dict = json.load(fp)

print(f"Tổng số mục: {len(q_dict)}")

# Tìm kiếm theo mã D0504, D0505 và từ M, N
for row in q_dict:
    sid = str(row[0]) if len(row) > 0 else ""
    word = str(row[1]) if len(row) > 1 else ""
    desc = str(row[3]) if len(row) > 3 else ""
    
    # Check if id starts with D0504 or D0505
    if sid.startswith("D0504") or sid.startswith("D0505"):
        print(f"Match ID: sid={sid}, word='{word}', desc='{desc}'")
    elif word.strip().upper() in ["M", "N", "CHỮ M", "CHỮ N", "M.", "N."]:
        print(f"Match Word: sid={sid}, word='{word}', desc='{desc}'")

# Tìm các video bảng chữ cái từ D0489 đến D0519
print("\n=== CÁC VIDEO BẢNG CHỮ CÁI XUNG QUANH D0504, D0505 (D0500 - D0510) ===")
for row in q_dict:
    sid = str(row[0]) if len(row) > 0 else ""
    word = str(row[1]) if len(row) > 1 else ""
    desc = str(row[3]) if len(row) > 3 else ""
    if any(sid.startswith(f"D050{i}") for i in range(10)) or any(sid.startswith(f"D051{i}") for i in range(10)):
        print(f"  {sid:8} | {word:10} | {desc[:60]}")

