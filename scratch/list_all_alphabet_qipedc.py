"""
In toàn bộ các mục bảng chữ cái trong qipedc_dictionary.json từ D0488 đến D0520.
"""
import os
import json

WORKSPACE = "d:/SIGN_LANGUAGE"
qipedc_path = os.path.join(WORKSPACE, "backend/data/sources/qipedc_dictionary.json")

with open(qipedc_path, "r", encoding="utf-8") as fp:
    q_dict = json.load(fp)

print("ID       | Word       | Region / Dialect info")
print("-" * 60)
for row in q_dict:
    sid = str(row[0]) if len(row) > 0 else ""
    word = str(row[1]) if len(row) > 1 else ""
    desc = str(row[3]) if len(row) > 3 else ""
    # Check if id between D0489 and D0520
    if sid.startswith("D048") or sid.startswith("D049") or sid.startswith("D050") or sid.startswith("D051"):
        suffix = sid[5:] if len(sid) > 5 else "None (Dùng chung toàn quốc)"
        print(f"{sid:<8} | {word:<10} | Hậu tố: {suffix:<10} | {desc[:40]}")

