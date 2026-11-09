"""
Tìm kiếm xem F, J, W, Z có đang được sử dụng ở đâu trong toàn bộ codebase.
"""
import os
import re

WORKSPACE = "d:/SIGN_LANGUAGE"

def search_fjwz():
    patterns = {
        "F": [r'"F"', r"'F'"],
        "J": [r'"J"', r"'J'"],
        "W": [r'"W"', r"'W'"],
        "Z": [r'"Z"', r"'Z'"],
    }
    
    results = {k: [] for k in patterns}
    
    for dirpath, _, filenames in os.walk(WORKSPACE):
        if any(skip in dirpath for skip in [".git", "node_modules", ".venv", "dist", ".gemini", "__pycache__"]):
            continue
        for f in filenames:
            if f.endswith((".py", ".jsx", ".js", ".json")):
                p = os.path.join(dirpath, f)
                rel = os.path.relpath(p, WORKSPACE)
                try:
                    with open(p, "r", encoding="utf-8") as fp:
                        content = fp.read()
                        for char, pats in patterns.items():
                            for pat in pats:
                                if re.search(pat, content):
                                    results[char].append(rel)
                                    break
                except Exception:
                    pass
    return results

res = search_fjwz()
for k, files in res.items():
    print(f"\n=== KÝ TỰ {k} ({len(files)} files) ===")
    for f in sorted(set(files))[:15]:
        print(f"  - {f}")
    if len(files) > 15:
        print(f"  ... và {len(files)-15} files khác")

