"""
Script khảo sát Frontend cho bảng chữ cái và ghép ký tự.
"""
import re
import os

WORKSPACE = "d:/SIGN_LANGUAGE"

def search_frontend(pattern, root="src"):
    matches = []
    regex = re.compile(pattern)
    for dirpath, _, filenames in os.walk(os.path.join(WORKSPACE, root)):
        for f in filenames:
            if f.endswith((".jsx", ".js", ".tsx", ".ts", ".vue", ".html")):
                path = os.path.join(dirpath, f)
                try:
                    with open(path, "r", encoding="utf-8") as fp:
                        for i, line in enumerate(fp):
                            if regex.search(line):
                                rel = os.path.relpath(path, WORKSPACE)
                                matches.append((rel, i + 1, line.strip()))
                except Exception:
                    pass
    return matches

print("=== SEARCH REGEX / LETTER FILTER IN FRONTEND ===")
for m in search_frontend(r"ALLOWED|filter|regex|test\(|replace|mode.*letter|char|alphabet"):
    if any(k in m[2].lower() for k in ["letter", "regex", "test", "alphabet", "mode"]):
        print(f"{m[0]}:{m[1]}: {m[2]}")

