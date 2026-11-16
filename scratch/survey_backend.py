"""
Script khảo sát chi tiết Backend về các định nghĩa chữ cái, pipeline, model, Diacritic FSM.
"""
import re
import os

WORKSPACE = "d:/SIGN_LANGUAGE"

def search_files(pattern, root="backend"):
    matches = []
    regex = re.compile(pattern)
    for dirpath, _, filenames in os.walk(os.path.join(WORKSPACE, root)):
        for f in filenames:
            if f.endswith(".py"):
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

print("=== SEARCH ALLOWED_LETTERS ===")
for m in search_files(r"ALLOWED_LETTERS"):
    print(f"{m[0]}:{m[1]}: {m[2]}")

print("\n=== SEARCH VIETNAMESE_ALPHABET or VN_29 ===")
for m in search_files(r"VIETNAMESE_ALPHABET|VN_29|ALPHABET_29"):
    print(f"{m[0]}:{m[1]}: {m[2]}")

print("\n=== SEARCH BASE_HANDSHAPES_22 ===")
for m in search_files(r"BASE_HANDSHAPES_22"):
    print(f"{m[0]}:{m[1]}: {m[2]}")

print("\n=== SEARCH DiacriticStateMachine ===")
for m in search_files(r"DiacriticStateMachine"):
    print(f"{m[0]}:{m[1]}: {m[2]}")

print("\n=== SEARCH AlphabetStaticHead / MLP classes ===")
for m in search_files(r"class AlphabetStaticHead|STATIC_CLASSES|MLP"):
    print(f"{m[0]}:{m[1]}: {m[2]}")

