import json, os

if os.path.exists("backend/data/sign_descriptions.json"):
    with open("backend/data/sign_descriptions.json", "r", encoding="utf-8") as f:
        desc = json.load(f)
    for k in ["P", "Q", "R", "X", "p", "q", "r", "x"]:
        if k in desc:
            print(f"sign_descriptions[{k}]: {desc[k]}")

# Also check ALL_SIGNS in vsl_dictionary.py
from backend.models.vsl_dictionary import ALL_SIGNS
for k in ["P", "Q", "R", "X"]:
    if k in ALL_SIGNS:
        print(f"ALL_SIGNS[{k}]: {ALL_SIGNS[k]}")
