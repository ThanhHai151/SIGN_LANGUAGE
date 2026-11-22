import os
import re
import json
from pathlib import Path

def analyze_app():
    app_text = Path("backend/app.py").read_text(encoding="utf-8")
    routes = re.findall(r'@app\.(post|get|put|delete)\(["\']([^"\']+)["\']', app_text)
    print("=== ROUTES IN backend/app.py ===")
    for method, path in routes:
        print(f"  {method.upper():6} {path}")

def analyze_qipedc_dict():
    p = Path("backend/data/sources/qipedc_dictionary.json")
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        print("\n=== QIPEDC DICTIONARY JSON ===")
        print(f"Total entries: {len(data)}")
        if isinstance(data, list) and len(data) > 0:
            print("First item sample:", data[0])
            # Check fields
            keys = set()
            video_count = 0
            words = set()
            for item in data:
                keys.update(item.keys())
                w = item.get("word") or item.get("gloss") or item.get("title")
                if w: words.add(w)
                v = item.get("video") or item.get("video_url") or item.get("file")
                if v: video_count += 1
            print("All keys present:", keys)
            print(f"Words: {len(words)}, Videos: {video_count}")
        elif isinstance(data, dict):
            print("Dict keys sample:", list(data.keys())[:10])
            first_val = next(iter(data.values()))
            print("First val sample:", first_val)
            print("Total entries:", len(data))

def analyze_lexicon():
    p = Path("backend/models/lexicon.py")
    print("\n=== backend/models/lexicon.py ===")
    lines = p.read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines[:50]):
        print(f"{i+1:3}: {line}")

if __name__ == "__main__":
    analyze_app()
    analyze_qipedc_dict()
    analyze_lexicon()
