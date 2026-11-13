import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/sstcn_102_dump.json", encoding="utf-8") as f:
    dump = json.load(f)

with open("backend/data/vsl_lexicon.json", encoding="utf-8") as f:
    lex = json.load(f)

by_sstcn = {e["sstcn_index"]: e for e in lex["entries"] if e.get("sstcn_index") is not None}

print(f"Total entries in lexicon linked to sstcn: {len(by_sstcn)}")

for i in range(39, 102):
    e = by_sstcn.get(i)
    meta = dump[i]
    term = meta["term"]
    if not e:
        print(f"MISSING ENTRY in lexicon for index {i} ({term})")
        continue
    
    # Check word match
    word = e["word"]
    variants = e.get("variants", [])
    v_codes = [v.get("id") or v.get("code") or v for v in variants]
    print(f"[{i:3d}] Term: \"{term}\" | Lexicon Word: \"{word}\" | ID: {e['id']} | Main Video: {e.get('video_url')} | Variants ({len(variants)}): {v_codes}")

