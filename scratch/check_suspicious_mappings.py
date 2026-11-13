import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/sstcn_102_dump.json", encoding="utf-8") as f:
    dump = json.load(f)

with open("backend/data/vsl_lexicon.json", encoding="utf-8") as f:
    lex = json.load(f)

by_sstcn = {e["sstcn_index"]: e for e in lex["entries"] if e.get("sstcn_index") is not None}

print("=== CHECKING CONSISTENCY FOR ALL 63 CORE CLASSES (39..101) ===")
anomalies = []
for i in range(39, 102):
    meta = dump[i]
    e = by_sstcn.get(i)
    term = meta["term"]
    word = e["word"] if e else None
    
    # Check if term and word differ significantly
    if word and term.lower() != word.lower():
        anomalies.append({
            "index": i,
            "term": term,
            "word": word,
            "id": e["id"],
            "reason": f"Tên term ('{term}') và word trong lexicon ('{word}') khác nhau"
        })
    
    # Check if entry ID has strange naming
    if e and ("giong" in e["id"] or "-2" in e["id"]):
        anomalies.append({
            "index": i,
            "term": term,
            "word": word,
            "id": e["id"],
            "reason": f"Entry ID mang hậu tố đặc biệt '{e['id']}'"
        })

for a in anomalies:
    print(f"Index {a['index']:3d}: Term='{a['term']}' vs Word='{a['word']}' | ID={a['id']} | Note: {a['reason']}")

print(f"\nTotal potential anomalies / notable mappings: {len(anomalies)}")

