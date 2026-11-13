import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.models.vsl_102_dictionary import SIGN_DICT_102, SIGN_METADATA_102
from backend.models.lexicon import load_lexicon, get_entry_by_sstcn

print(f"Total entries in SIGN_DICT_102: {len(SIGN_DICT_102)}")
print(f"Total entries in SIGN_METADATA_102: {len(SIGN_METADATA_102)}")

lex = load_lexicon()
print(f"Total entries in vsl_lexicon.json: {len(lex['entries'])}")
print(f"SSTCN linked in vsl_lexicon: {sum(1 for e in lex['entries'] if e.get('sstcn_index') is not None)}")

# List all 102 classes with their QIPEDC link and metadata
results_102 = []
for i in range(102):
    term = SIGN_DICT_102.get(i)
    meta = SIGN_METADATA_102[i] if i < len(SIGN_METADATA_102) else {}
    q_entry = get_entry_by_sstcn(i)
    
    entry_id = q_entry.get("id") if q_entry else None
    q_word = q_entry.get("word") if q_entry else None
    q_video = q_entry.get("video_url") if q_entry else None
    variants = q_entry.get("variants", []) if q_entry else []
    
    results_102.append({
        "index": i,
        "term": term,
        "label": meta.get("label", term),
        "category": meta.get("category", ""),
        "description": meta.get("description", ""),
        "video_url_meta": meta.get("video_url", ""),
        "q_entry_id": entry_id,
        "q_word": q_word,
        "q_video_url": q_video,
        "variants_count": len(variants),
        "variants": variants
    })

with open("scratch/sstcn_102_dump.json", "w", encoding="utf-8") as f:
    json.dump(results_102, f, ensure_ascii=False, indent=2)

print("Saved scratch/sstcn_102_dump.json successfully!")

