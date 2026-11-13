import json, sys
sys.stdout.reconfigure(encoding='utf-8')

# 30 benchmark indices in order S01-S30
BENCH = [
    ("S01", 52), ("S02", 53), ("S03", 51), ("S04", 60),
    ("S05", 55), ("S06", 56), ("S07", 58), ("S08", 94),
    ("S09", 44), ("S10", 67), ("S11", 68), ("S12", 69),
    ("S13", 71), ("S14", 73), ("S15", 75), ("S16", 76),
    ("S17", 77), ("S18", 80), ("S19", 81), ("S20", 82),
    ("S21", 83), ("S22", 84), ("S23", 91), ("S24", 92),
    ("S25", 86), ("S26", 95), ("S27", 98), ("S28", 100),
    ("S29", 96), ("S30", 101),
]

# Load SIGN_METADATA_102
import importlib.util
spec = importlib.util.spec_from_file_location("vsl_102", "backend/models/vsl_102_dictionary.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
METADATA = {e["index"]: e for e in mod.SIGN_METADATA_102}

# Load lexicon
with open("backend/data/vsl_lexicon.json", encoding="utf-8") as f:
    lexicon_data = json.load(f)

# Build sstcn_index -> entries map
lex_by_idx = {}
for entry in lexicon_data["entries"]:
    idx = entry.get("sstcn_index")
    if idx is not None:
        if idx not in lex_by_idx:
            lex_by_idx[idx] = []
        lex_by_idx[idx].append(entry)

# Output
result = {}
for sid, idx in BENCH:
    meta = METADATA.get(idx, {})
    lex_entries = lex_by_idx.get(idx, [])
    result[sid] = {
        "benchmark_id": sid,
        "sstcn_index": idx,
        "code_label": meta.get("label", "???"),
        "term": meta.get("term", "???"),
        "category": meta.get("category", "???"),
        "description": meta.get("description", ""),
        "supabase_video_url": meta.get("video_url", ""),
        "image_url": meta.get("image_url", ""),
        "lexicon_entries": lex_entries,
    }

print(json.dumps(result, ensure_ascii=False, indent=2))

