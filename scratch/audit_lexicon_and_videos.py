import os
import sys
import json
import glob

sys.path.insert(0, ".")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 1. Inspect vsl_lexicon.json
lex_path = "backend/data/vsl_lexicon.json"
entries = []
if os.path.exists(lex_path):
    with open(lex_path, encoding='utf-8') as f:
        lex_data = json.load(f)
    entries = lex_data.get('entries', [])
    print(f"Loaded vsl_lexicon.json: {len(entries)} entries")

letters = ['Ă', 'Â', 'Ê', 'Ô', 'Ơ', 'Ư', 'Đ']

print("\n--- LEXICON SEARCH FOR 7 LETTERS ---")
for v in entries:
    eid = str(v.get('id', '')).strip()
    word = str(v.get('word', '')).strip()
    how_to = str(v.get('how_to', '')).strip()
    source = str(v.get('source', '')).strip()
    category = str(v.get('category', '')).strip()
    variants = v.get('variants', [])
    
    # Check if this entry matches one of the 7 letters
    is_match = False
    for ch in letters:
        if word.upper() == ch or word.lower() == f"chữ {ch.lower()}" or eid.lower() == f"q-{ch.lower()}" or eid.lower().startswith(f"q-{ch.lower()}-"):
            is_match = True
            break
        if eid.startswith("q-") and ch.lower() in eid.lower():
            is_match = True
            break

    if is_match:
        print(f"ID: {eid:<15} | Word: {word:<8} | Source: {source:<20} | Cat: {category}")
        print(f"   How_to: {how_to}")
        print(f"   Video URL: {v.get('video_url')}")
        if variants:
            print(f"   Variants ({len(variants)}):")
            for var in variants:
                print(f"      Region: {var.get('region')} | Label: {var.get('label')} | URL: {var.get('url')}")
        print("-" * 80)

# 2. Check all video files in backend/data/media_cache/videos/ and subdirectories
print("\n--- VIDEO FILES IN REPOSITORY ---")
video_files = glob.glob("backend/data/**/videos/*.*", recursive=True) + glob.glob("backend/data/**/*.mp4", recursive=True)
print(f"Total video files found: {len(video_files)}")

# Check video files mentioning the codes D0490, D0491, D0495, D0497, D0507, D0508, D0515 or similar
target_codes = ["D0490", "D0491", "D0492", "D0493", "D0494", "D0495", "D0496", "D0497", "D0507", "D0508", "D0515", "D0516"]
matched_videos = []
for vf in video_files:
    fname = os.path.basename(vf)
    if any(code in fname for code in target_codes) or any(f"{ch}." in fname or f"{ch}_" in fname for ch in letters):
        matched_videos.append(vf)

for mv in sorted(matched_videos):
    print(f"  {mv}")

# 3. Check motion sequences in backend/data/alphabet_v2/motion_sequences/
print("\n--- MOTION SEQUENCES ---")
ms_files = glob.glob("backend/data/alphabet_v2/motion_sequences/*.json")
for mf in ms_files:
    fname = os.path.basename(mf)
    with open(mf, encoding='utf-8') as f:
        mdata = json.load(f)
    print(f"  {fname:<15} | seq_len={len(mdata.get('sequence', []))} | meta={mdata.get('metadata', mdata.get('video_id', 'no meta'))}")
