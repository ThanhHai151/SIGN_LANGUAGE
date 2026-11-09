import os
import json
import glob

print("=== CHECKING VSL_LEXICON.JSON FOR M AND N ===")
with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    lex = json.load(f)

for e in lex.get('entries', []):
    w = e.get('word', '').lower()
    if w in ['m', 'n', 'chữ m', 'chữ n']:
        print(f"Word: '{e.get('word')}', Code: {e.get('code')}, Video URL: {e.get('video_url')}, How-to: {e.get('how_to')}")
        print(f"  Variants: {e.get('variants')}")

print("\n=== CHECKING ALPHABET_V2 EXTRACTION / METADATA ===")
meta_files = glob.glob('backend/data/alphabet_v2/**/*.json', recursive=True)
for mf in meta_files:
    if any(k in os.path.basename(mf).upper() for k in ['M.JSON', 'N.JSON', 'METADATA', 'LABELS']):
        print("Meta file:", mf)

print("\n=== CHECKING VIDEOS IN MEDIA_CACHE/VIDEOS ===")
all_videos = glob.glob('backend/data/media_cache/videos/*.mp4')
print(f"Total videos in media_cache/videos: {len(all_videos)}")
m_n_vids = [v for v in all_videos if any(x in os.path.basename(v) for x in ['0504', '0505', 'M', 'N'])]
print("Matching video candidates:", m_n_vids[:20])

