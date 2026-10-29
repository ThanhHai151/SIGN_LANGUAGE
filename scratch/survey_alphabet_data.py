import sys
import os
import json

sys.stdout.reconfigure(encoding='utf-8')

lexicon_path = r"D:\SIGN_LANGUAGE\backend\data\vsl_lexicon.json"
video_dir = r"D:\SIGN_LANGUAGE\backend\data\media_cache\videos"

with open(lexicon_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

entries = data.get('entries', [])
alphabet_items = [item for item in entries if item.get('category') == 'Bảng chữ cái']

print(f"Tổng số mục 'Bảng chữ cái' trong vsl_lexicon.json: {len(alphabet_items)}")

available = []
missing = []

for item in alphabet_items:
    word = item.get('word', '')
    url = item.get('video_url', '')
    vid_name = os.path.basename(url) if url else ''
    local_path = os.path.join(video_dir, vid_name)
    exists = os.path.exists(local_path)
    size = os.path.getsize(local_path) if exists else 0
    info = {
        'id': item.get('id'),
        'word': word,
        'video_file': vid_name,
        'url': url,
        'exists': exists,
        'size': size,
        'sstcn_index': item.get('sstcn_index'),
        'how_to': item.get('how_to', '')
    }
    if exists:
        available.append(info)
    else:
        missing.append(info)

print(f"\n1. Video CÓ SẴN trên đĩa cục bộ: {len(available)} video")
for a in available:
    print(f"  - [{a['id']}] '{a['word']}': {a['video_file']} ({a['size']:,} bytes) | sstcn: {a['sstcn_index']} | {a['how_to'][:60]}...")

print(f"\n2. Video THIẾU trên đĩa cục bộ: {len(missing)} video")
for m in missing:
    print(f"  - [{m['id']}] '{m['word']}': {m['video_file']} ({m['url']})")
