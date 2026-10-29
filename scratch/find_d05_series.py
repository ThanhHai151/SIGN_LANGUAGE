import os
import sys
import json
import re

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

lex_path = 'backend/data/vsl_lexicon.json'
with open(lex_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lex = data.get('entries', [])

# Tìm tất cả các mục từ có url chứa D05
d05_entries = []
for e in lex:
    url = e.get('video_url', '') or ''
    m = re.search(r'(D05\d+[A-Z]?)', url)
    if m:
        d05_entries.append((m.group(1), e))

# Sắp xếp theo mã video
d05_entries.sort(key=lambda x: x[0])

print(f"Tìm thấy {len(d05_entries)} mục từ dãy D05xx trong QIPEDC:")
for vid, e in d05_entries:
    print(f"  Video: {vid:<7} | Word: {e.get('word'):<25} | ID: {e.get('id'):<20} | URL: {e.get('video_url')}")

