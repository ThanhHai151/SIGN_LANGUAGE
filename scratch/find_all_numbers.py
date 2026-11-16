import sys
import json
import re

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

entries = data['entries']
print(f"Tổng số entries: {len(entries)}")

matches = []
for item in entries:
    word = item.get('word', '')
    desc = item.get('description', '')
    if re.search(r'(^số\b|\b\d+\b|hai mươi|ba mươi|bốn mươi|năm mươi|sáu mươi|bảy mươi|tám mươi|chín mươi|\bmười\b)', word.lower()):
        matches.append((item.get('id'), word, item.get('video_url', '')))

for k, word, vurl in matches:
    print(f"{k}: {word} -> {vurl}")

print(f"Tổng số mục tìm thấy liên quan đến số: {len(matches)}")
