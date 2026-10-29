import os
import sys
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

lex_path = 'backend/data/vsl_lexicon.json'
if not os.path.exists(lex_path):
    print("vsl_lexicon.json không tồn tại!")
    sys.exit(0)

with open(lex_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lex = data.get('entries', [])
print(f"Tổng số mục từ trong lexicon: {len(lex)}")

keywords = ['số', 'mười', 'chục', 'trăm', '10', '20', '30', '40', '50', '60', '70', '80', '90',
            'mươi', 'hai mươi', 'ba mươi', 'bốn mươi', 'năm mươi', 'sáu mươi', 'bảy mươi', 'tám mươi', 'chín mươi']

found = []
for entry in lex:
    w = str(entry.get('word', '')).lower()
    eid = str(entry.get('id', '')).lower()
    for kw in keywords:
        if kw in w or kw in eid:
            found.append(entry)
            break

print(f"Tìm thấy {len(found)} mục từ khớp từ khóa số/hàng chục:")
for e in found:
    print(f"  ID: {e.get('id')} | Word: {e.get('word')} | Video: {e.get('video_id')} | File: {e.get('video_file')} | URL: {e.get('video_url')}")
