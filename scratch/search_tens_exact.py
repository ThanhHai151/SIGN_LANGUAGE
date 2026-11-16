import os
import sys
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

lex_path = 'backend/data/vsl_lexicon.json'
with open(lex_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

lex = data.get('entries', [])

tens = ['10', '20', '30', '40', '50', '60', '70', '80', '90',
        'mười', 'hai mươi', 'ba mươi', 'bốn mươi', 'năm mươi', 'sáu mươi', 'bảy mươi', 'tám mươi', 'chín mươi']

print("Tìm kiếm mục từ chính xác cho các số hàng chục:")
for target in tens:
    matches = [e for e in lex if e.get('word', '').strip().lower() == target or e.get('id', '') == f"q-{target}"]
    print(f"  Target '{target}': {len(matches)} mục từ tìm thấy")
    for m in matches:
        print(f"     -> ID: {m.get('id')}, Word: '{m.get('word')}', Video: {m.get('video_url')}")

