import sys
import json

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

with open('backend/data/sources/qipedc_dictionary.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for item in data:
    if any(item[0] == cid or item[1] in ['10', '40', '80', '90', 'mười', 'bốn mươi', 'tám mươi', 'chín mươi'] for cid in ['D0539', 'D0544', 'D0545', 'D0546N']):
        print(item)

