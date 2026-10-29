import sys
import json
import re
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

tens_words = ['hai mươi', 'ba mươi', 'năm mươi', 'sáu mươi', 'bảy mươi', '20', '30', '50', '60', '70']

print("Tìm kiếm trong qipedc_dictionary.json:")
q_data = json.load(open('backend/data/sources/qipedc_dictionary.json', encoding='utf-8'))
for item in q_data:
    code = item[0]
    word = item[1]
    for tw in tens_words:
        if tw == word.lower() or word.lower() == f"số {tw}":
            print(f"  Exact match in qipedc_dictionary: {item}")
        elif tw in word.lower() and ('mươi' in word.lower() or re.search(r'\b(20|30|50|60|70)\b', word)):
            # compound phrases
            pass

print("\nTìm kiếm mã D054* và D055* trong qipedc_dictionary.json:")
for item in q_data:
    code = item[0]
    if re.match(r'^D05[345]\d', code):
        print(f"  {item}")

