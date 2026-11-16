import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

search_dirs = ['docs', 'backend', 'src']
keywords = ['số 10', 'so 10', 'hàng chục', 'hang chuc', 'hai mươi', 'hai muoi', 'digit_10', 'number_10']

print("Tìm kiếm trong codebase:")
for s_dir in search_dirs:
    if not os.path.exists(s_dir):
        continue
    for root, _, files in os.walk(s_dir):
        for f in files:
            if f.endswith(('.py', '.md', '.json', '.jsx', '.js', '.txt')):
                p = os.path.join(root, f)
                try:
                    text = open(p, 'r', encoding='utf-8', errors='ignore').read()
                    for kw in keywords:
                        if kw in text.lower():
                            print(f"  Match '{kw}' in: {p}")
                            break
                except Exception:
                    pass

