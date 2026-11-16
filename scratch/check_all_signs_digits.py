import os
import sys

sys.path.insert(0, os.path.abspath("."))
site_packages = os.path.abspath(".venv/Lib/site-packages")
if os.path.exists(site_packages) and site_packages not in sys.path:
    sys.path.insert(0, site_packages)

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

from backend.models.vsl_dictionary import ALL_SIGNS

print(f"Tổng số signs trong ALL_SIGNS: {len(ALL_SIGNS)}")

digits_found = []
for k, v in ALL_SIGNS.items():
    if v.get('type') == 'digit' or k.isdigit():
        digits_found.append((k, v))

print(f"Các ký hiệu type='digit' hoặc key là số trong ALL_SIGNS ({len(digits_found)}):")
for k, v in digits_found:
    print(f"  Key: {k:<5} | Label: {v.get('label'):<15} | Type: {v.get('type')}")
