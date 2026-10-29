import os
import sys

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

cache_dir = 'backend/data/media_cache/videos'
files = os.listdir(cache_dir) if os.path.exists(cache_dir) else []
print(f"Tổng số video trong cache: {len(files)}")

d_nums = [f for f in files if any(f.startswith(f"D05{i}") for i in range(29, 60))]
print(f"Các video số D0529-D0555 đã có trong cache: {d_nums}")

