import os
import sys
import re

sys.path.insert(0, ".")
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

patterns = [
    re.compile(r'D0490B'),
    re.compile(r'D0491B'),
    re.compile(r'D0497B'),
    re.compile(r'D0507B'),
    re.compile(r'D0508N'),
    re.compile(r'D0515N'),
    re.compile(r'D0495'),
    re.compile(r'Miền Bắc', re.IGNORECASE),
    re.compile(r'Bắc', re.IGNORECASE),
]

scan_dirs = ['src', 'backend']
ignore_dirs = ['node_modules', '.git', '.venv', 'temp_frames', 'media_cache', '__pycache__', 'scratch']

results = []

for sdir in scan_dirs:
    for root, dirs, files in os.walk(sdir):
        # filter dirs
        dirs[:] = [d for d in dirs if d not in ignore_dirs]
        for f in files:
            if not f.endswith(('.py', '.jsx', '.js', '.json', '.md', '.html')):
                continue
            fpath = os.path.join(root, f)
            try:
                with open(fpath, 'r', encoding='utf-8') as fp:
                    for lno, line in enumerate(fp, 1):
                        for p in patterns[:7]:  # Check video codes first
                            if p.search(line):
                                results.append((fpath, lno, p.pattern, line.strip()[:100]))
            except Exception:
                pass

print(f"Total occurrences of video IDs: {len(results)}")
for fpath, lno, pat, ltxt in results:
    print(f"{fpath}:{lno} [{pat}] -> {ltxt}")

