import sys
sys.path.insert(0, '.')
import json
from pathlib import Path
from backend.models.vsl_102_dictionary import SIGN_DICT_102, SIGN_METADATA_102
from backend.models.lexicon import load_lexicon, get_entry_by_sstcn

lex = load_lexicon()
by_id = lex['by_id']

print(f"Total SIGN_METADATA_102: {len(SIGN_METADATA_102)}")
# 0-38: alphabet & digits
# 39-101: 63 words!

word_classes = [item for item in SIGN_METADATA_102 if int(item['index']) >= 39]
print(f"Total word classes in SSTCN 102: {len(word_classes)}")

# For each word class, let's see how get_entry_by_sstcn resolves it
# and what region / video_url it gets
q_count = 0
gt_count = 0
b_count = 0
t_count = 0
n_count = 0
nat_count = 0
unknown_count = 0

details = []
for item in word_classes:
    idx = int(item['index'])
    term = item['term']
    entry = get_entry_by_sstcn(idx)
    if entry:
        eid = entry.get('id', '')
        reg = entry.get('region')
        vurl = entry.get('video_url', '')
        if eid.startswith('q-'):
            q_count += 1
        elif eid.startswith('gt-'):
            gt_count += 1
        else:
            unknown_count += 1
            
        import re
        m = re.search(r'/([^/]+)\.mp4', vurl)
        vid = m.group(1) if m else ''
        if re.search(r'B$', vid, re.IGNORECASE) and not vid.startswith('sign_'):
            b_count += 1
            reg_type = 'B'
        elif re.search(r'T$', vid, re.IGNORECASE) and not vid.startswith('sign_'):
            t_count += 1
            reg_type = 'T'
        elif re.match(r'^[A-Z]\d+N$', vid):
            n_count += 1
            reg_type = 'N'
        elif re.match(r'^[A-Z]\d+$', vid):
            nat_count += 1
            reg_type = 'national'
        elif 'sign_' in vid:
            reg_type = 'GT_video'
        else:
            reg_type = 'other'
            
        details.append((idx, term, eid, reg, reg_type, vid))
    else:
        unknown_count += 1
        details.append((idx, term, None, None, 'NO_ENTRY', ''))

print(f"SSTCN 63 Word Classes resolution:")
print(f"  QIPEDC entries (q-): {q_count}")
print(f"  Giáo trình entries (gt-): {gt_count}")
print(f"  Unknown / no entry: {unknown_count}")
print(f"Video types:")
print(f"  National (no suffix): {nat_count}")
print(f"  N suffix (South): {n_count}")
print(f"  B suffix (North): {b_count}")
print(f"  T suffix (Central): {t_count}")

print("\nAll 36 GT classes in SSTCN:")
for d in details:
    if d[2] and d[2].startswith('gt-'):
        print(f"  idx={d[0]}, term={d[1]}, eid={d[2]}, region={d[3]}, reg_type={d[4]}, vid={d[5]}")

print("\nSample QIPEDC classes in SSTCN (first 10):")
for d in details:
    if d[2] and d[2].startswith('q-'):
        print(f"  idx={d[0]}, term={d[1]}, eid={d[2]}, region={d[3]}, reg_type={d[4]}, vid={d[5]}")
