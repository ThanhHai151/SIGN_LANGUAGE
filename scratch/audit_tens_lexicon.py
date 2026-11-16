import sys
import json
import re
import os

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')

data = json.load(open('backend/data/vsl_lexicon.json', encoding='utf-8'))['entries']

targets = ['10', '20', '30', '40', '50', '60', '70', '80', '90']
for t in targets:
    found = []
    for item in data:
        w = item.get('word', '').strip()
        vid = item.get('video_url', '') or ''
        patterns = [f'^{t}$', f'^số {t}$']
        if t == '10': patterns.extend(['^mười$', '^số mười$'])
        elif t == '20': patterns.extend(['^hai mươi$', '^số hai mươi$'])
        elif t == '30': patterns.extend(['^ba mươi$', '^số ba mươi$'])
        elif t == '40': patterns.extend(['^bốn mươi$', '^số bốn mươi$'])
        elif t == '50': patterns.extend(['^năm mươi$', '^số năm mươi$'])
        elif t == '60': patterns.extend(['^sáu mươi$', '^số sáu mươi$'])
        elif t == '70': patterns.extend(['^bảy mươi$', '^số bảy mươi$'])
        elif t == '80': patterns.extend(['^tám mươi$', '^số tám mươi$'])
        elif t == '90': patterns.extend(['^chín mươi$', '^số chín mươi$'])

        for pat in patterns:
            if re.search(pat, w, re.IGNORECASE):
                found.append(item)
                break
    print(f"=== SỐ {t} (Tìm thấy {len(found)} mục):")
    for f in found:
        vid_path = ""
        vurl = f.get('video_url') or ''
        if vurl:
            v_basename = os.path.basename(vurl)
            local_cache = os.path.join('backend/data/media_cache/videos', v_basename)
            has_cache = os.path.exists(local_cache)
            vid_path = f"{vurl} (cache: {has_cache})"
        print(f"  id: {f.get('id')}, word: '{f.get('word')}', video: {vid_path}, sstcn_index: {f.get('sstcn_index')}, motion_id: {f.get('motion_id')}")

