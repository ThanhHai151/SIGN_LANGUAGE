import json
import re
from pathlib import Path

with open('backend/data/vsl_lexicon.json', 'r', encoding='utf-8') as f:
    lex = json.load(f)

entries = lex['entries']
print(f"Total entries: {len(entries)}")

non_standard = []
b_suffix = []
t_suffix = []
n_suffix = []
national = []

for e in entries:
    eid = e.get('id', '')
    region = e.get('region')
    vurl = e.get('video_url') or ''
    m = re.search(r'/([^/]+)\.mp4', vurl)
    vid = m.group(1) if m else ''
    
    # Check if vid ends with B or T
    if re.search(r'B$', vid, re.IGNORECASE) and not vid.startswith('sign_'):
        b_suffix.append((eid, vid, region, vurl))
    elif re.search(r'T$', vid, re.IGNORECASE) and not vid.startswith('sign_'):
        t_suffix.append((eid, vid, region, vurl))
    elif re.match(r'^[A-Z]\d+N$', vid):
        n_suffix.append((eid, vid, region, vurl))
    elif re.match(r'^[A-Z]\d+$', vid):
        national.append((eid, vid, region, vurl))
    else:
        non_standard.append((eid, vid, region, vurl))

print(f"National (no suffix): {len(national)}")
print(f"N suffix (South): {len(n_suffix)}")
print(f"B suffix (North): {len(b_suffix)}")
print(f"T suffix (Central): {len(t_suffix)}")
print(f"Non-standard / other: {len(non_standard)}")

print("\nNon-standard sample (first 45):")
for item in non_standard[:45]:
    print(f"  id={item[0]}, vid={item[1]}, region={item[2]}, url={item[3][:60]}")

# Also check variants in all entries
b_vars = []
t_vars = []
n_vars = []
nat_vars = []
other_vars = []

for e in entries:
    eid = e.get('id')
    for v in e.get('variants', []):
        v_url = v.get('url') or ''
        v_reg = v.get('region')
        m = re.search(r'/([^/]+)\.mp4', v_url)
        v_vid = m.group(1) if m else ''
        if re.search(r'B$', v_vid, re.IGNORECASE) and not v_vid.startswith('sign_'):
            b_vars.append((eid, v_vid, v_reg, v_url))
        elif re.search(r'T$', v_vid, re.IGNORECASE) and not v_vid.startswith('sign_'):
            t_vars.append((eid, v_vid, v_reg, v_url))
        elif re.match(r'^[A-Z]\d+N$', v_vid):
            n_vars.append((eid, v_vid, v_reg, v_url))
        elif re.match(r'^[A-Z]\d+$', v_vid):
            nat_vars.append((eid, v_vid, v_reg, v_url))
        else:
            other_vars.append((eid, v_vid, v_reg, v_url))

print(f"\n--- VARIANTS ---")
print(f"Variants National: {len(nat_vars)}")
print(f"Variants N: {len(n_vars)}")
print(f"Variants B: {len(b_vars)}")
print(f"Variants T: {len(t_vars)}")
print(f"Variants other: {len(other_vars)}")

if b_vars:
    print("Sample B variants:", b_vars[:5])
if t_vars:
    print("Sample T variants:", t_vars[:5])
