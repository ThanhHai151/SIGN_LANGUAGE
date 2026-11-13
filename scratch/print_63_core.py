import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/sstcn_102_dump.json", encoding="utf-8") as f:
    dump = json.load(f)

core = dump[39:]
print(f"Index range: {core[0]['index']} to {core[-1]['index']} (Total: {len(core)} classes)")

print("| SSTCN index | Code label | Vietnamese display name | Category | QIPEDC Entry ID | Biến thể | QIPEDC Video URL |")
print("|:---:|:---|:---|:---|:---|:---:|:---|")
for c in core:
    vid = c['q_video_url'] or c['video_url_meta']
    print(f"| {c['index']} | `{c['term']}` | {c['label']} | {c['category']} | `{c['q_entry_id']}` | {c['variants_count']} | {vid} |")

