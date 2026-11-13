import json
import sys

sys.stdout.reconfigure(encoding='utf-8')

with open("scratch/sstcn_102_dump.json", encoding="utf-8") as f:
    dump = json.load(f)

core = dump[39:]

print("--- CANDIDATE DETAILS FOR 63 CORE WORDS ---")
for c in core:
    print(f"[{c['index']}] {c['term']} ({c['category']})")
    print(f"   Desc: {c['description']}")
    print(f"   QIPEDC: {c['q_entry_id']} | Video: {c['q_video_url']}")

