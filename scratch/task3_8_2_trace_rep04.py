import json
from collections import deque, Counter

log_path = "backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl"
records = []
with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        r["line_no"] = i + 1
        records.append(r)

# Find rep04 start and end
rep04_indices = [i for i, r in enumerate(records) if r.get("trial_id") and "rep04" in r.get("trial_id")]
start_idx = rep04_indices[0]
end_idx = rep04_indices[-1]

print(f"rep04 is at records index {start_idx} to {end_idx} (lines {records[start_idx]['line_no']} to {records[end_idx]['line_no']})")

# Let's see the 15 records before rep04
print("\n--- 10 RECORDS BEFORE REP04 ---")
for i in range(max(0, start_idx - 10), start_idx):
    r = records[i]
    print(f"line={r['line_no']:03d} | exp={r.get('expected_sign')} | trial={r.get('trial_id')} | final={r.get('final_sign')} | conf={r.get('final_confidence'):.2f} | confirmed={r.get('confirmed')}")

print("\n--- ALL REP04 RECORDS (LINE BY LINE AUDIT) ---")
# Let's inspect the recorded fields in JSONL for each frame of rep04
for i in rep04_indices:
    r = records[i]
    fidx = r.get("frame_idx")
    line_no = r["line_no"]
    geom_letter = r.get("geometric", {}).get("letter")
    final_sign = r.get("final_sign")
    final_conf = r.get("final_confidence")
    confirmed = r.get("confirmed")
    ts = r.get("timestamp")
    print(f"line={line_no:03d} | fidx={fidx:02d} | ts={ts:.4f} | geom={str(geom_letter):<4} | final={str(final_sign):<7} | conf={final_conf:.2f} | confirmed={str(confirmed):<5}")

