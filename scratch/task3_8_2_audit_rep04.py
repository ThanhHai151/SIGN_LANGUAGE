import json
import sys

log_path = "backend/data/diagnostics/pilot_pilot_20261109_004312.jsonl"
records = []
with open(log_path, "r", encoding="utf-8") as f:
    for i, line in enumerate(f):
        line = line.strip()
        if not line: continue
        r = json.loads(line)
        r["line_no"] = i + 1
        records.append(r)

rep04_recs = [r for r in records if r.get("trial_id") and "rep04" in r.get("trial_id")]
print(f"Total rep04 records: {len(rep04_recs)}")

for r in rep04_recs:
    line_no = r["line_no"]
    fidx = r.get("frame_idx")
    exp = r.get("expected_sign")
    geom = r.get("geometric", {}).get("letter")
    final = r.get("final_sign")
    conf = r.get("final_confidence")
    confirmed = r.get("confirmed")
    src = r.get("prediction_source")
    idx_s = r.get("geometric", {}).get("features", {}).get("idx_straight")
    d_ti = r.get("geometric", {}).get("features", {}).get("d_TI")
    print(f"line={line_no:03d} | fidx={fidx:02d} | exp={exp} | geom={str(geom):<4} | final={str(final):<7} | conf={conf:.2f} | confirmed={str(confirmed):<5} | src={src:<10} | idx_s={idx_s:.2f} | d_ti={d_ti:.2f}")

