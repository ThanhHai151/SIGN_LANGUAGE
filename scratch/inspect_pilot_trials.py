import json
from collections import Counter

def inspect_file(fname):
    print("=" * 60)
    print(f"FILE: {fname}")
    print("=" * 60)
    with open(fname, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    
    # Group by trial_id
    by_trial = {}
    for idx, d in enumerate(lines):
        tid = d.get("trial_id")
        by_trial.setdefault(tid, []).append((idx, d))
    
    for tid, records in by_trial.items():
        print(f"\n--- Trial ID: {tid} ({len(records)} frames) ---")
        signs = Counter(r[1].get("final_sign") for r in records)
        sources = Counter(r[1].get("prediction_source") for r in records)
        geom_statuses = Counter(r[1].get("geometric", {}).get("status") for r in records)
        geom_letters = Counter(r[1].get("geometric", {}).get("letter") for r in records)
        print(f"  Final Signs: {dict(signs)}")
        print(f"  Sources:     {dict(sources)}")
        print(f"  Geom Status: {dict(geom_statuses)}")
        print(f"  Geom Letter: {dict(geom_letters)}")
        if records:
            t_start = records[0][1].get("timestamp")
            t_end = records[-1][1].get("timestamp")
            print(f"  Timestamp:   {t_start} -> {t_end}")

inspect_file("backend/data/diagnostics/pilot_pilot_20261105_2024.jsonl")
inspect_file("backend/data/diagnostics/pilot_pilot_20261105_2032.jsonl")

