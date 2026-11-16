import json

def analyze_trial(filename, target_trial=None):
    print(f"\n=======================================================")
    print(f"ANALYZING: {filename} (target={target_trial})")
    print(f"=======================================================")
    records = []
    with open(filename, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if not line.strip():
                continue
            d = json.loads(line)
            tid = d.get("trial_id")
            if target_trial is None or tid == target_trial or target_trial in str(tid):
                records.append((idx, d))
    
    print(f"Total matching records: {len(records)}")
    if not records:
        return
    
    signs = {}
    for idx, d in records:
        fs = d.get("final_sign")
        gl = d.get("geometric", {}).get("letter")
        cand = d.get("geometric", {}).get("matched_candidates", [])
        key = f"final={fs} | geom={gl} | cand={cand}"
        signs[key] = signs.get(key, 0) + 1
    
    print("Distribution:")
    for k, v in signs.items():
        print(f"  {k}: {v} frames")
        
    return records

r_2024_rep02 = analyze_trial("backend/data/diagnostics/pilot_pilot_20261105_2024.jsonl", "pilot_20261105_2024_I_rep02")
r_2032_rep01 = analyze_trial("backend/data/diagnostics/pilot_pilot_20261105_2032.jsonl", "pilot_20261105_2032_I_rep01")
r_2032_none = analyze_trial("backend/data/diagnostics/pilot_pilot_20261105_2032.jsonl", None)

