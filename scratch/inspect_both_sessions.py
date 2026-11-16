import json

def inspect_lines(fname, start, end):
    print(f"=== {fname} lines {start} to {end} ===")
    with open(fname, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]
    for i in range(start, min(end, len(lines))):
        d = lines[i]
        ts = d.get("timestamp")
        tid = d.get("trial_id")
        fidx = d.get("frame_idx")
        exp = d.get("expected_sign")
        fs = d.get("final_sign")
        conf = d.get("final_confidence")
        geom = d.get("geometric", {})
        gl = geom.get("letter")
        gs = geom.get("status")
        matched_cand = geom.get("matched_candidates")
        print(f"l{i:03d} | ts={ts:.2f} | trial={tid} | f={fidx} | exp={exp} | sign={fs:7s}({conf:.2f}) | geom={gs}:{gl} | cand={matched_cand}")

inspect_lines("backend/data/diagnostics/pilot_pilot_20261105_2032.jsonl", 25, 65)

