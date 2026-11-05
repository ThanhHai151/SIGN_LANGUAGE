import sys
sys.path.insert(0, ".")
import json
from backend.vision.alphabet_classifier import alphabet_classifier

def test_frames(name, lines):
    print(f"\n==========================================")
    print(f"EVALUATING: {name} ({len(lines)} frames)")
    print(f"==========================================")
    res_counts = {}
    for i, d in enumerate(lines):
        feat = d.get("geometric", {}).get("features")
        if not feat:
            res_counts["NO_FEAT"] = res_counts.get("NO_FEAT", 0) + 1
            continue
        matches = alphabet_classifier.evaluate_rules(feat)
        logged_final = d.get("final_sign")
        logged_geom = d.get("geometric", {}).get("letter")
        
        match_letters = [m[0] for m in matches] if matches else ["NO_MATCH"]
        key = f"logged={logged_final} -> eval={match_letters}"
        res_counts[key] = res_counts.get(key, 0) + 1
    for k, v in sorted(res_counts.items()):
        print(f"  {k}: {v}")

with open("backend/data/diagnostics/pilot_pilot_20261105_2024.jsonl", "r", encoding="utf-8") as f:
    lines_2024 = [json.loads(line) for line in f if line.strip()]

with open("backend/data/diagnostics/pilot_pilot_20261105_2032.jsonl", "r", encoding="utf-8") as f:
    lines_2032 = [json.loads(line) for line in f if line.strip()]

rep02_2024 = [d for d in lines_2024 if d.get("trial_id") == "pilot_20261105_2024_I_rep02"]
test_frames("pilot_20261105_2024_I_rep02", rep02_2024)

rep01_2032 = [d for d in lines_2032 if d.get("trial_id") == "pilot_20261105_2032_I_rep01"]
test_frames("pilot_20261105_2032_I_rep01", rep01_2032)

none_2032 = [d for d in lines_2032 if d.get("trial_id") is None]
test_frames("pilot_20261105_2032 (trial_id=None, frames 257-329)", none_2032)

