import os
import json
import glob

diag_dir = "backend/data/diagnostics"
files = glob.glob(os.path.join(diag_dir, "pilot_eval_*.json"))
print(f"Found {len(files)} pilot eval JSON files:")

for f in sorted(files):
    with open(f, encoding="utf-8") as fp:
        try:
            d = json.load(fp)
            print(f"\n--- File: {os.path.basename(f)} ---")
            print("Keys:", list(d.keys())[:8])
            print("Session ID:", d.get("sessionId") or d.get("session_id"))
            summary = d.get("summary", {})
            print("Summary:", summary)
            letters = d.get("letters", {})
            print("Tested Letters:", list(letters.keys()))
            for lk, ldata in letters.items():
                print(f"  Letter {lk}: status={ldata.get('status')}, accuracy={ldata.get('accuracy_pct')}%")
        except Exception as e:
            print("Error reading:", f, e)

