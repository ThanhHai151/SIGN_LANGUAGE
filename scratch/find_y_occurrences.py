import json

# Check both files
for fname in ["backend/data/diagnostics/pilot_pilot_20261105_2024.jsonl", "backend/data/diagnostics/pilot_pilot_20261105_2032.jsonl"]:
    print(f"================ {fname} ================")
    with open(fname, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            d = json.loads(line)
            geom = d.get("geometric", {})
            cand = geom.get("matched_candidates", [])
            fs = d.get("final_sign")
            gl = geom.get("letter")
            i_diag = geom.get("i_rule_diagnostics")
            # If Y is in cand or fs == Y or gl == Y
            if fs == "Y" or gl == "Y" or any(c[0] == "Y" for c in cand):
                print(f"Line {idx} (trial={d.get('trial_id')}, f={d.get('frame_idx')}): final={fs}, geom={gl}, cand={cand}")
                feat = geom.get("features", {})
                print(f"   d_PW={feat.get('d_PW')}, d_TW={feat.get('d_TW')}, d_TI={feat.get('d_TI')}, d_TM={feat.get('d_TM')}, thumb_lat={feat.get('thumb_lat')}, ang_TP={feat.get('ang_TP')}")
                if i_diag:
                    print(f"   i_diag failed: {i_diag.get('failed_conditions')}")

