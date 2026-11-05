import json

with open("backend/data/diagnostics/pilot_pilot_20261105_2032.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

for idx, d in enumerate(lines[:35]):
    feat = d.get("geometric", {}).get("features", {})
    fs = d.get("final_sign")
    i_fail = d.get("geometric", {}).get("i_rule_diagnostics", {}).get("failed_conditions", [])
    d_iw = feat.get("d_IW")
    d_mw = feat.get("d_MW")
    d_rw = feat.get("d_RW")
    d_pw = feat.get("d_PW")
    print(f"l{idx:02d} | fs={fs:7s} | d_IW={d_iw} | d_MW={d_mw} | d_RW={d_rw} | d_PW={d_pw} | i_fail={i_fail}")

