import json

with open("backend/data/diagnostics/pilot_pilot_20261105_2024.jsonl", "r", encoding="utf-8") as f:
    lines = [json.loads(line) for line in f if line.strip()]

rep02 = [d for d in lines if d.get("trial_id") == "pilot_20261105_2024_I_rep02"]
print(f"Total frames in pilot_20261105_2024_I_rep02: {len(rep02)}")

for idx, d in enumerate(rep02):
    fidx = d.get("frame_idx")
    fs = d.get("final_sign")
    geom = d.get("geometric", {})
    gl = geom.get("letter")
    cand = geom.get("matched_candidates", [])
    feat = geom.get("features", {})
    
    d_pw = feat.get("d_PW")
    d_tw = feat.get("d_TW")
    d_ti = feat.get("d_TI")
    d_tm = feat.get("d_TM")
    thumb_lat = feat.get("thumb_lat")
    ang_tp = feat.get("ang_TP")
    d_iw = feat.get("d_IW")
    d_mw = feat.get("d_MW")
    d_rw = feat.get("d_RW")
    t_ext = feat.get("thumb_ext")
    p_ext = feat.get("pinky_ext")
    i_fail = geom.get("i_rule_diagnostics", {}).get("failed_conditions", [])
    
    print(f"f{fidx:02d} | final={fs:7s} | cand={cand} | d_PW={d_pw:.2f} d_TW={d_tw:.2f} d_TI={d_ti:.2f} d_TM={d_tm:.2f} t_lat={thumb_lat:.2f} ang_TP={ang_tp:.1f} | d_IW={d_iw:.2f} d_MW={d_mw:.2f} d_RW={d_rw:.2f} | i_fail={i_fail}")

