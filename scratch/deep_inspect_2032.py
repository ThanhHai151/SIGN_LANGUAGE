import sys
sys.path.insert(0, ".")
import json

def inspect_features_deep(fname):
    print(f"\n=======================================================")
    print(f"DEEP INSPECTION OF: {fname}")
    print(f"=======================================================")
    with open(fname, "r", encoding="utf-8") as f:
        lines = [json.loads(line) for line in f if line.strip()]

    for idx, d in enumerate(lines):
        tid = d.get("trial_id")
        fidx = d.get("frame_idx")
        fs = d.get("final_sign")
        geom = d.get("geometric", {})
        cand = geom.get("matched_candidates", [])
        feat = geom.get("features")
        if not feat:
            continue
        
        # Extract features
        d_pw = feat.get("d_PW")
        d_iw = feat.get("d_IW")
        d_mw = feat.get("d_MW")
        d_rw = feat.get("d_RW")
        d_tw = feat.get("d_TW")
        d_ti = feat.get("d_TI")
        d_tm = feat.get("d_TM")
        thumb_lat = feat.get("thumb_lat")
        ang_tp = feat.get("ang_TP")
        p_curl = feat.get("pinky_curl")
        p_ext = feat.get("pinky_ext")
        t_ext = feat.get("thumb_ext")
        i_ext = feat.get("index_ext")
        m_ext = feat.get("middle_ext")
        r_ext = feat.get("ring_ext")
        
        # Rule I diagnostic
        i_diag = geom.get("i_rule_diagnostics", {})
        i_failed = i_diag.get("failed_conditions", []) if i_diag else []
        
        # Check if Y conditions would match
        # Y: p_ext or p_curl<=0.12, d_pw>=1.30, not i_ext and not m_ext and not r_ext, d_iw<=0.90, d_mw<=0.80, d_rw<=0.80
        # and d_tw>=1.05 and d_ti>=0.40 and (thumb_lat>=0.36 or ang_tp>=33.0)
        y_cond_thumb = (d_tw is not None and d_tw >= 1.05) and (d_ti is not None and d_ti >= 0.40) and ((thumb_lat is not None and thumb_lat >= 0.36) or (ang_tp is not None and ang_tp >= 33.0))
        
        # Let's see
        print(f"[{idx:03d}|{tid}|f{fidx}] final={fs:7s} | cand={cand} | d_PW={d_pw:.2f} d_TW={d_tw:.2f} d_TI={d_ti:.2f} d_TM={d_tm:.2f} t_lat={thumb_lat:.2f} ang_TP={ang_tp:.1f} | p_ext={p_ext} t_ext={t_ext} | i_fail={i_failed} | y_thumb={y_cond_thumb}")

inspect_features_deep("backend/data/diagnostics/pilot_pilot_20261105_2032.jsonl")

