import json
import glob
import os

files = glob.glob('backend/data/diagnostics/*0150a8*frames.jsonl')
q_records = []
for f in sorted(files):
    with open(f, 'r', encoding='utf-8') as jf:
        for line in jf:
            rec = json.loads(line)
            tgt = rec.get('expected_sign') or rec.get('clean_expected_sign') or rec.get('target')
            if tgt == 'Q':
                rec['_src_file'] = os.path.basename(f)
                q_records.append(rec)

print(f"Total Q records found: {len(q_records)}")
for i, r in enumerate(q_records):
    final_s = r.get('final_sign')
    final_c = r.get('final_conf')
    src = r.get('prediction_source')
    gat = r.get('gating_status')
    rej = r.get('reject_reason')
    head_p = r.get('head_payload', {})
    geom_p = r.get('geom_payload', {})
    has_lm = bool(r.get('p_prim') or r.get('landmarks'))
    phase = r.get('phase')
    top_k = head_p.get('top_k', [])
    print(f"[{i:02d}] phase={phase} | final={final_s} (conf={final_c}) | src={src} | gat={gat} | head={head_p.get('raw_predicted_letter')} (conf={head_p.get('confidence')}) | geom={geom_p.get('status')} | rej={rej} | has_lm={has_lm}")
    if top_k:
        print(f"     top_k: {top_k[:3]}")

