import json

fpath = 'backend/data/diagnostics/pilot_eval_29_pilot_20261111_050708_b0af91_20261111_051044_frames.jsonl'
records = []
with open(fpath, 'r', encoding='utf-8') as f:
    for line in f:
        records.append(json.loads(line))

print(f"Total frame records: {len(records)}")
if records:
    print("Keys in record:", list(records[0].keys()))

for i, r in enumerate(records):
    f_idx = r.get('frame_idx')
    exp = r.get('expected_sign')
    pred = r.get('predicted_sign')
    conf = r.get('confidence')
    ts = r.get('timestamp')
    match = r.get('is_match')
    has_lm = r.get('landmarks') is not None
    print(f"[{i:02d}] frame_idx={f_idx} | expected={exp} | pred={pred} (conf={conf}) | match={match} | ts={ts:.2f} | has_lm={has_lm}")

