import glob, json, os

diag_files = sorted(glob.glob('backend/data/diagnostics/*.jsonl'))
print('Diagnostic files:', diag_files)
latest = 'backend/data/diagnostics/pilot_pilot_20261106_2347.jsonl'
if not os.path.exists(latest):
    # Find matching
    matches = [f for f in diag_files if '2347' in f]
    if matches:
        latest = matches[0]
    elif diag_files:
        latest = diag_files[-1]

print('Using file:', latest)
with open(latest, 'r', encoding='utf-8') as f:
    lines = [line.strip() for line in f if line.strip()]

print(f'Total lines in log: {len(lines)}')

for i, l in enumerate(lines[:35]):
    try:
        data = json.loads(l)
    except Exception as e:
        print(f"Error parsing line {i}: {e}")
        continue
    exp = data.get('expected_sign')
    pred = data.get('sign') or data.get('final_sign') or data.get('letter')
    geom = data.get('geometric_rule', {})
    s_diag = geom.get('s_rule_diagnostics')
    feats = geom.get('features', {})
    print(f"Line {i+1}: exp={exp}, pred={pred}, geom_letter={geom.get('letter')}")
    if s_diag:
        print(f"   s_diag: matched={s_diag.get('matched')}, failed={s_diag.get('failed_conditions')}")
    if feats:
        print(f"   key feats: d_TW={feats.get('d_TW')}, d_TI={feats.get('d_TI')}, d_TM={feats.get('d_TM')}, d_IW={feats.get('d_IW')}, d_MW={feats.get('d_MW')}, thumb_lat={feats.get('thumb_lat')}, ext={(feats.get('i_ext'), feats.get('m_ext'), feats.get('r_ext'), feats.get('p_ext'))}")

