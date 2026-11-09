import json

for letter in ['O', 'S', 'P', 'Q']:
    json_path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(json_path, 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(f'=== Letter {letter} ===')
    print('  Video ID:', d.get('video_id'))
    print('  Total samples in JSON:', len(d.get('samples', [])))
    phases = {}
    valid_cnt = 0
    for s in d.get('samples', []):
        p = s.get('phase', 'unknown')
        phases[p] = phases.get(p, 0) + 1
        if s.get('landmarks'):
            valid_cnt += 1
    print('  Phases:', phases)
    print(f'  Valid landmarks: {valid_cnt}/{len(d.get("samples", []))}')

