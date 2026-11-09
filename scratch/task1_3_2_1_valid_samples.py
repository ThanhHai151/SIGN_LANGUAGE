import json

for letter in ['P', 'K', 'Q', 'D']:
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    valid_samples = [s for s in samples if s.get('landmarks') is not None]
    print(f"\n==============================")
    print(f"FIXTURE {letter}.json: {len(valid_samples)} samples with landmarks:")
    print(f"==============================")
    for s in valid_samples:
        f_idx = s.get('frame_index') if s.get('frame_index') is not None else s.get('frame_idx')
        phase = s.get('phase')
        print(f"  Frame {f_idx:2d} | JSON phase: {phase}")

