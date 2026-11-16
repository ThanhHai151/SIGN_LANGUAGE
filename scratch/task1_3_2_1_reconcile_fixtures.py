import json

for letter in ['P', 'K', 'Q', 'D']:
    path = f'backend/data/alphabet_v2/static_landmarks/{letter}.json'
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    samples = data.get('samples', [])
    print(f"==================================================")
    print(f"FIXTURE {letter}.json: Total samples in JSON = {len(samples)}")
    print(f"==================================================")
    for idx, s in enumerate(samples):
        f_idx = s.get('frame_index') if s.get('frame_index') is not None else s.get('frame_idx')
        phase = s.get('phase')
        has_lms = s.get('landmarks') is not None
        print(f"Sample #{idx:2d} | frame_index={str(f_idx):5s} | phase={str(phase):12s} | has_landmarks={has_lms}")

