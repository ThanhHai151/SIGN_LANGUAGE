import json

for letter in ['P', 'Q']:
    with open(f'backend/data/alphabet_v2/static_landmarks/{letter}.json', 'r', encoding='utf-8') as f:
        d = json.load(f)
    print(f'=== Letter {letter} ({d.get("video_id")}) ===')
    stable_frames = []
    for s in d['samples']:
        if s.get('phase') == 'stable_pose' and s.get('landmarks'):
            stable_frames.append(s.get('frame_idx'))
    print(f'Stable frames with landmarks (total {len(stable_frames)}): {stable_frames}')

