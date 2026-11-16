import os

for f in ['backend/training/download_dataset.py', 'backend/training/extract_video_landmarks.py']:
    if os.path.exists(f):
        print(f"=== {f} ===")
        with open(f, 'r', encoding='utf-8') as fp:
            for l in fp.readlines()[:40]:
                print(l, end='')

