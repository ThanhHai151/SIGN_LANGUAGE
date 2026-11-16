import json
import os
import cv2

# Check R (D0511) and X (D0517B) videos
for vid_name in ['D0510.mp4', 'D0511.mp4', 'D0512.mp4', 'D0517B.mp4', 'D0518N.mp4']:
    vpath = os.path.join('backend/data/media_cache/videos', vid_name)
    if os.path.exists(vpath):
        cap = cv2.VideoCapture(vpath)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_f = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"{vid_name}: {w}x{h}, {total_f} frames, {fps:.1f} fps")
        cap.release()
    else:
        print(f"{vid_name}: NOT FOUND")

# Check R.json and X.json structure
for let in ['R', 'X']:
    jpath = f'backend/data/alphabet_v2/static_landmarks/{let}.json'
    with open(jpath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    print(f"\n{let}.json keys:", list(data.keys()))
    print(f"{let}.json metadata:", {k: data[k] for k in ['letter', 'type', 'video_id', 'total_frames', 'valid_frames'] if k in data})
    samples = data.get('samples', [])
    phases = {}
    for s in samples:
        ph = s.get('phase', 'unknown')
        phases[ph] = phases.get(ph, 0) + 1
    print(f"{let}.json phases:", phases)
