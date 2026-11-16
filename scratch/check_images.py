import cv2
import json

for vid, fn, fidx in [('T', 'backend/data/media_cache/inspect_frames/D0513B_f34.jpg', 34),
                      ('U', 'backend/data/media_cache/inspect_frames/D0514_f38.jpg', 38),
                      ('S', 'backend/data/media_cache/inspect_frames/D0512_f36.jpg', 36)]:
    img = cv2.imread(fn)
    print(f"Loaded {fn}: shape={img.shape}")

