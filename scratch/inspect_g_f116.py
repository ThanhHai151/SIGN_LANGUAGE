import cv2
import json

# Check landmarks at frame 116 of G.json
with open('backend/data/alphabet_v2/static_landmarks/G.json', 'r', encoding='utf-8') as f:
    g_data = json.load(f)

sample_116 = next((s for s in g_data['samples'] if s.get('frame_index') == 116), None)
if sample_116:
    print('Sample 116 phase:', sample_116.get('phase'))
    lm = sample_116['landmarks']
    img = cv2.imread('scratch/d0498b_f116.jpg')
    h, w, _ = img.shape
    for p in lm:
        cv2.circle(img, (int(p[0]*w), int(p[1]*h)), 3, (0, 0, 255), -1)
    cv2.imwrite('scratch/d0498b_f116_annotated.jpg', img)
    print('Annotated image saved.')

