import cv2, mediapipe as mp, numpy as np
from backend.vision.alphabet_classifier import VSLAlphabetClassifier

video_path = "backend/data/vsl_videos/D0511.mp4"
cap = cv2.VideoCapture(video_path)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

classifier = VSLAlphabetClassifier()

frame_idx = 0
r_frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    h, w, c = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)
    if results.multi_hand_landmarks:
        lms = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
        feat = classifier.extract_features(lms)
        diag = classifier.evaluate_rule_r_diagnostics(feat)
        if 20 <= frame_idx <= 45:
            print(f"Frame {frame_idx:2d}: d_IW={feat.get('d_IW'):.2f}, d_MW={feat.get('d_MW'):.2f}, idx_horiz={feat.get('idx_horiz_ratio'):.2f}, idx_dy={feat.get('idx_dy'):.2f}, cross={feat.get('cross_indicator'):.4f}, matched={diag.get('matched')}, failed={diag.get('failed_conditions')}")
    frame_idx += 1

cap.release()
hands.close()

