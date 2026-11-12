import os
import cv2
import mediapipe as mp
import numpy as np
from backend.vision.recognizer import VSLRecognizer

videos = {
    "Đ": "backend/data/media_cache/videos/D0495.mp4",
    "Ư": "backend/data/media_cache/videos/D0515N.mp4",
    "Ơ": "backend/data/media_cache/videos/D0508N.mp4",
    "Ă": "backend/data/media_cache/videos/D0490B.mp4",
    "Â": "backend/data/media_cache/videos/D0491B.mp4",
    "Ê": "backend/data/media_cache/videos/D0497B.mp4",
    "Ô": "backend/data/media_cache/videos/D0507B.mp4",
}

rec = VSLRecognizer()
mp_hands = mp.solutions.hands

print("=" * 90)
print(f"{'Letter':<6} | {'Video':<8} | {'Total':<6} | {'Valid':<6} | {'Base Stable':<15} | {'Candidate':<12} | {'Confirmed':<12} | {'Final Result'}")
print("=" * 90)

for expected_letter, vpath in videos.items():
    if not os.path.exists(vpath):
        print(f"{expected_letter:<6} | {os.path.basename(vpath):<8} | File Not Found!")
        continue

    cap = cv2.VideoCapture(vpath)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    dt = 1.0 / fps

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    session_id = f"replay_{expected_letter}_{os.path.basename(vpath)}"
    rec.reset_diacritic_fsm(session_id)

    frame_idx = 0
    valid_count = 0
    base_stable_frame = None
    base_stable_sign = None
    candidate_frame = None
    candidate_sign = None
    confirmed_frame = None
    confirmed_sign = None
    confirmed_source = None
    all_predictions = []

    t = 0.0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        lms = None
        if results.multi_hand_landmarks:
            lms = [[lm.x, lm.y, lm.z] for lm in results.multi_hand_landmarks[0].landmark]
            valid_count += 1

        res = rec.predict_landmarks(lms, mode="letters", session_id=session_id, timestamp=t)
        sign = res.get("sign")
        src = res.get("source")
        all_predictions.append((frame_idx, sign, src))

        fsm = rec.diacritic_fsms.get(session_id)
        if fsm is not None:
            if base_stable_frame is None and fsm.state in ("BASE_STABLE", "DIACRITIC_CANDIDATE", "COOLDOWN"):
                base_stable_frame = frame_idx
                base_stable_sign = fsm.current_base
            if candidate_frame is None and fsm.state == "DIACRITIC_CANDIDATE":
                candidate_frame = frame_idx
                candidate_sign = fsm.diacritic_candidate
            if confirmed_frame is None and (fsm.state == "COOLDOWN" or sign == expected_letter):
                confirmed_frame = frame_idx
                confirmed_sign = sign
                confirmed_source = src

        t += dt
        frame_idx += 1

    cap.release()
    hands.close()

    base_str = f"f{base_stable_frame} ({base_stable_sign})" if base_stable_frame is not None else "None"
    cand_str = f"f{candidate_frame} ({candidate_sign})" if candidate_frame is not None else "None"
    conf_str = f"f{confirmed_frame} ({confirmed_sign})" if confirmed_frame is not None else "None"

    # Status summary
    status = "SUCCESS" if confirmed_sign == expected_letter else "NOT_CONFIRMED"
    print(f"{expected_letter:<6} | {os.path.basename(vpath):<8} | {frame_idx:<6} | {valid_count:<6} | {base_str:<15} | {cand_str:<12} | {conf_str:<12} | {status} (src={confirmed_source})")

    # Diagnostic summary of signs
    from collections import Counter
    c = Counter([s for _, s, _ in all_predictions if s != "NEUTRAL"])
    print(f"       -> Top signs: {dict(c.most_common(5))}")

