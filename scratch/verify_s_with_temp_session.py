import sys, os
sys.path.insert(0, os.path.abspath("."))
import cv2
import numpy as np
from backend.tools.temp_frame_manager import TemporaryFrameSession, purge_orphaned_sessions
from backend.vision.alphabet_classifier import AlphabetClassifier
from backend.vision.recognizer import VSLRecognizer

print("=== TESTING TEMPORARY FRAME MANAGER WITH LETTER S VIDEO D0512.MP4 ===")

video_path = "backend/data/media_cache/videos/D0512.mp4"
assert os.path.exists(video_path), f"Video not found: {video_path}"

clf = AlphabetClassifier()
session_dir_path = None

with TemporaryFrameSession(session_prefix="s_verify_session_") as session:
    session_dir_path = session.session_dir
    print(f"Session directory created: {session_dir_path}")
    assert os.path.exists(session_dir_path), "Session dir should exist"

    # Extract stable frames 20, 28, 36
    cap = cv2.VideoCapture(video_path)
    target_frames = [20, 28, 36]
    extracted_paths = []

    f_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if f_idx in target_frames:
            saved_path = session.save_frame(frame, f"frame_{f_idx:03d}.jpg")
            extracted_paths.append(saved_path)
            print(f"  Extracted frame {f_idx} to temporary path: {saved_path}")
            assert os.path.exists(saved_path), f"Frame {saved_path} must exist"
        f_idx += 1
    cap.release()

    print(f"Total temporary frames saved in session: {len(session._created_files)}")
    assert len(session._created_files) == 3

# Now we have exited the with block (context manager has run cleanup)
print("\nExited TemporaryFrameSession context block.")
print(f"Checking if session directory still exists: {session_dir_path}")
assert not os.path.exists(session_dir_path), "Session directory must be completely cleaned up!"
for p in extracted_paths:
    assert not os.path.exists(p), f"Temporary frame {p} must be deleted!"

print("CLEANUP VERIFICATION PASSED: Zero residual frame clutter!")
