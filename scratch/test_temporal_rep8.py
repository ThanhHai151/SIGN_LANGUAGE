import os
import sys
import json

root_dir = os.path.abspath(".")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

frames_path = "backend/data/diagnostics/pilot_eval_29_pilot_20261111_172142_27052b_20261111_172433_frames.jsonl"
with open(frames_path, "r", encoding="utf-8") as f:
    frames = [json.loads(line) for line in f]

session_id = "test_session"
trial_id_7 = "trial_rep07"
trial_id_8 = "trial_rep08"

# Run 9 frames of rep 7
for idx in range(59, 68):
    lm = frames[idx].get("landmarks")
    if lm:
        res7 = rec.predict_landmarks(landmarks_list=lm, mode="letters", session_id=session_id, trial_id=trial_id_7)

print(f"After Rep 7, temporal state keys: {list(rec._pilot_temporal_states.keys())}")
print(f"Rep 7 history len: {len(rec._pilot_temporal_states[(session_id, trial_id_7, 'letters')]['prediction_history'])}")

# Now run frame 69 (Rep 8 frame 1) with trial_id_8
lm8_1 = frames[68].get("landmarks")
if lm8_1:
    res8_1 = rec.predict_landmarks(landmarks_list=lm8_1, mode="letters", session_id=session_id, trial_id=trial_id_8)
    print(f"Rep 8 frame 1 result: confirmed={res8_1.get('confirmed')}, new_token={res8_1.get('new_token')}, sign={res8_1.get('sign')}")
    print(f"Rep 8 temporal state: {rec._pilot_temporal_states.get((session_id, trial_id_8, 'letters'))}")
