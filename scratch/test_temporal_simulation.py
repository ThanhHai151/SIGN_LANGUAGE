import os
import sys
import collections

root_dir = os.path.abspath(".")
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

# Create dummy hand landmarks (21 points)
dummy_landmarks = [[0.5, 0.5 + i*0.01, 0.0] for i in range(21)]

session_id = "test_sess"
trial_7 = "trial_rep07"
trial_8 = "trial_rep08"

# Mock classify_single_hand or alphabet classifier to return Q with 0.82
# Instead of full neural net inference, let's trace lines 980-1008 of recognizer.py
print("Testing trial isolation in recognizer...")

# Step 1: Call predict_landmarks for trial_7 5 times
for i in range(5):
    res = rec.predict_landmarks(dummy_landmarks, session_id=session_id, trial_id=trial_7, mode="letters")

print(f"Keys in _pilot_temporal_states: {list(rec._pilot_temporal_states.keys())}")
t7_state = rec._pilot_temporal_states.get((session_id, trial_7, "letters"))
print(f"Trial 7 state: history_len={len(t7_state['prediction_history'])}, last_conf={t7_state['last_confirmed_sign']}")
print(f"Global recognizer state: history_len={len(rec.prediction_history)}, last_conf={rec.last_confirmed_sign}")

# Step 2: Now call predict_landmarks for trial_8 for the FIRST frame
res8_1 = rec.predict_landmarks(dummy_landmarks, session_id=session_id, trial_id=trial_8, mode="letters")
t8_state = rec._pilot_temporal_states.get((session_id, trial_8, "letters"))
print(f"Trial 8 state after 1 frame: history_len={len(t8_state['prediction_history'])}, last_conf={t8_state['last_confirmed_sign']}")
print(f"Trial 8 frame 1 result: confirmed={res8_1.get('confirmed')}, new_token={res8_1.get('new_token')}")

