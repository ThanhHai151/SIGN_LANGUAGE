import sys
sys.path.insert(0, ".")
import json
from backend.vision.recognizer import VSLRecognizer

recognizer = VSLRecognizer()
recognizer.reset()
recognizer.clear_sentence()

# Load stable sample from Y.json and I.json
data_y = json.load(open('backend/data/alphabet_v2/static_landmarks/Y.json', encoding='utf-8'))
s_y = [s for s in data_y['samples'] if s['frame_index'] == 24][0]['landmarks']

data_i = json.load(open('backend/data/alphabet_v2/static_landmarks/I.json', encoding='utf-8'))
s_i = [s for s in data_i['samples'] if s['frame_index'] == 24][0]['landmarks']

print("=== Simulating 3 transient frames of Y followed by Letter I ===")
# 3 frames of Y
for f_idx in range(1, 4):
    res = recognizer.predict_landmarks(landmarks_list=s_y, session_id="test_trans", mode="letters", frame_idx=f_idx)
    print(f"Transient Y f{f_idx}: sign={res['sign']} conf={res['confidence']} confirmed={res['confirmed']}")

# Now user forms I
confirmed_at = None
for f_idx in range(4, 12):
    res = recognizer.predict_landmarks(landmarks_list=s_i, session_id="test_trans", mode="letters", frame_idx=f_idx)
    print(f"I Frame f{f_idx}: sign={res['sign']} conf={res['confidence']} confirmed={res['confirmed']} token={res['new_token']} sentence='{res['sentence']}'")
    if res['confirmed'] and confirmed_at is None:
        confirmed_at = f_idx

print(f"\nResult: Letter I confirmed at Frame {confirmed_at} despite initial Y frames!")
assert confirmed_at is not None and confirmed_at <= 8
assert res['sentence'] == "I"
print("Transition test passed 100%!")

