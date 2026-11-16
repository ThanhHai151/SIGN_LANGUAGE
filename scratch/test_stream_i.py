import sys
sys.path.insert(0, ".")
import json
from backend.vision.recognizer import VSLRecognizer

recognizer = VSLRecognizer()
recognizer.reset()
recognizer.clear_sentence()

# Load stable sample from I.json
data = json.load(open('backend/data/alphabet_v2/static_landmarks/I.json', encoding='utf-8'))
s24 = [s for s in data['samples'] if s['frame_index'] == 24][0]
lm24 = s24['landmarks']

print("=== Simulating 10 consecutive frames of Letter I ===")
confirmed_at = None
for f_idx in range(1, 11):
    res = recognizer.predict_landmarks(
        landmarks_list=lm24,
        session_id="test_sim_i",
        mode="letters",
        frame_idx=f_idx,
    )
    sign = res.get("sign")
    conf = res.get("confidence")
    src = res.get("source")
    is_conf = res.get("confirmed")
    tok = res.get("new_token")
    sent = res.get("sentence")
    print(f"Frame {f_idx:02d}: sign={sign} conf={conf} src={src} confirmed={is_conf} token={tok} sentence='{sent}'")
    if is_conf and confirmed_at is None:
        confirmed_at = f_idx

print(f"\nResult: Letter I confirmed at Frame {confirmed_at} (~{confirmed_at * 66}ms latency)!")
assert confirmed_at is not None and confirmed_at <= 5, f"Confirmation took too long: {confirmed_at}"
assert res.get("sentence") == "I", f"Sentence token mismatch: {res.get('sentence')}"
assert res.get("source") == "geometric_rule", f"Source mismatch: {res.get('source')}"
print("Stream simulation for Letter I: PASSED 100%!")
