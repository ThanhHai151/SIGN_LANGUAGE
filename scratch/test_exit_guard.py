import sys, json
sys.path.insert(0, '.')
from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()
with open('backend/data/alphabet_v2/motion_sequences/Ă.json', encoding='utf-8') as f:
    data = json.load(f)

for item in data['sequence']:
    idx = item.get('frame_index')
    t = item.get('timestamp_sec', 0.0)
    lms = item.get('landmarks')
    res = rec.predict_landmarks(lms, mode='letters', session_id='test_a', trial_id='test_a', frame_idx=idx, timestamp=t, expected_sign='Ă')
    if res.get('confirmed'):
        print(f"Confirmed: frame={idx} ({t:.3f}s) token={res.get('new_token')} rep={res.get('replaced_token')}")

print("Final sentence tokens:", rec.sentence_tokens)

