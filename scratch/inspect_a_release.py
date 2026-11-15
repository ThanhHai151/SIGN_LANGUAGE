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
    if 30 <= idx <= 55:
        print(f"Frame {idx:2d} ({t:.3f}s): sign={res.get('sign')} confirmed={res.get('confirmed')} new_tok={res.get('new_token')} release={getattr(rec, '_seen_neutral_release', None)} n_cnt={getattr(rec, '_neutral_release_count', None)}")
