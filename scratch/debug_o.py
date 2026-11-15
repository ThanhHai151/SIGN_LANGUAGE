import sys
import os
sys.path.insert(0, ".")
os.environ["VSL_DIACRITIC_DEBUG"] = "1"
os.environ["VSL_PREFETCH"] = "0"
import json
from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()
with open('backend/data/alphabet_v2/motion_sequences/Ô.json', encoding='utf-8') as f:
    data = json.load(f)

for item in data['sequence']:
    idx = item.get('frame_index')
    t = item.get('timestamp_sec', 0.0)
    lms = item.get('landmarks')
    res = rec.predict_landmarks(lms, mode='letters', session_id='test_o', trial_id='test_o', frame_idx=idx, timestamp=t, expected_sign='Ô')
    diag = res.get('classifier_details', {}).get('diacritic_fsm', {})
    st = diag.get('state')
    diac = diag.get('diacritic_detected')
    re = diag.get('reason')
    print(f"Frame {idx:2d} ({t:.3f}s): sign={res.get('sign')} state={st} diac={diac} reason={re}")
