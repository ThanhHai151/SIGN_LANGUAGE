import json
import sys
sys.path.insert(0, ".")

from backend.vision.diacritic_state_machine import DiacriticStateMachine, COMPETING_RULES_MAP
from backend.vision.recognizer import VSLRecognizer

rec = VSLRecognizer()

# Temporarily patch recognizer's FSM class or test directly
with open('backend/data/alphabet_v2/motion_sequences/Ô.json', encoding='utf-8') as f:
    d = json.load(f)

seq = d['sequence']
print("Running full recognizer simulation on Ô.json...")

session_id = "test_o_sim"
rec.reset()
rec.reset_diacritic_fsm(session_id)

tokens = []
diac_frames = []

for item in seq:
    idx = item.get('frame_index')
    t = item.get('timestamp_sec')
    lms = item.get('landmarks')
    res = rec.predict_landmarks(lms, mode='letters', session_id=session_id, trial_id=session_id, frame_idx=idx, timestamp=t, expected_sign='Ô')
    sign = res.get('sign')
    tok = res.get('new_token')
    src = res.get('source')
    diag = res.get('classifier_details', {}).get('diacritic_fsm', {})
    if tok:
        tokens.append((idx, round(t, 2), tok, src))
    if sign == 'Ô' or diag.get('diacritic_detected') == 'Ô':
        diac_frames.append((idx, round(t, 2), sign, diag.get('state'), diag.get('reason')))

print("Tokens emitted:", tokens)
print("Diacritic Ô frames count:", len(diac_frames))
if diac_frames:
    print("First 3 Ô frames:", diac_frames[:3])

